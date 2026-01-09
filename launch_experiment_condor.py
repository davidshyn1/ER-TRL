
import pathlib
import numpy as np
import torch
import random
import hydra
import omegaconf
import os 
import sys

# Patch mujoco_py generated directory before any imports to avoid lock file conflicts
# This allows each condor job to use its own generated directory
job_generated_dir = os.environ.get('MUJOCO_PY_JOB_GENERATED_DIR')
if job_generated_dir and os.path.exists(job_generated_dir):
    # Import the patch module created in condor.sh
    # This will patch mujoco_py to use job-specific generated directory
    try:
        import mujoco_patch_init
        # Ensure patch is applied
        mujoco_patch_init.apply_mujoco_patch()
    except ImportError:
        # If patch module doesn't exist, continue anyway
        pass

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.agents.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.agents.csro import CSROPrediction
from rlkit.torch.agents.er_trl import ERTRL

from rlkit.torch.agents.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
import pdb



def global_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_max_trajs_per_task_combinations():
    """
    Generate all max_trajs_per_task combinations.
    Modify the list below to add/remove max_trajs_per_task values.
    """
    # Define max_trajs_per_task search space
    # You can customize this list to include only the values you want
    # Examples:
    # max_trajs_list = [None, 10, 20, 50, 100]  # None means no limit
    # max_trajs_list = [10, 20, 30, 40, 50]
    max_trajs_list = [20, 50, 100, 200, 500]  # Modify this list as needed
    return max_trajs_list


def experiment(algorithm, cfg):
    DEBUG = cfg.util_params.debug
    os.environ['DEBUG'] = str(int(DEBUG))
    
    exp_id = 'debug' if DEBUG else cfg.util_params.exp_name
    experiment_log_dir, wandb_logger = setup_logger(
        cfg.env_name,
        variant= omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        exp_id=exp_id,
        base_log_dir=cfg.util_params.base_log_dir, 
        seed=cfg.seed,
        snapshot_mode="gap_and_last",
        snapshot_gap=5, 
        use_wandb=cfg.use_wandb,
    )

    # optionally save eval trajectories as pkl files
    if cfg.algo_params.dump_eval_paths:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train(wandb_logger)
    if wandb_logger:
        wandb_logger.finish()

def initialize(cfg, ):
    # create multi-task environment and sample tasks, normalize obs if provided with 'normalizer.npz'
    env = NormalizedBoxEnv(ENVS[cfg.env_name](**dict(cfg.env_params)))
    global_seed(cfg.seed)
    env.seed(cfg.seed)

    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = cfg.latent_size
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if cfg.algo_params.use_next_obs_in_context else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if cfg.algo_params.use_information_bottleneck else latent_dim
    net_size = cfg.net_size
    recurrent = cfg.algo_params.recurrent
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=cfg.encoder_dims,
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        output_activation=torch.tanh,
    )

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )

    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    # critic network for divergence in dual form (see BRAC paper https://arxiv.org/abs/1911.11361)
    c = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1
    )

    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **dict(cfg.algo_params)
    )
    task_modes = env.task_modes()
    if cfg.algo_type == 'CSROPrediction':
        if cfg.algo_params.use_club_sa:
            club_input_dim = obs_dim + action_dim
        else:
            club_input_dim = obs_dim + action_dim + reward_dim if cfg.algo_params.use_next_obs_in_context else obs_dim + action_dim

        club_model = encoder_model(
            hidden_sizes=cfg.encoder_dims,
            input_size=club_input_dim,
            output_size=latent_dim * 2,
            output_activation=torch.tanh,
            # output_activation_half=True
        )

        decoder = FlattenMlp(
            hidden_sizes = [net_size, net_size],
            input_size = obs_dim + action_dim + latent_dim,
            output_size= obs_dim + 1,
        )

        algorithm = CSROPrediction(
            env=env,
            train_tasks=task_modes['train'],
            eval_tasks=task_modes['moderate'],
            extreme_tasks=task_modes['extreme'],
            nets=[agent, qf1, qf2, vf, c, club_model],
            latent_dim=latent_dim,
            decoder=decoder,
            **dict(cfg.algo_params)
        )

    if cfg.algo_type == 'ERTRL':
        generator = FlattenMlp(
            hidden_sizes = cfg.encoder_dims,
            input_size = latent_dim+obs_dim+cfg.algo_params.generator_dim, 
            output_size = action_dim,
            output_activation = torch.tanh
         )
        discriminator = FlattenMlp(
            hidden_sizes = [net_size, net_size],
            input_size = action_dim+obs_dim+latent_dim,
            output_size = 1,
            output_activation = torch.sigmoid,
        )

        algorithm = ERTRL(
            env=env,
            train_tasks=task_modes['train'],
            eval_tasks=task_modes['moderate'],
            extreme_tasks=task_modes['extreme'],
            nets=[agent, qf1, qf2, vf, c, generator, discriminator],
            latent_dim=latent_dim,
            **dict(cfg.algo_params)
        )
    # optional GPU mode
    ptu.set_gpu_mode(cfg.use_gpu, cfg.gpu_id)
    if ptu.gpu_enabled():
        algorithm.to()
    return algorithm


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="experiment")
def main(cfg):
    # Set HYDRA_FULL_ERROR for complete stack trace
    os.environ['HYDRA_FULL_ERROR'] = '1'
    
    # Get PID from Hydra config (passed as +PID=value from condor.sh)
    # Hydra will add it to cfg if passed as +PID=value
    pid = None
    
    # Check for PID in Hydra config (passed as +PID=value)
    if hasattr(cfg, 'PID'):
        try:
            pid = int(cfg.PID) if cfg.PID is not None else None
        except (ValueError, TypeError):
            pid = None
    
    # Fallback: check environment variable
    if pid is None and 'PID' in os.environ:
        try:
            pid = int(os.environ['PID'])
        except (ValueError, TypeError):
            pass
    
    # Fallback: check CONDOR_PROCESS_ID (Condor's built-in variable)
    if pid is None and 'CONDOR_PROCESS_ID' in os.environ:
        try:
            pid = int(os.environ['CONDOR_PROCESS_ID'])
        except (ValueError, TypeError):
            pass
    
    # If PID is provided, use it to select max_trajs_per_task from combinations
    if pid is not None:
        max_trajs_list = generate_max_trajs_per_task_combinations()
        if pid >= len(max_trajs_list):
            raise ValueError(f"PID {pid} is out of range. Total max_trajs_per_task combinations: {len(max_trajs_list)}. Valid PID range: 0 to {len(max_trajs_list)-1}")
        
        selected_max_trajs = max_trajs_list[pid]
        cfg.algo_params.max_trajs_per_task = selected_max_trajs
        
        print("=" * 60)
        print(f"PID {pid}: Selected max_trajs_per_task={cfg.algo_params.max_trajs_per_task} from {len(max_trajs_list)} combinations")
        print(f"Available max_trajs_per_task list: {max_trajs_list}")
        print(f"PID {pid} -> max_trajs_per_task {cfg.algo_params.max_trajs_per_task} (index {pid} in list)")
        print("=" * 60)
    else:
        # Fallback: use max_trajs_per_task from config or command line
        if not hasattr(cfg.algo_params, 'max_trajs_per_task') or cfg.algo_params.max_trajs_per_task is None:
            # Keep the default from config
            pass
    
    print(f"Max trajs per task: {cfg.algo_params.max_trajs_per_task}")
    print(f"Env name: {cfg.env_name}")
    print(f"Algo type: {cfg.algo_type}")
    print(f"Seed: {cfg.seed}")
    
    algorithm = initialize(cfg)
    experiment(algorithm, cfg)

if __name__ == "__main__":
    main()

