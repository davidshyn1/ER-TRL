import hydra 
import os
import sys
import itertools
import argparse

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

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs import ENVS
from rlkit.torch.pytorch_sac.train import Workspace
import pdb


def generate_goal_idx_combinations():
    """
    Generate all goal_idx combinations using itertools.
    Modify the list below to add/remove goal_idx values.
    """
    # Define goal_idx search space
    # You can customize this list to include only the goal_idx values you want
    goal_indices = list(range(20, 40))  # 0-39 (40 tasks)
    # Or specify specific indices: goal_indices = [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 39]
    # goal_indices = [30,31,32,33,34,36,37,38,39]
    # goal_indices = [35,36,37,38,39]
    return goal_indices


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="data_collection")
def main(cfg):
    # Set HYDRA_FULL_ERROR for complete stack trace
    os.environ['HYDRA_FULL_ERROR'] = '1'
    
    # Get PID from Hydra config (passed as +PID=value from condor.sh)
    # Hydra sill add it to cfg if passed as +PID=value
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
    
    # If PID is provided, use it to select goal_idx from combinations
    if pid is not None:
        goal_indices = generate_goal_idx_combinations()
        if pid >= len(goal_indices):
            raise ValueError(f"PID {pid} is out of range. Total goal_idx combinations: {len(goal_indices)}. Valid PID range: 0 to {len(goal_indices)-1}")
        
        selected_goal_idx = goal_indices[pid]
        cfg.goal_idx = selected_goal_idx
        
        print("=" * 60)
        print(f"PID {pid}: Selected goal_idx={cfg.goal_idx} from {len(goal_indices)} combinations")
        print(f"Available goal_idx list: {goal_indices}")
        print(f"PID {pid} -> goal_idx {cfg.goal_idx} (index {pid} in list)")
        print("=" * 60)
    else:
        # Fallback: use goal_idx from config or command line
        if not hasattr(cfg, 'goal_idx') or cfg.goal_idx == '' or cfg.goal_idx is None:
            cfg.goal_idx = 0  # Default to 0 if not set or empty
        elif isinstance(cfg.goal_idx, str):
            try:
                cfg.goal_idx = int(cfg.goal_idx)
            except (ValueError, TypeError):
                cfg.goal_idx = 0  # Default to 0 if conversion fails
    
    print(f"Goal index: {cfg.goal_idx}")
    print(f"Env name: {cfg.env_name}")
    print(f"Env params: {cfg.env_params}")
    print(f"Seed: {cfg.seed}")
    
    env = NormalizedBoxEnv(ENVS[cfg.env_name](**dict(cfg.env_params)))
    if cfg.seed is not None:
        env.seed(cfg.seed) 
    env.reset_task(cfg.goal_idx)
    workspace = Workspace(cfg=cfg, env=env,)
    workspace.run()
    

if __name__ == '__main__':
    main()

