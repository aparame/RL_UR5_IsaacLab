#!/usr/bin/env python3

"""Script to train UR5 pick and place with SKRL."""

import argparse
import sys
import os
import random
from datetime import datetime

from isaaclab.app import AppLauncher
# Import SKRL adapter to register custom feature extractors


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train UR5 pick and place RL agent with SKRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="RL-Ur5-PickAndPlace-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random).")
parser.add_argument("--distributed", action="store_true", default=False, help="Run with multiple GPUs or nodes.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="ML framework for training."
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "IPPO", "MAPPO", "AMP","SAC"],
    help="RL algorithm for training."
)

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
# Parse arguments
args_cli, hydra_args = parser.parse_known_args()
# Enable cameras for video recording
if args_cli.video:
    args_cli.enable_cameras = True

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import necessary modules
import gymnasium as gym
import numpy as np
import skrl
from packaging import version

# Check minimum supported SKRL version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported SKRL version: {skrl.__version__}. "
        f"Install supported version: 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

# Import framework-specific modules
if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

# Import Isaac Lab utilities
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import our tasks package to register environments
import RL_UR5  # noqa: F401

# Config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"

# Import Hydra task config utility
from isaaclab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train UR5 pick and place task with SKRL agent."""
    
    # Override configs with CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Multi-GPU training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    
    # Set max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    
    # Configure ML framework
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"
    
    # Handle random seed
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    
    # Set agent and environment seed
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    
    # Setup logging directories
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # Create experiment name
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Experiment name from command line: {log_dir}")
    
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    
    # Update agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    
    # Full log directory path
    full_log_dir = os.path.join(log_root_path, log_dir)
    
    # Dump configuration files
    dump_yaml(os.path.join(full_log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(full_log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(full_log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(full_log_dir, "params", "agent.pkl"), agent_cfg)
    
    # Get checkpoint path for resuming training
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    
    # Add video recording wrapper if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(full_log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap environment for SKRL
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    
    # Create SKRL runner
    runner = Runner(env, agent_cfg)
    
    # Load checkpoint if specified
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
    
    # Run training
    runner.run()
    
    # Close environment
    env.close()

if __name__ == "__main__":
    # Run main function
    main()
    # Close simulator
    simulation_app.close()