"""
Script to generate ground-truth depth CSV from Isaac Lab simulation.
"""

import argparse
from isaaclab.app import AppLauncher

# Create parser
parser = argparse.ArgumentParser(description="Generate depth CSV from simulation.")
# Add standard Isaac Lab args (includes --headless, --enable_cameras etc)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force enable cameras since we need them for depth generation
args.enable_cameras = True

# Launch App first
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
import os
import sys

# Add source to path if needed (simpler imports)
sys.path.append("/home/adi2440/Desktop/RL_UR5_IsaacLab/source")

# Import the task package to register environments
import RL_UR5.tasks


def main():
    # Instantiate config manually to avoid registration issues
    from RL_UR5.tasks.direct.rl_ur5.ur5_calibration_env import UR5CalibrationEnvCfg

    env_cfg = UR5CalibrationEnvCfg()

    print("[INFO] Creating environment...")
    # Pass cfg directly to env constructor via kwargs
    env = gym.make("UR5-Calibration-Depth", cfg=env_cfg, render_mode="rgb_array")

    print("[INFO] Resetting environment...")
    obs, _ = env.reset()

    # Run for a few steps to settle physics
    print("[INFO] Stepping to settle physics...")
    for _ in range(50):
        action = torch.zeros((env.unwrapped.num_envs, 6), device=env.unwrapped.device)
        obs, _, _, _, _ = env.step(action)

    # Extract depth
    # obs is a dict: {'depth': tensor(num_envs, height, width)}
    depth_tensor = obs["depth"]

    if isinstance(depth_tensor, torch.Tensor):
        depth_np = depth_tensor.cpu().numpy()
    else:
        depth_np = depth_tensor

    # Get first env
    depth_frame = depth_np[0]  # (540, 960)

    print(
        f"[INFO] Captured Depth Frame: Shape={depth_frame.shape}, Dtype={depth_frame.dtype}"
    )
    print(f"[INFO] Range: Min={np.min(depth_frame):.3f}, Max={np.max(depth_frame):.3f}")

    # Save to CSV
    save_path = "/home/adi2440/Desktop/RL_UR5_IsaacLab/sim_raw_depth.csv"
    print(f"[INFO] Saving to {save_path}...")

    # Use pandas for faster CSV writing if available, else numpy
    # User's numpy.savetxt might be slow for 960x540
    try:
        import pandas as pd

        df = pd.DataFrame(depth_frame)
        df.to_csv(save_path, header=False, index=False)
        print("Saved using Pandas.")
    except ImportError:
        np.savetxt(save_path, depth_frame, delimiter=",", fmt="%.4f")
        print("Saved using Numpy.")

    env.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
