#!/usr/bin/env python3

"""
Simplified play script that adds recording without modifying the environment configuration.
This bypasses the event system issues and directly integrates recording.
"""

import argparse
from isaaclab.app import AppLauncher
from datetime import datetime
# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games with recording.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# NEW: Custom recording arguments
parser.add_argument("--record", action="store_true", default=False, help="Enable custom recording system.")
parser.add_argument("--record_dir", type=str, default="./logs/play_recordings", help="Output directory for recordings.")
parser.add_argument("--record_episodes", type=int, default=3, help="Number of episodes to record.")
parser.add_argument("--camera_name", type=str, default="tiled_camera_right", help="Camera name for recording.")
# High-frequency recording arguments
parser.add_argument("--record_camera", action="store_true", default=False, help="Record camera at full simulation rate (120Hz).")
parser.add_argument("--record_duration", type=float, default=5.0, help="Duration to record each episode in seconds.")
parser.add_argument("--record_episodes_hf", type=int, default=3, help="Number of episodes to record at high frequency.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video or args_cli.record or args_cli.record_camera:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import time
import torch
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)
import RL_UR5.tasks  # noqa: F401


def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # Disable recording in environment config to avoid conflicts
    if hasattr(env_cfg, 'events') and hasattr(env_cfg.events, 'start_recording'):
        env_cfg.events.start_recording.params["recorder_config"]["enable_recording"] = False

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # NEW: Initialize high-frequency recorder
    hf_recorder = None
    if args_cli.record_camera:
        try:
            from RL_UR5.tasks.manager_based.rl_ur5.mdp.recorder import HighFrequencyRecorder
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = os.path.splitext(os.path.basename(resume_path))[0]
            output_dir = f"./logs/camera_recordings/{checkpoint_name}_{timestamp}"
            
            hf_recorder = HighFrequencyRecorder(
                env=env,
                output_dir=output_dir,
                camera_name="tiled_camera_right",
                record_duration_seconds=args_cli.record_duration
            )
            
            print(f"[INFO] High-frequency camera recording enabled")
            print(f"  - Output: {output_dir}")
            print(f"  - Duration per episode: {args_cli.record_duration}s")
            print(f"  - Episodes to record: {args_cli.record_episodes_hf}")
            
        except Exception as e:
            print(f"[WARNING] High-frequency recorder failed: {e}")
            hf_recorder = None

    # NEW: Initialize custom recording system AFTER environment creation
    custom_recorder = None
    if args_cli.record:
        try:
            # Import our custom recorder
            from RL_UR5.tasks.manager_based.rl_ur5.mdp.recorder import EnvironmentRecorder
            
            # Create timestamp for unique output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = os.path.splitext(os.path.basename(resume_path))[0]
            output_dir = os.path.join(args_cli.record_dir, f"{checkpoint_name}_{timestamp}")
            
            # IMPORTANT: Pass the unwrapped environment to the recorder
            # The recorder will handle the unwrapping internally now
            custom_recorder = EnvironmentRecorder(
                env=env,  # Pass the wrapped env, recorder will handle unwrapping
                output_dir=output_dir,
                record_video=True,
                record_robot_states=True,
                record_actions=True,
                camera_name=args_cli.camera_name
            )
            
            print(f"[INFO] Custom recording enabled:")
            print(f"  - Output directory: {output_dir}")
            print(f"  - Camera: {args_cli.camera_name}")
            print(f"  - Episodes to record: {args_cli.record_episodes}")
            
        except ImportError as e:
            print(f"[WARNING] Custom recorder not available: {e}")
            print("[WARNING] Continuing without custom recording...")
            custom_recorder = None
        except Exception as e:
            print(f"[ERROR] Failed to initialize custom recorder: {e}")
            print("[ERROR] Continuing without custom recording...")
            custom_recorder = None

    # wrap for standard video recording (if not using custom recorder)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.physics_dt

    # NEW: High-frequency recording state
    hf_episodes_recorded = 0
    hf_recording_active = False
    
    # Custom recording variables
    episodes_recorded = 0
    max_episodes_to_record = args_cli.record_episodes if args_cli.record else 0

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    
    # NEW: Start first episode recording if using custom recorder
    if custom_recorder and episodes_recorded < max_episodes_to_record:
        episode_name = f"episode_{episodes_recorded:04d}_{datetime.now().strftime('%H%M%S')}"
        custom_recorder.start_recording(episode_name)
        print(f"[INFO] Started recording episode: {episode_name}")
        episodes_recorded += 1
    
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        
        # FIXED: Start high-frequency recording if needed and not active
        if (hf_recorder and not hf_recording_active and 
            hf_episodes_recorded < args_cli.record_episodes_hf):
            episode_name = f"hf_episode_{hf_episodes_recorded:04d}_{datetime.now().strftime('%H%M%S')}"
            hf_recorder.start_recording(episode_name)
            hf_recording_active = True
            print(f"[INFO] Started high-frequency recording: {episode_name}")
            
        # FIXED: Capture frame at simulation rate (every step) if recording is active
        if hf_recorder and hf_recording_active:
            continue_recording = hf_recorder.capture_frame()
            if not continue_recording:
                # Recording duration completed
                print(f"[INFO] Stopping high-frequency recording after {hf_recorder.frame_count} frames")
                hf_recorder.stop_recording()
                hf_recording_active = False
                hf_episodes_recorded += 1
                print(f"[INFO] Completed high-frequency recording {hf_episodes_recorded}/{args_cli.record_episodes_hf}")
                
                # FIXED: Continue to next episode or exit if done
                if hf_episodes_recorded >= args_cli.record_episodes_hf:
                    print("[INFO] All high-frequency recordings completed!")
                    # Don't break here, let the simulation continue naturally
                    # The loop will exit when simulation_app.is_running() becomes False
        
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)

            # NEW: Record step data if using custom recorder
            if custom_recorder and custom_recorder.is_recording:
                custom_recorder.step(actions)

            # perform operations for terminated episodes
            if len(dones) > 0 and dones.any():
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
                
                # NEW: Handle episode completion for custom recording
                if custom_recorder and custom_recorder.is_recording:
                    custom_recorder.stop_recording()
                    print(f"[INFO] Episode recording completed ({episodes_recorded}/{max_episodes_to_record})")
                    
                    # Start next episode if we haven't reached the limit
                    if episodes_recorded < max_episodes_to_record:
                        episode_name = f"episode_{episodes_recorded:04d}_{datetime.now().strftime('%H%M%S')}"
                        custom_recorder.start_recording(episode_name)
                        print(f"[INFO] Started recording episode: {episode_name}")
                        episodes_recorded += 1

        # Standard video recording exit condition
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # FIXED: Exit condition for high-frequency recording
        if (args_cli.record_camera and 
            hf_episodes_recorded >= args_cli.record_episodes_hf and 
            not hf_recording_active):
            print("[INFO] High-frequency recording completed, exiting...")
            break

        # Exit condition for custom recording
        if (args_cli.record and episodes_recorded >= max_episodes_to_record and 
            (not custom_recorder or not custom_recorder.is_recording)):
            print("[INFO] Custom recording completed, exiting...")
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # FIXED: Cleanup recordings properly
    if hf_recorder and hf_recording_active:
        print("[INFO] Finalizing high-frequency recording...")
        hf_recorder.stop_recording()
        
    if custom_recorder and custom_recorder.is_recording:
        print("[INFO] Finalizing custom recording...")
        custom_recorder.stop_recording()
        
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()