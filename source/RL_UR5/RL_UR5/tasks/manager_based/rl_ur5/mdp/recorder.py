#!/usr/bin/env python3

"""
Custom recording system for Isaac Lab environments.

This module provides recording functionality for camera data, robot states, and actions
without relying on Isaac Lab's built-in RecorderManager (which doesn't exist).
"""

from __future__ import annotations

import torch
import numpy as np
import cv2
import os
import h5py
import json
from typing import TYPE_CHECKING, Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class EnvironmentRecorder:
    """Main recorder class for Isaac Lab environments."""
    
    def __init__(
        self, 
        env: ManagerBasedRLEnv, 
        output_dir: str = "./logs/recordings",
        record_video: bool = True,
        record_robot_states: bool = True,
        record_actions: bool = True,
        video_fps: int = 30,
        camera_name: str = "tiled_camera_right"
    ):
        """
        Initialize the environment recorder.
        
        Args:
            env: The RL environment instance
            output_dir: Directory to save recordings
            record_video: Whether to record camera video
            record_robot_states: Whether to record robot state data
            record_actions: Whether to record action data
            video_fps: FPS for video recording
            camera_name: Name of camera sensor to record
        """
        # Handle wrapped environments - get the unwrapped environment
        if hasattr(env, 'unwrapped'):
            self.env = env.unwrapped
        else:
            self.env = env
            
        self.output_dir = output_dir
        self.record_video = record_video
        self.record_robot_states = record_robot_states
        self.record_actions = record_actions
        self.video_fps = video_fps
        self.camera_name = camera_name
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Recording state
        self.is_recording = False
        self.episode_count = 0
        self.current_episode_data = defaultdict(list)
        
        # Video recording
        self.video_writers = {}
        self.frame_buffers = defaultdict(list)
        
        # Get camera sensor if video recording is enabled
        self.camera = None
        if self.record_video:
            try:
                # Try different ways to access the camera
                if hasattr(self.env, 'scene') and hasattr(self.env.scene, 'sensors'):
                    self.camera = self.env.scene.sensors[camera_name]
                    print(f"Camera '{camera_name}' found for video recording")
                elif hasattr(self.env, 'scene') and camera_name in self.env.scene:
                    self.camera = self.env.scene[camera_name]
                    print(f"Camera '{camera_name}' found for video recording")
                else:
                    print(f"Warning: Camera '{camera_name}' not found. Video recording disabled.")
                    print(f"Available scene entities: {list(self.env.scene.keys()) if hasattr(self.env, 'scene') else 'No scene'}")
                    self.record_video = False
            except Exception as e:
                print(f"Warning: Could not access camera '{camera_name}': {e}")
                self.record_video = False
        
        # Initialize episode tracking
        self.episode_ids = [0] * self.env.num_envs
        self.episode_step_counts = [0] * self.env.num_envs
        
    def start_recording(self, episode_name: Optional[str] = None):
        """Start recording data."""
        if episode_name is None:
            episode_name = f"episode_{self.episode_count:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_episode_name = episode_name
        self.is_recording = True
        self.current_episode_data.clear()
        
        # Clear frame buffers
        for env_idx in range(self.env.num_envs):
            self.frame_buffers[env_idx].clear()
        
        print(f"Started recording: {episode_name}")
    
    def stop_recording(self):
        """Stop recording and save data."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self._save_episode_data()
        self.episode_count += 1
        print(f"Stopped recording: {self.current_episode_name}")
    
    def step(self, actions: torch.Tensor):
        """Record data for the current step."""
        if not self.is_recording:
            return
        
        try:
            # Record actions
            if self.record_actions:
                self.current_episode_data['actions'].append(actions.cpu().numpy())
            
            # Record robot states
            if self.record_robot_states:
                try:
                    robot = self.env.scene["robot"]
                    self.current_episode_data['joint_positions'].append(robot.data.joint_pos.cpu().numpy())
                    self.current_episode_data['joint_velocities'].append(robot.data.joint_vel.cpu().numpy())
                    
                    # Record end-effector poses if available
                    try:
                        ee_frame = self.env.scene["ee_frame"]
                        self.current_episode_data['ee_positions'].append(
                            ee_frame.data.target_pos_w[..., 0, :].cpu().numpy()
                        )
                        self.current_episode_data['ee_orientations'].append(
                            ee_frame.data.target_quat_w[..., 0, :].cpu().numpy()
                        )
                    except (KeyError, IndexError):
                        # ee_frame not available or different structure, skip
                        pass
                        
                except Exception as e:
                    print(f"Warning: Failed to record robot states: {e}")
            
            # Record camera frames
            if self.record_video and self.camera is not None:
                self._record_camera_frame()
            
            # Record timestamps
            current_time = getattr(self.env, 'episode_length_buf', torch.zeros(self.env.num_envs))
            if isinstance(current_time, torch.Tensor):
                current_time = current_time.cpu().numpy()
            self.current_episode_data['timestamps'].append(current_time)
            
            # Update step counts
            for env_idx in range(self.env.num_envs):
                self.episode_step_counts[env_idx] += 1
                
        except Exception as e:
            print(f"Warning: Recording step failed: {e}")
            # Continue execution even if recording fails
    
    def _record_camera_frame(self):
        """Record current camera frame."""
        try:
            if not hasattr(self.camera.data, 'output') or "rgb" not in self.camera.data.output:
                return

            rgb_data = self.camera.data.output["rgb"]
            
            # Convert to numpy
            if isinstance(rgb_data, torch.Tensor):
                rgb_data = rgb_data.cpu().numpy()
            
            # Process each environment
            for env_idx in range(self.env.num_envs):
                if len(rgb_data.shape) == 4:  # (batch, height, width, channels)
                    frame = rgb_data[env_idx]
                else:  # Single frame for all environments
                    frame = rgb_data
                
                # Convert to uint8 if needed
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                self.frame_buffers[env_idx].append(frame.copy())
                
        except Exception as e:
            print(f"Warning: Failed to record camera frame: {e}")
    
    def _save_episode_data(self):
        """Save recorded episode data."""
        episode_dir = os.path.join(self.output_dir, self.current_episode_name)
        os.makedirs(episode_dir, exist_ok=True)
        
        # Save robot state data
        if self.record_robot_states and self.current_episode_data:
            self._save_robot_states(episode_dir)
        
        # Save action data
        if self.record_actions and 'actions' in self.current_episode_data:
            self._save_actions(episode_dir)
        
        # Save videos
        if self.record_video:
            self._save_videos(episode_dir)
        
        # Save metadata
        self._save_metadata(episode_dir)
    
    def _save_robot_states(self, episode_dir: str):
        """Save robot state data as HDF5."""
        filename = os.path.join(episode_dir, "robot_states.h5")
        
        with h5py.File(filename, "w") as f:
            for key, data_list in self.current_episode_data.items():
                if data_list and key != 'actions':  # Skip empty lists and actions
                    try:
                        # Convert list of arrays to single array
                        data_array = np.array(data_list)
                        f.create_dataset(key, data=data_array)
                    except Exception as e:
                        print(f"Warning: Could not save {key}: {e}")
        
        print(f"Robot states saved: {filename}")
    
    def _save_actions(self, episode_dir: str):
        """Save action data as NPZ."""
        filename = os.path.join(episode_dir, "actions.npz")
        actions_array = np.array(self.current_episode_data['actions'])
        np.savez(filename, actions=actions_array)
        print(f"Actions saved: {filename}")
    
    def _save_videos(self, episode_dir: str):
        """Save camera videos."""
        videos_dir = os.path.join(episode_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        for env_idx in range(self.env.num_envs):
            frames = self.frame_buffers[env_idx]
            if not frames:
                continue
            
            filename = os.path.join(videos_dir, f"env_{env_idx}_{self.camera_name}.mp4")
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(filename, fourcc, self.video_fps, (width, height))
            
            if video_writer.isOpened():
                for frame in frames:
                    video_writer.write(frame)
                video_writer.release()
                print(f"Video saved: {filename} ({len(frames)} frames)")
            else:
                print(f"Warning: Failed to create video writer for {filename}")
    
    def _save_metadata(self, episode_dir: str):
        """Save episode metadata."""
        metadata = {
            "episode_name": self.current_episode_name,
            "timestamp": datetime.now().isoformat(),
            "num_environments": self.env.num_envs,
            "total_steps": len(self.current_episode_data.get('timestamps', [])),
            "episode_length_s": self.env.cfg.episode_length_s,
            "decimation": self.env.cfg.decimation,
            "sim_dt": self.env.cfg.sim.dt,
            "camera_name": self.camera_name,
            "video_fps": self.video_fps,
            "recorded_data": {
                "robot_states": self.record_robot_states,
                "actions": self.record_actions,
                "video": self.record_video,
            }
        }
        
        filename = os.path.join(episode_dir, "metadata.json")
        with open(filename, "w") as f:
            json.dump(metadata, f, indent=2)


#!/usr/bin/env python3

"""
Custom recording system for Isaac Lab environments.

This module provides recording functionality for camera data, robot states, and actions
without relying on Isaac Lab's built-in RecorderManager (which doesn't exist).
"""




class HighFrequencyRecorder:
    """High-frequency recorder that captures camera data at simulation rate (120Hz)."""
    
    def __init__(
        self,
        env,
        output_dir: str = "./logs/recordings",
        camera_name: str = "tiled_camera_right",
        record_duration_seconds: float = 5.0,
    ):
        """
        Initialize high-frequency recorder.
        
        Args:
            env: The environment instance
            output_dir: Directory to save recordings
            camera_name: Name of camera to record
            record_duration_seconds: How long to record each episode in seconds
        """
        # Handle wrapped environments
        if hasattr(env, 'unwrapped'):
            self.env = env.unwrapped
        else:
            self.env = env
            
        self.output_dir = output_dir
        self.camera_name = camera_name
        self.record_duration_seconds = record_duration_seconds
        
        # Get simulation timestep (e.g., 1/120 = 0.00833 seconds)
        self.sim_dt = getattr(self.env.cfg, 'sim', None)
        if self.sim_dt and hasattr(self.sim_dt, 'dt'):
            self.sim_dt = self.sim_dt.dt
        else:
            self.sim_dt = 1.0 / 120.0  # Default to 120Hz
        
        # Calculate total frames to record
        self.target_frames = int(record_duration_seconds / self.sim_dt)
        
        print(f"[INFO] High-frequency recorder initialized:")
        print(f"  - Simulation rate: {1/self.sim_dt:.1f} Hz ({self.sim_dt:.6f}s per step)")
        print(f"  - Record duration: {record_duration_seconds}s")
        print(f"  - Target frames per episode: {self.target_frames}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get camera
        self.camera = None
        try:
            if hasattr(self.env, 'scene') and hasattr(self.env.scene, 'sensors'):
                self.camera = self.env.scene.sensors[camera_name]
                print(f"  - Camera '{camera_name}' found for recording")
            elif hasattr(self.env, 'scene') and camera_name in self.env.scene:
                self.camera = self.env.scene[camera_name]
                print(f"  - Camera '{camera_name}' found for recording")
            else:
                print(f"  - Warning: Camera '{camera_name}' not found")
                available = list(self.env.scene.keys()) if hasattr(self.env, 'scene') else []
                print(f"  - Available entities: {available}")
        except Exception as e:
            print(f"  - Error accessing camera: {e}")
        
        # Recording state
        self.is_recording = False
        self.current_episode_name = ""
        self.frame_buffer = []
        self.frame_count = 0
        self.sim_step_count = 0
        
    def start_recording(self, episode_name: str):
        """Start recording a new episode."""
        self.current_episode_name = episode_name
        self.is_recording = True
        self.frame_buffer = []
        self.frame_count = 0
        self.sim_step_count = 0
        print(f"[INFO] Started high-frequency recording: {episode_name}")
        print(f"  - Will record {self.target_frames} frames over {self.record_duration_seconds}s")
        
    def capture_frame(self):
        """Capture a single frame - call this every simulation step."""
        if not self.is_recording or self.camera is None:
            return False
            
        self.sim_step_count += 1
        
        # Check if we should stop recording
        if self.frame_count >= self.target_frames:
            return False  # Recording complete
            
        try:
            # Get camera data
            if hasattr(self.camera.data, 'output') and "rgb" in self.camera.data.output:
                rgb_data = self.camera.data.output["rgb"]
                
                if isinstance(rgb_data, torch.Tensor):
                    rgb_data = rgb_data.cpu().numpy()
                
                # Process first environment only
                if len(rgb_data.shape) == 4:
                    frame = rgb_data[0]
                else:
                    frame = rgb_data
                
                # Convert to uint8
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Store frame with timestamp
                frame_data = {
                    'frame': frame.copy(),
                    'frame_index': self.frame_count,
                    'sim_step': self.sim_step_count,
                    'timestamp': self.sim_step_count * self.sim_dt
                }
                self.frame_buffer.append(frame_data)
                self.frame_count += 1
                
                # Progress indicator
                if self.frame_count % 60 == 0:  # Every ~0.5 seconds at 120Hz
                    progress = (self.frame_count / self.target_frames) * 100
                    print(f"  - Recording progress: {progress:.1f}% ({self.frame_count}/{self.target_frames} frames)")
                
                return True  # Continue recording
                
        except Exception as e:
            print(f"Warning: Failed to capture frame: {e}")
            
        return True  # Continue despite errors
        
    def stop_recording(self):
        """Stop recording and save video."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        if not self.frame_buffer:
            print(f"[WARNING] No frames recorded for {self.current_episode_name}")
            return
            
        try:
            # Create episode directory
            episode_dir = os.path.join(self.output_dir, self.current_episode_name)
            os.makedirs(episode_dir, exist_ok=True)
            
            # Save high-framerate video
            video_file = os.path.join(episode_dir, f"{self.camera_name}_120hz.mp4")
            self._save_high_quality_video(video_file)
            
            # Save metadata
            metadata = {
                "episode_name": self.current_episode_name,
                "timestamp": datetime.now().isoformat(),
                "simulation_rate_hz": 1 / self.sim_dt,
                "recorded_frames": len(self.frame_buffer),
                "target_frames": self.target_frames,
                "actual_duration_s": len(self.frame_buffer) * self.sim_dt,
                "target_duration_s": self.record_duration_seconds,
                "camera_name": self.camera_name,
                "video_file": f"{self.camera_name}_120hz.mp4"
            }
            
            metadata_file = os.path.join(episode_dir, "recording_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"[INFO] High-frequency recording saved:")
            print(f"  - Episode: {self.current_episode_name}")
            print(f"  - Frames: {len(self.frame_buffer)}/{self.target_frames}")
            print(f"  - Duration: {len(self.frame_buffer) * self.sim_dt:.2f}s")
            print(f"  - Video: {video_file}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save recording: {e}")
            
    def _save_high_quality_video(self, filename: str):
        """Save frames as high-quality video."""
        if not self.frame_buffer:
            return
            
        try:
            # Get frame dimensions
            first_frame = self.frame_buffer[0]['frame']
            height, width = first_frame.shape[:2]
            
            # Use high-quality encoding
            # H.264 with high bitrate for good quality
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Video FPS = simulation rate (120 fps)
            video_fps = 1 / self.sim_dt
            
            video_writer = cv2.VideoWriter(filename, fourcc, video_fps, (width, height))
            
            if not video_writer.isOpened():
                print(f"[ERROR] Failed to create video writer for {filename}")
                return
            
            # Write all frames
            for frame_data in self.frame_buffer:
                video_writer.write(frame_data['frame'])
            
            video_writer.release()
            
            print(f"  - Video saved: {filename}")
            print(f"    * Resolution: {width}x{height}")
            print(f"    * FPS: {video_fps}")
            print(f"    * Total frames: {len(self.frame_buffer)}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save video: {e}")
    """Main recorder class for Isaac Lab environments."""
    
    def __init__(
        self, 
        env: ManagerBasedRLEnv, 
        output_dir: str = "./logs/recordings",
        record_video: bool = True,
        record_robot_states: bool = True,
        record_actions: bool = True,
        video_fps: int = 30,
        camera_name: str = "tiled_camera_right"
    ):
        """
        Initialize the environment recorder.
        
        Args:
            env: The RL environment instance
            output_dir: Directory to save recordings
            record_video: Whether to record camera video
            record_robot_states: Whether to record robot state data
            record_actions: Whether to record action data
            video_fps: FPS for video recording
            camera_name: Name of camera sensor to record
        """
        # Handle wrapped environments - get the unwrapped environment
        if hasattr(env, 'unwrapped'):
            self.env = env.unwrapped
        else:
            self.env = env
            
        self.output_dir = output_dir
        self.record_video = record_video
        self.record_robot_states = record_robot_states
        self.record_actions = record_actions
        self.video_fps = video_fps
        self.camera_name = camera_name
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Recording state
        self.is_recording = False
        self.episode_count = 0
        self.current_episode_data = defaultdict(list)
        
        # Video recording
        self.video_writers = {}
        self.frame_buffers = defaultdict(list)
        
        # Get camera sensor if video recording is enabled
        self.camera = None
        if self.record_video:
            try:
                # Try different ways to access the camera
                if hasattr(self.env, 'scene') and hasattr(self.env.scene, 'sensors'):
                    self.camera = self.env.scene.sensors[camera_name]
                    print(f"Camera '{camera_name}' found for video recording")
                elif hasattr(self.env, 'scene') and camera_name in self.env.scene:
                    self.camera = self.env.scene[camera_name]
                    print(f"Camera '{camera_name}' found for video recording")
                else:
                    print(f"Warning: Camera '{camera_name}' not found. Video recording disabled.")
                    print(f"Available scene entities: {list(self.env.scene.keys()) if hasattr(self.env, 'scene') else 'No scene'}")
                    self.record_video = False
            except Exception as e:
                print(f"Warning: Could not access camera '{camera_name}': {e}")
                self.record_video = False
        
        # Initialize episode tracking
        self.episode_ids = [0] * self.env.num_envs
        self.episode_step_counts = [0] * self.env.num_envs
        
    def start_recording(self, episode_name: Optional[str] = None):
        """Start recording data."""
        if episode_name is None:
            episode_name = f"episode_{self.episode_count:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_episode_name = episode_name
        self.is_recording = True
        self.current_episode_data.clear()
        
        # Clear frame buffers
        for env_idx in range(self.env.num_envs):
            self.frame_buffers[env_idx].clear()
        
        print(f"Started recording: {episode_name}")
    
    def stop_recording(self):
        """Stop recording and save data."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self._save_episode_data()
        self.episode_count += 1
        print(f"Stopped recording: {self.current_episode_name}")
    
    def step(self, actions: torch.Tensor):
        """Record data for the current step."""
        if not self.is_recording:
            return
        
        try:
            # Record actions
            if self.record_actions:
                self.current_episode_data['actions'].append(actions.cpu().numpy())
            
            # Record robot states
            if self.record_robot_states:
                try:
                    robot = self.env.scene["robot"]
                    self.current_episode_data['joint_positions'].append(robot.data.joint_pos.cpu().numpy())
                    self.current_episode_data['joint_velocities'].append(robot.data.joint_vel.cpu().numpy())
                    
                    # Record end-effector poses if available
                    try:
                        ee_frame = self.env.scene["ee_frame"]
                        self.current_episode_data['ee_positions'].append(
                            ee_frame.data.target_pos_w[..., 0, :].cpu().numpy()
                        )
                        self.current_episode_data['ee_orientations'].append(
                            ee_frame.data.target_quat_w[..., 0, :].cpu().numpy()
                        )
                    except (KeyError, IndexError):
                        # ee_frame not available or different structure, skip
                        pass
                        
                except Exception as e:
                    print(f"Warning: Failed to record robot states: {e}")
            
            # Record camera frames
            if self.record_video and self.camera is not None:
                self._record_camera_frame()
            
            # Record timestamps
            current_time = getattr(self.env, 'episode_length_buf', torch.zeros(self.env.num_envs))
            if isinstance(current_time, torch.Tensor):
                current_time = current_time.cpu().numpy()
            self.current_episode_data['timestamps'].append(current_time)
            
            # Update step counts
            for env_idx in range(self.env.num_envs):
                self.episode_step_counts[env_idx] += 1
                
        except Exception as e:
            print(f"Warning: Recording step failed: {e}")
            # Continue execution even if recording fails
    
    def _record_camera_frame(self):
        """Record current camera frame."""
        try:
            if not hasattr(self.camera.data, 'output') or "rgb" not in self.camera.data.output:
                return

            rgb_data = self.camera.data.output["rgb"]
            
            # Convert to numpy
            if isinstance(rgb_data, torch.Tensor):
                rgb_data = rgb_data.cpu().numpy()
            
            # Process each environment
            for env_idx in range(self.env.num_envs):
                if len(rgb_data.shape) == 4:  # (batch, height, width, channels)
                    frame = rgb_data[env_idx]
                else:  # Single frame for all environments
                    frame = rgb_data
                
                # Convert to uint8 if needed
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                self.frame_buffers[env_idx].append(frame.copy())
                
        except Exception as e:
            print(f"Warning: Failed to record camera frame: {e}")
    
    def _save_episode_data(self):
        """Save recorded episode data."""
        episode_dir = os.path.join(self.output_dir, self.current_episode_name)
        os.makedirs(episode_dir, exist_ok=True)
        
        # Save robot state data
        if self.record_robot_states and self.current_episode_data:
            self._save_robot_states(episode_dir)
        
        # Save action data
        if self.record_actions and 'actions' in self.current_episode_data:
            self._save_actions(episode_dir)
        
        # Save videos
        if self.record_video:
            self._save_videos(episode_dir)
        
        # Save metadata
        self._save_metadata(episode_dir)
    
    def _save_robot_states(self, episode_dir: str):
        """Save robot state data as HDF5."""
        filename = os.path.join(episode_dir, "robot_states.h5")
        
        with h5py.File(filename, "w") as f:
            for key, data_list in self.current_episode_data.items():
                if data_list and key != 'actions':  # Skip empty lists and actions
                    try:
                        # Convert list of arrays to single array
                        data_array = np.array(data_list)
                        f.create_dataset(key, data=data_array)
                    except Exception as e:
                        print(f"Warning: Could not save {key}: {e}")
        
        print(f"Robot states saved: {filename}")
    
    def _save_actions(self, episode_dir: str):
        """Save action data as NPZ."""
        filename = os.path.join(episode_dir, "actions.npz")
        actions_array = np.array(self.current_episode_data['actions'])
        np.savez(filename, actions=actions_array)
        print(f"Actions saved: {filename}")
    
    def _save_videos(self, episode_dir: str):
        """Save camera videos."""
        videos_dir = os.path.join(episode_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        for env_idx in range(self.env.num_envs):
            frames = self.frame_buffers[env_idx]
            if not frames:
                continue
            
            filename = os.path.join(videos_dir, f"env_{env_idx}_{self.camera_name}.mp4")
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(filename, fourcc, self.video_fps, (width, height))
            
            if video_writer.isOpened():
                for frame in frames:
                    video_writer.write(frame)
                video_writer.release()
                print(f"Video saved: {filename} ({len(frames)} frames)")
            else:
                print(f"Warning: Failed to create video writer for {filename}")
    
    def _save_metadata(self, episode_dir: str):
        """Save episode metadata."""
        metadata = {
            "episode_name": self.current_episode_name,
            "timestamp": datetime.now().isoformat(),
            "num_environments": self.env.num_envs,
            "total_steps": len(self.current_episode_data.get('timestamps', [])),
            "episode_length_s": self.env.cfg.episode_length_s,
            "decimation": self.env.cfg.decimation,
            "sim_dt": self.env.cfg.sim.dt,
            "camera_name": self.camera_name,
            "video_fps": self.video_fps,
            "recorded_data": {
                "robot_states": self.record_robot_states,
                "actions": self.record_actions,
                "video": self.record_video,
            }
        }
        
        filename = os.path.join(episode_dir, "metadata.json")
        with open(filename, "w") as f:
            json.dump(metadata, f, indent=2)


class RecordingEventManager:
    """Manages recording through environment events."""
    
    def __init__(self, recorder: EnvironmentRecorder):
        self.recorder = recorder
        self.auto_record = True
        self.episodes_recorded = 0
        self.max_episodes = 10  # Maximum episodes to record
    
    def on_reset(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        """Handle episode resets."""
        if self.auto_record and self.episodes_recorded < self.max_episodes:
            # Stop current recording if active
            if self.recorder.is_recording:
                self.recorder.stop_recording()
            
            # Start new recording
            episode_name = f"auto_episode_{self.episodes_recorded:04d}"
            self.recorder.start_recording(episode_name)
            self.episodes_recorded += 1
    
    def on_step(self, env: ManagerBasedRLEnv, actions: torch.Tensor):
        """Handle environment steps."""
        if self.recorder.is_recording:
            self.recorder.step(actions)


# Event functions for use in environment configuration

def start_recording_on_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    recorder_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Event function to start recording on episode reset.
    
    Args:
        env: The RL environment instance
        env_ids: Environment IDs that are resetting
        recorder_config: Optional recorder configuration
        
    Returns:
        Empty dictionary
    """
    # Initialize recorder if not exists
    if not hasattr(env, '_recorder'):
        config = recorder_config or {}
        env._recorder = EnvironmentRecorder(
            env=env,
            output_dir=config.get('output_dir', './logs/recordings'),
            record_video=config.get('record_video', True),
            record_robot_states=config.get('record_robot_states', True),
            record_actions=config.get('record_actions', True),
            video_fps=config.get('video_fps', 30),
            camera_name=config.get('camera_name', 'tiled_camera_right')
        )
        env._recording_manager = RecordingEventManager(env._recorder)
    
    # Handle recording on reset
    env._recording_manager.on_reset(env, env_ids)
    
    return {}


def record_step_data(
    env: ManagerBasedRLEnv,
    actions: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Function to record step data.
    
    Args:
        env: The RL environment instance
        actions: Current actions (if available)
        
    Returns:
        Empty dictionary
    """
    if hasattr(env, '_recording_manager') and actions is not None:
        env._recording_manager.on_step(env, actions)
    
    return {}


# Utility functions

def create_simple_recorder(
    env: ManagerBasedRLEnv,
    output_dir: str = "./logs/recordings",
    camera_name: str = "tiled_camera_right"
) -> EnvironmentRecorder:
    """
    Create a simple recorder for the environment.
    
    Args:
        env: The RL environment instance
        output_dir: Directory to save recordings
        camera_name: Name of camera to record
        
    Returns:
        EnvironmentRecorder instance
    """
    return EnvironmentRecorder(
        env=env,
        output_dir=output_dir,
        record_video=True,
        record_robot_states=True,
        record_actions=True,
        camera_name=camera_name
    )


def manual_recording_session(
    env: ManagerBasedRLEnv,
    episode_name: str,
    num_steps: int = 1000,
    output_dir: str = "./logs/manual_recording"
) -> str:
    """
    Manually record a single episode.
    
    Args:
        env: The RL environment instance
        episode_name: Name for the episode
        num_steps: Number of steps to record
        output_dir: Directory to save recording
        
    Returns:
        Path to saved episode directory
    """
    recorder = create_simple_recorder(env, output_dir)
    recorder.start_recording(episode_name)
    
    try:
        # Reset environment
        obs, _ = env.reset()
        
        for step in range(num_steps):
            # Get random actions (replace with actual policy)
            actions = torch.randn(env.num_envs, env.action_space.shape[0], device=env.device)
            actions = torch.clamp(actions, -1, 1)  # Clip to action space
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Record step
            recorder.step(actions)
            
            # Check if any environment terminated
            if terminated.any() or truncated.any():
                break
    
    finally:
        recorder.stop_recording()
    
    episode_dir = os.path.join(output_dir, episode_name)
    return episode_dir


