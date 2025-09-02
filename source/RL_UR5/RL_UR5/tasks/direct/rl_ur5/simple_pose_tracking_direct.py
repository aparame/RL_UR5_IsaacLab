"""
Simple UR5 Camera-based Pose Tracking Environment

A minimal environment for UR5 pose tracking with camera observations.
No obstacles, no table - just pure pose tracking with visual feedback.
"""

from __future__ import annotations

import torch
import numpy as np
import math
from typing import Dict, Any, Tuple, Optional, Sequence
import gymnasium as gym

# IsaacLab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import sample_uniform
from isaaclab.envs.ui import BaseEnvWindow
# Robot configuration
from isaaclab_assets.robots.ur5 import UR5_GRIPPER_CFG



class UR5EnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: SimpleCameraPoseTrackingEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)




@configclass
class SimpleCameraPoseTrackingEnvCfg(DirectRLEnvCfg):
    """Configuration for the simple camera-based pose tracking environment."""
    
    # Environment settings
    episode_length_s = 8.0
    decimation = 4
    action_scale = 0.5
    num_envs = 8
    env_spacing = 4.0
    state_dim = 19

    # Observation and action spaces
    action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(6,))
    state_space = 0
    observation_space = gym.spaces.Dict({
        "image": gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(80, 100, 3)),
        "state": gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(state_dim,)),
    })
    
    # Debug visualization
    debug_vis = True
    
    # UR5 Robot
    robot_cfg: ArticulationCfg = UR5_GRIPPER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Frame transformer for end-effector
    ee_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/ee_link",
                name="end_effector",
                offset=OffsetCfg(pos=[0.12, 0.0, 0.0]),
            ),
        ],
    )
    
    # Camera configuration
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.0,
            focus_distance=10.0,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 50.0)
        ),
        width=80,
        height=100,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.5, 0.5, 1.5),
            rot=(0.679, 0.281, 0.281, 0.613),
            convention="opengl"
        )
    )
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0/120.0,
        render_interval=decimation,
    )
    
    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=True,
    )
    
    # Viewer settings
    viewer = ViewerCfg(eye=(7.5, 7.5, 7.5), origin_type="world", env_index=0)
    
    # Target pose ranges (in robot base frame)
    position_range = {
        "x": (0.3, 0.8),
        "y": (-0.5, 0.5),
        "z": (-0.2, 0.2),
    }
    orientation_range = {
        "roll": (-3.14, 3.14),
        "pitch": (0.0, 3.14),
        "yaw": (-3.14, 3.14),
    }
    
    # Reward weights
    position_reward_weight = 1.0
    orientation_reward_weight = 0.0
    action_penalty_weight = -0.001
    velocity_penalty_weight = -0.001
    
    # Success thresholds
    position_threshold = 0.05
    orientation_threshold = 0.1
    
    # Robot reset configuration
    robot_base_pose = [-0.568, -0.658, 1.602, -2.585, -1.606, -1.641]
    robot_reset_noise = 0.1


class SimpleCameraPoseTrackingEnv(DirectRLEnv):
    """Simple camera-based pose tracking environment for UR5."""
    
    cfg: SimpleCameraPoseTrackingEnvCfg
    
    def __init__(self, cfg: SimpleCameraPoseTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        # Store config
        self.cfg = cfg
        
        # Initialize parent
        super().__init__(cfg, render_mode, **kwargs)
        
        # Joint names and indices
        self._joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self._joint_indices, _ = self._robot.find_joints(self._joint_names)
        
        # Get DOF limits
        self._robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, self._joint_indices, 0].to(self.device)
        self._robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, self._joint_indices, 1].to(self.device)
        
        # Initialize buffers
        self._robot_dof_targets = torch.zeros((self.num_envs, len(self._joint_indices)), device=self.device)
        
        # Target poses (similar to quadcopter's _desired_pos_w)
        self._desired_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_quat_b = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Previous actions for penalty
        self._previous_actions = torch.zeros(self.num_envs, 6, device=self.device)
        
        # Logging
        self._episode_sums = {
            "position_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "orientation_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "total_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }
        
        # Setup debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        
        print(f"[INFO] Simple Camera Pose Tracking Environment initialized with {self.num_envs} environments")
        
    def _setup_scene(self):
        """Set up the scene with robot and camera."""
        # Add robot
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        
        # Add camera
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        
        # Add end-effector frame
        self._ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Add ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)
        
    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        # Clamp and scale actions
        self._actions = actions.clone().clamp(-5.0, 5.0)
        self._previous_actions = self._actions.clone()
        
    def _apply_action(self):
        """Apply actions to the robot."""
        # Get current joint positions
        # current_joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        
        # # Add scaled actions to current positions
        self._robot_dof_targets =  self._actions[:,self._joint_indices] * self.cfg.action_scale
        
        # # Clamp to joint limits
        self._robot_dof_targets = torch.clamp(
            self._robot_dof_targets,
            self._robot_dof_lower_limits,
            self._robot_dof_upper_limits
        )
        
        # Set joint position targets
        self._robot.set_joint_position_target(
            self._robot_dof_targets, 
            joint_ids=self._joint_indices
        )
        
    def _get_observations(self) -> dict:
        """Compute and return observations."""
        # Get joint states
        joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        joint_vel = self._robot.data.joint_vel[:, self._joint_indices]
        
        # Get target pose (already in robot base frame)
        target_pose = torch.cat([self._desired_pos_b, self._desired_quat_b], dim=-1)
        
        # Create state observation
        state_obs = torch.cat([
            joint_pos,      # 6 dims
            joint_vel,      # 6 dims
            target_pose,    # 7 dims
        ], dim=-1)
        
        # Get camera observation
        camera_data = self._tiled_camera.data.output["rgb"]  # Shape: (num_envs, H, W, C)
        # Normalize to [0, 1]
        camera_obs = camera_data / 255.0
        # Convert to CHW format
        camera_obs = camera_obs.permute(0, 3, 1, 2)
        
        # Create observation dictionary
        obs = {"image": camera_obs, "state": state_obs}
        observations = {"policy": obs}
        
        return observations
        
    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards based on pose tracking performance."""
        # Get end-effector pose
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat = self._ee_frame.data.target_quat_w[..., 0, :]
        
        # Transform desired pose to world frame
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        
        desired_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, self._desired_pos_b
        )
        desired_quat_w = math_utils.quat_mul(robot_quat, self._desired_quat_b)
        
        # Position tracking reward (using tanh for smooth reward)
        position_error = torch.norm(ee_position - desired_pos_w, p=2, dim=1)
        position_reward = (1.0 - torch.tanh(position_error / 0.5)) * self.cfg.position_reward_weight
        
        # Orientation tracking reward
        orientation_error = math_utils.quat_error_magnitude(ee_quat, desired_quat_w)
        orientation_reward = (1.0 - torch.tanh(orientation_error / 0.5)) * self.cfg.orientation_reward_weight
        
        # # Action penalty (encourage smooth movements)
        # action_penalty = torch.sum(torch.square(self._actions), dim=1) * self.cfg.action_penalty_weight
        
        # # Velocity penalty (encourage stability)
        # velocity_penalty = torch.sum(torch.square(self._robot.data.joint_vel[:, self._joint_indices]), dim=1) * self.cfg.velocity_penalty_weight
        
        # Total reward
        rewards = position_reward 
        
        # Update logging
        self._episode_sums["position_error"] += position_error
        self._episode_sums["orientation_error"] += orientation_error
        self._episode_sums["total_reward"] += rewards
        
        return rewards
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        # Time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Get end-effector pose
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat = self._ee_frame.data.target_quat_w[..., 0, :]
        
        # Transform desired pose to world frame
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        
        desired_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, self._desired_pos_b
        )
        desired_quat_w = math_utils.quat_mul(robot_quat, self._desired_quat_b)
        
        # Check success criteria
        position_error = torch.norm(ee_position - desired_pos_w, p=2, dim=-1)
        orientation_error = math_utils.quat_error_magnitude(ee_quat, desired_quat_w)
        
        # Success when both position and orientation are within thresholds
        success = (position_error < self.cfg.position_threshold) & (orientation_error < self.cfg.orientation_threshold)
        
        return success, time_out
        
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
            
        super()._reset_idx(env_ids)
        
        # Log episode statistics
        if len(env_ids) > 0:
            extras = {}
            for key, value in self._episode_sums.items():
                if key.endswith("_error"):
                    # Average error over episode
                    extras[f"Episode/{key}"] = torch.mean(value[env_ids] / self.episode_length_buf[env_ids]).item()
                else:
                    # Total reward
                    extras[f"Episode/{key}"] = torch.mean(value[env_ids]).item()
                # Reset sums
                self._episode_sums[key][env_ids] = 0.0
            
            self.extras["log"] = extras
        
        # Reset robot joint positions
        num_resets = len(env_ids)
        
        # Base pose with noise
        base_pose = torch.tensor(self.cfg.robot_base_pose, device=self.device, dtype=torch.float32)
        joint_pos = base_pose.unsqueeze(0).repeat(num_resets, 1)
        
        # Add noise using the same pattern as quadcopter
        noise = torch.zeros_like(joint_pos).uniform_(-self.cfg.robot_reset_noise, self.cfg.robot_reset_noise)
        joint_pos += noise
        joint_vel = torch.zeros_like(joint_pos)
        
        # Set joint state
        self._robot.write_joint_state_to_sim(
            joint_pos, joint_vel,
            joint_ids=self._joint_indices,
            env_ids=env_ids
        )
        
        # Sample new target poses
        self._sample_target_poses(env_ids)
        
        # Reset actions
        self._previous_actions[env_ids] = 0.0
        
    def _sample_target_poses(self, env_ids: Sequence[int]):
        """Sample new target poses for specified environments."""
        # Sample positions using the quadcopter pattern
        self._desired_pos_b[env_ids, 0] = torch.zeros_like(self._desired_pos_b[env_ids, 0]).uniform_(
            self.cfg.position_range["x"][0], 
            self.cfg.position_range["x"][1]
        )
        self._desired_pos_b[env_ids, 1] = torch.zeros_like(self._desired_pos_b[env_ids, 1]).uniform_(
            self.cfg.position_range["y"][0], 
            self.cfg.position_range["y"][1]
        )
        self._desired_pos_b[env_ids, 2] = torch.zeros_like(self._desired_pos_b[env_ids, 2]).uniform_(
            self.cfg.position_range["z"][0], 
            self.cfg.position_range["z"][1]
        )
        
        # Sample orientations using the same pattern
        roll = torch.zeros(len(env_ids), device=self.device).uniform_(
            self.cfg.orientation_range["roll"][0], 
            self.cfg.orientation_range["roll"][1]
        )
        pitch = torch.zeros(len(env_ids), device=self.device).uniform_(
            self.cfg.orientation_range["pitch"][0], 
            self.cfg.orientation_range["pitch"][1]
        )
        yaw = torch.zeros(len(env_ids), device=self.device).uniform_(
            self.cfg.orientation_range["yaw"][0], 
            self.cfg.orientation_range["yaw"][1]
        )
        
        # Convert to quaternion
        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        self._desired_quat_b[env_ids] = quat
        
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Setup debug visualization."""
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                # Create marker for target pose
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.markers["frame"].size = (0.01, 0.01, 0.01)
                marker_cfg.prim_path = "/Visuals/target_pose"
                self.target_visualizer = VisualizationMarkers(marker_cfg)
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)
                
    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        # Transform target positions to world frame
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        
        desired_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, self._desired_pos_b
        )
        
        # Visualize target positions
        self.target_visualizer.visualize(desired_pos_w)