"""
Direct RL Environment for Object Camera Pose Tracking with UR5 Robot
Modified to use sphere obstacle and control density function rewards

This implements a multi-observation space environment compatible with skrl.
"""

from __future__ import annotations

import torch
import numpy as np
import math
import csv
import os
from datetime import datetime
import random
from typing import Dict, Any, Tuple, Optional, Sequence
import gymnasium as gym

# IsaacLab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import sample_uniform

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# Robot configuration
from isaaclab_assets.robots.ur5 import UR5_GRIPPER_CFG

# Custom utilities - with fallback
try:
    from .thresholds import *
except ImportError:
    # Define minimal thresholds if file not found
    TABLE_HEIGHT = 0.72
    CUBE_HEIGHT = 0.0382
    CUBE_WIDTH = 0.0286
    CUBE_LENGTH = 0.0635
    CUBE_START_HEIGHT = TABLE_HEIGHT + (CUBE_HEIGHT / 2)
    PLACEMENT_POS_THRESHOLD = 0.05
    GRIPPER_OPEN_THRESHOLD = 5.0
    GRIPPER_CLOSED_THRESHOLD = 25.0
    GRIPPER_CLOSING_THRESHOLD = 15.0
    POSITION_THRESHOLD = 0.05
    ORIENTATION_THRESHOLD = 0.9
    CUBE_HOVER_HEIGHT = 0.3
    PRE_GRASP_HEIGHT = 0.1
    VELOCITY_THRESHOLD = 0.05
    TORQUE_THRESHOLD = 1.0
    CUBE_MAX_HEIGHT = 1.0
    DISTANCE_SCALE = 0.1


# Control Density Function Implementation
class SphereObstacle:
    """Sphere obstacle with control density function for reward computation."""
    
    def __init__(self, center: torch.Tensor, radius: float, sensing: float,
                 alpha: float = 0.1, target_state: torch.Tensor = None,
                 max_density: float = 1e3, device: str = 'cuda'):
        """
        Initialize sphere obstacle for density computation.
        
        Args:
            center: 3D position of sphere center (torch tensor)
            radius: Sphere radius
            sensing: Sensing radius (should be > radius)
            alpha: Exponent for Lyapunov term
            target_state: Target 3D position
            max_density: Maximum cap for density value
            device: Computation device
        """
        self.device = device
        self.center = center.to(device)
        self.r = float(radius)
        self.s = float(sensing)
        self.alpha = float(alpha)
        self.max_density = float(max_density)
        self.target_state = target_state.to(device) if target_state is not None else None
        
    def _bump(self, val: torch.Tensor) -> torch.Tensor:
        """Bump function for smooth transitions."""
        return torch.where(val > 0, torch.exp(-1.0 / val), torch.zeros_like(val))
    
    def Phi_function(self, states: torch.Tensor) -> torch.Tensor:
        """
        Inverse bump function for safe density.
        
        Args:
            states: Batch of 3D positions (N x 3)
        
        Returns:
            Phi values for each state (N,)
        """
        # Compute distance to obstacle center
        diff = states - self.center.unsqueeze(0)
        shape = torch.sum(diff ** 2, dim=1) - self.r ** 2
        denom = self.s ** 2 - self.r ** 2
        
        if denom == 0:
            return torch.zeros(states.shape[0], device=self.device)
        
        temp1 = shape / denom
        bump1 = self._bump(temp1)
        bump2 = self._bump(1 - temp1)
        
        denominator = bump1 + bump2
        return torch.where(denominator != 0, bump1 / denominator, torch.zeros_like(bump1))
    
    def V_function(self, states: torch.Tensor) -> torch.Tensor:
        """
        Squared 2-norm distance from current to target state.
        
        Args:
            states: Batch of 3D positions (N x 3)
        
        Returns:
            V values for each state (N,)
        """
        if self.target_state is None:
            raise ValueError("Target state must be set for V_function computation")
        
        diff = states - self.target_state.unsqueeze(0)
        return torch.sum(diff ** 2, dim=1)
    
    def density(self, states: torch.Tensor) -> torch.Tensor:
        """
        Full density calculation with capped value.
        
        Args:
            states: Batch of 3D positions (N x 3)
        
        Returns:
            Density values for each state (N,)
        """
        V = self.V_function(states)
        Phi = self.Phi_function(states)
        
        # Avoid division by zero
        V_safe = torch.where(V != 0, V, torch.ones_like(V) * 1e-6)
        # rho = Phi / (V_safe ** self.alpha)
        rho = Phi
        
        return torch.clamp(rho, max=self.max_density)
    
    def update_center(self, new_center: torch.Tensor):
        """Update sphere center position."""
        self.center = new_center.to(self.device)
    
    def update_target(self, new_target: torch.Tensor):
        """Update target position."""
        self.target_state = new_target.to(self.device)




##
# Environment Configuration
##

@configclass
class SphereObstacleCDFEnvCfg(DirectRLEnvCfg):
    """Configuration for the direct RL environment with sphere obstacle and CDF rewards."""
    
    # Visualization settings
    debug_vis = False  # Enable/disable debug visualization

    # Sphere obstacle settings
    sphere_radius = 0.05
    sphere_sensing_radius = 2*sphere_radius
    sphere_position_bounds = {
        "x": (0.2, 0.6),
        "y": (-0.3, 0.3),
        "z": (0.8, 1.0),
    }

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    
    # UR5 Robot
    robot_cfg: ArticulationCfg = UR5_GRIPPER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Sphere obstacle configuration - red colored rigid sphere
    sphere_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/sphere_obstacle",
        spawn=sim_utils.SphereCfg(
            radius=sphere_radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,  # Static obstacle
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Red color
                metallic=0.2,
                roughness=0.5,
                opacity=1.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.0, 0.9),  # Will be randomized on reset
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # White plane configuration
    white_plane_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/white_plane",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 2.81, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
                metallic=0.0,
                roughness=0.1,
                opacity=1.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2, 0.0, 0.8),
            rot=(0.70711, 0.0, 0.70711, 0.0)
        ),
    )

    # Frame transformer for end-effector
    ee_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/ee_link",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.1226, 0.0, 0.0],
                ),
            ),
        ],
    )
    
    # Camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.955,
            focus_distance=28.0,
            horizontal_aperture=5.229,
            vertical_aperture=2.942,
            clipping_range=(0.1, 1000.0)
        ),
        width=640,
        height=360,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.27, 0.06, 1.143),
            rot=(0.59637, 0.37993, 0.37993, 0.59637),
            convention="opengl"
        )
    )

    # Basic environment settings
    episode_length_s = 8.0
    decimation = 2
    action_scale = 0.5
    state_dim = 19
    camera_target_height = 180
    camera_target_width = 320    

    # Observation and action spaces
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
    state_space = 0
    observation_space = gym.spaces.Dict({
        "image": gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(camera_target_height, camera_target_width, 3)),
        "state": gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(state_dim,)),
    })
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0/120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        device="cuda:0",
    )

    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=2.0,
    )
    
    # Command settings
    command_resampling_time = 16.0
    target_pose_range = {
        "x": (0.4, 0.7),
        "y": (-0.5, 0.5),
        "z": (0.8, 1.1),
        "roll": (0.0, 0.0),
        "pitch": (1.57, 1.57),
        "yaw": (0.0, 0.0),
    }
    
    # Control Density Function parameters
    cdf_alpha = 0.1
    cdf_max_density = 100
    cdf_reward_weight = 10.0
    
    # Reward settings
    reward_distance_weight = -2.5
    reward_distance_tanh_weight = 1.5
    reward_distance_tanh_std = 0.1
    reward_orientation_weight = -1.0
    reward_sphere_avoidance_weight = 7.0

    # Artificial Potential Field parameters
    apf_critical_distance = 0.05
    apf_smoothness = 0.1
    energy_reward_weight = -1.0

    # Huber loss parameters
    huber_delta = 0.08
    success_bonus = 5.0
    
    # Termination settings
    position_threshold = 0.01
    orientation_threshold = 0.05
    velocity_threshold = 0.05
    bounds_safety_margin = 0.1
    
    # Camera preprocessing settings
    camera_crop_top = 30
    camera_crop_bottom = 20
    
    # Visualization settings
    visualize_camera_interval = 20000
    visualization_save_path = "/home/adi2440/Desktop/camera_obs"
    
    # Noise settings
    joint_pos_noise_min = -0.01
    joint_pos_noise_max = 0.01
    joint_vel_noise_min = -0.001
    joint_vel_noise_max = 0.001
    
    # Reset settings
    robot_base_pose = [-0.568, -0.858, 1.402, -2.185, -1.6060665, 1.64142667]
    robot_reset_noise_range = 0.1


class SphereObstacleCDFEnv(DirectRLEnv):
    """Direct RL environment with sphere obstacle and control density function rewards."""
    
    cfg: SphereObstacleCDFEnvCfg
    
    def __init__(self, cfg: SphereObstacleCDFEnvCfg, render_mode: str | None = None, **kwargs):
        # Store config
        self.cfg = cfg

        # Episode / logging bookkeeping
        self._episode_counter = 0
        self._state_obs_file = None
        self._state_csv_writer = None
        self._image_obs_dir = None
        
        # Initialize parent
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize extras dictionary for logging
        self.extras = {"log": {}}
        
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
        self._robot_dof_targets = torch.zeros(
            (self.num_envs, len(self._joint_indices)), device=self.device
        )
        self._target_poses = torch.zeros((self.num_envs, 7), device=self.device)
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)
        
        # Sphere obstacle positions
        self._sphere_positions = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Initialize control density functions for each environment
        self._cdf_obstacles = []
        for i in range(self.num_envs):
            cdf = SphereObstacle(
                center=self._sphere_positions[i],
                radius=self.cfg.sphere_radius,
                sensing=self.cfg.sphere_sensing_radius,
                alpha=self.cfg.cdf_alpha,
                target_state=self._target_poses[i, :3],
                max_density=self.cfg.cdf_max_density,
                device=self.device
            )
            self._cdf_obstacles.append(cdf)
        
        # Performance tracking
        self._episode_sums = {
            "position_error": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
            "success_count": torch.zeros(self.num_envs, device=self.device),
            "min_sphere_distance": torch.ones(self.num_envs, device=self.device) * float('inf'),
        }
        
        # Initialize gradient computation buffers
        self._prev_potentials = None
        self._prev_densities = None
        self._prev_ee_positions = None
        self._reset_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Log initial information
        print(f"[INFO] Environment initialized with {self.num_envs} environments")
        print(f"[INFO] Using sphere obstacle with radius: {self.cfg.sphere_radius}m")
        print(f"[INFO] CDF alpha parameter: {self.cfg.cdf_alpha}")
        
        # Setup debug visualization if enabled
        self.set_debug_vis(self.cfg.debug_vis)
        
        # Create visualization directory
        if not os.path.exists(self.cfg.visualization_save_path):
            os.makedirs(self.cfg.visualization_save_path)
        
        # Initialize visualization counter
        self._vis_counter = 0

    def close(self):
        """Cleanup for the environment."""
        super().close()
        
    def _setup_scene(self):
        """Set up the scene with robots, sphere obstacle, cameras, etc."""
        # Spawn all prims in the source environment only
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        self._sphere = RigidObject(self.cfg.sphere_cfg)
        self._white_plane = RigidObject(self.cfg.white_plane_cfg)

        # Clone to other environments
        self.scene.clone_environments(copy_from_source=False)

        # Register handles in IsaacLab's scene registry
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.rigid_objects["sphere"] = self._sphere
        self.scene.rigid_objects["white_plane"] = self._white_plane

        # Add static geometry and lighting
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/DomeLight", light_cfg)
        
        dir_light_cfg = sim_utils.DistantLightCfg(intensity=1000.0, color=(1.0, 1.0, 0.9), angle=0.53)
        dir_light_cfg.func("/World/DirectionalLight", dir_light_cfg)
            
    def _sample_commands(self, env_ids: Sequence[int]) -> None:
        """Sample new target poses for specified environments."""
        num_envs = len(env_ids)
        
        # Sample positions randomly within full target range
        self._target_poses[env_ids, 0] = sample_uniform(
            self.cfg.target_pose_range["x"][0],
            self.cfg.target_pose_range["x"][1],
            (num_envs,), self.device
        )
        
        self._target_poses[env_ids, 1] = sample_uniform(
            self.cfg.target_pose_range["y"][0],
            self.cfg.target_pose_range["y"][1],
            (num_envs,), self.device
        )
        
        self._target_poses[env_ids, 2] = sample_uniform(
            self.cfg.target_pose_range["z"][0],
            self.cfg.target_pose_range["z"][1],
            (num_envs,), self.device
        )

        # Sample orientations
        roll = sample_uniform(
            self.cfg.target_pose_range["roll"][0],
            self.cfg.target_pose_range["roll"][1],
            (num_envs,), self.device
        )
        pitch = sample_uniform(
            self.cfg.target_pose_range["pitch"][0],
            self.cfg.target_pose_range["pitch"][1],
            (num_envs,), self.device
        )
        yaw = sample_uniform(
            self.cfg.target_pose_range["yaw"][0],
            self.cfg.target_pose_range["yaw"][1],
            (num_envs,), self.device
        )

        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        self._target_poses[env_ids, 3:7] = quat
        
        # Update CDF target states
        for env_id in env_ids:
            self._cdf_obstacles[env_id].update_target(self._target_poses[env_id, :3])
            
    def _get_observations(self) -> dict:
        """Get multi-modal observations."""
        # Get camera observation
        camera_obs = self._get_camera_observation()
        
        # Get state observation
        state_obs = self._get_state_observation()
        
        # Add joint position noise
        joint_pos_noise = sample_uniform(
            self.cfg.joint_pos_noise_min,
            self.cfg.joint_pos_noise_max,
            (self.num_envs, 6),
            self.device
        )
        
        # Add joint velocity noise
        joint_vel_noise = sample_uniform(
            self.cfg.joint_vel_noise_min,
            self.cfg.joint_vel_noise_max,
            (self.num_envs, 6),
            self.device
        )
        
        # Apply noise to joint observations in state
        state_obs[:, :6] += joint_pos_noise
        state_obs[:, 6:12] += joint_vel_noise
        
        # Create observation dict compatible with skrl
        observations = {
            "policy": {
                "image": camera_obs.permute(0, 2, 3, 1),  # Convert to HWC format
                "state": state_obs,
            }
        }
        
        return observations
        
    def _get_state_observation(self) -> torch.Tensor:
        """Get proprioceptive state observation including joint velocities."""
        # Get joint positions (6 values)
        joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        
        # Get joint velocities (6 values)
        joint_vel = self._robot.data.joint_vel[:, self._joint_indices]
        
        # Get end-effector pose relative to robot base (7 values)
        ee_pos_b = self._ee_frame.data.target_pos_source[..., 0, :]
        ee_quat_b = self._ee_frame.data.target_quat_source[..., 0, :]
        
        # Combine into state vector (19 values: 6 + 6 + 7)
        state = torch.cat([
            joint_pos,      # 6 joint positions
            joint_vel,      # 6 joint velocities
            ee_pos_b,       # 3 EE position
            ee_quat_b,      # 4 EE orientation
        ], dim=-1)
        
        return state
        
    def _get_camera_observation(self) -> torch.Tensor:
        """Process camera observation with cropping and resizing."""
        # Get camera data
        camera_data = self._tiled_camera.data.output["rgb"] / 255.0
        
        # Store raw image for visualization
        raw_camera_data = camera_data.clone()
        
        # Mean subtraction for normalization
        mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        camera_data = camera_data - mean_tensor
        
        # Crop image (top and bottom)
        cropped = camera_data[
            :,
            self.cfg.camera_crop_top:-self.cfg.camera_crop_bottom,
            :,
            :
        ]
        
        # Resize to target size using interpolation
        # Convert to NCHW format for processing
        cropped = cropped.permute(0, 3, 1, 2)

        # Resize using torch interpolation
        resized = torch.nn.functional.interpolate(
            cropped,
            size=(self.cfg.camera_target_height, self.cfg.camera_target_width),
            mode='bilinear',
            align_corners=False
        )

        # Visualize camera observation periodically
        if self.common_step_counter % self.cfg.visualize_camera_interval == 0:
            self._visualize_camera_observation(raw_camera_data, resized, env_id=0)
        
        return resized
        
    def _visualize_camera_observation(self, raw_obs: torch.Tensor, processed_obs: torch.Tensor, env_id: int = 0):
        """Visualize camera observations for debugging."""
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Raw observation
        raw_img = raw_obs[env_id].cpu().numpy()
        axes[0].imshow(raw_img)
        axes[0].set_title(f'Raw Camera (640x360)')
        axes[0].axis('off')
        
        # Add crop lines
        crop_top = self.cfg.camera_crop_top
        crop_bottom = raw_obs.shape[1] - self.cfg.camera_crop_bottom
        axes[0].axhline(y=crop_top, color='r', linestyle='--', linewidth=2)
        axes[0].axhline(y=crop_bottom, color='r', linestyle='--', linewidth=2)
        
        # Processed observation
        processed_img = processed_obs[env_id].permute(1, 2, 0).cpu().numpy()
        # Denormalize for visualization
        processed_img = (processed_img - processed_img.min()) / (processed_img.max() - processed_img.min() + 1e-8)
        axes[1].imshow(processed_img)
        axes[1].set_title(f'Processed ({self.cfg.camera_target_width}x{self.cfg.camera_target_height})')
        axes[1].axis('off')
        
        plt.suptitle(f'Camera Observation - Step {self.common_step_counter}')
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(
            self.cfg.visualization_save_path,
            f'camera_obs_step_{self.common_step_counter}_vis_{self._vis_counter}.png'
        )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self._vis_counter += 1
        print(f"[VIS] Saved camera observation to: {save_path}")
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        # Store raw actions
        self.actions = actions.clone().clamp(-1.0, 1.0) * self.cfg.action_scale
        
        # Update command timer
        self._command_time_left -= self.physics_dt
        
        # Resample target poses when timer runs out
        expired_mask = self._command_time_left <= 0.0
        if torch.any(expired_mask):
            expired_ids = torch.nonzero(expired_mask, as_tuple=False).squeeze(-1)
            env_ids = expired_ids.cpu().tolist()
            self._sample_commands(env_ids)
            # Reset their countdown
            self._command_time_left[expired_mask] = self.cfg.command_resampling_time

        # Update debug visualization if enabled
        self._update_debug_visualization()
            
    def _apply_action(self) -> None:
        """Apply the processed actions to the robot with safety checks."""
        # Get current joint positions
        current_joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        
        # Add actions to current positions for position control
        self._robot_dof_targets = current_joint_pos + self.actions
        
        # Clamp to joint limits with safety margin
        safety_margin = 0.05
        self._robot_dof_targets = torch.clamp(
            self._robot_dof_targets,
            self._robot_dof_lower_limits + safety_margin,
            self._robot_dof_upper_limits - safety_margin
        )
        
        # Apply velocity limits for safety
        max_velocity = 1.5
        velocity_command = (self._robot_dof_targets - current_joint_pos) / self.physics_dt
        velocity_command = torch.clamp(velocity_command, -max_velocity, max_velocity)
        self._robot_dof_targets = current_joint_pos + velocity_command * self.physics_dt
        
        # Set joint position targets
        self._robot.set_joint_position_target(
            self._robot_dof_targets, joint_ids=self._joint_indices
        )
        
    def _huber_loss(self, x: torch.Tensor, delta: float) -> torch.Tensor:
        """Compute Huber loss for robust distance penalty."""
        abs_x = torch.abs(x)
        return torch.where(
            abs_x <= delta,
            0.5 * x * x,
            delta * (abs_x - 0.5 * delta)
        )
        
    def _compute_cdf_gradient_reward(self, ee_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute reward based on the gradient (change) of CDF density.
        Positive change in density = moving toward better configuration = positive reward
        """
        cdf_rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Compute current densities for all environments
        current_densities = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            density = self._cdf_obstacles[i].density(ee_positions[i:i+1])
            current_densities[i] = density[0]
        
        # If this is the first step after reset, initialize previous densities
        if self._prev_densities is None:
            self._prev_densities = current_densities.clone()
            return cdf_rewards
        
        # Compute density gradients (change in density)
        density_gradients = current_densities - self._prev_densities
        
        # Normalize gradients
        gradient_scale = 1.0
        normalized_gradients = torch.tanh(density_gradients / gradient_scale)
        
        # Apply reward weight
        cdf_rewards = normalized_gradients * self.cfg.cdf_reward_weight
        
        # Optional: Add small penalty for low absolute density
        density_baseline_penalty = -0.1 * torch.exp(-current_densities / 10.0)
        cdf_rewards += density_baseline_penalty * self.cfg.cdf_reward_weight
        
        # Update previous densities for next step
        self._prev_densities = current_densities.clone()
        
        # Log gradient info occasionally
        if self.common_step_counter % 500 == 0 and self.num_envs > 0:
            print(f"[CDF GRADIENT] Env 0 - Current density: {current_densities[0]:.3f}, "
                  f"Gradient: {density_gradients[0]:.3f}, "
                  f"Normalized gradient: {normalized_gradients[0]:.3f}")
        
        return cdf_rewards

    def _initialize_gradient_buffers_after_reset(self, ee_positions: torch.Tensor):
        """Initialize gradient computation buffers for recently reset environments."""
        if hasattr(self, '_reset_env_mask') and self._reset_env_mask.any():
            reset_envs = torch.where(self._reset_env_mask)[0]
            
            # Initialize previous densities
            if hasattr(self, '_prev_densities'):
                if self._prev_densities is None:
                    self._prev_densities = torch.zeros(self.num_envs, device=self.device)
                
                for env_id in reset_envs:
                    density = self._cdf_obstacles[env_id].density(ee_positions[env_id:env_id+1])
                    self._prev_densities[env_id] = density[0]
            
            # Initialize previous positions
            if hasattr(self, '_prev_ee_positions'):
                if self._prev_ee_positions is None:
                    self._prev_ee_positions = ee_positions.clone()
                else:
                    self._prev_ee_positions[reset_envs] = ee_positions[reset_envs]
            
            # Initialize previous potentials
            if hasattr(self, '_prev_potentials'):
                if self._prev_potentials is None:
                    self._prev_potentials = torch.zeros(self.num_envs, device=self.device)
                
                for env_id in reset_envs:
                    density = self._cdf_obstacles[env_id].density(ee_positions[env_id:env_id+1])
                    self._prev_potentials[env_id] = torch.log1p(density[0])
            
            # Clear reset mask
            self._reset_env_mask[reset_envs] = False

    def _compute_beta_transition(self, min_distances: torch.Tensor) -> torch.Tensor:
        """Compute smooth transition factor β for adaptive reward mixing."""
        x = (min_distances - self.cfg.apf_critical_distance) / self.cfg.apf_smoothness
        beta = (torch.tanh(x) + 1.0) / 2.0
        return beta

    def _compute_energy_reward(self) -> torch.Tensor:
        """Compute energy-based reward from joint velocities."""
        joint_velocities = self._robot.data.joint_vel[:, self._joint_indices]
        # Compute norm squared for each joint
        velocity_norms_squared = joint_velocities ** 2
        # Sum tanh over all 6 joints for each environment
        energy_reward = -torch.sum(torch.tanh(velocity_norms_squared), dim=1)
        return energy_reward

    def _compute_sphere_avoidance_rewards(self) -> torch.Tensor:
        """Compute sphere avoidance rewards with dynamic collision risk assessment."""
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Safety parameters
        critical_distance = 0.03
        danger_distance = 0.08
        safe_distance = 0.12
        
        # Get sphere pose and velocity
        sphere_world_positions = self._sphere_positions + self.scene.env_origins
        sphere_velocity = torch.zeros_like(sphere_world_positions)
        
        # Get end effector position and velocity
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_velocity = self._ee_frame.data.target_lin_vel_w[..., 0, :] if hasattr(self._ee_frame.data, 'target_lin_vel_w') else torch.zeros_like(ee_position)
        
        # Calculate minimum distance from end effector to sphere surface
        ee_to_sphere = ee_position - sphere_world_positions
        min_distances = torch.norm(ee_to_sphere, dim=-1) - self.cfg.sphere_radius
        
        # Calculate relative velocity
        relative_vel = ee_velocity - sphere_velocity
        relative_speed = torch.sum(ee_to_sphere * relative_vel, dim=-1) / (torch.norm(ee_to_sphere, dim=-1) + 1e-6)
        
        # Dynamic penalty based on both distance and relative motion
        for i in range(self.num_envs):
            distance = min_distances[i]
            
            if distance < critical_distance:
                rewards[i] = -15.0
            elif distance < danger_distance:
                base_penalty = -8.0 * (1.0 - (distance - critical_distance) / (danger_distance - critical_distance))
                
                if relative_speed[i] > 0:
                    motion_factor = torch.clamp(relative_speed[i] / 0.5, 0.0, 0.7)
                    rewards[i] = base_penalty * (1.0 - motion_factor)
                else:
                    motion_factor = torch.clamp(-relative_speed[i] / 0.5, 0.0, 0.5)
                    rewards[i] = base_penalty * (1.0 + motion_factor)
            elif distance < safe_distance:
                base_penalty = -2.0 * (1.0 - (distance - danger_distance) / (safe_distance - danger_distance))
                
                if relative_speed[i] < 0:
                    rewards[i] = base_penalty
                else:
                    rewards[i] = 0.0
            else:
                rewards[i] = 0.0
        
        return rewards

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards using Artificial Potential Field approach with Huber loss."""
        # Initialize rewards
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Get end-effector position and orientation
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat = self._ee_frame.data.target_quat_w[..., 0, :]
        
        # Transform target pose to world frame
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        
        des_pos_b = self._target_poses[:, :3]
        des_quat_b = self._target_poses[:, 3:7]
        
        des_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, des_pos_b
        )
        des_quat_w = math_utils.quat_mul(robot_quat, des_quat_b)
        
        # Calculate distances to sphere obstacle
        sphere_world_positions = self._sphere_positions + self.scene.env_origins
        ee_to_sphere = ee_position - sphere_world_positions
        min_distances_to_sphere = torch.norm(ee_to_sphere, dim=-1) - self.cfg.sphere_radius
        
        # Compute β transition factor for APF
        beta = self._compute_beta_transition(min_distances_to_sphere)
        
        # === Traditional Rewards (Rt) ===
        traditional_rewards = torch.zeros_like(rewards)
        
        # 1. Position tracking with Huber loss
        position_error = torch.norm(ee_position - des_pos_w, dim=-1)
        position_huber_loss = self._huber_loss(position_error, self.cfg.huber_delta)
        position_reward = self.cfg.reward_distance_weight * position_huber_loss
        traditional_rewards += position_reward
        
        # 2. Position tracking tanh reward (smooth near goal)
        position_reward_tanh = 1.0 - torch.tanh(position_error / self.cfg.reward_distance_tanh_std)
        position_reward_tanh_scaled = self.cfg.reward_distance_tanh_weight * position_reward_tanh
        traditional_rewards += position_reward_tanh_scaled
        
        # 3. Orientation tracking reward with Huber loss
        orientation_error = math_utils.quat_error_magnitude(ee_quat, des_quat_w)
        orientation_huber_loss = self._huber_loss(orientation_error, self.cfg.huber_delta * 0.5)
        orientation_reward = self.cfg.reward_orientation_weight * orientation_huber_loss
        traditional_rewards += orientation_reward
        
        # 4. Sphere avoidance rewards
        sphere_reward = self._compute_sphere_avoidance_rewards() * self.cfg.reward_sphere_avoidance_weight
        traditional_rewards += sphere_reward

        # 5. Success bonus
        joint_velocities = torch.norm(self._robot.data.joint_vel, p=2, dim=-1)
        success_mask = (position_error < 0.05) & (min_distances_to_sphere > 0.08) & (joint_velocities < self.cfg.velocity_threshold)
        traditional_rewards += torch.where(success_mask, self.cfg.success_bonus, 0.0)
        
        # === Energy-based Rewards (Renergy) ===
        energy_rewards = self._compute_energy_reward() * self.cfg.energy_reward_weight
        
        # === Adaptive Combination using APF ===
        # R_adaptive = β · Rt + (1 - β) · Renergy
        rewards = beta * traditional_rewards + (1.0 - beta) * energy_rewards
        
        # Track reward components for logging
        if hasattr(self, '_episode_sums'):
            self._episode_sums["total_reward"] += rewards
            self._episode_sums["position_error"] += position_error
            self._episode_sums["min_sphere_distance"] = torch.minimum(
                self._episode_sums["min_sphere_distance"], min_distances_to_sphere
            )
            
            # Check for success
            success_mask = (position_error < 0.05) & (min_distances_to_sphere > 0.08)
            self._episode_sums["success_count"] += success_mask.float()
        
        # Log detailed reward breakdown occasionally
        if self.common_step_counter % 500 == 0 and self.num_envs > 0:
            env_0_data = {
                "position_error": position_error[0].item(),
                "position_huber": position_huber_loss[0].item(),
                "orientation_error": orientation_error[0].item(),
                "min_dist_to_sphere": min_distances_to_sphere[0].item(),
                "beta": beta[0].item(),
                "energy_reward": energy_rewards[0].item(),
                "total_reward": rewards[0].item()
            }
        
        return rewards
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get termination signals."""
        # Time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Get end-effector position
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        
        # Check for collisions with table
        table_collision = ee_position[:, 2] < (TABLE_HEIGHT - 0.01)
        
        # Combine termination conditions
        terminated = table_collision 
        
        return terminated, time_out
        
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # Call parent reset first
        super()._reset_idx(env_ids)
        
        # Print episode statistics for completed environments
        if len(env_ids) > 0 and hasattr(self, '_episode_sums'):
            avg_position_error = self._episode_sums["position_error"][env_ids].mean().item()
            avg_reward = self._episode_sums["total_reward"][env_ids].mean().item()
            success_rate = self._episode_sums["success_count"][env_ids].mean().item()
            min_sphere_dist = self._episode_sums["min_sphere_distance"][env_ids].mean().item()
            
            if self.common_step_counter % 1000 == 0:
                print(f"[INFO] Episode stats - Pos error: {avg_position_error:.4f}, "
                    f"Reward: {avg_reward:.2f}, Success: {success_rate:.2f}, "
                    f"Min sphere dist: {min_sphere_dist:.3f}")
        
        # Reset episode tracking
        if hasattr(self, '_episode_sums'):
            for key in self._episode_sums:
                if key == "min_sphere_distance":
                    self._episode_sums[key][env_ids] = float('inf')
                else:
                    self._episode_sums[key][env_ids] = 0.0
        
        # Reset robot joint positions
        num_resets = len(env_ids)
        
        # Base joint positions
        base_pose = torch.tensor(
            self.cfg.robot_base_pose,
            device=self.device,
            dtype=torch.float32
        )
        
        # Add noise to joint positions
        joint_pos = base_pose.unsqueeze(0).repeat(num_resets, 1)
        if self.cfg.robot_reset_noise_range > 0:
            joint_pos += sample_uniform(
                -self.cfg.robot_reset_noise_range,
                self.cfg.robot_reset_noise_range,
                joint_pos.shape,
                self.device
            )
        
        # Clamp to joint limits
        joint_pos = torch.clamp(
            joint_pos,
            self._robot_dof_lower_limits.unsqueeze(0),
            self._robot_dof_upper_limits.unsqueeze(0)
        )
        
        joint_vel = torch.zeros_like(joint_pos)
        
        # Set joint state
        self._robot.write_joint_state_to_sim(
            joint_pos, joint_vel,
            joint_ids=self._joint_indices,
            env_ids=env_ids
        )
        
        # Reset sphere obstacle positions
        sphere_poses = torch.zeros((num_resets, 7), device=self.device)
        
        for i, env_id in enumerate(env_ids):
            # Random position within bounds
            self._sphere_positions[env_id, 0] = sample_uniform(
                self.cfg.sphere_position_bounds["x"][0],
                self.cfg.sphere_position_bounds["x"][1],
                (1,), self.device
            )
            
            self._sphere_positions[env_id, 1] = sample_uniform(
                self.cfg.sphere_position_bounds["y"][0],
                self.cfg.sphere_position_bounds["y"][1],
                (1,), self.device
            )
            
            self._sphere_positions[env_id, 2] = sample_uniform(
                self.cfg.sphere_position_bounds["z"][0],
                self.cfg.sphere_position_bounds["z"][1],
                (1,), self.device
            )
            
            # Add environment origin offset
            sphere_poses[i, :3] = self._sphere_positions[env_id] + self.scene.env_origins[env_id]
            sphere_poses[i, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            
            # Update CDF obstacle center
            self._cdf_obstacles[env_id].update_center(self._sphere_positions[env_id])
        
        # Set sphere positions in simulation
        self._sphere.write_root_pose_to_sim(
            root_pose=sphere_poses,
            env_ids=env_ids
        )
        
        # Reset target poses and timers
        self._command_time_left[env_ids] = 0.0
        
        # Sample new target poses - randomly within full range
        num_envs = len(env_ids)
        
        x = sample_uniform(
            self.cfg.target_pose_range["x"][0],
            self.cfg.target_pose_range["x"][1],
            (num_envs,), self.device
        )
        
        y = sample_uniform(
            self.cfg.target_pose_range["y"][0],
            self.cfg.target_pose_range["y"][1],
            (num_envs,), self.device
        )
        
        z = sample_uniform(
            self.cfg.target_pose_range["z"][0],
            self.cfg.target_pose_range["z"][1],
            (num_envs,), self.device
        )
        
        # Sample orientations
        roll = sample_uniform(
            self.cfg.target_pose_range["roll"][0],
            self.cfg.target_pose_range["roll"][1],
            (num_envs,), self.device
        )
        pitch = sample_uniform(
            self.cfg.target_pose_range["pitch"][0],
            self.cfg.target_pose_range["pitch"][1],
            (num_envs,), self.device
        )
        yaw = sample_uniform(
            self.cfg.target_pose_range["yaw"][0],
            self.cfg.target_pose_range["yaw"][1],
            (num_envs,), self.device
        )
        
        # Convert euler to quaternion
        target_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        
        # Update target pose buffers
        self._target_poses[env_ids, 0] = x
        self._target_poses[env_ids, 1] = y
        self._target_poses[env_ids, 2] = z
        self._target_poses[env_ids, 3:7] = target_quat
        
        # Update CDF target states
        for idx, env_id in enumerate(env_ids):
            target_pos = torch.tensor([x[idx], y[idx], z[idx]], device=self.device)
            self._cdf_obstacles[env_id].update_target(target_pos)
        
        # Reset joint targets
        self._robot_dof_targets[env_ids] = joint_pos
        
        # Reset gradient computation buffers
        if hasattr(self, '_prev_densities'):
            if self._prev_densities is None:
                pass
            else:
                if not hasattr(self, '_reset_env_mask'):
                    self._reset_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                self._reset_env_mask[env_ids] = True
        
        # Reset command timer
        self._command_time_left[env_ids] = self.cfg.command_resampling_time

    def set_debug_vis(self, debug_vis: bool) -> None:
        """Set debug visualization mode."""
        self.cfg.debug_vis = debug_vis
        if hasattr(self, "_ee_frame") and self._ee_frame is not None:
            self._ee_frame.set_debug_vis(debug_vis)
        
        self._set_debug_vis_impl(debug_vis)

    def _update_debug_visualization(self):
        """Update debug visualization markers."""
        if not self.cfg.debug_vis or not hasattr(self, "target_pos_visualizer"):
            return
            
        # Update target pose markers
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        
        des_pos_b = self._target_poses[:, :3]
        des_quat_b = self._target_poses[:, 3:7]
        
        # Transform to world frame
        des_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, des_pos_b
        )
        des_quat_w = math_utils.quat_mul(robot_quat, des_quat_b)
        
        # Visualize the target positions
        self.target_pos_visualizer.visualize(translations=des_pos_w, orientations=des_quat_w)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or hide debug visualization markers."""
        if debug_vis:
            if not hasattr(self, "target_pos_visualizer"):
                target_marker_cfg = FRAME_MARKER_CFG.copy()
                target_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
                target_marker_cfg.prim_path = "/Visuals/Command/target_position"
                self.target_pos_visualizer = VisualizationMarkers(target_marker_cfg)
            self.target_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_pos_visualizer"):
                self.target_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization markers."""
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        
        des_pos_b = self._target_poses[:, :3]
        des_quat_b = self._target_poses[:, 3:7]
        
        des_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, des_pos_b
        )
        des_quat_w = math_utils.quat_mul(robot_quat, des_quat_b)
        
        self.target_pos_visualizer.visualize(translations=des_pos_w, orientations=des_quat_w)


# Factory function for creating the environment
def create_sphere_obstacle_cdf_env(
    cfg: SphereObstacleCDFEnvCfg = None,
    render_mode: str = None,
    **kwargs
) -> SphereObstacleCDFEnv:
    """Factory function to create the environment with default config if none provided."""
    if cfg is None:
        cfg = SphereObstacleCDFEnvCfg()
    
    return SphereObstacleCDFEnv(cfg, render_mode=render_mode, **kwargs)