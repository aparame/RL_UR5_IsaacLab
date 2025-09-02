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
        rho = Phi / (V_safe ** self.alpha)
        
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
        "x": (0.2, 0.6),  # Modified x bounds as requested
        "y": (-0.3, 0.3),
        "z": (0.8, 1.0),
    }

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    
    # UR5 Robot
    robot_cfg: ArticulationCfg = UR5_GRIPPER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # # Table configuration
    # table_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="/home/adi2440/Desktop/ur5_isaacsim/usd/table.usd",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             rigid_body_enabled=True,
    #             kinematic_enabled=True,
    #             disable_gravity=True,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             collision_enabled=False
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.6, 0.0, -0.0234), 
    #         rot=(1.0, 0.0, 0.0, 0.0)
    #     ),
    # )

    # Sphere obstacle configuration - red colored rigid sphere
    sphere_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/sphere_obstacle",
        spawn=sim_utils.SphereCfg(
            radius=sphere_radius,  # 0.1m radius
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
        width=320,
        height=180,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.27, 0.06, 1.143),
            rot=(0.59637, 0.37993, 0.37993, 0.59637),
            convention="opengl"
        )
    )

    # Basic environment settings
    episode_length_s = 6.0
    decimation = 4
    action_scale = 0.3
    state_dim = 19  # Increased from 13 to include 6 joint velocities
    camera_target_height = 100
    camera_target_width = 180    

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

    # Environment-specific settings
    success_window_size = 100
    success_threshold = 0.8
    max_curriculum_level = 3
    
    # Command settings
    command_resampling_time = 5.0
    target_pose_range = {
        "x": (0.4, 0.7),
        "y": (0.5, 0.7),
        "z": (-0.2, 0.2),  # wrt base link of robot [-80mm to +320mm] irl
        "roll": (0.0, 0.0),
        "pitch": (1.57, 1.57),
        "yaw": (0.0, 0.0),
    }
    

    
    # Control Density Function parameters
    cdf_alpha = 0.1  # Alpha parameter for density function
    cdf_max_density = 100  # Maximum density value
    cdf_reward_weight = 5.0  # Weight for CDF reward component
    
    # Reward settings
    reward_distance_weight = -1.5
    reward_distance_tanh_weight = 1.5
    reward_distance_tanh_std = 0.1
    reward_orientation_weight = -1.0
    reward_velocity_weight = -0.0001  # Joint velocity penalty weight
    reward_table_collision_weight = -5.0
    success_bonus = 10.0
    
    # Huber loss parameters
    huber_delta = 0.1
    
    # Termination settings
    position_threshold = 0.01
    orientation_threshold = 0.05
    velocity_threshold = 0.05
    bounds_safety_margin = 0.1
    
    # Camera preprocessing settings
    camera_crop_top = 15
    camera_crop_bottom = 5
    
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
    robot_reset_noise_range = 0.05


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
                target_state=self._target_poses[i, :3],  # Will be updated
                max_density=self.cfg.cdf_max_density,
                device=self.device
            )
            self._cdf_obstacles.append(cdf)
        
        
        # Curriculum learning state
        self._curriculum_level = 0
        self._success_buffer = torch.zeros(self.cfg.success_window_size, device=self.device)
        self._success_buffer_idx = 0
        
        # Apply initial curriculum settings
        self._update_curriculum_settings()
        
        # Performance tracking
        self._episode_sums = {
            "position_error": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
            "success_count": torch.zeros(self.num_envs, device=self.device),
            "min_sphere_distance": torch.ones(self.num_envs, device=self.device) * float('inf'),
        }
        
        # Log initial information
        print(f"[INFO] Environment initialized with {self.num_envs} environments")
        print(f"[INFO] Using sphere obstacle with radius: {self.cfg.sphere_radius}m")
        print(f"[INFO] CDF alpha parameter: {self.cfg.cdf_alpha}")
        print(f"[INFO] Joint velocity penalty weight: {self.cfg.reward_velocity_weight}")
        
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
        """Set up the scene with robots, table, sphere obstacle, cameras, etc."""
        # Spawn all prims in the source environment only
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        self._sphere = RigidObject(self.cfg.sphere_cfg)
        
        # Create static assets
        # self._table = RigidObject(self.cfg.table_cfg)
        self._white_plane = RigidObject(self.cfg.white_plane_cfg)

        # Clone source env_1&env_N (env_0 keeps its prims)
        self.scene.clone_environments(copy_from_source=False)

        # Register handles in IsaacLab's scene registry
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.rigid_objects["sphere"] = self._sphere
        
        # Add static assets to scene registry
        # self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["white_plane"] = self._white_plane

        # Add static geometry and lighting
        # Ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        # Multiple lights for better scene illumination
        # Main dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/DomeLight", light_cfg)
        
        # Additional directional light for shadows
        dir_light_cfg = sim_utils.DistantLightCfg(intensity=1000.0, color=(1.0, 1.0, 0.9), angle=0.53)
        dir_light_cfg.func("/World/DirectionalLight", dir_light_cfg)
        
        
    def _update_curriculum_settings(self):
        """Update environment settings based on curriculum level."""
        base_range = 0.3
        range_increase = 0.1
        
        # Expand target range based on curriculum level
        expansion = self._curriculum_level * range_increase
        
        # Update x range
        x_center = (self.cfg.target_pose_range["x"][0] + self.cfg.target_pose_range["x"][1]) / 2
        self._current_x_range = (
            max(0.3, x_center - base_range/2 - expansion),
            min(0.9, x_center + base_range/2 + expansion)
        )
        
        # Update y range
        y_center = (self.cfg.target_pose_range["y"][0] + self.cfg.target_pose_range["y"][1]) / 2
        self._current_y_range = (
            max(-0.4, y_center - base_range/2 - expansion),
            min(0.4, y_center + base_range/2 + expansion)
        )
        
        # Update z range
        z_center = (self.cfg.target_pose_range["z"][0] + self.cfg.target_pose_range["z"][1]) / 2
        self._current_z_range = (
            max(0.5, z_center - base_range/2 - expansion),
            min(1.2, z_center + base_range/2 + expansion)
        )
        
    def _check_curriculum_advancement(self):
        """Check if curriculum should advance based on success rate."""
        if self._curriculum_level >= self.cfg.max_curriculum_level:
            return
            
        success_rate = self._success_buffer.mean().item()
        
        if success_rate >= self.cfg.success_threshold:
            self._curriculum_level += 1
            self._update_curriculum_settings()
            
            print(f"[CURRICULUM] Advanced to level {self._curriculum_level}")
            print(f"[CURRICULUM] New ranges - X: {self._current_x_range}, "
                  f"Y: {self._current_y_range}, Z: {self._current_z_range}")
            
            # Reset success buffer
            self._success_buffer.zero_()
            self._success_buffer_idx = 0
            
    def _sample_commands(self, env_ids: Sequence[int]) -> None:
        """Sample new target poses for specified environments."""
        num_envs = len(env_ids)
        
        # Sample positions within current curriculum range
        self._target_poses[env_ids, 0] = torch.rand(num_envs, device=self.device) * (
            self._current_x_range[1] - self._current_x_range[0]
        ) + self._current_x_range[0]
        
        self._target_poses[env_ids, 1] = torch.rand(num_envs, device=self.device) * (
            self._current_y_range[1] - self._current_y_range[0]
        ) + self._current_y_range[0]
        
        self._target_poses[env_ids, 2] = torch.rand(num_envs, device=self.device) * (
            self._current_z_range[1] - self._current_z_range[0]
        ) + self._current_z_range[0]
        

        # sample orientations (roll, pitch, yaw)
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

        # Fixed orientation (pointing down)
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
        state_obs[:, 6:12] += joint_vel_noise  # Joint velocities now included
        
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
        
        # Get joint velocities (6 values) - NEW
        joint_vel = self._robot.data.joint_vel[:, self._joint_indices]
        
        # Get end-effector pose relative to robot base (7 values)
        ee_pos_b = self._ee_frame.data.target_pos_source[..., 0, :]
        ee_quat_b = self._ee_frame.data.target_quat_source[..., 0, :]
        
        # Combine into state vector (now 19 values: 6 + 6 + 7)
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
        camera_data = self._tiled_camera.data.output["rgb"] / 255.0  # Shape: (num_envs, H, W, C)
        
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
        cropped = cropped.permute(0, 3, 1, 2)  # (N, C, H, W)

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
        
        # Check curriculum advancement
        self._check_curriculum_advancement()

        # Update debug visualization if enabled
        self._update_debug_visualization()
        
        # Check if robot is stuck at table and reset if needed
        # self._reset_robot_when_stuck_at_table()
        
    # def _reset_robot_when_stuck_at_table(self):
    #     """Reset robot to a safe position when it gets stuck at the table."""
    #     # Default safe poses (well above table)
    #     safe_poses = [
    #         [-0.71055204, -1.3046993, 1.9, -2.23, -1.59000665, 1.76992667],
    #         [-0.568, -0.658, 1.602, -2.585, -1.6060665, -1.64142667],  # Alternative safe pose
    #     ]
        
    #     # Get end-effector position
    #     ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
    #     ee_height = ee_position[:, 2]
        
    #     # Check which environments have robots stuck at table
    #     table_height = TABLE_HEIGHT
    #     safety_margin = 0.05
    #     stuck_at_table = ee_height < (table_height + safety_margin)
        
    #     # Get environment IDs that are stuck
    #     stuck_env_ids = torch.nonzero(stuck_at_table, as_tuple=False).squeeze(-1)
        
    #     if len(stuck_env_ids) == 0:
    #         return
        
    #     # Reset stuck robots to safe positions
    #     for env_id in stuck_env_ids:
    #         # Choose a random safe pose
    #         import random
    #         base_pose = random.choice(safe_poses)
            
    #         # Convert pose to tensor and add noise
    #         joint_pos_base = torch.tensor(base_pose, device=self.device, dtype=torch.float32)
    #         noise_range = 0.02
    #         noise = torch.rand(len(base_pose), device=self.device) * 2 * noise_range - noise_range
    #         joint_pos = joint_pos_base + noise
    #         joint_vel = torch.zeros_like(joint_pos)
            
    #         # Clamp to joint limits
    #         joint_pos = torch.clamp(
    #             joint_pos,
    #             self._robot_dof_lower_limits,
    #             self._robot_dof_upper_limits
    #         )
            
    #         # Convert env_id to tensor
    #         env_id_tensor = torch.tensor([env_id], device=self.device, dtype=torch.long)
            
    #         # Reset robot to safe position
    #         self._robot.set_joint_position_target(
    #             joint_pos.unsqueeze(0), 
    #             joint_ids=self._joint_indices, 
    #             env_ids=env_id_tensor
    #         )
    #         self._robot.write_joint_state_to_sim(
    #             joint_pos.unsqueeze(0), 
    #             joint_vel.unsqueeze(0), 
    #             joint_ids=self._joint_indices, 
    #             env_ids=env_id_tensor
    #         )
            
    #         # Update target to prevent immediate re-collision
    #         self._robot_dof_targets[env_id] = joint_pos
            
    #         # Log the reset
    #         if len(stuck_env_ids) <= 2:  # Avoid spam
    #             print(f"[INFO] Reset stuck robot in environment {env_id.item()}")
            
    def _apply_action(self) -> None:
        """Apply the processed actions to the robot with safety checks."""
        # Get current joint positions
        current_joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        
        # Add actions to current positions for position control
        self._robot_dof_targets = current_joint_pos + self.actions
        
        # Clamp to joint limits with safety margin
        safety_margin = 0.05  # radians
        self._robot_dof_targets = torch.clamp(
            self._robot_dof_targets,
            self._robot_dof_lower_limits + safety_margin,
            self._robot_dof_upper_limits - safety_margin
        )
        
        # Apply velocity limits for safety
        max_velocity = 1.5  # rad/s
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
        
    def _compute_cdf_reward(self, ee_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized CDF reward that balances target reaching and obstacle avoidance.
        """
        cdf_rewards = torch.zeros(self.num_envs, device=self.device)
        
        for i in range(self.num_envs):
            # Get raw density value
            density = self._cdf_obstacles[i].density(ee_positions[i:i+1])
            
            # Apply log-scale normalization for better gradient flow
            # This maps large density values to a reasonable range
            log_density = torch.log1p(density[0])  # log(1 + density)
            max_log_density = torch.log1p(torch.tensor(self.cfg.cdf_max_density, device=self.device))
            normalized_density = log_density / max_log_density
            
            # Scale by weight (make this the primary reward component)
            cdf_rewards[i] = normalized_density * self.cfg.cdf_reward_weight
            
        return cdf_rewards
        
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards including CDF, position, orientation, and velocity penalties."""
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
        # Get sphere world positions by adding environment origins
        sphere_world_positions = self._sphere_positions + self.scene.env_origins
        ee_to_sphere = ee_position - sphere_world_positions
        min_distances_to_sphere = torch.norm(ee_to_sphere, dim=-1) - self.cfg.sphere_radius
        
        # 1. Position tracking with Huber loss
        position_error = torch.norm(ee_position - des_pos_w, dim=-1)
        position_huber_loss = self._huber_loss(position_error, self.cfg.huber_delta)
        position_reward = self.cfg.reward_distance_weight * position_huber_loss
        rewards += position_reward
        
        # 2. Position tracking tanh reward (smooth near goal)
        position_reward_tanh = 1.0 - torch.tanh(position_error / self.cfg.reward_distance_tanh_std)
        position_reward_tanh_scaled = self.cfg.reward_distance_tanh_weight * position_reward_tanh
        rewards += position_reward_tanh_scaled
        
        # 3. Orientation tracking reward with Huber loss
        orientation_error = math_utils.quat_error_magnitude(ee_quat, des_quat_w)
        orientation_huber_loss = self._huber_loss(orientation_error, self.cfg.huber_delta * 0.5)
        orientation_reward = self.cfg.reward_orientation_weight * orientation_huber_loss
        rewards += orientation_reward
        
        # # 4. Joint velocity penalty (replacing torque penalty)
        joint_velocities = self._robot.data.joint_vel[:, self._joint_indices]
        # velocity_penalty = torch.sum(torch.square(joint_velocities), dim=1)
        # velocity_reward = self.cfg.reward_velocity_weight * velocity_penalty
        # rewards += velocity_reward
        
        # # 5. Table collision penalty
        # ee_height = ee_position[:, 2]
        # table_height = TABLE_HEIGHT
        # safety_margin = 0.05
        
        # table_penalty = torch.where(
        #     ee_height < (table_height + safety_margin),
        #     torch.ones_like(ee_height) * self.cfg.reward_table_collision_weight,
        #     torch.zeros_like(ee_height)
        # )
        # rewards += table_penalty
        
        # 6. Control Density Function reward
        cdf_reward = self._compute_cdf_reward(ee_position)
        rewards += cdf_reward
        
        # 7. Success bonus for reaching goal while avoiding sphere
        joint_vel_norm = torch.norm(joint_velocities, p=2, dim=-1)
        success_mask = (
            (position_error < 0.05) & 
            (min_distances_to_sphere > 0.05) & 
            (joint_vel_norm < self.cfg.velocity_threshold)
        )
        rewards += torch.where(success_mask, 5.0, 0.0)
        
        # Track reward components for logging
        if hasattr(self, '_episode_sums'):
            self._episode_sums["total_reward"] += rewards
            self._episode_sums["position_error"] += position_error
            self._episode_sums["min_sphere_distance"] = torch.minimum(
                self._episode_sums["min_sphere_distance"], min_distances_to_sphere
            )
            
            # Update success buffer for curriculum learning
            self._episode_sums["success_count"] += success_mask.float()
            if torch.any(success_mask):
                success_rate = success_mask.float().mean()
                self._success_buffer[self._success_buffer_idx] = success_rate
                self._success_buffer_idx = (self._success_buffer_idx + 1) % self.cfg.success_window_size
        
        # Log detailed reward breakdown occasionally
        if self.common_step_counter % 500 == 0 and self.num_envs > 0:
            env_0_data = {
                "position_error": position_error[0].item(),
                "orientation_error": orientation_error[0].item(),
                # "velocity_penalty": velocity_penalty[0].item(),
                "min_dist_to_sphere": min_distances_to_sphere[0].item(),
                "cdf_reward": cdf_reward[0].item(),
                "total_reward": rewards[0].item()
            }
            print(f"[REWARD] Env 0 - Pos err: {env_0_data['position_error']:.3f}, "
                  f"Dist to sphere: {env_0_data['min_dist_to_sphere']:.3f}, "
                  f"CDF: {env_0_data['cdf_reward']:.3f}, "
                  f"Total: {env_0_data['total_reward']:.3f}")
        
        # Add this at the end:
        if self.cfg.debug_vis and hasattr(self, "target_pos_visualizer"):
            self._update_debug_visualization()
        
        return rewards

    # def _get_rewards(self) -> torch.Tensor:
    #     """Compute rewards using CDF as the primary navigation reward."""
    #     # Initialize rewards
    #     rewards = torch.zeros(self.num_envs, device=self.device)
        
    #     # Get end-effector position and orientation
    #     ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
    #     ee_quat = self._ee_frame.data.target_quat_w[..., 0, :]
        
    #     # Transform target pose to world frame
    #     robot_pos = self._robot.data.root_state_w[:, :3]
    #     robot_quat = self._robot.data.root_state_w[:, 3:7]
    #     des_pos_b = self._target_poses[:, :3]
    #     des_quat_b = self._target_poses[:, 3:7]
    #     des_pos_w, _ = math_utils.combine_frame_transforms(
    #         robot_pos, robot_quat, des_pos_b
    #     )
    #     des_quat_w = math_utils.quat_mul(robot_quat, des_quat_b)
        
    #     # Calculate distances for safety checks
    #     sphere_world_positions = self._sphere_positions + self.scene.env_origins
    #     ee_to_sphere = ee_position - sphere_world_positions
    #     min_distances_to_sphere = torch.norm(ee_to_sphere, dim=-1) - self.cfg.sphere_radius
        
    #     # 1. Control Density Function reward (PRIMARY NAVIGATION REWARD)
    #     # This already includes both obstacle avoidance AND target reaching
    #     cdf_reward = self._compute_cdf_reward(ee_position)
    #     rewards += cdf_reward
        
    #     # 2. Orientation tracking reward
    #     # Keep this separate as CDF only handles position
    #     orientation_error = math_utils.quat_error_magnitude(ee_quat, des_quat_w)
    #     # orientation_reward = -self.cfg.reward_orientation_weight * orientation_error
    #     # rewards += orientation_reward
        
    #     # # 3. Smoothness penalty (joint velocity)
    #     joint_velocities = self._robot.data.joint_vel[:, self._joint_indices]
    #     # velocity_penalty = torch.sum(torch.square(joint_velocities), dim=1)
    #     # velocity_reward = self.cfg.reward_velocity_weight * velocity_penalty
    #     # rewards += velocity_reward
        
    #     # # 4. Safety penalties (only for constraint violations)
    #     # # Table collision penalty
    #     # ee_height = ee_position[:, 2]
    #     # table_height = TABLE_HEIGHT
    #     # safety_margin = 0.05
    #     # table_penalty = torch.where(
    #     #     ee_height < (table_height + safety_margin),
    #     #     torch.ones_like(ee_height) * self.cfg.reward_table_collision_weight,
    #     #     torch.zeros_like(ee_height)
    #     # )
    #     # rewards += table_penalty
        
    #     # 5. Success bonus (discrete reward at goal)
    #     position_error = torch.norm(ee_position - des_pos_w, dim=-1)
    #     joint_vel_norm = torch.norm(joint_velocities, p=2, dim=-1)
    #     success_mask = (
    #         (position_error < self.cfg.position_threshold) & 
    #         (orientation_error < self.cfg.orientation_threshold) &
    #         (min_distances_to_sphere > 0.05) & 
    #         (joint_vel_norm < self.cfg.velocity_threshold)
    #     )
    #     rewards += torch.where(success_mask, self.cfg.success_bonus, 0.0)
        
    #     # Track for logging
    #     if hasattr(self, '_episode_sums'):
    #         self._episode_sums["total_reward"] += rewards
    #         self._episode_sums["position_error"] += position_error
    #         self._episode_sums["min_sphere_distance"] = torch.minimum(
    #             self._episode_sums["min_sphere_distance"], min_distances_to_sphere
    #         )
    #         self._episode_sums["success_count"] += success_mask.float()
        
    #     return rewards



        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get termination signals."""
        # Time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Get end-effector position
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        
        
        # Check for collisions with table
        table_collision = ee_position[:, 2] < (TABLE_HEIGHT - 0.01)
        
        
        # Combine termination conditions
        terminated =  table_collision 
        
        return terminated, time_out
        
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)
        
        # Print episode statistics for completed environments
        if len(env_ids) > 0 and hasattr(self, '_episode_sums'):
            avg_position_error = self._episode_sums["position_error"][env_ids].mean().item()
            avg_reward = self._episode_sums["total_reward"][env_ids].mean().item()
            success_rate = self._episode_sums["success_count"][env_ids].mean().item()
            min_sphere_dist = self._episode_sums["min_sphere_distance"][env_ids].mean().item()
            
            if self.common_step_counter % 1000 == 0:  # Log every 1000 steps
                print(f"[INFO] Episode stats - Pos error: {avg_position_error:.4f}, "
                      f"Reward: {avg_reward:.2f}, Success: {success_rate:.2f}, "
                      f"Min sphere dist: {min_sphere_dist:.3f}, "
                      f"Curriculum: L{self._curriculum_level}")
        
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
        
        # Add noise
        joint_pos = base_pose.unsqueeze(0).repeat(num_resets, 1)
        if self.cfg.robot_reset_noise_range > 0:
            joint_pos += sample_uniform(
                -self.cfg.robot_reset_noise_range,
                self.cfg.robot_reset_noise_range,
                joint_pos.shape,
                self.device
            )
        joint_vel = torch.zeros_like(joint_pos)
        
        # Set joint state
        self._robot.write_joint_state_to_sim(
            joint_pos, joint_vel,
            joint_ids=self._joint_indices,
            env_ids=env_ids
        )
        
        # Reset sphere positions
        sphere_poses = torch.zeros((num_resets, 7), device=self.device)
        
        for i, env_id in enumerate(env_ids):
            # Random position within bounds
            self._sphere_positions[env_id, 0] = torch.rand(1, device=self.device) * (
                self.cfg.sphere_position_bounds["x"][1] - self.cfg.sphere_position_bounds["x"][0]
            ) + self.cfg.sphere_position_bounds["x"][0]
            
            self._sphere_positions[env_id, 1] = torch.rand(1, device=self.device) * (
                self.cfg.sphere_position_bounds["y"][1] - self.cfg.sphere_position_bounds["y"][0]
            ) + self.cfg.sphere_position_bounds["y"][0]
            
            self._sphere_positions[env_id, 2] = torch.rand(1, device=self.device) * (
                self.cfg.sphere_position_bounds["z"][1] - self.cfg.sphere_position_bounds["z"][0]
            ) + self.cfg.sphere_position_bounds["z"][0]
            
            # Add environment origin offset for proper positioning
            sphere_poses[i, :3] = self._sphere_positions[env_id] + self.scene.env_origins[env_id]
            # Set orientation (quaternion w,x,y,z)
            sphere_poses[i, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            
            # Update CDF obstacle center (use local position, not world position)
            self._cdf_obstacles[env_id].update_center(self._sphere_positions[env_id])
        
        # Set sphere positions in simulation
        self._sphere.write_root_pose_to_sim(
            root_pose=sphere_poses,
            env_ids=env_ids
        )
        
        # Reset target poses and timers
        self._command_time_left[env_ids] = 0.0  # Force immediate resampling
        self._sample_target_poses_for_reset(env_ids)
        
        
        # Reset joint targets
        self._robot_dof_targets[env_ids] = joint_pos

    def _sample_target_poses_for_reset(self, env_ids: Sequence[int]):
        """Sample new target poses for reset environments."""
        num_resets = len(env_ids)
        
        # Sample target poses
        x = torch.rand(num_resets, device=self.device) * (
            self.cfg.target_pose_range["x"][1] - self.cfg.target_pose_range["x"][0]
        ) + self.cfg.target_pose_range["x"][0]
        
        y = torch.rand(num_resets, device=self.device) * (
            self.cfg.target_pose_range["y"][1] - self.cfg.target_pose_range["y"][0]
        ) + self.cfg.target_pose_range["y"][0]
        
        z = torch.rand(num_resets, device=self.device) * (
            self.cfg.target_pose_range["z"][1] - self.cfg.target_pose_range["z"][0]
        ) + self.cfg.target_pose_range["z"][0]
        
        # Fixed orientation for now
        roll = torch.full((num_resets,), self.cfg.target_pose_range["roll"][0], device=self.device)
        pitch = torch.full((num_resets,), self.cfg.target_pose_range["pitch"][0], device=self.device)
        yaw = torch.full((num_resets,), self.cfg.target_pose_range["yaw"][0], device=self.device)
        
        # Convert euler to quaternion
        target_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        
        # Update buffers
        self._target_poses[env_ids, :3] = torch.stack([x, y, z], dim=-1)
        self._target_poses[env_ids, 3:7] = target_quat

       
    def set_debug_vis(self, debug_vis: bool) -> None:
        """Set debug visualization mode."""
        self.cfg.debug_vis = debug_vis
        if hasattr(self, "_ee_frame") and self._ee_frame is not None:
            self._ee_frame.set_debug_vis(debug_vis)
        
        # Add this line:
        self._set_debug_vis_impl(debug_vis)


    def _update_debug_visualization(self):
        """Update debug visualization markers - call this in your step loop."""
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

    def _set_debug_vis_impl(self, debug_vis:bool):
        # Create markers for visualizing the goal poses
        if debug_vis:
            if not hasattr(self, "target_pos_visualizer"):
                # Create a separate marker config for target visualization with different color
                target_marker_cfg = FRAME_MARKER_CFG.copy()
                target_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)  # Slightly larger
                target_marker_cfg.prim_path = "/Visuals/Command/target_position"
                self.target_pos_visualizer = VisualizationMarkers(target_marker_cfg)
            # Set target visibility to true
            self.target_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_pos_visualizer"):
                self.target_pos_visualizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        """Update debug visualization markers and save joint targets."""
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