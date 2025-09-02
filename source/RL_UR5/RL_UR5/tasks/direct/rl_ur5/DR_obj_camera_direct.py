"""
Direct RL Environment for Object Camera Pose Tracking with UR5 Robot
Enhanced with Domain Randomization for Sim-to-Real Transfer

This implements a multi-observation space environment compatible with skrl.
"""

from __future__ import annotations

import torch
import numpy as np
import math
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


##
# Environment Configuration
##

@configclass
class ObjCameraPoseTrackingDirectEnvCfg(DirectRLEnvCfg):
    """Configuration for the direct RL environment with domain randomization."""
    
    # Visualization settings - MOVED TO TOP to fix reference issue
    debug_vis = False # Enable/disable debug visualization
    
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    
    # UR5 Robot
    robot_cfg: ArticulationCfg = UR5_GRIPPER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Table configuration
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/adi2440/Desktop/ur5_isaacsim/usd/table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, -0.0234), 
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # Arm configuration - dynamic object for collision avoidance
    arm_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/arm",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/adi2440/Desktop/ur5_isaacsim/usd/arm.usd",
            scale=(0.01, 0.01, 0.01),  # Ensure no scaling
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),  # Give it some mass
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True  # Enable collision
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, 0.0, 0.9), 
            rot=(0.70711, 0.0,0.70711, 0.0)
        ),
    )

    # White plane configuration
    white_plane_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/white_plane",
        spawn=sim_utils.CuboidCfg(
            size=(0.42112, 2.81, 0.01),
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
        debug_vis=debug_vis,  # Now this works since enable_debug_vis is defined above
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
            focal_length=2.208,
            focus_distance=28.0,
            horizontal_aperture=5.76,
            vertical_aperture=3.24,
            clipping_range=(0.1, 1000.0)
        ),
        width=224,
        height=224,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.27, 0.06, 1.143),
            rot=(0.56099,0.43046,0.43406,0.56099),
            convention="opengl"
        )
    )

    # Basic environment settings
    episode_length_s = 6.0
    decimation = 40
    action_scale = 0.3  # Reduced for smoother movements
    state_dim = 13
    camera_target_height = 100
    camera_target_width = 120    

    # Observation and action spaces
    action_space = gym.spaces.Box(low=-3.5, high=3.5, shape=(6,))
    state_space = 0
    ## For PPO
    observation_space = gym.spaces.Dict({
        "image": gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(camera_target_height, camera_target_width, 3)),
        "state": gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(state_dim,)),
    })
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0/120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8, 
        env_spacing=4.0,
        replicate_physics=True,
    )
    
    # Viewer settings
    viewer = ViewerCfg(eye=(7.5, 7.5, 7.5), origin_type="world", env_index=0)
    
    # Curriculum learning settings
    curriculum_enabled = True
    curriculum_steps = [5000, 10000, 20000, 40000]  # Steps at which to increase difficulty
    curriculum_arm_speeds = [0.0, 0.1, 0.2, 0.3]  # Progressive arm movement speeds
    curriculum_target_ranges = [
        {"x": (0.55, 0.65), "y": (0.45, 0.5), "z": (-0.1, 0.1)},   # Easy
        {"x": (0.5, 0.7), "y": (0.4, 0.5), "z": (-0.15, 0.15)},    # Medium
        {"x": (0.5, 0.7), "y": (0.35, 0.55), "z": (-0.2, 0.2)},    # Hard
        {"x": (0.45, 0.75), "y": (0.3, 0.6), "z": (-0.2, 0.2)},    # Expert
    ]
    
    # Success tracking for adaptive curriculum
    success_window_size = 100
    curriculum_advance_threshold = 0.7  # Advance when success rate > 70%

    # Command/target pose settings
    target_pose_range = {
        "x": (0.5, 0.7),
        "y": (0.45, 0.55),
        "z": (-0.2, 0.2),  # wrt base link of robot [-80mm to +320mm] irl
        "roll": (0.0, 0.0),
        "pitch": (1.57, 1.57),
        "yaw": (0.0, 0.0),
    }

    # target_ee_bounds = {
    #     "x": (0.35, 0.85),
    #     "y": (-0.6, 0.6),
    #     "z": (-0.4, 0.4),  # wrt base link of robot [-80mm to +320mm] irl
    # }

    command_resampling_time = 6.0
    
    # Human arm movement settings
    arm_position_bounds = {
        "x": (0.80, 1.2),
        "y": (-0.5, 0.5),
        "z": (0.80, 1.2),
    }
    arm_movement_speed = 0.0 # Speed of random movement
    
    # Reward settings
    reward_distance_weight = -2.5
    reward_distance_tanh_weight = 1.5
    reward_distance_tanh_std = 0.1
    reward_orientation_weight = -0.2
    reward_torque_weight = -0.0001  # Replaced torque with action penalty
    reward_table_collision_weight = -4.0
    reward_arm_avoidance_weight = 7.0  # Changed from obstacle
    
    # Artificial Potential Field parameters
    apf_critical_distance = 0.15  # db - critical distance for obstacle avoidance
    apf_smoothness = 0.1  # ko - smoothness parameter for beta transition
    energy_reward_weight = -1.0  # Weight for energy component
    
    # Huber loss parameters
    huber_delta = 0.08  # Delta parameter for Huber loss
    
    # Action filter settings
    action_filter_order = 2
    action_filter_cutoff_freq = 3.0
    action_filter_damping_ratio = 0.707
    
    # Termination settings
    position_threshold = 0.05
    orientation_threshold = 0.1
    velocity_threshold = 0.05
    torque_threshold = 1.0
    bounds_safety_margin = 0.1  # 0.1m margin for bounds checking
    
    # Camera preprocessing settings
    camera_crop_top = 80
    camera_crop_bottom = 20
    
    # Visualization settings
    visualize_camera_interval = 10000  # Visualize camera every N steps
    visualization_save_path = "/home/adi2440/Desktop/camera_obs"  # Path to save visualizations

    
    # Noise settings
    joint_pos_noise_min = -0.01
    joint_pos_noise_max = 0.01
    joint_vel_noise_min = -0.001
    joint_vel_noise_max = 0.001
    
    # Reset settings
    robot_base_pose = [-0.568, -0.858,  1.402, -2.185, -1.6060665,  1.64142667]
    robot_reset_noise_range = 0.05
    
    # ===== DOMAIN RANDOMIZATION PARAMETERS =====
    # Friction randomization
    friction_range = (0.5, 1.5)  # Min and max friction coefficients
    
    # Action noise parameters
    action_noise_std = 0.01  # Same magnitude as joint observation noise
    
    # Camera noise parameters
    camera_noise_std = 0.02  # Gaussian noise standard deviation (2% of pixel values)
    camera_brightness_range = (0.8, 1.2)  # Random brightness scaling
    camera_contrast_range = (0.9, 1.1)  # Random contrast adjustment
    
    # Enable/disable specific DR components
    enable_friction_randomization = False
    enable_action_noise = True
    enable_camera_noise = True


class ObjCameraPoseTrackingDirectEnv(DirectRLEnv):
    """Direct RL environment for object camera pose tracking with multi-observation space and domain randomization."""
    
    cfg: ObjCameraPoseTrackingDirectEnvCfg
    
    def __init__(self, cfg: ObjCameraPoseTrackingDirectEnvCfg, render_mode: str | None = None, **kwargs):
        # Store config
        self.cfg = cfg
        
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
        
        # Arm movement state
        self._arm_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Initialize action filter
        self._setup_action_filter()
        
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
            "min_arm_distance": torch.ones(self.num_envs, device=self.device) * float('inf'),
        }
        
        # === DOMAIN RANDOMIZATION BUFFERS ===
        # Store randomized friction values per environment
        self._current_friction = torch.ones(self.num_envs, device=self.device)
        
        # Store camera augmentation parameters per environment
        self._camera_brightness = torch.ones(self.num_envs, device=self.device)
        self._camera_contrast = torch.ones(self.num_envs, device=self.device)
        
        # Log initial information
        print(f"[INFO] Environment initialized with {self.num_envs} environments")
        print(f"[INFO] Action scale: {self.cfg.action_scale}")
        print(f"[INFO] Target pose range X: {self.cfg.target_pose_range['x']}")
        print(f"[INFO] Target pose range Y: {self.cfg.target_pose_range['y']}")
        print(f"[INFO] Target pose range Z: {self.cfg.target_pose_range['z']}")
        print(f"[INFO] Arm bounds X: {self.cfg.arm_position_bounds['x']}")
        print(f"[INFO] Arm bounds Y: {self.cfg.arm_position_bounds['y']}")
        print(f"[INFO] Arm bounds Z: {self.cfg.arm_position_bounds['z']}")
        
        # Log domain randomization settings
        print(f"[INFO] === Domain Randomization Settings ===")
        print(f"[INFO] Friction randomization: {self.cfg.enable_friction_randomization} (range: {self.cfg.friction_range})")
        print(f"[INFO] Action noise: {self.cfg.enable_action_noise} (std: {self.cfg.action_noise_std})")
        print(f"[INFO] Camera noise: {self.cfg.enable_camera_noise} (std: {self.cfg.camera_noise_std})")
        
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
        """Set up the scene with robots, table, obstacles, cameras, etc."""
        # --- spawn all prims in the source environment only ---
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        self._arm = RigidObject(self.cfg.arm_cfg)
        
        # Create static assets
        self._table = RigidObject(self.cfg.table_cfg)
        self._white_plane = RigidObject(self.cfg.white_plane_cfg)

        # --- clone source → env_1…env_N (env_0 keeps its prims) ---
        self.scene.clone_environments(copy_from_source=False)

        # --- register handles in IsaacLab's scene registry ---
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.rigid_objects["arm"] = self._arm
        
        # Add static assets to scene registry
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["white_plane"] = self._white_plane

        # --- add static geometry and lighting ---
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

    def _randomize_friction(self, env_ids: Sequence[int]):
        """Randomize friction coefficients for specified environments."""
        if not self.cfg.enable_friction_randomization:
            return
            
        num_envs = len(env_ids)
        
        # Sample new friction values
        new_friction = torch.rand(num_envs, device=self.device) * (
            self.cfg.friction_range[1] - self.cfg.friction_range[0]
        ) + self.cfg.friction_range[0]
        
        # Store for logging
        self._current_friction[env_ids] = new_friction
        
        # Apply to robot joints
        # Note: In IsaacLab, we need to modify the physics material properties
        # This is typically done through the articulation configuration
        # For now, we'll store the values and they can be used in physics calculations
        
        # Log friction values for first few environments
        if env_ids[0] < 3:
            print(f"[DR] Env {env_ids[0]} friction: {new_friction[0]:.3f}")
            
    def _randomize_camera_parameters(self, env_ids: Sequence[int]):
        """Randomize camera augmentation parameters for specified environments."""
        if not self.cfg.enable_camera_noise:
            return
            
        num_envs = len(env_ids)
        
        # Sample brightness and contrast adjustments
        self._camera_brightness[env_ids] = torch.rand(num_envs, device=self.device) * (
            self.cfg.camera_brightness_range[1] - self.cfg.camera_brightness_range[0]
        ) + self.cfg.camera_brightness_range[0]
        
        self._camera_contrast[env_ids] = torch.rand(num_envs, device=self.device) * (
            self.cfg.camera_contrast_range[1] - self.cfg.camera_contrast_range[0]
        ) + self.cfg.camera_contrast_range[0]
        
    def _setup_action_filter(self):
        """Initialize action filter states and coefficients."""
        # Initialize filter memory for each environment and joint
        num_joints = len(self._joint_indices)
        self._action_filter_x1 = torch.zeros((self.num_envs, num_joints), device=self.device)
        self._action_filter_x2 = torch.zeros((self.num_envs, num_joints), device=self.device)
        self._action_filter_y1 = torch.zeros((self.num_envs, num_joints), device=self.device)
        self._action_filter_y2 = torch.zeros((self.num_envs, num_joints), device=self.device)
        
        # Calculate filter coefficients
        if self.cfg.action_filter_order == 2:
            omega = 2.0 * math.pi * self.cfg.action_filter_cutoff_freq
            dt = self.cfg.sim.dt
            k = omega * dt
            a1 = k * k
            a2 = k * 2.0 * self.cfg.action_filter_damping_ratio
            a3 = a1 + a2 + 1.0
            
            self._filter_b0 = a1 / a3
            self._filter_b1 = 2.0 * a1 / a3
            self._filter_b2 = a1 / a3
            self._filter_a1 = (2.0 * a1 - 2.0) / a3
            self._filter_a2 = (a1 - a2 + 1.0) / a3
            
    def _reset_idx(self, env_ids: Sequence[int] | None = None):
        """Reset the environment for the given environment indices."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
            
        super()._reset_idx(env_ids)
        
        # Print episode statistics for completed environments
        if len(env_ids) > 0 and hasattr(self, '_episode_sums'):
            avg_position_error = self._episode_sums["position_error"][env_ids].mean().item()
            avg_reward = self._episode_sums["total_reward"][env_ids].mean().item()
            success_rate = self._episode_sums["success_count"][env_ids].mean().item()
            min_arm_dist = self._episode_sums["min_arm_distance"][env_ids].mean().item()
            
            if self.common_step_counter % 1000 == 0:  # Log every 1000 steps
                print(f"[INFO] Episode stats - Pos error: {avg_position_error:.4f}, "
                      f"Reward: {avg_reward:.2f}, Success: {success_rate:.2f}, "
                      f"Min arm dist: {min_arm_dist:.3f}, "
                      f"Curriculum: L{self._curriculum_level}")
        
        # Reset episode tracking
        if hasattr(self, '_episode_sums'):
            for key in self._episode_sums:
                self._episode_sums[key][env_ids] = 0.0
        
        # === DOMAIN RANDOMIZATION ON RESET ===
        # Randomize friction for these environments
        self._randomize_friction(env_ids)
        
        # Randomize camera parameters
        self._randomize_camera_parameters(env_ids)
        
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
        
        # Reset arm position and orientation targets
        for i, env_id in enumerate(env_ids):
            # Random position within bounds
            self._arm_target_pos[env_id, 0] = torch.rand(1, device=self.device) * (
                self.cfg.arm_position_bounds["x"][1] - self.cfg.arm_position_bounds["x"][0]
            ) + self.cfg.arm_position_bounds["x"][0]
            
            self._arm_target_pos[env_id, 1] = torch.rand(1, device=self.device) * (
                self.cfg.arm_position_bounds["y"][1] - self.cfg.arm_position_bounds["y"][0]
            ) + self.cfg.arm_position_bounds["y"][0]
            
            self._arm_target_pos[env_id, 2] = torch.rand(1, device=self.device) * (
                self.cfg.arm_position_bounds["z"][1] - self.cfg.arm_position_bounds["z"][0]
            ) + self.cfg.arm_position_bounds["z"][0]
        
        # Set initial arm pose
        new_positions = self._arm_target_pos[env_ids] + self.scene.env_origins[env_ids, :3]
        
        # Keep default orientation (identity quaternion)
        default_quat = torch.tensor([0.0, 1.0, 0.0,0.0], device=self.device)
        new_quats = default_quat.unsqueeze(0).repeat(len(env_ids), 1)
        
        # Combine position and orientation
        new_poses = torch.cat([new_positions, new_quats], dim=-1)
        
        # Write to simulation
        self._arm.write_root_pose_to_sim(new_poses, env_ids=env_ids)
        self._arm.write_root_velocity_to_sim(
            torch.zeros((num_resets, 6), device=self.device),
            env_ids=env_ids
        )
        
        # Reset target poses
        self._sample_target_poses_for_reset(env_ids)
        
        # Reset action filter states
        self._action_filter_x1[env_ids] = 0.0
        self._action_filter_x2[env_ids] = 0.0
        self._action_filter_y1[env_ids] = 0.0
        self._action_filter_y2[env_ids] = 0.0
        
        # Reset timers
        self._command_time_left[env_ids] = self.cfg.command_resampling_time
        
    def _sample_target_poses_for_reset(self, env_ids: Sequence[int]):
        """Sample new target poses for reset environments."""
        num_resets = len(env_ids)
        
        # Get current curriculum target range
        if self.cfg.curriculum_enabled:
            target_range = self.cfg.curriculum_target_ranges[self._curriculum_level]
        else:
            target_range = self.cfg.target_pose_range
        
        # Sample random positions
        target_x = torch.rand(num_resets, device=self.device) * (
            target_range["x"][1] - target_range["x"][0]
        ) + target_range["x"][0]
        
        target_y = torch.rand(num_resets, device=self.device) * (
            target_range["y"][1] - target_range["y"][0]
        ) + target_range["y"][0]
        
        target_z = torch.rand(num_resets, device=self.device) * (
            target_range["z"][1] - target_range["z"][0]
        ) + target_range["z"][0]
        
        # Fixed orientation (pointing down)
        target_quat = torch.tensor([0.707, 0.0, 0.707, 0.0], device=self.device)
        target_quat = target_quat.unsqueeze(0).repeat(num_resets, 1)
        
        # Combine position and orientation
        self._target_poses[env_ids, 0] = target_x
        self._target_poses[env_ids, 1] = target_y
        self._target_poses[env_ids, 2] = target_z
        self._target_poses[env_ids, 3:7] = target_quat
        
        # Reset command timer
        self._command_time_left[env_ids] = self.cfg.command_resampling_time
        
        # Initialize arm movement pattern and time
        if not hasattr(self, '_arm_motion_pattern'):
            self._arm_motion_pattern = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._arm_motion_time = torch.zeros(self.num_envs, device=self.device)
        
        # Randomize movement pattern for reset envs
        self._arm_motion_pattern[env_ids] = torch.randint(0, 3, (num_resets,), device=self.device)
        self._arm_motion_time[env_ids] = torch.rand(num_resets, device=self.device) * 4.0
        
    def _update_curriculum_settings(self):
        """Update settings based on curriculum level."""
        if not self.cfg.curriculum_enabled:
            return
            
        # Update arm movement speed
        self.cfg.arm_movement_speed = self.cfg.curriculum_arm_speeds[self._curriculum_level]
        
        # Log curriculum change
        if self._curriculum_level > 0:
            target_range = self.cfg.curriculum_target_ranges[self._curriculum_level]
            print(f"[CURRICULUM] Level {self._curriculum_level}: "
                  f"Arm speed={self.cfg.arm_movement_speed:.2f}, "
                  f"Target X={target_range['x']}, Y={target_range['y']}, Z={target_range['z']}")
    
    def _check_curriculum_advancement(self):
        """Check if curriculum should advance based on success rate."""
        if not self.cfg.curriculum_enabled:
            return
            
        # Calculate current success rate
        success_rate = self._success_buffer.mean().item()
        
        # Check if we should advance
        if success_rate > self.cfg.curriculum_advance_threshold:
            if self._curriculum_level < len(self.cfg.curriculum_steps) - 1:
                # Check if we've reached the step threshold
                step_threshold = self.cfg.curriculum_steps[self._curriculum_level + 1]
                if self.common_step_counter >= step_threshold:
                    self._curriculum_level += 1
                    self._update_curriculum_settings()
                    # Reset success buffer for new level
                    self._success_buffer.zero_()
                    print(f"[CURRICULUM] Advanced to level {self._curriculum_level} at step {self.common_step_counter}")
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply actions before physics step with domain randomization."""
        # Store raw actions
        self.actions = actions.clone().clamp(-3.5, 3.5)
        self._previous_actions = self.actions.clone()
        
        # === ADD ACTION NOISE (Domain Randomization) ===
        if self.cfg.enable_action_noise:
            action_noise = torch.randn_like(self.actions) * self.cfg.action_noise_std
            self.actions = self.actions + action_noise
            # Re-clamp after adding noise
            self.actions = self.actions.clamp(-3.5, 3.5)
        
        # Apply action filtering
        filtered_actions = self._apply_action_filter(self.actions)
        
        # Scale actions
        self.actions = filtered_actions * self.cfg.action_scale
        
        # Update command timer
        self._command_time_left -= self.physics_dt
        
        # --- resample target poses when timer runs out ---
        expired_mask = self._command_time_left <= 0.0
        if torch.any(expired_mask):
            expired_ids = torch.nonzero(expired_mask, as_tuple=False).squeeze(-1)
            env_ids = expired_ids.cpu().tolist()
            self._sample_commands(env_ids)
            # reset their countdown
            self._command_time_left[expired_mask] = self.cfg.command_resampling_time
        
        # Check curriculum advancement
        self._check_curriculum_advancement()
        
        #IF robot is stuck at the table, reset it
        self._reset_robot_when_stuck_at_table()

        # Update arm position
        self._update_arm_position()

    def _apply_action(self) -> None:
        """Apply the processed actions to the robot with safety checks and friction simulation."""
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
        
        # === FRICTION SIMULATION (Domain Randomization) ===
        # Scale velocity commands by friction factor to simulate varying joint friction
        if self.cfg.enable_friction_randomization:
            # Expand friction values to match joint dimensions
            friction_factor = self._current_friction.unsqueeze(1).expand(-1, len(self._joint_indices))
            # Higher friction means slower movement
            velocity_command = velocity_command / friction_factor
        
        self._robot_dof_targets = current_joint_pos + velocity_command * self.physics_dt
        
        # Set joint position targets
        self._robot.set_joint_position_target(
            self._robot_dof_targets, joint_ids=self._joint_indices
        )
        
    def _apply_action_filter(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply second-order Butterworth filter to actions."""
        if self.cfg.action_filter_order == 2:
            filtered_actions = (
                self._filter_b0 * actions +
                self._filter_b1 * self._action_filter_x1 +
                self._filter_b2 * self._action_filter_x2 -
                self._filter_a1 * self._action_filter_y1 -
                self._filter_a2 * self._action_filter_y2
            )
            
            # Update filter memory
            self._action_filter_x2 = self._action_filter_x1.clone()
            self._action_filter_x1 = actions.clone()
            self._action_filter_y2 = self._action_filter_y1.clone()
            self._action_filter_y1 = filtered_actions.clone()
            
            return filtered_actions
        else:
            return actions

    def _sample_commands(self, env_ids: Sequence[int]) -> None:
        """Randomize the target poses for the given env indices."""
        # Get current curriculum target range
        if self.cfg.curriculum_enabled:
            target_range = self.cfg.curriculum_target_ranges[self._curriculum_level]
        else:
            target_range = self.cfg.target_pose_range
            
        for env_id in env_ids:
            # Sample new position
            self._target_poses[env_id, 0] = torch.rand(1, device=self.device) * (
                target_range["x"][1] - target_range["x"][0]
            ) + target_range["x"][0]
            
            self._target_poses[env_id, 1] = torch.rand(1, device=self.device) * (
                target_range["y"][1] - target_range["y"][0]
            ) + target_range["y"][0]
            
            self._target_poses[env_id, 2] = torch.rand(1, device=self.device) * (
                target_range["z"][1] - target_range["z"][0]
            ) + target_range["z"][0]
            
            # Keep fixed orientation
            self._target_poses[env_id, 3:7] = torch.tensor([0.707, 0.0, 0.707, 0.0], device=self.device)
    
    def _reset_robot_when_stuck_at_table(self):
        """Reset robot when end-effector is stuck at or below table height."""
        # Get end-effector position
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        
        # Check if z-position is at or below table height with some margin
        table_height = 0.77
        stuck_threshold = table_height + 0.03  # 3cm above table
        
        # Find environments where robot is stuck
        stuck_mask = ee_position[:, 2] < stuck_threshold
        
        if torch.any(stuck_mask):
            stuck_env_ids = torch.nonzero(stuck_mask, as_tuple=False).squeeze(-1)
            stuck_env_ids = stuck_env_ids.cpu().tolist()
            
            # Log warning for first few environments
            for i, env_id in enumerate(stuck_env_ids[:3]):
                print(f"[WARNING] Robot stuck at table in env {env_id}, z={ee_position[env_id, 2]:.3f}m. Resetting...")
            
            # Reset these environments
            self._reset_idx(stuck_env_ids)
            
    def _update_arm_position(self):
        """Update arm position with smooth movement patterns."""
        # Get arm movement speed from curriculum
        movement_speed = self.cfg.arm_movement_speed
        
        if movement_speed <= 0:
            # No movement if speed is 0
            return
            
        # Update motion time
        self._arm_motion_time += self.physics_dt
        
        # Calculate new positions based on movement pattern
        new_positions = self._arm_target_pos.clone()
        
        for i in range(self.num_envs):
            # Period for one complete motion cycle
            period = 4.0  # seconds
            phase = (self._arm_motion_time[i] % period) / period
            
            # Smooth sinusoidal motion
            if self._arm_motion_pattern[i] == 0:  # X-axis movement
                offset = 0.3 * math.sin(2 * math.pi * phase)
                new_positions[i, 0] = (self.cfg.arm_position_bounds["x"][0] + 
                                      self.cfg.arm_position_bounds["x"][1]) / 2.0 + offset
            elif self._arm_motion_pattern[i] == 1:  # Y-axis movement
                offset = 0.3 * math.sin(2 * math.pi * phase)
                new_positions[i, 1] = (self.cfg.arm_position_bounds["y"][0] + 
                                      self.cfg.arm_position_bounds["y"][1]) / 2.0 + offset
            else:  # Diagonal movement
                offset_x = 0.2 * math.sin(2 * math.pi * phase)
                offset_y = 0.2 * math.cos(2 * math.pi * phase)
                new_positions[i, 0] = (self.cfg.arm_position_bounds["x"][0] + 
                                      self.cfg.arm_position_bounds["x"][1]) / 2.0 + offset_x
                new_positions[i, 1] = (self.cfg.arm_position_bounds["y"][0] + 
                                      self.cfg.arm_position_bounds["y"][1]) / 2.0 + offset_y
        
        # Apply positions to simulation
        world_positions = new_positions + self.scene.env_origins[:, :3]
        
        # Get current orientations
        arm_quats = self._arm.data.root_quat_w
        
        # Set new poses
        self._arm.write_root_pose_to_sim(
            torch.cat([world_positions, arm_quats], dim=-1)
        )
        
        # Calculate and set velocities for smooth physics
        if self.cfg.arm_movement_speed > 0:
            velocities = torch.zeros((self.num_envs, 6), device=self.device)
            
            for i in range(self.num_envs):
                # Calculate velocity based on motion pattern and phase
                period = 4.0
                phase = (self._arm_motion_time[i] % period) / period
                
                # Velocity direction changes at phase 0.5
                direction = 1.0 if phase < 0.5 else -1.0
                
                if self._arm_motion_pattern[i] == 0:  # X-axis
                    velocities[i, 0] = direction * self.cfg.arm_movement_speed
                elif self._arm_motion_pattern[i] == 1:  # Y-axis
                    velocities[i, 1] = direction * self.cfg.arm_movement_speed
                else:  # Diagonal
                    velocities[i, 0] = direction * self.cfg.arm_movement_speed * 0.7
                    velocities[i, 1] = direction * self.cfg.arm_movement_speed * 0.7
            
            self._arm.write_root_velocity_to_sim(velocities)
        else:
            self._arm.write_root_velocity_to_sim(
                torch.zeros((self.num_envs, 6), device=self.device)
            )
        
    def _get_observations(self) -> dict:
        """Compute and return observations as a dictionary."""
        # Get state observations
        state_obs = self._get_state_observations()
        
        # Get camera observations with domain randomization
        camera_obs = self._get_camera_observations()
        
        obs = {"image": camera_obs, "state": state_obs}
        observations = {"policy": obs}
        
        return observations

    def _get_state_observations(self) -> torch.Tensor:
        """Get state-based observations."""
        # Get joint positions with noise
        joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        if self.cfg.joint_pos_noise_max > 0:
            joint_pos_noise = torch.rand_like(joint_pos) * (
                self.cfg.joint_pos_noise_max - self.cfg.joint_pos_noise_min
            ) + self.cfg.joint_pos_noise_min
            joint_pos_noisy = joint_pos + joint_pos_noise
        else:
            joint_pos_noisy = joint_pos
        
        # Get joint velocities with noise
        # joint_vel = self._robot.data.joint_vel[:, self._joint_indices]
        # if self.cfg.joint_vel_noise_max > 0:
        #     joint_vel_noise = torch.rand_like(joint_vel) * (
        #         self.cfg.joint_vel_noise_max - self.cfg.joint_vel_noise_min
        #     ) + self.cfg.joint_vel_noise_min
        #     joint_vel_noisy = joint_vel + joint_vel_noise
        # else:
        #     joint_vel_noisy = joint_vel
        
        # Get target pose (already in robot base frame)
        target_pose = self._target_poses
        
        # Concatenate all state observations
        state_obs = torch.cat([
            joint_pos_noisy,      # 6 dims
            # joint_vel_noisy,      # 6 dims
            target_pose,          # 7 dims
        ], dim=-1)
        
        return state_obs
    
    def _visualize_camera_observation(self, raw_image: torch.Tensor, processed_image: torch.Tensor, env_id: int = 0):
        """Visualize raw and processed camera observations for debugging."""
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Get data for specified environment
        raw_env = raw_image[env_id].cpu().numpy()
        processed_env = processed_image[env_id].cpu().numpy()
        
        # Raw image
        axes[0].imshow(raw_env)
        axes[0].set_title(f'Raw Camera Image (224x224)\nEnv {env_id}')
        axes[0].axis('off')
        
        # Add crop region visualization on raw image
        crop_rect = Rectangle((0, self.cfg.camera_crop_top), 224, 
                            224 - self.cfg.camera_crop_top - self.cfg.camera_crop_bottom,
                            linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(crop_rect)
        axes[0].text(5, self.cfg.camera_crop_top - 5, 'Crop Region', color='red', fontsize=10)
        
        # Processed image (CHW to HWC for visualization)
        processed_vis = processed_env.transpose(1, 2, 0)
        
        # Show normalized values
        axes[1].imshow(processed_vis)
        axes[1].set_title(f'Processed Image ({self.cfg.camera_target_height}x{self.cfg.camera_target_width})')
        axes[1].axis('off')
        
        # Show value distribution
        axes[2].hist(processed_env.flatten(), bins=50, alpha=0.7)
        axes[2].set_xlabel('Pixel Value')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Pixel Value Distribution')
        axes[2].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = processed_env.mean()
        std_val = processed_env.std()
        axes[2].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        axes[2].axvline(mean_val + std_val, color='orange', linestyle='--', label=f'Std: {std_val:.3f}')
        axes[2].axvline(mean_val - std_val, color='orange', linestyle='--')
        axes[2].legend()
        
        # Add environment info
        fig.suptitle(f'Camera Observation Debug - Step {self.common_step_counter}', fontsize=14)
        
        # Save figure
        save_path = os.path.join(self.cfg.visualization_save_path, f'camera_obs_step_{self.common_step_counter}_env_{env_id}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[VIS] Saved camera observation visualization to: {save_path}")
    
    def _apply_camera_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        """Apply domain randomization augmentations to camera images."""
        if not self.cfg.enable_camera_noise:
            return image
            
        augmented = image.clone()
        
        # Apply brightness adjustment
        brightness = self._camera_brightness.view(-1, 1, 1, 1)
        augmented = augmented * brightness
        
        # Apply contrast adjustment
        contrast = self._camera_contrast.view(-1, 1, 1, 1)
        mean = augmented.mean(dim=(1, 2, 3), keepdim=True)
        augmented = (augmented - mean) * contrast + mean
        
        # Add Gaussian noise
        noise = torch.randn_like(augmented) * self.cfg.camera_noise_std
        augmented = augmented + noise
        
        # Clamp values to valid range
        augmented = torch.clamp(augmented, 0.0, 1.0)
        
        return augmented
    
    def _get_camera_observations(self) -> torch.Tensor:
        """Get camera observations with domain randomization."""
        # Render and get camera data
        self._tiled_camera.update(dt=self.scene.physics_dt)
        data = self._tiled_camera.data.output["rgb"][:, :, :, :3]  # [num_envs, H, W, 3]
        
        # Convert to float and normalize to [0, 1]
        camera_obs = data.float() / 255.0
        
        # === APPLY CAMERA AUGMENTATIONS (Domain Randomization) ===
        camera_obs = self._apply_camera_augmentation(camera_obs)
        
        # Crop the image
        cropped = camera_obs[:, 
                           self.cfg.camera_crop_top:camera_obs.shape[1]-self.cfg.camera_crop_bottom, 
                           :, :]
        
        # Resize to target dimensions
        # For now, we'll use a simple interpolation
        # In practice, you might want to use proper image resizing
        import torch.nn.functional as F
        
        # Convert to CHW format for resize
        cropped_chw = cropped.permute(0, 3, 1, 2)  # [num_envs, 3, H_crop, W]
        
        # Resize
        resized = F.interpolate(
            cropped_chw, 
            size=(self.cfg.camera_target_height, self.cfg.camera_target_width),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert back to HWC format
        camera_processed = resized.permute(0, 2, 3, 1)  # [num_envs, H_target, W_target, 3]
        
        # Visualize periodically if enabled
        if self.cfg.debug_vis and self.common_step_counter % self.cfg.visualize_camera_interval == 0:
            self._visualize_camera_observation(camera_obs, resized, env_id=0)
            self._vis_counter += 1
        
        return camera_processed
    
    def _compute_position_error(self) -> torch.Tensor:
        """Compute position error between end-effector and target."""
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        
        # Target position is in robot base frame, need to transform to world
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        
        target_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, self._target_poses[:, :3]
        )
        
        return torch.norm(ee_position - target_pos_w, dim=-1)
    
    def _compute_orientation_error(self) -> torch.Tensor:
        """Compute orientation error between end-effector and target."""
        ee_quat = self._ee_frame.data.target_quat_w[..., 0, :]
        
        # Target orientation is in robot base frame
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        target_quat_w = math_utils.quat_mul(robot_quat, self._target_poses[:, 3:7])
        
        # Compute relative quaternion
        rel_quat = math_utils.quat_mul(math_utils.quat_inv(ee_quat), target_quat_w)
        
        # Convert to angle
        return 2.0 * torch.acos(torch.clamp(torch.abs(rel_quat[:, 0]), -1.0, 1.0))
    
    
    def _point_to_box_distance(self, point: torch.Tensor, box_pos: torch.Tensor, 
                              box_quat: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor:
        """Calculate minimum distance from point to oriented box."""
        # Transform point to box local frame
        rel_pos = point - box_pos
        
        # Convert quaternion to rotation matrix
        rot_matrix = math_utils.matrix_from_quat(box_quat)
        
        # Transform to box local coordinates
        local_point = torch.matmul(rot_matrix.transpose(-2, -1), rel_pos.unsqueeze(-1)).squeeze(-1)
        
        # Calculate distance to box surface
        # Clamp point to box bounds
        clamped = torch.clamp(local_point, -half_extents, half_extents)
        
        # If point is inside box, distance is negative (distance to nearest face)
        inside = torch.all(torch.abs(local_point) <= half_extents, dim=-1)
        
        # Distance to surface
        surface_dist = torch.norm(local_point - clamped, dim=-1)
        
        # For points inside, calculate distance to nearest face
        distances_to_faces = half_extents - torch.abs(local_point)
        min_face_dist = torch.min(distances_to_faces, dim=-1)[0]
        
        # Set negative distance for points inside
        distance = torch.where(inside, -min_face_dist, surface_dist)
        
        return distance
    
    def _huber_loss(self, x: torch.Tensor, delta: float) -> torch.Tensor:
        """Compute Huber loss for robust distance penalty."""
        abs_x = torch.abs(x)
        return torch.where(
            abs_x <= delta,
            0.5 * x * x,
            delta * (abs_x - 0.5 * delta)
        )
    
    def _compute_arm_avoidance_rewards(self) -> torch.Tensor:
        """Compute arm avoidance rewards with dynamic collision risk assessment."""
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Arm dimensions (half-extents for easier calculation)
        arm_half_extents = torch.tensor([0.25, 0.1, 0.06], device=self.device)  # [0.5, 0.2, 0.12] / 2
        
        # Safety parameters - reduced to allow closer approach when needed
        critical_distance = 0.03  # Very close - high penalty
        danger_distance = 0.08    # Close - moderate penalty
        safe_distance = 0.12      # Reduced from 0.15 to allow more flexibility
        
        # Get arm pose and velocity
        arm_position = self._arm.data.root_pos_w[:, :3]
        arm_quat = self._arm.data.root_quat_w
        arm_velocity = self._arm.data.root_lin_vel_w if hasattr(self._arm.data, 'root_lin_vel_w') else torch.zeros_like(arm_position)
        
        # Get end effector position and velocity
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_velocity = self._ee_frame.data.target_lin_vel_w[..., 0, :] if hasattr(self._ee_frame.data, 'target_lin_vel_w') else torch.zeros_like(ee_position)
        
        # Calculate minimum distance from end effector to arm cuboid
        min_distances = self._point_to_box_distance(
            ee_position, 
            arm_position, 
            arm_quat, 
            arm_half_extents
        )
        
        # Calculate relative velocity (positive means moving away from each other)
        relative_pos = ee_position - arm_position
        relative_vel = ee_velocity - arm_velocity
        relative_speed = torch.sum(relative_pos * relative_vel, dim=-1) / (torch.norm(relative_pos, dim=-1) + 1e-6)
        
        # Dynamic penalty based on both distance and relative motion
        for i in range(self.num_envs):
            distance = min_distances[i]
            
            if distance < critical_distance:
                # Very close - high penalty regardless of motion
                rewards[i] = -15.0
            elif distance < danger_distance:
                # In danger zone - penalty depends on relative motion
                base_penalty = -8.0 * (1.0 - (distance - critical_distance) / (danger_distance - critical_distance))
                
                # Reduce penalty if moving away from arm
                if relative_speed[i] > 0:
                    motion_factor = torch.clamp(relative_speed[i] / 0.5, 0.0, 0.7)  # Max 70% reduction
                    rewards[i] = base_penalty * (1.0 - motion_factor)
                else:
                    # Increase penalty if moving toward arm
                    motion_factor = torch.clamp(-relative_speed[i] / 0.5, 0.0, 0.5)  # Max 50% increase
                    rewards[i] = base_penalty * (1.0 + motion_factor)
            elif distance < safe_distance:
                # In safe zone - small penalty that decreases with distance
                base_penalty = -2.0 * (1.0 - (distance - danger_distance) / (safe_distance - danger_distance))
                
                # Only apply penalty if moving toward arm
                if relative_speed[i] < 0:
                    rewards[i] = base_penalty
                else:
                    rewards[i] = 0.0  # No penalty if moving away
            else:
                # Outside safe distance - no penalty
                rewards[i] = 0.0
        
        return rewards
    
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
        
        # Calculate distances to arm obstacle
        arm_half_extents = torch.tensor([0.25, 0.1, 0.06], device=self.device)
        arm_position = self._arm.data.root_pos_w[:, :3]
        arm_quat = self._arm.data.root_quat_w
        
        min_distances_to_arm = self._point_to_box_distance(
            ee_position, 
            arm_position, 
            arm_quat, 
            arm_half_extents
        )
        
        # Compute β transition factor for APF
        beta = self._compute_beta_transition(min_distances_to_arm)
        
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
        orientation_huber_loss = self._huber_loss(orientation_error, self.cfg.huber_delta * 0.5)  # Smaller delta for orientation
        orientation_reward = self.cfg.reward_orientation_weight * orientation_huber_loss
        traditional_rewards += orientation_reward
        
        # 4. Joint torque penalty
        if hasattr(self._robot.data, 'applied_torque') and self._robot.data.applied_torque is not None:
            joint_torques = self._robot.data.applied_torque[:, self._joint_indices]
            torque_penalty = torch.sum(torch.square(joint_torques), dim=1)
            torque_reward = self.cfg.reward_torque_weight * torque_penalty
            rewards += torque_reward
        else:
            torque_reward = torch.zeros_like(rewards)
        
        # 5. Table collision penalty
        ee_height = ee_position[:, 2]
        table_height = TABLE_HEIGHT
        safety_margin = 0.05
        
        table_penalty = torch.where(
            ee_height < (table_height + safety_margin),
            torch.ones_like(ee_height) * self.cfg.reward_table_collision_weight,
            torch.zeros_like(ee_height)
        )
        traditional_rewards += table_penalty
        
        # 6. Arm avoidance rewards (part of traditional rewards)
        arm_reward = self._compute_arm_avoidance_rewards() * self.cfg.reward_arm_avoidance_weight
        traditional_rewards += arm_reward

        # 7 Success for reaching the end goal and avoiding the arm
        # Calculate minimum distance from end effector to arm cuboid
        # Arm dimensions (half-extents for easier calculation)
        arm_half_extents = torch.tensor([0.25, 0.1, 0.06], device=self.device)  # [0.5, 0.2, 0.12] / 2
        arm_position = self._arm.data.root_pos_w[:, :3]
        arm_quat = self._arm.data.root_quat_w

        min_distances = self._point_to_box_distance(
            ee_position, 
            arm_position, 
            arm_quat, 
            arm_half_extents
        )
        success_mask = (position_error < 0.05) & (min_distances > 0.08)
        traditional_rewards += torch.where(success_mask, 5.0, 0.0)
        
        # === Energy-based Rewards (Renergy) ===
        energy_rewards = self._compute_energy_reward() * self.cfg.energy_reward_weight
        
        # === Adaptive Combination using APF ===
        # Rada = β · Rt + (1 − β) · Renergy
        rewards = beta * traditional_rewards + (1.0 - beta) * energy_rewards
        
        # Track reward components for logging
        if hasattr(self, '_episode_sums'):
            self._episode_sums["total_reward"] += rewards
            self._episode_sums["position_error"] += position_error
            self._episode_sums["min_arm_distance"] = torch.minimum(
                self._episode_sums["min_arm_distance"], min_distances_to_arm
            )
            
            # Check for success
            success_mask = (position_error < 0.05) & (min_distances_to_arm > 0.08)
            self._episode_sums["success_count"] += success_mask.float()
            
            # Update success buffer for curriculum learning
            if torch.any(success_mask):
                success_rate = success_mask.float().mean()
                self._success_buffer[self._success_buffer_idx] = success_rate
                self._success_buffer_idx = (self._success_buffer_idx + 1) % self.cfg.success_window_size
        
        # Log detailed reward breakdown for first environment occasionally
        if self.common_step_counter % 500 == 0 and self.num_envs > 0:
            env_0_data = {
                "position_error": position_error[0].item(),
                "position_huber": position_huber_loss[0].item(),
                "orientation_error": orientation_error[0].item(),
                # "action_penalty": torque_penalty[0].item(),
                "min_dist_to_arm": min_distances_to_arm[0].item(),
                "beta": beta[0].item(),
                "energy_reward": energy_rewards[0].item(),
                "total_reward": rewards[0].item()
            }
            # print(f"[REWARD] Env 0 - Beta: {env_0_data['beta']:.3f}, "
            #       f"Dist to arm: {env_0_data['min_dist_to_arm']:.3f}, "
            #       f"Energy: {env_0_data['energy_reward']:.3f}, "
            #       f"Total: {env_0_data['total_reward']:.3f}")
        
        return rewards
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return termination conditions."""
        # Time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Success condition
        position_error = self._compute_position_error()
        orientation_error = self._compute_orientation_error()
        
        success_position = position_error < self.cfg.position_threshold
        success_orientation = orientation_error < self.cfg.orientation_threshold
        
        # Joint velocity check
        joint_vel = self._robot.data.joint_vel[:, self._joint_indices]
        velocity_converged = torch.all(
            torch.abs(joint_vel) < self.cfg.velocity_threshold, dim=-1
        )
        
        # Boundary check - reset if robot goes out of bounds
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        x_bounds = torch.logical_or(
            ee_position[:, 0] < 0.3 - self.cfg.bounds_safety_margin,
            ee_position[:, 0] > 0.8 + self.cfg.bounds_safety_margin
        )
        y_bounds = torch.logical_or(
            ee_position[:, 1] < -0.6 - self.cfg.bounds_safety_margin,
            ee_position[:, 1] > 0.6 + self.cfg.bounds_safety_margin
        )
        z_bounds = torch.logical_or(
            ee_position[:, 2] < 0.4,  # Well above table
            ee_position[:, 2] > 1.5
        )
        out_of_bounds = torch.logical_or(torch.logical_or(x_bounds, y_bounds), z_bounds)
        
        # Check for arm collision (very close to arm)
        arm_half_extents = torch.tensor([0.25, 0.1, 0.06], device=self.device)
        arm_position = self._arm.data.root_pos_w[:, :3]
        arm_quat = self._arm.data.root_quat_w
        min_distances = self._point_to_box_distance(
            ee_position, arm_position, arm_quat, arm_half_extents
        )
        arm_collision = min_distances < 0.02  # 2cm threshold
        
        # Success only if at target with low velocity
        success = success_position & success_orientation & velocity_converged
        
        # Episode termination
        terminated = time_out | out_of_bounds | arm_collision
        
        # Log termination reasons
        if torch.any(terminated):
            term_envs = torch.nonzero(terminated).squeeze(-1)
            for env_id in term_envs[:3]:  # Log first 3
                if time_out[env_id]:
                    reason = "timeout"
                elif out_of_bounds[env_id]:
                    reason = f"bounds (pos: {ee_position[env_id].cpu().numpy()})"
                elif arm_collision[env_id]:
                    reason = f"arm collision (dist: {min_distances[env_id]:.3f}m)"
                else:
                    reason = "unknown"
                print(f"[DONE] Env {env_id} terminated: {reason}, Success: {success[env_id]}")
        
        return terminated, time_out
    
    def set_debug_vis(self, debug_vis: bool) -> None:
        """Set debug visualization mode."""
        self.cfg.debug_vis = debug_vis
        if hasattr(self, '_ee_frame'):
            self._ee_frame.set_debug_vis(debug_vis)
            
        # Create target pose visualizer if in debug mode
        if debug_vis and not hasattr(self, 'target_pos_visualizer'):
            from isaaclab.markers import VisualizationMarkersCfg, CUBOID_MARKER_CFG
            
            marker_cfg = CUBOID_MARKER_CFG.copy()
            marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
            marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0)
            )
            marker_cfg.prim_path = "/Visuals/TargetPose"
            
            self.target_pos_visualizer = VisualizationMarkers(marker_cfg)
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization mode implementation."""
        # Handle end-effector frame visualization
        if hasattr(self, '_ee_frame'):
            self._ee_frame.set_debug_vis(debug_vis)
            
        # Handle target pose visualization
        if debug_vis:
            if not hasattr(self, 'target_pos_visualizer'):
                # Create visualizer
                from isaaclab.markers import VisualizationMarkersCfg, CUBOID_MARKER_CFG
                
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0)
                )
                marker_cfg.prim_path = "/Visuals/TargetPose"
                
                self.target_pos_visualizer = VisualizationMarkers(marker_cfg)
        else:
            # Hide visualization if it exists
            if hasattr(self, 'target_pos_visualizer'):
                self.target_pos_visualizer.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        """Callback for debug visualization."""
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
        
        # Additionally log APF beta values for first few environments
        if self.common_step_counter % 10 == 0:  # Every 100 steps

            # Calculate current beta values
            ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
            arm_half_extents = torch.tensor([0.25, 0.1, 0.06], device=self.device)
            arm_position = self._arm.data.root_pos_w[:, :3]
            arm_quat = self._arm.data.root_quat_w
            
            min_distances = self._point_to_box_distance(
                ee_position, arm_position, arm_quat, arm_half_extents
            )
            beta_values = self._compute_beta_transition(min_distances)

            joint_vel = self._robot.data.joint_vel[:, self._joint_indices]
            max_velocity = 1.5  # rad/s
            current_joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
            velocity_command = (self._robot_dof_targets - current_joint_pos) / self.physics_dt
            velocity_command = torch.clamp(velocity_command, -max_velocity, max_velocity)
            self._robot_dof_targets = current_joint_pos + velocity_command * self.physics_dt

            # Log for first 3 environments
            for i in range(min(3, self.num_envs)):
                print(f"[APF] Env {i}: dist={min_distances[i]:.3f}m, β={beta_values[i]:.3f}")
                # print(f"Joint Velocities sent as observation to Env{i}: vel {joint_vel}")
                # print(f"Joint Velocities command added to action to Env{i}: vel {velocity_command}")
                # print(f"Action Target sent to robot Env{i}: action {self._robot_dof_targets}")


# Factory function for creating the environment
def create_obj_camera_pose_tracking_env(
    cfg: ObjCameraPoseTrackingDirectEnvCfg = None,
    render_mode: str = None,
    **kwargs
) -> ObjCameraPoseTrackingDirectEnv:
    """Factory function to create the environment with default config if none provided."""
    if cfg is None:
        cfg = ObjCameraPoseTrackingDirectEnvCfg()
    
    return ObjCameraPoseTrackingDirectEnv(cfg, render_mode=render_mode, **kwargs)