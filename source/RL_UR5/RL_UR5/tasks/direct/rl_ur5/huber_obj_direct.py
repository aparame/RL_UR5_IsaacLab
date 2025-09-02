"""
Direct RL Environment for Object Camera Pose Tracking with UR5 Robot

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
print([attr for attr in dir(sim_utils) if 'material' in attr.lower() or 'surface' in attr.lower()])
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
    """Configuration for the direct RL environment."""
    
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
            size=(0.762, 2.5, 0.01),
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
            pos=(0.2, 0.0, 0.74),
            rot=(0.70711, 0.0, 0.70711, 0.0)
        ),
    )

    # I2R plane configuration with image texture
    i2r_plane_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/i2r_plane",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/adi2440/Desktop/ur5_isaacsim/usd/i2r_plane.usd",
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
            pos=(0.21, -0.2, 0.93),
            rot=(0.50000, 0.50000, 0.50000, 0.50000)
        ),
    )

    # Clemson plane configuration with image texture
    clemson_plane_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/clemson_plane",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/adi2440/Desktop/ur5_isaacsim/usd/clemson_plane.usd",
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
            pos=(0.21, 0.2, 0.93),
            rot=(0.50000, 0.50000, 0.50000, 0.50000)
        ),
    )


    # Frame transformer for end-effector
    ee_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,  # Now this works since enable_debug_vis is defined above
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
        data_types=["rgb"],  # No depth as requested
        spawn=sim_utils.PinholeCameraCfg(
            # ZED2 calculated parameters:
            # fx = fy = 361.63 pixels -> focal_length = 2.955mm
            # 1/3" CMOS sensor: 5.229 x 2.942 mm
            focal_length=2.955,           # Changed from 2.208
            focus_distance=28.0,          # Keep existing (focus distance for rendering)
            horizontal_aperture=5.229,    # Changed from 5.76 (ZED2 sensor width)
            vertical_aperture=2.942,      # Changed from 3.24 (ZED2 sensor height)
            clipping_range=(0.1, 1000.0)  # Keep existing
        ),
        # ZED2 resolution from camera_info
        width=224,    # Matches ZED2 exactly
        height=224,   # Matches ZED2 exactly
        # Keep your existing camera pose
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.27, 0.06, 1.143),                    # Unchanged
            rot=(0.59637,0.37993,0.37993,0.59637),     # Unchanged
            convention="opengl"                         # Unchanged
        )
    )

    # Basic environment settings
    episode_length_s = 6.0
    decimation = 4
    action_scale = 0.3  # Reduced for smoother movements
    state_dim = 13
    camera_target_height = 100
    camera_target_width = 120    

    # Observation and action spaces
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
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
    curriculum_advance_threshold = 0.5  # Advance when success rate > 70%

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
        "x": (1.0, 1.2),
        "y": (-0.5, 0.5),
        "z": (0.80, 1.2),
    }
    arm_movement_speed = 0.3  # Speed of random movement
    
    # Reward settings
    reward_distance_weight = -2.5
    reward_distance_tanh_weight = 1.5
    reward_distance_tanh_std = 0.1
    reward_orientation_weight = -1.0
    reward_torque_weight = -0.001  # Replaced torque with action penalty
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
    action_filter_cutoff_freq = 8.0
    action_filter_damping_ratio = 0.707
    
    # Termination settings
    position_threshold = 0.01
    orientation_threshold = 0.05
    velocity_threshold = 0.05
    torque_threshold = 1.0
    bounds_safety_margin = 0.1  # 0.1m margin for bounds checking
    
    # Camera preprocessing settings
    camera_crop_top = 60
    camera_crop_bottom = 20
    
    # Visualization settings
    visualize_camera_interval = 20000  # Visualize camera every N steps
    visualization_save_path = "/home/adi2440/Desktop/ur5_isaacsim/camera_obs"  # Path to save visualizations

    
    # Noise settings
    joint_pos_noise_min = -0.01
    joint_pos_noise_max = 0.01
    joint_vel_noise_min = -0.001
    joint_vel_noise_max = 0.001
    
    # Reset settings
    robot_base_pose = [-0.568, -0.858,  1.402, -2.185, -1.6060665,  1.64142667]
    robot_reset_noise_range = 0.05


class ObjCameraPoseTrackingDirectEnv(DirectRLEnv):
    """Direct RL environment for object camera pose tracking with multi-observation space."""
    
    cfg: ObjCameraPoseTrackingDirectEnvCfg
    
    def __init__(self, cfg: ObjCameraPoseTrackingDirectEnvCfg, render_mode: str | None = None, **kwargs):
        # Store config
        self.cfg = cfg

        # === episode / logging bookkeeping ===
        self._episode_counter = 0

        # file handles, writers, directories (initialized on first save)
        self._state_obs_file    = None
        self._state_csv_writer  = None
        self._image_obs_dir     = None
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
        
        # Log initial information
        print(f"[INFO] Environment initialized with {self.num_envs} environments")
        print(f"[INFO] Action scale: {self.cfg.action_scale}")
        print(f"[INFO] Target pose range X: {self.cfg.target_pose_range['x']}")
        print(f"[INFO] Target pose range Y: {self.cfg.target_pose_range['y']}")
        print(f"[INFO] Target pose range Z: {self.cfg.target_pose_range['z']}")
        print(f"[INFO] Arm bounds X: {self.cfg.arm_position_bounds['x']}")
        print(f"[INFO] Arm bounds Y: {self.cfg.arm_position_bounds['y']}")
        print(f"[INFO] Arm bounds Z: {self.cfg.arm_position_bounds['z']}")
        
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
        self._clemson_plane = RigidObject(self.cfg.clemson_plane_cfg)
        self._i2r_plane = RigidObject(self.cfg.i2r_plane_cfg)

        # --- clone source  env_1&env_N (env_0 keeps its prims) ---
        self.scene.clone_environments(copy_from_source=False)

        # --- register handles in IsaacLab's scene registry ---
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.rigid_objects["arm"] = self._arm
        
        # Add static assets to scene registry
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["white_plane"] = self._white_plane
        self.scene.rigid_objects["clemson_plane"] = self._clemson_plane
        self.scene.rigid_objects["i2r_plane"] = self._i2r_plane

        # --- add static geometry and lighting ---
        # Ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        # Multiple lights for better scene illumination
        # Main dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=1600.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/DomeLight", light_cfg)
        
        # Additional directional light for shadows
        dir_light_cfg = sim_utils.DistantLightCfg(intensity=1000.0, color=(1.0, 1.0, 0.9), angle=0.53)
        dir_light_cfg.func("/World/DirectionalLight", dir_light_cfg)

        
    def _setup_action_filter(self):
        """Initialize action filter states and coefficients."""
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
    
    def _update_curriculum_settings(self):
        """Update environment settings based on curriculum level."""
        if not self.cfg.curriculum_enabled:
            return
            
        level = self._curriculum_level
        
        # Update arm movement speed
        if level < len(self.cfg.curriculum_arm_speeds):
            self.cfg.arm_movement_speed = self.cfg.curriculum_arm_speeds[level]
        
        # Update target pose ranges
        if level < len(self.cfg.curriculum_target_ranges):
            self.cfg.target_pose_range.update(self.cfg.curriculum_target_ranges[level])
        
        print(f"[CURRICULUM] Level {level}: arm_speed={self.cfg.arm_movement_speed:.2f}, "
              f"target_x={self.cfg.target_pose_range['x']}, "
              f"target_y={self.cfg.target_pose_range['y']}")
    
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
        """Apply actions before physics step."""
        # Store raw actions
        self.actions = actions.clone().clamp(-1.0, 1.0)

        
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

        # Update debug visualization if enabled
        self._update_debug_visualization()
        
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
        num = len(env_ids)
        if num == 0:
            return

        # sample positions within configured ranges
        x = sample_uniform(
            self.cfg.target_pose_range["x"][0],
            self.cfg.target_pose_range["x"][1],
            (num,), self.device
        )
        y = sample_uniform(
            self.cfg.target_pose_range["y"][0],
            self.cfg.target_pose_range["y"][1],
            (num,), self.device
        )
        z = sample_uniform(
            self.cfg.target_pose_range["z"][0],
            self.cfg.target_pose_range["z"][1],
            (num,), self.device
        )

        # sample orientations (roll, pitch, yaw)
        roll = sample_uniform(
            self.cfg.target_pose_range["roll"][0],
            self.cfg.target_pose_range["roll"][1],
            (num,), self.device
        )
        pitch = sample_uniform(
            self.cfg.target_pose_range["pitch"][0],
            self.cfg.target_pose_range["pitch"][1],
            (num,), self.device
        )
        yaw = sample_uniform(
            self.cfg.target_pose_range["yaw"][0],
            self.cfg.target_pose_range["yaw"][1],
            (num,), self.device
        )
        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        # write into the buffer
        self._target_poses[env_ids, :3] = torch.stack([x, y, z], dim=-1)
        self._target_poses[env_ids, 3:7] = quat
        
        # Debug print for first environment
        if 0 in env_ids and len(env_ids) <= 4:  # Only log for small resets
            idx = env_ids.index(0)
            print(f"[DEBUG] New target for env 0: pos=[{x[idx].item():.3f}, {y[idx].item():.3f}, {z[idx].item():.3f}]")

    def _update_arm_position(self):
        """Update the human arm position with simple linear motion pattern."""
        # Initialize motion parameters if not exists
        if not hasattr(self, '_arm_motion_time'):
            self._arm_motion_time = torch.zeros(self.num_envs, device=self.device)
            self._arm_motion_pattern = torch.randint(0, 3, (self.num_envs,), device=self.device)  # 0: X-axis, 1: Y-axis, 2: diagonal
        
        # Get current arm positions and orientations
        arm_positions = self._arm.data.root_pos_w.clone()
        arm_quats = self._arm.data.root_quat_w.clone()
        
        # Update motion time
        self._arm_motion_time += self.physics_dt
        
        for i in range(self.num_envs):
            # Calculate base position (center of motion range)
            base_x = (self.cfg.arm_position_bounds["x"][0] + self.cfg.arm_position_bounds["x"][1]) / 2
            base_y = (self.cfg.arm_position_bounds["y"][0] + self.cfg.arm_position_bounds["y"][1]) / 2
            base_z = (self.cfg.arm_position_bounds["z"][0] + self.cfg.arm_position_bounds["z"][1]) / 2
            
            # Calculate motion amplitudes
            amp_x = (self.cfg.arm_position_bounds["x"][1] - self.cfg.arm_position_bounds["x"][0]) / 2 * 0.8
            amp_y = (self.cfg.arm_position_bounds["y"][1] - self.cfg.arm_position_bounds["y"][0]) / 2 * 0.8
            amp_z = (self.cfg.arm_position_bounds["z"][1] - self.cfg.arm_position_bounds["z"][0]) / 2 * 0.8
            
            # Linear motion with triangle wave (back and forth)
            # Period of 4 seconds for complete back-and-forth motion
            period = 6.0
            phase = (self._arm_motion_time[i] % period) / period
            
            # Triangle wave: 0->1->0
            if phase < 0.5:
                motion_factor = phase * 2.0
            else:
                motion_factor = 2.0 - phase * 2.0
            
            # Apply motion pattern
            if self._arm_motion_pattern[i] == 0:  # X-axis motion
                new_x = base_x + (motion_factor - 0.5) * 2 * amp_x
                new_y = base_y
                new_z = base_z
            elif self._arm_motion_pattern[i] == 1:  # Y-axis motion
                new_x = base_x
                new_y = base_y + (motion_factor - 0.5) * 2 * amp_y
                new_z = base_z
            else:  # Diagonal motion (X-Y plane)
                new_x = base_x + (motion_factor - 0.5) * 2 * amp_x * 0.7
                new_y = base_y + (motion_factor - 0.5) * 2 * amp_y * 0.7
                new_z = base_z
            
            # Update position in world frame
            local_pos = torch.tensor([new_x, new_y, new_z], device=self.device)
            arm_positions[i, :3] = local_pos + self.scene.env_origins[i, :3]
        
        # Apply new poses (keep original orientation)
        self._arm.write_root_pose_to_sim(
            torch.cat([arm_positions, arm_quats], dim=-1)
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
        
        # Get camera observations
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
        
        # Show normalized image
        axes[1].imshow(processed_vis + 0.5)  # Add 0.5 since we subtracted mean
        axes[1].set_title(f'Processed & Resized\n({self.cfg.camera_target_height}x{self.cfg.camera_target_width})')
        axes[1].axis('off')
        
        # Show channel statistics
        axes[2].axis('off')
        stats_text = f"Processed Image Statistics (Env {env_id}):\n\n"
        stats_text += f"Shape: {processed_env.shape}\n"
        stats_text += f"Min value: {processed_env.min():.3f}\n"
        stats_text += f"Max value: {processed_env.max():.3f}\n"
        stats_text += f"Mean value: {processed_env.mean():.3f}\n"
        stats_text += f"Std value: {processed_env.std():.3f}\n\n"
        
        # Add current state info
        ee_pos = self._ee_frame.data.target_pos_w[env_id, 0, :].cpu().numpy()
        target_pos = self._target_poses[env_id, :3].cpu().numpy()
        arm_pos = self._arm.data.root_pos_w[env_id, :3].cpu().numpy()
        
        stats_text += f"End-effector pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]\n"
        stats_text += f"Target pos: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]\n"
        stats_text += f"Arm obstacle pos: [{arm_pos[0]:.3f}, {arm_pos[1]:.3f}, {arm_pos[2]:.3f}]\n"
        
        axes[2].text(0.1, 0.5, stats_text, transform=axes[2].transAxes, 
                    fontsize=11, verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{self.cfg.visualization_save_path}/camera_obs_step_{self.common_step_counter:06d}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        if self._vis_counter % 10 == 0:  # Log every 10th visualization
            print(f"[VIS] Saved camera observation to: {filename}")
        
        self._vis_counter += 1
    
    def _get_camera_observations(self) -> torch.Tensor:
        """Get and preprocess camera observations."""
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
    
    def _huber_loss(self, x: torch.Tensor, delta: float) -> torch.Tensor:
        """Compute Huber loss for robust distance penalty."""
        abs_x = torch.abs(x)
        return torch.where(
            abs_x <= delta,
            0.5 * x * x,
            delta * (abs_x - 0.5 * delta)
        )
    
    def _compute_beta_transition(self, min_distances: torch.Tensor) -> torch.Tensor:
        """Compute smooth transition factor ² for adaptive reward mixing."""
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
        
        # Compute ² transition factor for APF
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
        
        # # 4. Joint torque penalty
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
        joint_velocities = torch.norm(self._robot.data.joint_vel, p=2, dim=-1)

        success_mask = (position_error < 0.05) & (min_distances > 0.08) & (joint_velocities < self.cfg.velocity_threshold)
        traditional_rewards += torch.where(success_mask, 5.0, 0.0)
        
        # === Energy-based Rewards (Renergy) ===
        energy_rewards = self._compute_energy_reward() * self.cfg.energy_reward_weight
        
        # === Adaptive Combination using APF ===
        # Rada = ² · Rt + (1  ²) · Renergy
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
        
                # Add this at the end:
        if self.cfg.debug_vis and hasattr(self, "target_pos_visualizer"):
            self._update_debug_visualization()


        return rewards
    
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

    def _point_to_box_distance(self, points: torch.Tensor, box_pos: torch.Tensor, 
                            box_quat: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor:
        """
        Calculate minimum distance from points to oriented boxes.
        
        Args:
            points: Points to check (N, 3)
            box_pos: Box center positions (N, 3)
            box_quat: Box orientations as quaternions (N, 4)
            half_extents: Box half-extents (3,)
        
        Returns:
            Minimum distances (N,)
        """
        # Transform points to box local frame
        relative_pos = points - box_pos
        
        # Convert quaternion to rotation matrix for inverse transform
        # Using math_utils to handle quaternion operations
        inv_box_quat = math_utils.quat_inv(box_quat)
        
        # Rotate points to box local frame
        local_points = math_utils.quat_apply(inv_box_quat, relative_pos)
        
        # Find closest point on box surface
        # Clamp to box bounds
        closest_point_local = torch.clamp(
            local_points, 
            -half_extents.unsqueeze(0), 
            half_extents.unsqueeze(0)
        )
        
        # Check if point is inside the box
        inside_box = torch.all(
            torch.abs(local_points) <= half_extents.unsqueeze(0), 
            dim=-1
        )
        
        # For points inside the box, find distance to nearest face
        # For points outside, use standard distance
        distances = torch.zeros(self.num_envs, device=self.device)
        
        for i in range(self.num_envs):
            if inside_box[i]:
                # Find distance to each face and take minimum
                distances_to_faces = torch.zeros(6, device=self.device)
                distances_to_faces[0] = half_extents[0] - local_points[i, 0]  # +X face
                distances_to_faces[1] = local_points[i, 0] + half_extents[0]  # -X face
                distances_to_faces[2] = half_extents[1] - local_points[i, 1]  # +Y face
                distances_to_faces[3] = local_points[i, 1] + half_extents[1]  # -Y face
                distances_to_faces[4] = half_extents[2] - local_points[i, 2]  # +Z face
                distances_to_faces[5] = local_points[i, 2] + half_extents[2]  # -Z face
                
                # Minimum distance to any face (negative to indicate inside)
                distances[i] = -torch.min(distances_to_faces)
            else:
                # Standard distance calculation for outside points
                distances[i] = torch.norm(local_points[i] - closest_point_local[i])
        
        return distances


    def _reset_robot_when_stuck_at_table(self):
        """Reset robot to a safe position when it gets stuck at the table."""
        # Default safe poses (well above table)
        safe_poses = [
            [-0.71055204, -1.3046993, 1.9, -2.23, -1.59000665, 1.76992667],
            [-0.568, -0.658, 1.602, -2.585, -1.6060665, -1.64142667],  # Alternative safe pose
        ]
        
        # Get end-effector position
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_height = ee_position[:, 2]
        
        # Check which environments have robots stuck at table
        table_height = TABLE_HEIGHT
        safety_margin = 0.05
        stuck_at_table = ee_height < (table_height + safety_margin)
        
        # Get environment IDs that are stuck
        stuck_env_ids = torch.nonzero(stuck_at_table, as_tuple=False).squeeze(-1)
        
        if len(stuck_env_ids) == 0:
            return
        
        # Reset stuck robots to safe positions
        for env_id in stuck_env_ids:
            # Choose a random safe pose
            import random
            base_pose = random.choice(safe_poses)
            
            # Convert pose to tensor and add noise
            joint_pos_base = torch.tensor(base_pose, device=self.device, dtype=torch.float32)
            noise_range = 0.02
            noise = torch.rand(len(base_pose), device=self.device) * 2 * noise_range - noise_range
            joint_pos = joint_pos_base + noise
            joint_vel = torch.zeros_like(joint_pos)
            
            # Clamp to joint limits
            joint_pos = torch.clamp(
                joint_pos,
                self._robot_dof_lower_limits,
                self._robot_dof_upper_limits
            )
            
            # Convert env_id to tensor
            env_id_tensor = torch.tensor([env_id], device=self.device, dtype=torch.long)
            
            # Reset robot to safe position
            self._robot.set_joint_position_target(
                joint_pos.unsqueeze(0), 
                joint_ids=self._joint_indices, 
                env_ids=env_id_tensor
            )
            self._robot.write_joint_state_to_sim(
                joint_pos.unsqueeze(0), 
                joint_vel.unsqueeze(0), 
                joint_ids=self._joint_indices, 
                env_ids=env_id_tensor
            )
            
            # Update target to prevent immediate re-collision
            self._robot_dof_targets[env_id] = joint_pos
            
            # Log the reset
            if len(stuck_env_ids) <= 2:  # Avoid spam
                print(f"[INFO] Reset stuck robot in environment {env_id.item()}")

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return termination flags."""
        # Time limit truncation
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Pose tracking success termination
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
        
        # Check position and orientation errors
        position_error = torch.norm(ee_position - des_pos_w, p=2, dim=-1)
        orientation_error = math_utils.quat_error_magnitude(ee_quat, des_quat_w)
        
        # Check joint velocities for stability
        joint_velocities = torch.norm(self._robot.data.joint_vel, p=2, dim=-1)
        
        # Success criteria
        position_success = position_error < self.cfg.position_threshold
        orientation_success = orientation_error < self.cfg.orientation_threshold
        velocity_success = joint_velocities < self.cfg.velocity_threshold
        
        # Task success
        task_success = position_success & orientation_success & velocity_success
        
        
        # Early termination combines time out and bounds violation
        early_termination = time_out 
        return task_success, early_termination
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self._mark_episode_end()

        super()._reset_idx(env_ids)
        
        self._mark_episode_start()

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


    # -------------------------------------------------------------------------
    # 1) MARK EPISODE BOUNDARIES
    # -------------------------------------------------------------------------
    def _mark_episode_start(self):
        """Write a comment line to each CSV and bump episode counter."""
        self._episode_counter += 1
        step = self.common_step_counter

        # joint-target CSV
        if hasattr(self, "_joint_targets_file"):
            self._joint_targets_file.write(
                f"# --- EPISODE {self._episode_counter} START at step {step} ---\n"
            )
            self._joint_targets_file.flush()

        # state-observation CSV
        if self._state_obs_file:
            self._state_obs_file.write(
                f"# --- EPISODE {self._episode_counter} START at step {step} ---\n"
            )
            self._state_obs_file.flush()

    def _mark_episode_end(self):
        """Write a comment line to each CSV at episode end."""
        step = self.common_step_counter

        if hasattr(self, "_joint_targets_file"):
            self._joint_targets_file.write(
                f"# --- EPISODE {self._episode_counter}  END  at step {step} ---\n"
            )
            self._joint_targets_file.flush()

        if self._state_obs_file:
            self._state_obs_file.write(
                f"# --- EPISODE {self._episode_counter}  END  at step {step} ---\n"
            )
            self._state_obs_file.flush()

    # -------------------------------------------------------------------------
    # 2) SAVE STATE OBSERVATIONS
    # -------------------------------------------------------------------------
    def _save_state_observations(self):
        """Dump the current state vector for each env to a CSV."""
        # lazily open CSV
        if self._state_obs_file is None:
            os.makedirs('/home/adi2440/Desktop/state_data', exist_ok=True)
            fname = datetime.now().strftime("state_obs_%Y%m%d_%H%M%S.csv")
            path = os.path.join('/home/adi2440/Desktop/state_data', fname)
            self._state_obs_file = open(path, 'w', newline='')
            self._state_csv_writer = csv.writer(self._state_obs_file)
            # header: step, env_id, state_0 & state_N
            header = ['step', 'env_id'] + [f'state_{i}' for i in range(self.cfg.state_dim)]
            self._state_csv_writer.writerow(header)

        step = self.common_step_counter
        states = self._get_state_observations().cpu().numpy()  # (num_envs, state_dim)
        for env_id in range(self.num_envs):
            row = [step, env_id] + states[env_id].tolist()
            self._state_csv_writer.writerow(row)

        if step % 100 == 0:
            self._state_obs_file.flush()

    # -------------------------------------------------------------------------
    # 3) SAVE IMAGE OBSERVATIONS
    # -------------------------------------------------------------------------
    def _save_image_observations(self):
        """Save the processed camera observations (normalized, cropped, resized) as PNGs."""
        # Lazily create output directory
        if self._image_obs_dir is None:
            self._image_obs_dir = '/home/adi2440/Desktop/image_data'
            os.makedirs(self._image_obs_dir, exist_ok=True)

        step = self.common_step_counter
        # 1) Grab the processed tensor: shape (num_envs, C, H, W)
        processed = self._get_camera_observations().cpu().numpy()

        # 2) For each env, convert to H×W×C and shift back into [0,1] for saving
        for env_id in range(self.num_envs):
            proc = processed[env_id].transpose(1, 2, 0)      #  (H, W, C)
            vis  = np.clip(proc + 0.5, 0.0, 1.0)             # undo mean subtraction
            fname = f"ep{self._episode_counter:03d}_step{step:06d}_env{env_id}.png"
            path  = os.path.join(self._image_obs_dir, fname)
            plt.imsave(path, vis)

        if step % 10 == 0:
            print(f"[SAVE] Processed images saved to {self._image_obs_dir} at step {step}")




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
        
        # Calculate and update joint targets (moved outside the conditional block)
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

        # self._save_joint_targets()

        # # # new: save state & image
        # self._save_state_observations()
        # self._save_image_observations()
        
        # Additionally log APF beta values for first few environments (every 10 steps)
        if self.common_step_counter % 10 == 0:
            # Log for first 3 environments
            for i in range(min(3, self.num_envs)):
                print(f"[APF] Env {i}: dist={min_distances[i]:.3f}m, β={beta_values[i]:.3f}")

    def _save_joint_targets(self):
        """Save joint targets for all environments at current timestep."""
        
        # Initialize file and tracking variables if not already done
        if not hasattr(self, '_joint_targets_file'):
            # Create output directory
            os.makedirs('/home/adi2440/Desktop/joint_targets_data', exist_ok=True)
            
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._joint_targets_filename = f'/home/adi2440/Desktop/joint_targets_data/joint_targets_{timestamp}.csv'
            
            # Open file and write header
            self._joint_targets_file = open(self._joint_targets_filename, 'w', newline='')
            self._csv_writer = csv.writer(self._joint_targets_file)
            
            # Write header row
            num_joints = len(self._joint_indices)
            header = ['step', 'env_id'] + [f'joint_{i}_target' for i in range(num_joints)]
            self._csv_writer.writerow(header)
            
            print(f"Started saving joint targets to: {self._joint_targets_filename}")
        
        # Save joint targets for all environments
        current_step = self.common_step_counter
        joint_targets_cpu = self._robot_dof_targets.cpu().numpy()
        
        for env_id in range(self.num_envs):
            row = [current_step, env_id] + joint_targets_cpu[env_id].tolist()
            self._csv_writer.writerow(row)
        
        # Flush to ensure data is written (optional, for safety)
        if current_step % 100 == 0:  # Flush every 100 steps to balance performance and safety
            self._joint_targets_file.flush()

    def _cleanup_joint_targets_file(self):
        """Close the joint targets file when simulation ends."""
        if hasattr(self, '_joint_targets_file') and self._joint_targets_file:
            self._joint_targets_file.close()
            print(f"Joint targets data saved to: {self._joint_targets_filename}")

    def set_debug_vis(self, debug_vis: bool) -> None:
        """Set debug visualization mode."""
        self.cfg.debug_vis = debug_vis
        if hasattr(self, "_ee_frame") and self._ee_frame is not None:
            self._ee_frame.set_debug_vis(debug_vis)
        
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