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
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    TiledCamera,
    TiledCameraCfg,
    FrameTransformer,
    FrameTransformerCfg,
)
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import sample_uniform

# Visualization imports
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import json
from PIL import Image
import itertools


# Robot configuration
from .assets.ur5 import UR5_GRIPPER_CFG
from .rollout_logger import RolloutLogger

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
class AGANDataCollectionEnvCfg(DirectRLEnvCfg):
    """Configuration for the direct RL environment with Gray+Depth observations."""

    # Visualization settings - MOVED TO TOP to fix reference issue
    debug_vis = False  # Enable/disable debug visualization

    # AGAN Data Collection Switch
    save_agan_images = False  # Legacy PNG/JSONL export. HDF5 RGBD logging is preferred.
    agan_data_dir = "agan_dataset_run_2"
    agan_save_interval = 3  # Save every 3rd step (30Hz / 3 = 10Hz)

    # Synchronized RGBD/proprio/action rollout logging for latent dynamics training.
    save_rgbd_rollouts = False
    rgbd_rollout_log_path = "rgbd_dataset_watermark_1"  # Empty means create an HDF5 file inside agan_data_dir.
    rgbd_rollout_flush_interval = 100
    rgbd_rollout_stride = agan_save_interval
    rgbd_depth_max = 10.0
    max_rgbd_rollout_episodes = (
        30  # Counted across all parallel envs. Set <= 0 to disable auto-shutdown.
    )

    # Live replay attack settings. The policy receives replayed camera
    # observations after the trigger; HDF5 logging keeps both spoofed and live RGB-D.
    replay_attack_enabled = False
    replay_attack_trigger_step = (
        0  # Earliest per-episode step to search for a state-matched replay source.
    )
    replay_attack_delay_steps = 120
    replay_attack_warmup_steps = 0
    replay_attack_buffer_capacity = 4096
    replay_attack_match_threshold = 1.0
    replay_attack_joint_match_scale = 0.05
    replay_attack_arm_match_scale = 0.05
    replay_attack_print_start = True
    replay_attack_print_max_envs = 8

    # Watermark settings
    watermark_enabled = False
    watermark_covariance = 0.05  # Standard deviation of 0.01 matches the magnitude of joint_pos_noise bounds

    # Arm dimensions for BBox approximation (approximate dimensions in meters)
    arm_approx_dims = [0.2, 0.65, 0.1]  # Width, Length, Depth

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    # UR5 Robot
    robot_cfg: ArticulationCfg = UR5_GRIPPER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Table configuration
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/adi2440/Desktop/RL_UR5_IsaacLab/source/RL_UR5/RL_UR5/tasks/direct/rl_ur5/assets/table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, -0.0234), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # Arm configuration - dynamic object for collision avoidance
    arm_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/arm",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/adi2440/Desktop/RL_UR5_IsaacLab/source/RL_UR5/RL_UR5/tasks/direct/rl_ur5/assets/arm.usd",
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
            rot=(0.9, 0.0, 0.484, 0.0),
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
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.1, opacity=1.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.15, 0.0, 0.74), rot=(0.70711, 0.0, 0.70711, 0.0)
        ),
    )

    # I2R plane configuration with image texture
    i2r_plane_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/i2r_plane",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/adi2440/Desktop/RL_UR5_IsaacLab/source/RL_UR5/RL_UR5/tasks/direct/rl_ur5/assets/i2r_plane.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.16, -0.2, 0.93), rot=(0.50000, 0.50000, 0.50000, 0.50000)
        ),
    )

    # Clemson plane configuration with image texture
    clemson_plane_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/clemson_plane",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/adi2440/Desktop/RL_UR5_IsaacLab/source/RL_UR5/RL_UR5/tasks/direct/rl_ur5/assets/clemson_plane.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.16, 0.2, 0.93), rot=(0.50000, 0.50000, 0.50000, 0.50000)
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

    # Grayscale Camera
    tiled_camera_gray: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CameraGray",
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.82706,
            focus_distance=30.0,
            horizontal_aperture=5.229,
            vertical_aperture=2.942,
            clipping_range=(0.1, 1000.0),
        ),
        width=640,
        height=480,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.27, -0.06, 1.143),
            rot=(0.59637, 0.37993, 0.37993, 0.59637),
            convention="opengl",
        ),
    )

    # Depth Camera
    tiled_camera_depth: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CameraDepth",
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.82706,
            focus_distance=30.0,
            horizontal_aperture=5.229,
            vertical_aperture=2.942,
            clipping_range=(0.1, 1000.0),
        ),
        width=640,
        height=480,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.27, 0.0, 1.143),
            rot=(0.59637, 0.37993, 0.37993, 0.59637),
            convention="opengl",
        ),
    )

    # Basic environment settings
    episode_length_s = 6.0
    decimation = 4
    action_scale = 0.1  # Reduced for smoother movements
    state_dim = 13
    camera_target_height = 120
    camera_target_width = 160

    # Observation and action spaces
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
    state_space = 0
    # For PPO
    observation_space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(
                low=float("-inf"),
                high=float("inf"),
                shape=(camera_target_height, camera_target_width, 2),
            ),
            "state": gym.spaces.Box(
                low=float("-inf"), high=float("inf"), shape=(state_dim,)
            ),
        }
    )

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
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
    curriculum_enabled = False
    curriculum_steps = [
        5000,
        10000,
        20000,
        40000,
    ]  # Steps at which to increase difficulty
    curriculum_arm_speeds = [0.0, 0.1, 0.2, 0.3]  # Progressive arm movement speeds
    curriculum_target_ranges = [
        {"x": (0.55, 0.65), "y": (0.45, 0.5), "z": (-0.1, 0.1)},  # Easy
        {"x": (0.5, 0.7), "y": (0.4, 0.5), "z": (-0.15, 0.15)},  # Medium
        {"x": (0.5, 0.7), "y": (0.35, 0.55), "z": (-0.2, 0.2)},  # Hard
        {"x": (0.45, 0.75), "y": (0.3, 0.6), "z": (-0.2, 0.2)},  # Expert
    ]

    # Success tracking for adaptive curriculum
    success_window_size = 100
    curriculum_advance_threshold = 0.5  # Advance when success rate > 70%

    # Command/target pose settings
    target_pose_range = {
        "x": (0.6, 0.8),
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

    command_resampling_time = 16.0

    # Human arm movement settings
    arm_position_bounds = {
        "x": (0.9, 1.1),
        "y": (-0.5, 0.5),
        "z": (0.7, 1.0),
    }
    arm_movement_speed = 0.15  # Speed of random movement

    # Reward settings
    reward_distance_weight = -2.5
    reward_distance_tanh_weight = 1.5
    reward_distance_tanh_std = 0.1
    reward_orientation_weight = -1.0  # Increased to enforce downward orientation
    # reward_torque_weight removed
    reward_table_collision_weight = -4.0
    reward_arm_avoidance_weight = 5.0  # Changed from obstacle
    reward_action_rate_weight = -1.0  # Increased penalty for jagged movements

    # Artificial Potential Field parameters
    apf_critical_distance = 0.15  # db - critical distance for obstacle avoidance
    apf_smoothness = 0.1  # ko - smoothness parameter for beta transition
    energy_reward_weight = -1.0  # Weight for energy component

    # Huber loss parameters
    huber_delta = 0.08  # Delta parameter for Huber loss

    # Action filter settings - REMOVED
    # action_filter_order = 2
    # action_filter_cutoff_freq = 8.0
    # action_filter_damping_ratio = 0.707

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
    visualization_save_path = "./visualize_camera_images"  # Path to save visualizations

    # Noise settings
    joint_pos_noise_min = 0.0
    joint_pos_noise_max = 0.0
    joint_vel_noise_min = 0.0
    joint_vel_noise_max = 0.0

    # Reset settings
    robot_base_pose = [-0.568, -0.858, 1.402, -2.185, -1.6060665, 1.64142667]
    robot_reset_noise_range = 0.1


class AGANDataCollectionEnv(DirectRLEnv):
    """Direct RL environment for object camera pose tracking with multi-observation space (Gray+Depth)."""

    cfg: AGANDataCollectionEnvCfg

    def __init__(
        self,
        cfg: AGANDataCollectionEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        # Store config
        self.cfg = cfg
        self._rollout_logger = None
        self._rollout_episode_ids = None
        self._next_rollout_episode_id = 0
        self._replay_policy_image_buffer = None
        self._replay_rgbd_buffer = None
        self._replay_state_buffer = None
        self._replay_episode_id_buffer = None
        self._replay_step_id_buffer = None
        self._replay_env_id_buffer = None
        self._replay_buffer_write_idx = 0
        self._replay_buffer_count = 0
        self._replay_attack_was_active = None
        self._rgbd_rollout_completed_episodes = 0
        self._rgbd_rollout_completed_attacked_episodes = 0
        self._rgbd_rollout_waiting_for_attacks_printed = False
        self._rgbd_rollout_shutdown_requested = False

        # === episode / logging bookkeeping ===
        self._episode_counter = 0

        # file handles, writers, directories (initialized on first save)
        self._state_obs_file = None
        self._state_csv_writer = None
        self._image_obs_dir = None
        self.num_actions = 6
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
        self._robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[
            0, self._joint_indices, 0
        ].to(self.device)
        self._robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[
            0, self._joint_indices, 1
        ].to(self.device)

        # Initialize buffers
        self._robot_dof_targets = torch.zeros(
            (self.num_envs, len(self._joint_indices)), device=self.device
        )
        self._target_poses = torch.zeros((self.num_envs, 7), device=self.device)
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)
        self.raw_actions = torch.zeros_like(self._robot_dof_targets)
        self.actions = torch.zeros_like(self._robot_dof_targets)
        self.base_actions = torch.zeros_like(self._robot_dof_targets)
        self.sampled_watermarks = torch.zeros_like(self._robot_dof_targets)
        self.effective_action_deltas = torch.zeros_like(self._robot_dof_targets)
        self._rollout_episode_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.int64
        )
        self._next_rollout_episode_id = int(self.num_envs)
        self._latest_attack_active = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._latest_attack_trigger = torch.zeros_like(self._latest_attack_active)
        self._latest_attack_source_episode_index = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.int64
        )
        self._latest_attack_source_step_index = torch.full_like(
            self._latest_attack_source_episode_index, -1
        )
        self._latest_attack_source_env_id = torch.full_like(
            self._latest_attack_source_episode_index, -1
        )
        self._latest_attack_match_distance = torch.full(
            (self.num_envs,), float("inf"), device=self.device, dtype=torch.float32
        )
        self._replay_attack_was_active = torch.zeros_like(self._latest_attack_active)
        self._replay_attack_source_env_ids = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.int64
        )
        self._replay_attack_source_start_slots = torch.full_like(
            self._replay_attack_source_env_ids, -1
        )
        self._replay_attack_start_steps = torch.full_like(
            self._replay_attack_source_env_ids, -1
        )
        self._replay_attack_match_distances = torch.full(
            (self.num_envs,), float("inf"), device=self.device, dtype=torch.float32
        )
        self._init_replay_attack_buffers()

        # Arm movement state
        self._arm_target_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Initialize previous actions for smoothness penalty
        self.previous_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )

        # Curriculum learning state
        self._curriculum_level = 0
        self._success_buffer = torch.zeros(
            self.cfg.success_window_size, device=self.device
        )
        self._success_buffer_idx = 0

        # Apply initial curriculum settings
        self._update_curriculum_settings()

        # Performance tracking
        self._episode_sums = {
            "position_error": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
            "success_count": torch.zeros(self.num_envs, device=self.device),
            "min_arm_distance": torch.ones(self.num_envs, device=self.device)
            * float("inf"),
        }

        if self.cfg.save_rgbd_rollouts:
            rollout_path = self.cfg.rgbd_rollout_log_path or self.cfg.agan_data_dir
            rollout_run_prefix = "rgbd_proprio_actions"
            if self.cfg.replay_attack_enabled:
                rollout_run_prefix = f"{rollout_run_prefix}_replay_attacked"
                path_root, path_ext = os.path.splitext(str(rollout_path))
                if path_ext.lower() in {
                    ".h5",
                    ".hdf5",
                } and "replay_attacked" not in os.path.basename(path_root):
                    rollout_path = f"{path_root}_replay_attacked{path_ext}"
            self._rollout_logger = RolloutLogger(
                path=rollout_path,
                run_prefix=rollout_run_prefix,
                flush_interval=self.cfg.rgbd_rollout_flush_interval,
                metadata={
                    "task": "agan_rgbd_data_collection",
                    "num_envs": int(self.num_envs),
                    "state_dim": int(self.cfg.state_dim),
                    "action_dim": int(self.cfg.action_space.shape[0]),
                    "rgbd_channels": 4,
                    "proprio_dim": 19,
                    "camera_target_height": int(self.cfg.camera_target_height),
                    "camera_target_width": int(self.cfg.camera_target_width),
                    "rgbd_depth_units": "normalized_clipped_depth",
                    "rgbd_depth_max_m": float(self.cfg.rgbd_depth_max),
                    "max_rgbd_rollout_episodes": int(
                        self.cfg.max_rgbd_rollout_episodes
                    ),
                    "watermark_enabled": bool(self.cfg.watermark_enabled),
                    "watermark_covariance": float(self.cfg.watermark_covariance),
                    "replay_attack_enabled": bool(self.cfg.replay_attack_enabled),
                    "replay_attack_trigger_step": int(
                        self.cfg.replay_attack_trigger_step
                    ),
                    "replay_attack_delay_steps": int(
                        self.cfg.replay_attack_delay_steps
                    ),
                    "replay_attack_warmup_steps": int(
                        self.cfg.replay_attack_warmup_steps
                    ),
                    "replay_attack_buffer_capacity": int(
                        self.cfg.replay_attack_buffer_capacity
                    ),
                    "replay_attack_match_threshold": float(
                        self.cfg.replay_attack_match_threshold
                    ),
                    "replay_attack_joint_match_scale": float(
                        self.cfg.replay_attack_joint_match_scale
                    ),
                    "replay_attack_arm_match_scale": float(
                        self.cfg.replay_attack_arm_match_scale
                    ),
                    "replay_attack_print_start": bool(
                        self.cfg.replay_attack_print_start
                    ),
                },
            )
            print(f"[INFO] RGBD rollout logging enabled: {self._rollout_logger.path}")

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
        if self._rollout_logger is not None:
            self._rollout_logger.close()
            self._rollout_logger = None
        if self._state_obs_file is not None:
            self._state_obs_file.close()
            self._state_obs_file = None
        self._cleanup_joint_targets_file()
        super().close()

    def _setup_scene(self):
        """Set up the scene with robots, table, obstacles, cameras, etc."""
        # --- spawn all prims in the source environment only ---
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera_gray = TiledCamera(self.cfg.tiled_camera_gray)
        self._tiled_camera_depth = TiledCamera(self.cfg.tiled_camera_depth)

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
        self.scene.sensors["tiled_camera_gray"] = self._tiled_camera_gray
        self.scene.sensors["tiled_camera_depth"] = self._tiled_camera_depth
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
        dir_light_cfg = sim_utils.DistantLightCfg(
            intensity=1000.0, color=(1.0, 1.0, 0.9), angle=0.53
        )
        dir_light_cfg.func("/World/DirectionalLight", dir_light_cfg)

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

        print(
            f"[CURRICULUM] Level {level}: arm_speed={self.cfg.arm_movement_speed:.2f}, "
            f"target_x={self.cfg.target_pose_range['x']}, "
            f"target_y={self.cfg.target_pose_range['y']}"
        )

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
                    print(
                        f"[CURRICULUM] Advanced to level {self._curriculum_level} at step {self.common_step_counter}"
                    )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply actions before physics step."""
        # Update previous actions (before overwriting self.actions)
        if hasattr(self, "actions"):
            self.previous_actions = self.actions.clone()
        else:
            self.previous_actions = torch.zeros_like(actions)

        # Store raw actions
        self.raw_actions = actions.clone().clamp(-1.0, 1.0)

        # Action filter removed for direct control
        # filtered_actions = self._apply_action_filter(self.actions)

        # Scale actions
        self.base_actions = self.raw_actions * self.cfg.action_scale

        if self.cfg.watermark_enabled:
            # Sample additive Gaussian watermark
            std = self.cfg.watermark_covariance
            self.sampled_watermarks = torch.randn_like(self.base_actions) * std

            # Apply watermark to the scaled actions
            self.actions = self.base_actions + self.sampled_watermarks
            self.effective_action_deltas = self.actions - self.base_actions
        else:
            self.actions = self.base_actions.clone()
            self.sampled_watermarks.zero_()
            self.effective_action_deltas.zero_()

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

        # IF robot is stuck at the table, reset it
        self._reset_robot_when_stuck_at_table()

        # Update arm position
        self._update_arm_position()

        # Update debug visualization if enabled
        self._update_debug_visualization()

        # Save data for GAN training

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
            self._robot_dof_upper_limits - safety_margin,
        )

        # Apply velocity limits for safety
        max_velocity = 1.5  # rad/s
        velocity_command = (
            self._robot_dof_targets - current_joint_pos
        ) / self.physics_dt
        velocity_command = torch.clamp(velocity_command, -max_velocity, max_velocity)
        self._robot_dof_targets = current_joint_pos + velocity_command * self.physics_dt

        # Set joint position targets
        self._robot.set_joint_position_target(
            self._robot_dof_targets, joint_ids=self._joint_indices
        )

    def _sample_commands(self, env_ids: Sequence[int]) -> None:
        """Randomize the target poses for the given env indices."""
        num = len(env_ids)
        if num == 0:
            return

        # sample positions within configured ranges
        x = sample_uniform(
            self.cfg.target_pose_range["x"][0],
            self.cfg.target_pose_range["x"][1],
            (num,),
            self.device,
        )
        y = sample_uniform(
            self.cfg.target_pose_range["y"][0],
            self.cfg.target_pose_range["y"][1],
            (num,),
            self.device,
        )
        z = sample_uniform(
            self.cfg.target_pose_range["z"][0],
            self.cfg.target_pose_range["z"][1],
            (num,),
            self.device,
        )

        # sample orientations (roll, pitch, yaw)
        roll = sample_uniform(
            self.cfg.target_pose_range["roll"][0],
            self.cfg.target_pose_range["roll"][1],
            (num,),
            self.device,
        )
        pitch = sample_uniform(
            self.cfg.target_pose_range["pitch"][0],
            self.cfg.target_pose_range["pitch"][1],
            (num,),
            self.device,
        )
        yaw = sample_uniform(
            self.cfg.target_pose_range["yaw"][0],
            self.cfg.target_pose_range["yaw"][1],
            (num,),
            self.device,
        )
        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        # write into the buffer
        self._target_poses[env_ids, :3] = torch.stack([x, y, z], dim=-1)
        self._target_poses[env_ids, 3:7] = quat

        # Debug print for first environment
        if 0 in env_ids and len(env_ids) <= 4:  # Only log for small resets
            idx = env_ids.index(0)
            print(
                f"[DEBUG] New target for env 0: pos=[{x[idx].item():.3f}, {y[idx].item():.3f}, {z[idx].item():.3f}]"
            )

    def _update_arm_position(self):
        """Update the human arm position with Lissajous curve motion pattern for dense coverage."""
        # Initialize motion parameters if not exists
        if not hasattr(self, "_arm_motion_time"):
            self._arm_motion_time = torch.zeros(self.num_envs, device=self.device)
            # Add unique phase offsets per environment so they explore different parts of space
            self._phase_offsets_x = (
                torch.rand(self.num_envs, device=self.device) * 2 * math.pi
            )
            self._phase_offsets_y = (
                torch.rand(self.num_envs, device=self.device) * 2 * math.pi
            )
            self._phase_offsets_z = (
                torch.rand(self.num_envs, device=self.device) * 2 * math.pi
            )

            # Use non-commensurate frequencies for dense Lissajous coverage
            base_freq = self.cfg.arm_movement_speed
            self._freq_x = base_freq * 0.31
            self._freq_y = base_freq * 0.43
            self._freq_z = base_freq * 0.59

        # Get current arm positions and orientations
        arm_positions = self._arm.data.root_pos_w.clone()

        # Update motion time
        self._arm_motion_time += self.physics_dt

        # Calculate bounding box centers and amplitudes
        bounds_x = self.cfg.arm_position_bounds["x"]
        bounds_y = self.cfg.arm_position_bounds["y"]
        bounds_z = self.cfg.arm_position_bounds["z"]

        center_x = (bounds_x[0] + bounds_x[1]) / 2.0
        center_y = (bounds_y[0] + bounds_y[1]) / 2.0
        center_z = (bounds_z[0] + bounds_z[1]) / 2.0

        amp_x = (bounds_x[1] - bounds_x[0]) / 2.0
        amp_y = (bounds_y[1] - bounds_y[0]) / 2.0
        amp_z = (bounds_z[1] - bounds_z[0]) / 2.0

        # Calculate positions using sine waves for Lissajous curves
        t = self._arm_motion_time

        # Calculate local positions (before env origin offset)
        local_x = center_x + amp_x * torch.sin(
            2 * math.pi * self._freq_x * t + self._phase_offsets_x
        )
        local_y = center_y + amp_y * torch.sin(
            2 * math.pi * self._freq_y * t + self._phase_offsets_y
        )
        local_z = center_z + amp_z * torch.sin(
            2 * math.pi * self._freq_z * t + self._phase_offsets_z
        )

        local_pos = torch.stack([local_x, local_y, local_z], dim=-1)

        # Apply environment origin offsets
        arm_positions[:, :3] = local_pos + self.scene.env_origins[:, :3]

        # Apply new poses with fixed orientation quaternion (w, x, y, z)
        fixed_quat = torch.tensor([0.0, 0.99144, -0.0, -0.13053], device=self.device)
        fixed_quat = fixed_quat / torch.norm(fixed_quat)  # normalize
        arm_quats = fixed_quat.unsqueeze(0).expand(self.num_envs, -1)
        self._arm.write_root_pose_to_sim(torch.cat([arm_positions, arm_quats], dim=-1))

        # Calculate and set velocities (derivatives of the sine waves)
        if self.cfg.arm_movement_speed > 0:
            velocities = torch.zeros((self.num_envs, 6), device=self.device)
            # Velocity = Amplitude * 2*pi*Freq * cos(2*pi*Freq*t + phase)
            vel_x = (
                amp_x
                * 2
                * math.pi
                * self._freq_x
                * torch.cos(2 * math.pi * self._freq_x * t + self._phase_offsets_x)
            )
            vel_y = (
                amp_y
                * 2
                * math.pi
                * self._freq_y
                * torch.cos(2 * math.pi * self._freq_y * t + self._phase_offsets_y)
            )
            vel_z = (
                amp_z
                * 2
                * math.pi
                * self._freq_z
                * torch.cos(2 * math.pi * self._freq_z * t + self._phase_offsets_z)
            )

            velocities[:, :3] = torch.stack([vel_x, vel_y, vel_z], dim=-1)
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
        self._log_rgbd_rollout_batch(state_obs)

        obs = {"image": camera_obs, "state": state_obs}
        observations = {"policy": obs}

        return observations

    def _get_state_observations(self) -> torch.Tensor:
        """Get state-based observations."""
        # Get joint positions with noise
        joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        if self.cfg.joint_pos_noise_max > 0:
            joint_pos_noise = (
                torch.rand_like(joint_pos)
                * (self.cfg.joint_pos_noise_max - self.cfg.joint_pos_noise_min)
                + self.cfg.joint_pos_noise_min
            )
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
        state_obs = torch.cat(
            [
                joint_pos_noisy,  # 6 dims
                # joint_vel_noisy,      # 6 dims
                target_pose,  # 7 dims
            ],
            dim=-1,
        )

        return state_obs

    def _visualize_camera_observation(
        self, raw_image: torch.Tensor, processed_image: torch.Tensor, env_id: int = 0
    ):
        """Visualize raw and processed camera observations for debugging."""
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Get data for specified environment
        # raw_image is a dict/tensor depending on how we call it. Assuming it's the RGB one for now for vis.
        raw_env = raw_image[env_id].cpu().numpy()
        processed_env = processed_image[env_id].cpu().numpy()  # (2, H, W)

        # Raw image (Gray) - Displaying channel 0 of RGB input for simplicity
        axes[0].imshow(raw_env)
        axes[0].set_title(f"Raw RGB (Env {env_id})")
        axes[0].axis("off")

        # Add crop region visualization on raw image
        crop_rect = Rectangle(
            (0, self.cfg.camera_crop_top),
            224,
            224 - self.cfg.camera_crop_top - self.cfg.camera_crop_bottom,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        axes[0].add_patch(crop_rect)

        # Processed image (Combine channels for VIS or just show Gray)
        # Show Gray channel
        # processed_env is (2, H, W).
        axes[1].imshow(processed_env[0], cmap="gray")
        axes[1].set_title(
            f"Processed Gray\n({self.cfg.camera_target_height}x{self.cfg.camera_target_width})"
        )
        axes[1].axis("off")

        # Show Depth channel
        axes[2].imshow(processed_env[1], cmap="viridis")
        axes[2].set_title(f"Processed Depth")
        axes[2].axis("off")

        plt.tight_layout()

        # Save figure
        filename = f"{self.cfg.visualization_save_path}/camera_obs_step_{self.common_step_counter:06d}.png"
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close()

        if self._vis_counter % 10 == 0:  # Log every 10th visualization
            print(f"[VIS] Saved camera observation to: {filename}")

        self._vis_counter += 1

    def _huber_loss(self, x: torch.Tensor, delta: float) -> torch.Tensor:
        """Compute Huber loss for robust distance penalty."""
        abs_x = torch.abs(x)
        return torch.where(abs_x <= delta, 0.5 * x * x, delta * (abs_x - 0.5 * delta))

    def _compute_beta_transition(self, min_distances: torch.Tensor) -> torch.Tensor:
        """Compute smooth transition factor ² for adaptive reward mixing."""
        x = (min_distances - self.cfg.apf_critical_distance) / self.cfg.apf_smoothness
        beta = (torch.tanh(x) + 1.0) / 2.0
        return beta

    def _compute_energy_reward(self) -> torch.Tensor:
        """Compute energy-based reward from joint velocities."""
        joint_velocities = self._robot.data.joint_vel[:, self._joint_indices]
        # Compute norm squared for each joint
        velocity_norms_squared = joint_velocities**2
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
            ee_position, arm_position, arm_quat, arm_half_extents
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
        position_reward_tanh = 1.0 - torch.tanh(
            position_error / self.cfg.reward_distance_tanh_std
        )
        position_reward_tanh_scaled = (
            self.cfg.reward_distance_tanh_weight * position_reward_tanh
        )
        traditional_rewards += position_reward_tanh_scaled

        # 3. Orientation tracking reward with Huber loss
        orientation_error = math_utils.quat_error_magnitude(ee_quat, des_quat_w)
        orientation_huber_loss = self._huber_loss(
            orientation_error, self.cfg.huber_delta * 0.5
        )  # Smaller delta for orientation
        orientation_reward = self.cfg.reward_orientation_weight * orientation_huber_loss
        traditional_rewards += orientation_reward

        # 4. Joint torque penalty - Removed
        # torque_reward removed from calculation

        # 5. Table collision penalty
        ee_height = ee_position[:, 2]
        table_height = TABLE_HEIGHT
        safety_margin = 0.05

        table_penalty = torch.where(
            ee_height < (table_height + safety_margin),
            torch.ones_like(ee_height) * self.cfg.reward_table_collision_weight,
            torch.zeros_like(ee_height),
        )
        traditional_rewards += table_penalty

        # 6. Arm avoidance rewards (part of traditional rewards)
        arm_reward = (
            self._compute_arm_avoidance_rewards() * self.cfg.reward_arm_avoidance_weight
        )
        traditional_rewards += arm_reward

        # 7. Action Rate Penalty (Smoothness)
        # Penalize large changes in action between steps
        # Use simple difference norm
        if hasattr(self, "previous_actions"):
            # Use raw unscaled actions for penalty calculation to be scale-invariant relative to policy output
            # current_actions = self.actions / self.cfg.action_scale # Reconstruct or use stored?
            # Actually, self.actions IS scaled now. Let's compare scaled actions or unscaled?
            # Typically unscaled is better for policy smoothness, but scaled is better for physical smoothness.
            # Using scaled actions (actual command change)
            action_diff = self.actions - self.previous_actions
            action_rate_penalty = torch.sum(action_diff**2, dim=-1)
            traditional_rewards += (
                action_rate_penalty * self.cfg.reward_action_rate_weight
            )

        # 7 Success for reaching the end goal and avoiding the arm
        # Calculate minimum distance from end effector to arm cuboid
        # Arm dimensions (half-extents for easier calculation)
        arm_half_extents = torch.tensor(
            [0.25, 0.1, 0.06], device=self.device
        )  # [0.5, 0.2, 0.12] / 2
        arm_position = self._arm.data.root_pos_w[:, :3]
        arm_quat = self._arm.data.root_quat_w

        min_distances = self._point_to_box_distance(
            ee_position, arm_position, arm_quat, arm_half_extents
        )
        joint_velocities = torch.norm(self._robot.data.joint_vel, p=2, dim=-1)

        success_mask = (
            (position_error < 0.05)
            & (min_distances > 0.08)
            & (joint_velocities < self.cfg.velocity_threshold)
        )
        traditional_rewards += torch.where(success_mask, 5.0, 0.0)

        # === Energy-based Rewards (Renergy) ===
        energy_rewards = self._compute_energy_reward() * self.cfg.energy_reward_weight

        # === Adaptive Combination using APF ===
        # Rada = ² · Rt + (1  ²) · Renergy
        rewards = beta * traditional_rewards + (1.0 - beta) * energy_rewards

        # Track reward components for logging
        if hasattr(self, "_episode_sums"):
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
                self._success_buffer_idx = (
                    self._success_buffer_idx + 1
                ) % self.cfg.success_window_size

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
                "total_reward": rewards[0].item(),
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
        arm_half_extents = torch.tensor(
            [0.25, 0.1, 0.06], device=self.device
        )  # [0.5, 0.2, 0.12] / 2

        # Safety parameters - reduced to allow closer approach when needed
        critical_distance = 0.03  # Very close - high penalty
        danger_distance = 0.08  # Close - moderate penalty
        safe_distance = 0.12  # Reduced from 0.15 to allow more flexibility

        # Get arm pose and velocity
        arm_position = self._arm.data.root_pos_w[:, :3]
        arm_quat = self._arm.data.root_quat_w
        arm_velocity = (
            self._arm.data.root_lin_vel_w
            if hasattr(self._arm.data, "root_lin_vel_w")
            else torch.zeros_like(arm_position)
        )

        # Get end effector position and velocity
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_velocity = (
            self._ee_frame.data.target_lin_vel_w[..., 0, :]
            if hasattr(self._ee_frame.data, "target_lin_vel_w")
            else torch.zeros_like(ee_position)
        )

        # Calculate minimum distance from end effector to arm cuboid
        min_distances = self._point_to_box_distance(
            ee_position, arm_position, arm_quat, arm_half_extents
        )

        # Calculate relative velocity (positive means moving away from each other)
        relative_pos = ee_position - arm_position
        relative_vel = ee_velocity - arm_velocity
        relative_speed = torch.sum(relative_pos * relative_vel, dim=-1) / (
            torch.norm(relative_pos, dim=-1) + 1e-6
        )

        # Dynamic penalty based on both distance and relative motion
        for i in range(self.num_envs):
            distance = min_distances[i]

            if distance < critical_distance:
                # Very close - high penalty regardless of motion
                rewards[i] = -15.0
            elif distance < danger_distance:
                # In danger zone - penalty depends on relative motion
                base_penalty = -8.0 * (
                    1.0
                    - (distance - critical_distance)
                    / (danger_distance - critical_distance)
                )

                # Reduce penalty if moving away from arm
                if relative_speed[i] > 0:
                    motion_factor = torch.clamp(
                        relative_speed[i] / 0.5, 0.0, 0.7
                    )  # Max 70% reduction
                    rewards[i] = base_penalty * (1.0 - motion_factor)
                else:
                    # Increase penalty if moving toward arm
                    motion_factor = torch.clamp(
                        -relative_speed[i] / 0.5, 0.0, 0.5
                    )  # Max 50% increase
                    rewards[i] = base_penalty * (1.0 + motion_factor)
            elif distance < safe_distance:
                # In safe zone - small penalty that decreases with distance
                base_penalty = -2.0 * (
                    1.0
                    - (distance - danger_distance) / (safe_distance - danger_distance)
                )

                # Only apply penalty if moving toward arm
                if relative_speed[i] < 0:
                    rewards[i] = base_penalty
                else:
                    rewards[i] = 0.0  # No penalty if moving away
            else:
                # Outside safe distance - no penalty
                rewards[i] = 0.0

        return rewards

    def _point_to_box_distance(
        self,
        points: torch.Tensor,
        box_pos: torch.Tensor,
        box_quat: torch.Tensor,
        half_extents: torch.Tensor,
    ) -> torch.Tensor:
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
            local_points, -half_extents.unsqueeze(0), half_extents.unsqueeze(0)
        )

        # Check if point is inside the box
        inside_box = torch.all(
            torch.abs(local_points) <= half_extents.unsqueeze(0), dim=-1
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
            [
                -0.568,
                -0.658,
                1.602,
                -2.585,
                -1.6060665,
                -1.64142667,
            ],  # Alternative safe pose
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
            joint_pos_base = torch.tensor(
                base_pose, device=self.device, dtype=torch.float32
            )
            noise_range = 0.02
            noise = (
                torch.rand(len(base_pose), device=self.device) * 2 * noise_range
                - noise_range
            )
            joint_pos = joint_pos_base + noise
            joint_vel = torch.zeros_like(joint_pos)

            # Clamp to joint limits
            joint_pos = torch.clamp(
                joint_pos, self._robot_dof_lower_limits, self._robot_dof_upper_limits
            )

            # Convert env_id to tensor
            env_id_tensor = torch.tensor([env_id], device=self.device, dtype=torch.long)

            # Reset robot to safe position
            self._robot.set_joint_position_target(
                joint_pos.unsqueeze(0),
                joint_ids=self._joint_indices,
                env_ids=env_id_tensor,
            )
            self._robot.write_joint_state_to_sim(
                joint_pos.unsqueeze(0),
                joint_vel.unsqueeze(0),
                joint_ids=self._joint_indices,
                env_ids=env_id_tensor,
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
        if len(env_ids) > 0 and hasattr(self, "_episode_sums"):
            avg_position_error = (
                self._episode_sums["position_error"][env_ids].mean().item()
            )
            avg_reward = self._episode_sums["total_reward"][env_ids].mean().item()
            success_rate = self._episode_sums["success_count"][env_ids].mean().item()
            min_arm_dist = self._episode_sums["min_arm_distance"][env_ids].mean().item()

            if self.common_step_counter % 1000 == 0:  # Log every 1000 steps
                print(
                    f"[INFO] Episode stats - Pos error: {avg_position_error:.4f}, "
                    f"Reward: {avg_reward:.2f}, Success: {success_rate:.2f}, "
                    f"Min arm dist: {min_arm_dist:.3f}, "
                    f"Curriculum: L{self._curriculum_level}"
                )

        # Reset episode tracking
        if hasattr(self, "_episode_sums"):
            for key in self._episode_sums:
                self._episode_sums[key][env_ids] = 0.0

        # Reset robot joint positions
        num_resets = len(env_ids)

        # Base joint positions
        base_pose = torch.tensor(
            self.cfg.robot_base_pose, device=self.device, dtype=torch.float32
        )

        # Add noise
        joint_pos = base_pose.unsqueeze(0).repeat(num_resets, 1)
        if self.cfg.robot_reset_noise_range > 0:
            joint_pos += sample_uniform(
                -self.cfg.robot_reset_noise_range,
                self.cfg.robot_reset_noise_range,
                joint_pos.shape,
                self.device,
            )
        joint_vel = torch.zeros_like(joint_pos)

        completed_attacked_episodes = 0
        if (
            self.cfg.replay_attack_enabled
            and self._replay_attack_was_active is not None
        ):
            completed_attacked_episodes = int(
                torch.count_nonzero(self._replay_attack_was_active[env_ids]).item()
            )

        # Set joint state
        self._robot.write_joint_state_to_sim(
            joint_pos, joint_vel, joint_ids=self._joint_indices, env_ids=env_ids
        )
        self._robot_dof_targets[env_ids] = joint_pos
        self.raw_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.base_actions[env_ids] = 0.0
        self.sampled_watermarks[env_ids] = 0.0
        self.effective_action_deltas[env_ids] = 0.0
        if self._replay_attack_was_active is not None:
            self._replay_attack_was_active[env_ids] = False
        self._replay_attack_source_env_ids[env_ids] = -1
        self._replay_attack_source_start_slots[env_ids] = -1
        self._replay_attack_start_steps[env_ids] = -1
        self._replay_attack_match_distances[env_ids] = float("inf")

        # Reset arm position and orientation targets
        for i, env_id in enumerate(env_ids):
            # Random position within bounds
            self._arm_target_pos[env_id, 0] = (
                torch.rand(1, device=self.device)
                * (
                    self.cfg.arm_position_bounds["x"][1]
                    - self.cfg.arm_position_bounds["x"][0]
                )
                + self.cfg.arm_position_bounds["x"][0]
            )

            self._arm_target_pos[env_id, 1] = (
                torch.rand(1, device=self.device)
                * (
                    self.cfg.arm_position_bounds["y"][1]
                    - self.cfg.arm_position_bounds["y"][0]
                )
                + self.cfg.arm_position_bounds["y"][0]
            )

            self._arm_target_pos[env_id, 2] = (
                torch.rand(1, device=self.device)
                * (
                    self.cfg.arm_position_bounds["z"][1]
                    - self.cfg.arm_position_bounds["z"][0]
                )
                + self.cfg.arm_position_bounds["z"][0]
            )

        # Set initial arm pose
        new_positions = (
            self._arm_target_pos[env_ids] + self.scene.env_origins[env_ids, :3]
        )

        # Keep default orientation (identity quaternion)
        default_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
        new_quats = default_quat.unsqueeze(0).repeat(len(env_ids), 1)

        # Combine position and orientation
        new_poses = torch.cat([new_positions, new_quats], dim=-1)

        # Write to simulation
        self._arm.write_root_pose_to_sim(new_poses, env_ids=env_ids)
        self._arm.write_root_velocity_to_sim(
            torch.zeros((num_resets, 6), device=self.device), env_ids=env_ids
        )

        # Reset target poses
        self._sample_target_poses_for_reset(env_ids)

        # Reset previous actions for smoothness penalty
        if hasattr(self, "previous_actions"):
            self.previous_actions[env_ids] = 0.0

        # Reset timers
        self._command_time_left[env_ids] = self.cfg.command_resampling_time

        if self._rollout_episode_ids is not None:
            episode_ids = torch.arange(
                self._next_rollout_episode_id,
                self._next_rollout_episode_id + num_resets,
                device=self.device,
                dtype=torch.int64,
            )
            self._rollout_episode_ids[env_ids] = episode_ids
            self._next_rollout_episode_id += num_resets

        self._record_completed_rgbd_rollout_episodes(
            num_resets, completed_attacked_episodes
        )

    def _record_completed_rgbd_rollout_episodes(
        self, num_completed: int, num_attacked_completed: int
    ) -> None:
        """Stop collection after the configured number of completed episodes."""
        if not self.cfg.save_rgbd_rollouts or self._rollout_logger is None:
            return
        if self._rgbd_rollout_shutdown_requested:
            return

        max_episodes = int(self.cfg.max_rgbd_rollout_episodes)
        if max_episodes <= 0:
            return

        self._rgbd_rollout_completed_episodes += int(num_completed)
        self._rgbd_rollout_completed_attacked_episodes += int(num_attacked_completed)

        if self.cfg.replay_attack_enabled:
            completed_for_limit = self._rgbd_rollout_completed_attacked_episodes
            if (
                completed_for_limit == 0
                and self._rgbd_rollout_completed_episodes >= max_episodes
                and not self._rgbd_rollout_waiting_for_attacks_printed
            ):
                self._rgbd_rollout_waiting_for_attacks_printed = True
                print(
                    "[INFO] RGBD replay collection has completed "
                    f"{self._rgbd_rollout_completed_episodes} source episodes, "
                    "but no replay-attacked episode has finished yet. "
                    "Continuing until matched replay attacks are collected."
                )
        else:
            completed_for_limit = self._rgbd_rollout_completed_episodes

        if completed_for_limit < max_episodes:
            return

        self._request_rgbd_rollout_shutdown(max_episodes, completed_for_limit)

    def _request_rgbd_rollout_shutdown(
        self, max_episodes: int, completed_for_limit: int
    ) -> None:
        """Flush rollout data and request a clean Isaac Lab app shutdown."""
        self._rgbd_rollout_shutdown_requested = True
        episode_kind = (
            "replay-attacked episodes" if self.cfg.replay_attack_enabled else "episodes"
        )
        print(
            "[INFO] RGBD rollout episode limit reached: "
            f"{int(completed_for_limit)}/{int(max_episodes)} completed "
            f"{episode_kind} across all parallel environments "
            f"({self._rgbd_rollout_completed_episodes} total episodes). "
            "Flushing rollout data and shutting down Isaac Lab."
        )

        if self._rollout_logger is not None:
            self._rollout_logger.close()
            self._rollout_logger = None

        try:
            app = None
            import omni.kit.app

            app = omni.kit.app.get_app()
            if app is not None:
                app.post_quit()
        except Exception as exc:
            print(f"[WARN] Could not request Isaac Kit app quit: {exc}")

        sim = getattr(self, "sim", None)
        if sim is not None and hasattr(sim, "stop"):
            try:
                sim.stop()
            except Exception as exc:
                print(f"[WARN] Could not stop simulation context: {exc}")

        raise SystemExit(0)

    def _sample_target_poses_for_reset(self, env_ids: Sequence[int]):
        """Sample new target poses for reset environments."""
        num_resets = len(env_ids)

        # Sample target poses
        x = (
            torch.rand(num_resets, device=self.device)
            * (self.cfg.target_pose_range["x"][1] - self.cfg.target_pose_range["x"][0])
            + self.cfg.target_pose_range["x"][0]
        )

        y = (
            torch.rand(num_resets, device=self.device)
            * (self.cfg.target_pose_range["y"][1] - self.cfg.target_pose_range["y"][0])
            + self.cfg.target_pose_range["y"][0]
        )

        z = (
            torch.rand(num_resets, device=self.device)
            * (self.cfg.target_pose_range["z"][1] - self.cfg.target_pose_range["z"][0])
            + self.cfg.target_pose_range["z"][0]
        )

        # Fixed orientation for now
        roll = torch.full(
            (num_resets,), self.cfg.target_pose_range["roll"][0], device=self.device
        )
        pitch = torch.full(
            (num_resets,), self.cfg.target_pose_range["pitch"][0], device=self.device
        )
        yaw = torch.full(
            (num_resets,), self.cfg.target_pose_range["yaw"][0], device=self.device
        )

        # Convert euler to quaternion
        target_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        # Update buffers
        self._target_poses[env_ids, :3] = torch.stack([x, y, z], dim=-1)
        self._target_poses[env_ids, 3:7] = target_quat

    def _init_replay_attack_buffers(self) -> None:
        """Allocate replay source buffers when live replay attacks are enabled."""
        if not self.cfg.replay_attack_enabled:
            return

        capacity = int(self.cfg.replay_attack_buffer_capacity)
        if capacity <= 0:
            raise ValueError("replay_attack_buffer_capacity must be positive")

        height = int(self.cfg.camera_target_height)
        width = int(self.cfg.camera_target_width)
        self._replay_policy_image_buffer = torch.zeros(
            (self.num_envs, capacity, 2, height, width),
            device=self.device,
            dtype=torch.float32,
        )
        self._replay_rgbd_buffer = torch.zeros(
            (self.num_envs, capacity, 4, height, width),
            device=self.device,
            dtype=torch.float32,
        )
        self._replay_state_buffer = torch.zeros(
            (self.num_envs, capacity, 9), device=self.device, dtype=torch.float32
        )
        self._replay_episode_id_buffer = torch.full(
            (self.num_envs, capacity), -1, device=self.device, dtype=torch.int64
        )
        self._replay_step_id_buffer = torch.full_like(
            self._replay_episode_id_buffer, -1
        )
        self._replay_env_id_buffer = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
            .unsqueeze(1)
            .repeat(1, capacity)
        )
        self._replay_buffer_write_idx = 0
        self._replay_buffer_count = 0

    def _get_replay_match_states(self) -> torch.Tensor:
        """Return robot joint positions plus env-relative human arm position."""
        joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        arm_pos = self._arm.data.root_pos_w[:, :3] - self.scene.env_origins[:, :3]
        return torch.cat([joint_pos, arm_pos], dim=-1).detach()

    def _apply_replay_attack(
        self, policy_images: torch.Tensor, rgbd_images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy/RGB-D observations after optional live replay spoofing."""
        self._latest_true_rgbd_obs = rgbd_images.detach()

        self._latest_attack_active.zero_()
        self._latest_attack_trigger.zero_()
        self._latest_attack_source_episode_index.fill_(-1)
        self._latest_attack_source_step_index.fill_(-1)
        self._latest_attack_source_env_id.fill_(-1)
        self._latest_attack_match_distance.fill_(float("inf"))

        if not self.cfg.replay_attack_enabled:
            self._latest_rgbd_obs = rgbd_images.detach()
            return policy_images, rgbd_images

        if self._replay_policy_image_buffer is None:
            self._init_replay_attack_buffers()

        capacity = int(self.cfg.replay_attack_buffer_capacity)
        current_states = self._get_replay_match_states()
        can_replay = (
            self._replay_buffer_count >= capacity
            and self.common_step_counter >= int(self.cfg.replay_attack_warmup_steps)
        )
        local_step_ids = self.episode_length_buf.to(torch.int64)
        eligible_envs = (
            local_step_ids >= int(self.cfg.replay_attack_trigger_step)
        ) & bool(can_replay)

        newly_matched_env_ids = torch.nonzero(
            eligible_envs & ~self._replay_attack_was_active, as_tuple=False
        ).squeeze(-1)
        if newly_matched_env_ids.numel() > 0:
            self._select_replay_attack_sources(
                newly_matched_env_ids, current_states[newly_matched_env_ids]
            )

        has_source = self._replay_attack_source_env_ids >= 0
        attack_active = eligible_envs & has_source

        attacked_policy_images = policy_images
        attacked_rgbd_images = rgbd_images
        if torch.any(attack_active):
            active_env_ids = torch.nonzero(attack_active, as_tuple=False).squeeze(-1)
            source_env_ids = self._replay_attack_source_env_ids[active_env_ids]
            attack_offsets = (
                local_step_ids[active_env_ids]
                - self._replay_attack_start_steps[active_env_ids]
            )
            source_slots = (
                self._replay_attack_source_start_slots[active_env_ids] + attack_offsets
            ) % capacity

            attacked_policy_images = policy_images.clone()
            attacked_rgbd_images = rgbd_images.clone()
            attacked_policy_images[active_env_ids] = self._replay_policy_image_buffer[
                source_env_ids, source_slots
            ]
            attacked_rgbd_images[active_env_ids] = self._replay_rgbd_buffer[
                source_env_ids, source_slots
            ]
            self._latest_attack_source_episode_index[active_env_ids] = (
                self._replay_episode_id_buffer[source_env_ids, source_slots]
            )
            self._latest_attack_source_step_index[active_env_ids] = (
                self._replay_step_id_buffer[source_env_ids, source_slots]
            )
            self._latest_attack_source_env_id[active_env_ids] = (
                self._replay_env_id_buffer[source_env_ids, source_slots]
            )
            self._latest_attack_match_distance[active_env_ids] = (
                self._replay_attack_match_distances[active_env_ids]
            )

        self._latest_attack_active = attack_active
        self._latest_attack_trigger = attack_active & ~self._replay_attack_was_active
        self._replay_attack_was_active = attack_active.clone()
        self._latest_rgbd_obs = attacked_rgbd_images.detach()
        if self.cfg.replay_attack_print_start and torch.any(
            self._latest_attack_trigger
        ):
            self._print_replay_attack_start_indicator()

        self._append_replay_attack_buffer(policy_images, rgbd_images)
        return attacked_policy_images, attacked_rgbd_images

    def _select_replay_attack_sources(
        self, env_ids: torch.Tensor, current_states: torch.Tensor
    ) -> None:
        """Choose closest buffered source frames across all parallel environments."""
        if self._replay_state_buffer is None or env_ids.numel() == 0:
            return

        source_states = self._replay_state_buffer.reshape(-1, 9)
        source_episode_ids = self._replay_episode_id_buffer.reshape(-1)
        source_step_ids = self._replay_step_id_buffer.reshape(-1)
        source_env_ids = self._replay_env_id_buffer.reshape(-1)
        source_slots = (
            torch.arange(
                int(self.cfg.replay_attack_buffer_capacity),
                device=self.device,
                dtype=torch.int64,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
            .reshape(-1)
        )

        scales = torch.tensor(
            [float(self.cfg.replay_attack_joint_match_scale)] * 6
            + [float(self.cfg.replay_attack_arm_match_scale)] * 3,
            device=self.device,
            dtype=torch.float32,
        ).clamp_min(1e-6)
        threshold = float(self.cfg.replay_attack_match_threshold)

        for row, env_id in enumerate(env_ids):
            current_state = current_states[row]
            deltas = (source_states - current_state.unsqueeze(0)) / scales
            distances = torch.linalg.norm(deltas, dim=-1) / math.sqrt(
                float(source_states.shape[-1])
            )

            # Avoid starting from the same episode, which can produce a trivial
            # replay of the current run instead of a stored source trajectory.
            current_episode_id = self._rollout_episode_ids[env_id]
            distances = torch.where(
                source_episode_ids == current_episode_id,
                torch.full_like(distances, float("inf")),
                distances,
            )
            remaining_episode_steps = int(
                max(
                    1,
                    int(self.max_episode_length)
                    - int(self.episode_length_buf[env_id].item()),
                )
            )
            available_future_steps = (
                int(self._replay_buffer_write_idx) - source_slots - 1
            ) % int(self.cfg.replay_attack_buffer_capacity)
            has_future_buffer = available_future_steps >= (remaining_episode_steps - 1)
            has_future_episode = (source_step_ids + remaining_episode_steps) <= int(
                self.max_episode_length
            )
            distances = torch.where(
                has_future_buffer & has_future_episode,
                distances,
                torch.full_like(distances, float("inf")),
            )

            best_distance, best_index = torch.min(distances, dim=0)
            if not torch.isfinite(best_distance) or best_distance > threshold:
                continue

            self._replay_attack_source_env_ids[env_id] = source_env_ids[best_index]
            self._replay_attack_source_start_slots[env_id] = source_slots[best_index]
            self._replay_attack_start_steps[env_id] = self.episode_length_buf[env_id]
            self._replay_attack_match_distances[env_id] = best_distance

    def _print_replay_attack_start_indicator(self) -> None:
        """Print one terminal line for each env whose replay attack just started."""
        trigger_env_ids = torch.nonzero(
            self._latest_attack_trigger, as_tuple=False
        ).squeeze(-1)
        max_envs = max(1, int(self.cfg.replay_attack_print_max_envs))
        shown_env_ids = trigger_env_ids[:max_envs]

        env_ids = shown_env_ids.detach().cpu().tolist()
        episode_ids = self._rollout_episode_ids[shown_env_ids].detach().cpu().tolist()
        episode_steps = self.episode_length_buf[shown_env_ids].detach().cpu().tolist()
        source_env_ids = (
            self._latest_attack_source_env_id[shown_env_ids].detach().cpu().tolist()
        )
        source_episode_ids = (
            self._latest_attack_source_episode_index[shown_env_ids]
            .detach()
            .cpu()
            .tolist()
        )
        source_step_ids = (
            self._latest_attack_source_step_index[shown_env_ids].detach().cpu().tolist()
        )
        match_distances = (
            self._latest_attack_match_distance[shown_env_ids].detach().cpu().tolist()
        )

        for (
            env_id,
            episode_id,
            episode_step,
            source_env_id,
            source_episode_id,
            source_step_id,
            match_distance,
        ) in zip(
            env_ids,
            episode_ids,
            episode_steps,
            source_env_ids,
            source_episode_ids,
            source_step_ids,
            match_distances,
        ):
            print(
                "[REPLAY ATTACK START] "
                f"global_step={int(self.common_step_counter)} | "
                f"env={int(env_id)} | "
                f"episode_id={int(episode_id)} | "
                f"episode_step={int(episode_step)} | "
                f"source_env={int(source_env_id)} | "
                f"source_episode_id={int(source_episode_id)} | "
                f"source_step={int(source_step_id)} | "
                f"match_distance={float(match_distance):.4f}"
            )

        omitted = int(trigger_env_ids.numel()) - len(env_ids)
        if omitted > 0:
            print(
                "[REPLAY ATTACK START] "
                f"{omitted} additional envs triggered at "
                f"global_step={int(self.common_step_counter)}"
            )

    def _append_replay_attack_buffer(
        self, policy_images: torch.Tensor, rgbd_images: torch.Tensor
    ) -> None:
        """Store live camera observations as future replay-attack sources."""
        if self._replay_policy_image_buffer is None:
            return

        slot = self._replay_buffer_write_idx
        self._replay_policy_image_buffer[:, slot] = policy_images.detach()
        self._replay_rgbd_buffer[:, slot] = rgbd_images.detach()
        self._replay_state_buffer[:, slot] = self._get_replay_match_states()
        self._replay_episode_id_buffer[:, slot] = self._rollout_episode_ids
        self._replay_step_id_buffer[:, slot] = self.episode_length_buf.to(torch.int64)
        self._replay_buffer_write_idx = (slot + 1) % int(
            self.cfg.replay_attack_buffer_capacity
        )
        self._replay_buffer_count = min(
            self._replay_buffer_count + 1, int(self.cfg.replay_attack_buffer_capacity)
        )

    def _log_rgbd_rollout_batch(self, state_obs: torch.Tensor) -> None:
        """Append synchronized image, proprioception, and action samples to HDF5."""
        if self._rollout_logger is None:
            return
        if self.common_step_counter % max(1, int(self.cfg.rgbd_rollout_stride)) != 0:
            return
        if not hasattr(self, "_latest_rgbd_obs"):
            return

        joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        joint_vel = self._robot.data.joint_vel[:, self._joint_indices]
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        ee_pos_w = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat_w = self._ee_frame.data.target_quat_w[..., 0, :]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            robot_pos, robot_quat, ee_pos_w, ee_quat_w
        )
        ee_pose = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
        proprio_obs = torch.cat([joint_pos, joint_vel, ee_pose], dim=-1)
        arm_pose = torch.cat(
            [self._arm.data.root_pos_w[:, :3], self._arm.data.root_quat_w], dim=-1
        )

        rollout_batch = dict(
            rgbd_images=self._latest_rgbd_obs.detach().cpu().numpy().astype(np.float32),
            states=state_obs.detach().cpu().numpy().astype(np.float32),
            proprio_observations=proprio_obs.detach().cpu().numpy().astype(np.float32),
            raw_actions=self.raw_actions.detach().cpu().numpy().astype(np.float32),
            base_actions=self.base_actions.detach().cpu().numpy().astype(np.float32),
            scaled_actions=self.actions.detach().cpu().numpy().astype(np.float32),
            sampled_watermarks=self.sampled_watermarks.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            effective_action_deltas=self.effective_action_deltas.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            joint_positions=joint_pos.detach().cpu().numpy().astype(np.float32),
            joint_velocities=joint_vel.detach().cpu().numpy().astype(np.float32),
            joint_targets=self._robot_dof_targets.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            ee_poses=ee_pose.detach().cpu().numpy().astype(np.float32),
            target_poses=self._target_poses.detach().cpu().numpy().astype(np.float32),
            arm_poses=arm_pose.detach().cpu().numpy().astype(np.float32),
            episode_ids=self._rollout_episode_ids.detach()
            .cpu()
            .numpy()
            .astype(np.int64),
            step_ids=self.episode_length_buf.detach().cpu().numpy().astype(np.int64),
            global_step_ids=np.full(
                self.num_envs, self.common_step_counter, dtype=np.int64
            ),
            env_ids=np.arange(self.num_envs, dtype=np.int64),
        )
        if self.cfg.replay_attack_enabled:
            rollout_batch.update(
                true_rgbd_images=self._latest_true_rgbd_obs.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                attack_active=self._latest_attack_active.detach()
                .cpu()
                .numpy()
                .astype(np.bool_),
                attack_trigger=self._latest_attack_trigger.detach()
                .cpu()
                .numpy()
                .astype(np.bool_),
                attack_source_episode_index=self._latest_attack_source_episode_index.detach()
                .cpu()
                .numpy()
                .astype(np.int64),
                attack_source_step_index=self._latest_attack_source_step_index.detach()
                .cpu()
                .numpy()
                .astype(np.int64),
                attack_source_env_id=self._latest_attack_source_env_id.detach()
                .cpu()
                .numpy()
                .astype(np.int64),
                attack_match_distance=self._latest_attack_match_distance.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
            )
        self._rollout_logger.append_batch(**rollout_batch)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # Create markers for visualizing the goal poses
        if debug_vis:
            if not hasattr(self, "target_pos_visualizer"):
                # Create a separate marker config for target visualization with different color
                target_marker_cfg = FRAME_MARKER_CFG.copy()
                target_marker_cfg.markers["frame"].scale = (
                    0.05,
                    0.05,
                    0.05,
                )  # Slightly larger
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
            os.makedirs("./state_data", exist_ok=True)
            fname = datetime.now().strftime("state_obs_%Y%m%d_%H%M%S.csv")
            path = os.path.join("./state_data", fname)
            self._state_obs_file = open(path, "w", newline="")
            self._state_csv_writer = csv.writer(self._state_obs_file)
            # header: step, env_id, state_0 & state_N
            header = ["step", "env_id"] + [
                f"state_{i}" for i in range(self.cfg.state_dim)
            ]
            self._state_csv_writer.writerow(header)

        step = self.common_step_counter
        states = self._get_state_observations().cpu().numpy()  # (num_envs, state_dim)
        for env_id in range(self.num_envs):
            row = [step, env_id] + states[env_id].tolist()
            self._state_csv_writer.writerow(row)

        if step % 100 == 0:
            self._state_obs_file.flush()

    # -------------------------------------------------------------------------
    # 3) SAVE AGAN DATA (IMAGES + METADATA)
    # -------------------------------------------------------------------------
    def _world_to_screen(
        self, points_3d, camera_pos, camera_quat, intrinsics, width, height
    ):
        """
        Project 3D points to 2D screen coordinates.
        points_3d: (N, 3) tensor
        camera_pos: (3,) tensor
        camera_quat: (4,) tensor (w, x, y, z)
        intrinsics: (3, 3) tensor
        """
        # Transform to camera frame
        # R_cw = R_wc^T
        rot_mat = math_utils.matrix_from_quat(camera_quat)
        points_cam = torch.matmul(points_3d - camera_pos, rot_mat.T)

        # Convert OpenGL (Isaac) to OpenCV (Pinhole) convention
        # OpenGL: -Z forward, +Y up
        # OpenCV: +Z forward, +Y down
        # x_cv = x_gl, y_cv = -y_gl, z_cv = -z_gl
        points_cam = points_cam * torch.tensor(
            [1.0, -1.0, -1.0], device=points_cam.device
        )

        # Project to image plane: p_pix = K * (P_cam / P_cam_z)
        # Handle division by zero
        depths = points_cam[:, 2].clone()
        depths[depths < 1e-5] = 1e-5

        points_2d_homo = torch.matmul(points_cam, intrinsics.T)
        u = points_2d_homo[:, 0] / depths
        v = points_2d_homo[:, 1] / depths

        return torch.stack([u, v], dim=1)

    def _get_camera_observations(self) -> torch.Tensor:
        """Get and preprocess camera observations."""
        # 1. Get Grayscale Data (from RGB)
        rgb_data = (
            self._tiled_camera_gray.data.output["rgb"][..., :3] / 255.0
        )  # (N, H, W, 3)
        # Convert to grayscale by averaging channels
        gray_data = torch.mean(rgb_data, dim=-1, keepdim=True)  # (N, H, W, 1)

        # 2. Get Depth Data
        depth_data = self._tiled_camera_depth.data.output[
            "distance_to_image_plane"
        ]  # (N, H, W, 1)

        # --- Fix for depth stability ---
        # Replace infinity/nan with max range (e.g. 10.0m)
        max_depth = float(self.cfg.rgbd_depth_max)
        depth_data = torch.nan_to_num(
            depth_data, nan=max_depth, posinf=max_depth, neginf=0.0
        )
        depth_data = torch.clamp(depth_data, 0.0, max_depth)

        # Normalize depth to [0, 1] range roughly to match grayscale intensity distribution
        depth_data = depth_data / max_depth

        # 3. Concatenate
        combined_data = torch.cat([gray_data, depth_data], dim=-1)  # (N, H, W, 2)
        rgbd_data = torch.cat([rgb_data, depth_data], dim=-1)  # (N, H, W, 4)

        # Store raw RGB for visualization (optional)
        raw_camera_data = rgb_data.clone()

        # 4. Mean subtraction (Center the data)
        mean_tensor = torch.mean(combined_data, dim=(1, 2), keepdim=True)
        combined_data = combined_data - mean_tensor

        # 5. Crop image (top and bottom)
        cropped = combined_data[
            :, self.cfg.camera_crop_top : -self.cfg.camera_crop_bottom, :, :
        ]

        # 6. Resize
        # Convert to NCHW for interpolation
        cropped = cropped.permute(0, 3, 1, 2)  # (N, 2, H, W)

        # Resize using torch interpolation
        resized = torch.nn.functional.interpolate(
            cropped,
            size=(self.cfg.camera_target_height, self.cfg.camera_target_width),
            mode="bilinear",
            align_corners=False,
        )

        rgbd_cropped = rgbd_data[
            :, self.cfg.camera_crop_top : -self.cfg.camera_crop_bottom, :, :
        ].permute(0, 3, 1, 2)
        rgbd_resized = torch.nn.functional.interpolate(
            rgbd_cropped,
            size=(self.cfg.camera_target_height, self.cfg.camera_target_width),
            mode="bilinear",
            align_corners=False,
        )

        resized, _ = self._apply_replay_attack(resized, rgbd_resized)

        # Visualize camera observation periodically
        if self.common_step_counter % self.cfg.visualize_camera_interval == 0:
            self._visualize_camera_observation(raw_camera_data, resized, env_id=0)

        # --- AGAN Data Saving (Integrated) ---
        if self.cfg.save_agan_images and (
            self.common_step_counter % self.cfg.agan_save_interval == 0
        ):
            # Ensure directory exists
            if self._image_obs_dir is None:
                self._image_obs_dir = os.path.join(
                    self.cfg.agan_data_dir, f"run_{self._episode_counter}"
                )
                os.makedirs(self._image_obs_dir, exist_ok=True)

            step = self.common_step_counter
            processed_np = resized.cpu().numpy()
            metadata = {}

            # Arm BBox Prep
            half_dims = torch.tensor(self.cfg.arm_approx_dims, device=self.device) * 0.5
            corners_local = (
                torch.tensor(
                    list(itertools.product([-1, 1], repeat=3)), device=self.device
                )
                * half_dims
            )

            # Original dimensions for BBox projection (before crop/resize)
            orig_width = self.cfg.tiled_camera_gray.width
            orig_height = self.cfg.tiled_camera_gray.height

            for env_id in range(self.num_envs):
                # Save Images
                # Env data: (C, H, W) -> (H, W, C)
                env_data = processed_np[env_id].transpose(1, 2, 0)

                # Undo mean subtraction (add 0.5) and clip
                vis_data = np.clip(env_data + 0.5, 0.0, 1.0)

                gray_img = (vis_data[..., 0] * 255).astype(np.uint8)
                depth_img = (vis_data[..., 1] * 255).astype(np.uint8)

                # File names
                prefix = f"ep{self._episode_counter:03d}_step{step:06d}_env{env_id}"
                gray_path = os.path.join(self._image_obs_dir, f"{prefix}_gray.png")
                depth_path = os.path.join(self._image_obs_dir, f"{prefix}_depth.png")

                Image.fromarray(gray_img).save(gray_path)
                Image.fromarray(depth_img).save(depth_path)

                # Calculate BBox
                # 3D -> 2D (Original)
                arm_pos = self._arm.data.root_pos_w[env_id]
                arm_quat = self._arm.data.root_quat_w[env_id]
                rot_mat_arm = math_utils.matrix_from_quat(arm_quat)
                corners_world = torch.matmul(corners_local, rot_mat_arm.T) + arm_pos

                cam_pos = self._tiled_camera_gray.data.pos_w[env_id]
                cam_quat = self._tiled_camera_gray.data.quat_w_world[env_id]
                intrinsics = self._tiled_camera_gray.data.intrinsic_matrices[env_id]

                pts_2d = self._world_to_screen(
                    corners_world,
                    cam_pos,
                    cam_quat,
                    intrinsics,
                    orig_width,
                    orig_height,
                )

                # Adjust for Crop and Resize
                pts_2d[:, 1] -= self.cfg.camera_crop_top

                crop_h = (
                    orig_height - self.cfg.camera_crop_top - self.cfg.camera_crop_bottom
                )
                crop_w = orig_width

                scale_y = self.cfg.camera_target_height / crop_h
                scale_x = self.cfg.camera_target_width / crop_w

                pts_2d[:, 0] *= scale_x
                pts_2d[:, 1] *= scale_y

                # Bounds
                u_min = pts_2d[:, 0].min().item()
                u_max = pts_2d[:, 0].max().item()
                v_min = pts_2d[:, 1].min().item()
                v_max = pts_2d[:, 1].max().item()

                # Clamp
                u_min = max(0, min(self.cfg.camera_target_width, u_min))
                u_max = max(0, min(self.cfg.camera_target_width, u_max))
                v_min = max(0, min(self.cfg.camera_target_height, v_min))
                v_max = max(0, min(self.cfg.camera_target_height, v_max))

                is_valid = (
                    (u_max > u_min)
                    and (v_max > v_min)
                    and (u_max - u_min > 2)
                    and (v_max - v_min > 2)
                )

                # Metadata
                meta_key = os.path.abspath(gray_path)
                metadata[meta_key] = {
                    "env": env_id,
                    "step": step,
                    "bbox": [u_min, v_min, u_max, v_max],
                    "success": is_valid,
                    "arm_pos_3d": arm_pos.cpu().tolist(),
                    "episode": self._episode_counter,
                }

            if step % 10 == 0:
                print(
                    f"[SAVE] Saved AGAN data (imgs+meta) to {self._image_obs_dir} at step {step}"
                )

            # Append metadata
            jsonl_path = os.path.join(self.cfg.agan_data_dir, "metadata.jsonl")
            with open(jsonl_path, "a") as f:
                for k, v in metadata.items():
                    v["image_path"] = k
                    f.write(json.dumps(v) + "\n")

        return resized

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
        self.target_pos_visualizer.visualize(
            translations=des_pos_w, orientations=des_quat_w
        )

        # Calculate metrics for logging only (do not update targets)
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        arm_half_extents = torch.tensor([0.25, 0.1, 0.06], device=self.device)
        arm_position = self._arm.data.root_pos_w[:, :3]
        arm_quat = self._arm.data.root_quat_w

        min_distances = self._point_to_box_distance(
            ee_position, arm_position, arm_quat, arm_half_extents
        )
        beta_values = self._compute_beta_transition(min_distances)

        # Additionally log APF beta values for first few environments (every 10 steps)
        if self.common_step_counter % 10 == 0:
            # Log for first 3 environments
            for i in range(min(3, self.num_envs)):
                print(
                    f"[APF] Env {i}: dist={min_distances[i]:.3f}m, β={beta_values[i]:.3f}"
                )

    def _save_joint_targets(self):
        """Save joint targets for all environments at current timestep."""

        # Initialize file and tracking variables if not already done
        if not hasattr(self, "_joint_targets_file"):
            # Create output directory
            os.makedirs("./joint_targets_data", exist_ok=True)

            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._joint_targets_filename = (
                f"./joint_targets_data/joint_targets_{timestamp}.csv"
            )

            # Open file and write header
            self._joint_targets_file = open(
                self._joint_targets_filename, "w", newline=""
            )
            self._csv_writer = csv.writer(self._joint_targets_file)

            # Write header row
            num_joints = len(self._joint_indices)
            header = ["step", "env_id"] + [
                f"joint_{i}_target" for i in range(num_joints)
            ]
            self._csv_writer.writerow(header)

            print(f"Started saving joint targets to: {self._joint_targets_filename}")

        # Save joint targets for all environments
        current_step = self.common_step_counter
        joint_targets_cpu = self._robot_dof_targets.cpu().numpy()

        for env_id in range(self.num_envs):
            row = [current_step, env_id] + joint_targets_cpu[env_id].tolist()
            self._csv_writer.writerow(row)

        # Flush to ensure data is written (optional, for safety)
        if (
            current_step % 100 == 0
        ):  # Flush every 100 steps to balance performance and safety
            self._joint_targets_file.flush()

    def _cleanup_joint_targets_file(self):
        """Close the joint targets file when simulation ends."""
        if hasattr(self, "_joint_targets_file") and self._joint_targets_file:
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
        self.target_pos_visualizer.visualize(
            translations=des_pos_w, orientations=des_quat_w
        )


# Factory function for creating the environment
def create_obj_camera_pose_tracking_env(
    cfg: AGANDataCollectionEnvCfg = None,
    render_mode: str = None,
    **kwargs,
) -> AGANDataCollectionEnv:
    """Factory function to create the environment with default config if none provided."""
    if cfg is None:
        cfg = AGANDataCollectionEnvCfg()

    return AGANDataCollectionEnv(cfg, render_mode=render_mode, **kwargs)
