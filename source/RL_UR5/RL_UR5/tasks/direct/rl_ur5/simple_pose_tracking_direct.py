"""
Simple UR5 pose-tracking environment aligned with the gray-depth task setup,
without the human arm obstacle.
"""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    FrameTransformer,
    FrameTransformerCfg,
    TiledCamera,
    TiledCameraCfg,
)
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .assets.ur5 import UR5_GRIPPER_CFG
from .rollout_logger import RolloutLogger

try:
    from .thresholds import TABLE_HEIGHT
except ImportError:
    TABLE_HEIGHT = 0.72


class UR5EnvWindow(BaseEnvWindow):
    """Window manager for the simple UR5 task."""

    def __init__(self, env: SimpleCameraPoseTrackingEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class SimpleCameraPoseTrackingEnvCfg(DirectRLEnvCfg):
    """Configuration for the simple pose-tracking environment."""

    debug_vis = True

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    episode_length_s = 4.0
    decimation = 2
    action_scale = 0.1
    num_envs = 8
    env_spacing = 4.0
    state_dim = 19
    camera_target_height = 224
    camera_target_width = 224
    camera_crop_top = 60
    camera_crop_bottom = 20
    rollout_log_enabled = True
    rollout_log_path = "/home/adi2440/moveit2_UR5/src/rl_ur5_controller/rl_ur5_controller/isaac_logs/simple_v4_sim.hdf5"
    rollout_log_flush_interval = 100
    rollout_log_stride = 1

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
    state_space = gym.spaces.Box(
        low=float("-inf"), high=float("inf"), shape=(state_dim,)
    )
    observation_space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(
                low=float("-inf"),
                high=float("inf"),
                shape=(camera_target_height, camera_target_width, 3),
            ),
            "state": gym.spaces.Box(
                low=float("-inf"), high=float("inf"), shape=(state_dim,)
            ),
        }
    )

    robot_cfg: ArticulationCfg = UR5_GRIPPER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

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

    ee_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/ee_link",
                name="end_effector",
                offset=OffsetCfg(pos=[0.1226, 0.0, 0.0]),
            ),
        ],
    )

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
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
            pos=(1.5, -0.06, 1.143),
            rot=(0.59637, 0.37993, 0.37993, 0.59637),
            convention="opengl",
        ),
    )

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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=True,
    )

    viewer = ViewerCfg(eye=(7.5, 7.5, 7.5), origin_type="world", env_index=0)

    target_pose_range = {
        "x": (0.6, 0.8),
        "y": (0.45, 0.55),
        "z": (-0.2, 0.2),
        "roll": (0.0, 0.0),
        "pitch": (1.57, 1.57),
        "yaw": (0.0, 0.0),
    }

    reward_distance_weight = -1.0
    reward_distance_tanh_weight = 1.0
    reward_distance_tanh_std = 0.1
    reward_orientation_weight = -2.0
    # reward_action_rate_weight = 0.5
    reward_joint_velocity_weight = -0.5
    reward_table_collision_weight = -2.0
    huber_delta = 0.08
    success_reward = 5.0

    position_threshold = 0.1
    orientation_threshold = 0.15
    velocity_threshold = 0.5
    error_print_interval = 500

    joint_pos_noise_min = -0.005
    joint_pos_noise_max = 0.005

    robot_base_pose = [-0.568, -0.858, 1.402, -2.185, -1.6060665, 0.0]
    robot_reset_noise_range = 0.05


class SimpleCameraPoseTrackingEnv(DirectRLEnv):
    """Simple pose-tracking task without the human arm obstacle."""

    cfg: SimpleCameraPoseTrackingEnvCfg

    def __init__(
        self,
        cfg: SimpleCameraPoseTrackingEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        self.cfg = cfg
        self._rollout_logger = None
        self._rollout_episode_ids = None
        self._next_rollout_episode_id = 0
        super().__init__(cfg, render_mode, **kwargs)

        self.extras = {"log": {}}

        self._joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self._joint_indices, _ = self._robot.find_joints(self._joint_names)

        self._robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[
            0, self._joint_indices, 0
        ].to(self.device)
        self._robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[
            0, self._joint_indices, 1
        ].to(self.device)

        self._robot_dof_targets = torch.zeros(
            (self.num_envs, len(self._joint_indices)), device=self.device
        )
        self._target_poses = torch.zeros((self.num_envs, 7), device=self.device)
        self._actions = torch.zeros_like(self._robot_dof_targets)
        self._previous_actions = torch.zeros_like(self._robot_dof_targets)
        self._rollout_episode_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.int64
        )
        self._next_rollout_episode_id = int(self.num_envs)

        self._episode_sums = {
            "position_error": torch.zeros(self.num_envs, device=self.device),
            "orientation_error": torch.zeros(self.num_envs, device=self.device),
            "velocity_error": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
            "success_count": torch.zeros(self.num_envs, device=self.device),
        }

        if self.cfg.rollout_log_enabled:
            self._rollout_logger = RolloutLogger(
                path=self.cfg.rollout_log_path,
                run_prefix="simple_pose_tracking",
                flush_interval=self.cfg.rollout_log_flush_interval,
                metadata={
                    "task": "simple_pose_tracking",
                    "num_envs": int(self.num_envs),
                    "state_dim": int(self.cfg.state_dim),
                    "action_dim": int(self.cfg.action_space.shape[0]),
                    "camera_target_height": int(self.cfg.camera_target_height),
                    "camera_target_width": int(self.cfg.camera_target_width),
                },
            )
            print(f"[INFO] Rollout logging enabled: {self._rollout_logger.path}")

        self.set_debug_vis(self.cfg.debug_vis)

        print(f"[INFO] Environment initialized with {self.num_envs} environments")
        print(f"[INFO] Action scale: {self.cfg.action_scale}")
        print(f"[INFO] Target pose range X: {self.cfg.target_pose_range['x']}")
        print(f"[INFO] Target pose range Y: {self.cfg.target_pose_range['y']}")
        print(f"[INFO] Target pose range Z: {self.cfg.target_pose_range['z']}")

    def _setup_scene(self):
        """Set up the scene with the UR5, camera, and static assets."""
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)

        self._table = RigidObject(self.cfg.table_cfg)
        self._white_plane = RigidObject(self.cfg.white_plane_cfg)
        self._i2r_plane = RigidObject(self.cfg.i2r_plane_cfg)
        self._clemson_plane = RigidObject(self.cfg.clemson_plane_cfg)

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["white_plane"] = self._white_plane
        self.scene.rigid_objects["i2r_plane"] = self._i2r_plane
        self.scene.rigid_objects["clemson_plane"] = self._clemson_plane

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=1600.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/DomeLight", light_cfg)

        dir_light_cfg = sim_utils.DistantLightCfg(
            intensity=1000.0, color=(1.0, 1.0, 0.9), angle=0.53
        )
        dir_light_cfg.func("/World/DirectionalLight", dir_light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics stepping."""
        self._actions = actions.clone().clamp(-1.0, 1.0) * self.cfg.action_scale

    def _apply_action(self):
        """Apply delta joint actions with the same safety logic as the gray-depth task."""
        current_joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        self._robot_dof_targets = current_joint_pos + self._actions

        safety_margin = 0.05
        self._robot_dof_targets = torch.clamp(
            self._robot_dof_targets,
            self._robot_dof_lower_limits + safety_margin,
            self._robot_dof_upper_limits - safety_margin,
        )

        # max_velocity = 1.5
        # velocity_command = (
        #     self._robot_dof_targets - current_joint_pos
        # ) / self.physics_dt
        # velocity_command = torch.clamp(velocity_command, -max_velocity, max_velocity)
        # self._robot_dof_targets = current_joint_pos + velocity_command * self.physics_dt

        self._robot.set_joint_position_target(
            self._robot_dof_targets, joint_ids=self._joint_indices
        )

    def _get_observations(self) -> dict:
        """Return state and RGB camera observations."""
        camera_obs = self._get_camera_observations()
        state_obs = self._get_state_observations()
        self._log_rollout_batch(state_obs, camera_obs)
        return {"policy": {"image": camera_obs, "state": state_obs}}

    def _get_state_observations(self) -> torch.Tensor:
        """Return state observations aligned with the gray-depth task state layout."""
        joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        if self.cfg.joint_pos_noise_max > 0.0:
            joint_pos_noise = (
                torch.rand_like(joint_pos)
                * (self.cfg.joint_pos_noise_max - self.cfg.joint_pos_noise_min)
                + self.cfg.joint_pos_noise_min
            )
            joint_pos = joint_pos + joint_pos_noise

        return torch.cat(
            [joint_pos, self._target_poses, self._actions],
            dim=-1,
        )

    def _get_camera_observations(self) -> torch.Tensor:
        """Return resized RGB observations in NHWC layout for the camera PPO config."""
        camera_data = self._tiled_camera.data.output["rgb"] / 255.0
        mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        camera_data = camera_data - mean_tensor

        cropped = camera_data[
            :, self.cfg.camera_crop_top : -self.cfg.camera_crop_bottom, :, :
        ]
        cropped = cropped.permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            cropped,
            size=(self.cfg.camera_target_height, self.cfg.camera_target_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.permute(0, 2, 3, 1).contiguous()

    def _huber_loss(self, x: torch.Tensor, delta: float) -> torch.Tensor:
        abs_x = torch.abs(x)
        return torch.where(abs_x <= delta, 0.5 * x * x, delta * (abs_x - 0.5 * delta))

    def _quat_l2_error(
        self, quat_a: torch.Tensor, quat_b: torch.Tensor
    ) -> torch.Tensor:
        """Quaternion Euclidean distance (L2) with sign-invariant matching."""
        quat_a = torch.nn.functional.normalize(quat_a, dim=-1)
        quat_b = torch.nn.functional.normalize(quat_b, dim=-1)
        dot = torch.sum(quat_a * quat_b, dim=-1).abs().clamp(max=1.0)
        return torch.sqrt(torch.clamp(2.0 - 2.0 * dot, min=0.0))

    def _get_rewards(self) -> torch.Tensor:
        """Reward pose tracking using the gray-depth task shaping minus the arm terms."""
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat = self._ee_frame.data.target_quat_w[..., 0, :]

        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]

        desired_pos_b = self._target_poses[:, :3]
        desired_quat_b = self._target_poses[:, 3:7]
        desired_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, desired_pos_b
        )
        desired_quat_w = math_utils.quat_mul(robot_quat, desired_quat_b)

        position_error = torch.norm(ee_position - desired_pos_w, dim=-1)
        position_huber = self._huber_loss(position_error, self.cfg.huber_delta)
        position_reward = self.cfg.reward_distance_weight * position_huber

        position_reward_tanh = 1.0 - torch.tanh(
            position_error / self.cfg.reward_distance_tanh_std
        )
        position_reward_tanh = (
            self.cfg.reward_distance_tanh_weight * position_reward_tanh
        )

        orientation_error = self._quat_l2_error(ee_quat, desired_quat_w)
        orientation_stage_mask = position_error < self.cfg.position_threshold
        orientation_reward = (
            self.cfg.reward_orientation_weight
            * orientation_error
            * orientation_stage_mask.float()
        )

        joint_velocity_values = self._robot.data.joint_vel[:, self._joint_indices]
        velocity_stage_mask = orientation_stage_mask & (
            orientation_error < self.cfg.orientation_threshold
        )
        joint_velocity_penalty = (
            torch.mean(joint_velocity_values**2, dim=-1)
            * self.cfg.reward_joint_velocity_weight
            * velocity_stage_mask.float()
        )

        ee_height = ee_position[:, 2]
        table_penalty = torch.where(
            ee_height < (TABLE_HEIGHT + 0.05),
            torch.full_like(ee_height, self.cfg.reward_table_collision_weight),
            torch.zeros_like(ee_height),
        )

        joint_velocities = torch.norm(joint_velocity_values, p=2, dim=-1)
        success_mask = (
            (position_error < self.cfg.position_threshold)
            & (orientation_error < self.cfg.orientation_threshold)
            & (joint_velocities < self.cfg.velocity_threshold)
        )
        success_reward = torch.where(
            success_mask,
            torch.full_like(position_error, self.cfg.success_reward),
            torch.zeros_like(position_error),
        )

        rewards = (
            position_reward
            + position_reward_tanh
            + orientation_reward
            + joint_velocity_penalty
            + table_penalty
            + success_reward
        )

        self._episode_sums["position_error"] += position_error
        self._episode_sums["orientation_error"] += orientation_error
        self._episode_sums["velocity_error"] += joint_velocities
        self._episode_sums["total_reward"] += rewards
        self._episode_sums["success_count"] += success_mask.float()

        if self.common_step_counter % self.cfg.error_print_interval == 0:
            print(
                f"[TRACK_ERR] step={self.common_step_counter} "
                f"pos={position_error.mean().item():.4f}/{self.cfg.position_threshold:.4f}, "
                f"ori={orientation_error.mean().item():.4f}/{self.cfg.orientation_threshold:.4f}, "
                f"vel={joint_velocities.mean().item():.4f}/{self.cfg.velocity_threshold:.4f}, "
                f"ori_stage={orientation_stage_mask.float().mean().item():.2f}, "
                f"vel_stage={velocity_stage_mask.float().mean().item():.2f}"
            )

        # Update action history after reward computation so observations carry the most recent action.
        self._previous_actions.copy_(self._actions)

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check success and timeout termination."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat = self._ee_frame.data.target_quat_w[..., 0, :]

        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]

        desired_pos_b = self._target_poses[:, :3]
        desired_quat_b = self._target_poses[:, 3:7]
        desired_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, desired_pos_b
        )
        desired_quat_w = math_utils.quat_mul(robot_quat, desired_quat_b)

        position_error = torch.norm(ee_position - desired_pos_w, p=2, dim=-1)
        orientation_error = self._quat_l2_error(ee_quat, desired_quat_w)
        joint_velocities = torch.norm(
            self._robot.data.joint_vel[:, self._joint_indices], p=2, dim=-1
        )

        success = (
            (position_error < self.cfg.position_threshold)
            & (orientation_error < self.cfg.orientation_threshold)
            & (joint_velocities < self.cfg.velocity_threshold)
        )
        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset the requested environments."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)

        if len(env_ids) > 0:
            episode_lengths = torch.clamp(
                self.episode_length_buf[env_ids].float(), min=1.0
            )
            self.extras["log"] = {
                "Episode/position_error": torch.mean(
                    self._episode_sums["position_error"][env_ids] / episode_lengths
                ).item(),
                "Episode/orientation_error": torch.mean(
                    self._episode_sums["orientation_error"][env_ids] / episode_lengths
                ).item(),
                "Episode/velocity_error": torch.mean(
                    self._episode_sums["velocity_error"][env_ids] / episode_lengths
                ).item(),
                "Episode/total_reward": torch.mean(
                    self._episode_sums["total_reward"][env_ids]
                ).item(),
                "Episode/success_rate": torch.mean(
                    self._episode_sums["success_count"][env_ids] / episode_lengths
                ).item(),
            }

            for key in self._episode_sums:
                self._episode_sums[key][env_ids] = 0.0

        num_resets = len(env_ids)
        if num_resets == 0:
            return

        base_pose = torch.tensor(
            self.cfg.robot_base_pose, device=self.device, dtype=torch.float32
        )
        joint_pos = base_pose.unsqueeze(0).repeat(num_resets, 1)
        if self.cfg.robot_reset_noise_range > 0.0:
            joint_pos += sample_uniform(
                -self.cfg.robot_reset_noise_range,
                self.cfg.robot_reset_noise_range,
                joint_pos.shape,
                self.device,
            )
        joint_vel = torch.zeros_like(joint_pos)

        self._robot.write_joint_state_to_sim(
            joint_pos, joint_vel, joint_ids=self._joint_indices, env_ids=env_ids
        )
        self._robot.set_joint_position_target(
            joint_pos, joint_ids=self._joint_indices, env_ids=env_ids
        )
        self._robot_dof_targets[env_ids] = joint_pos

        self._sample_target_poses(env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        if self._rollout_episode_ids is not None:
            episode_ids = torch.arange(
                self._next_rollout_episode_id,
                self._next_rollout_episode_id + num_resets,
                device=self.device,
                dtype=torch.int64,
            )
            self._rollout_episode_ids[env_ids] = episode_ids
            self._next_rollout_episode_id += num_resets

    def _sample_target_poses(self, env_ids: Sequence[int]):
        """Sample target poses from the same distribution as the gray-depth task."""
        num_envs = len(env_ids)
        if num_envs == 0:
            return

        x = sample_uniform(
            self.cfg.target_pose_range["x"][0],
            self.cfg.target_pose_range["x"][1],
            (num_envs,),
            self.device,
        )
        y = sample_uniform(
            self.cfg.target_pose_range["y"][0],
            self.cfg.target_pose_range["y"][1],
            (num_envs,),
            self.device,
        )
        z = sample_uniform(
            self.cfg.target_pose_range["z"][0],
            self.cfg.target_pose_range["z"][1],
            (num_envs,),
            self.device,
        )
        roll = sample_uniform(
            self.cfg.target_pose_range["roll"][0],
            self.cfg.target_pose_range["roll"][1],
            (num_envs,),
            self.device,
        )
        pitch = sample_uniform(
            self.cfg.target_pose_range["pitch"][0],
            self.cfg.target_pose_range["pitch"][1],
            (num_envs,),
            self.device,
        )
        yaw = sample_uniform(
            self.cfg.target_pose_range["yaw"][0],
            self.cfg.target_pose_range["yaw"][1],
            (num_envs,),
            self.device,
        )

        self._target_poses[env_ids, :3] = torch.stack([x, y, z], dim=-1)
        self._target_poses[env_ids, 3:7] = math_utils.quat_from_euler_xyz(
            roll, pitch, yaw
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create a marker for the target pose."""
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/target_position"
                self.target_visualizer = VisualizationMarkers(marker_cfg)
            self.target_visualizer.set_visibility(True)
        elif hasattr(self, "target_visualizer"):
            self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update the target pose marker in the world frame."""
        del event
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        desired_pos_b = self._target_poses[:, :3]
        desired_quat_b = self._target_poses[:, 3:7]

        desired_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, desired_pos_b
        )
        desired_quat_w = math_utils.quat_mul(robot_quat, desired_quat_b)

        self.target_visualizer.visualize(
            translations=desired_pos_w, orientations=desired_quat_w
        )

    def close(self):
        """Cleanup for the environment."""
        if self._rollout_logger is not None:
            self._rollout_logger.close()
            self._rollout_logger = None
        super().close()

    def _log_rollout_batch(
        self, state_obs: torch.Tensor, camera_obs: torch.Tensor
    ) -> None:
        if self._rollout_logger is None:
            return
        if self.common_step_counter % max(1, int(self.cfg.rollout_log_stride)) != 0:
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

        self._rollout_logger.append_batch(
            states=state_obs.detach().cpu().numpy().astype(np.float32),
            images=camera_obs.detach().cpu().numpy().astype(np.float32),
            base_actions=self._actions.detach().cpu().numpy().astype(np.float32),
            policy_actions=(self._actions / max(float(self.cfg.action_scale), 1e-6))
            .detach()
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
            episode_ids=self._rollout_episode_ids.detach()
            .cpu()
            .numpy()
            .astype(np.int64),
            step_ids=self.episode_length_buf.detach().cpu().numpy().astype(np.int64),
            env_ids=np.arange(self.num_envs, dtype=np.int64),
        )
