"""
Hierarchical UR5 waypoint environments.

This module adds two staged tasks:

* UR5WaypointLowLevelEnv: a state-only joint controller that learns to track
  sampled end-effector waypoints.
* UR5HierarchicalGrayDepthEnv: a visual high-level policy that chooses
  Cartesian waypoints, executed by a damped differential IK tracker.
"""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .huber_obj_direct_gray_depth import (
    ObjCameraGrayDepthPoseTrackingDirectEnv,
    ObjCameraGrayDepthPoseTrackingDirectEnvCfg,
    TABLE_HEIGHT,
)


@configclass
class UR5WaypointLowLevelEnvCfg(ObjCameraGrayDepthPoseTrackingDirectEnvCfg):
    """State-only low-level waypoint tracker configuration."""

    debug_vis = False
    episode_length_s = 4.0
    decimation = 4
    action_scale = 0.06
    state_dim = 32

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
    state_space = 0
    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(
                low=float("-inf"), high=float("inf"), shape=(state_dim,)
            ),
        }
    )

    command_resampling_time = 6.0
    curriculum_enabled = True
    curriculum_arm_speeds = [0.0, 0.05, 0.10, 0.15]
    curriculum_target_ranges = [
        {"x": (0.55, 0.68), "y": (0.40, 0.52), "z": (-0.08, 0.12)},
        {"x": (0.50, 0.75), "y": (0.35, 0.56), "z": (-0.15, 0.18)},
        {"x": (0.45, 0.82), "y": (0.25, 0.62), "z": (-0.20, 0.25)},
        {"x": (0.35, 0.90), "y": (-0.45, 0.65), "z": (-0.25, 0.35)},
    ]

    max_joint_velocity = 1.2
    joint_limit_safety_margin = 0.05
    waypoint_position_threshold = 0.025
    waypoint_orientation_threshold = 0.08
    waypoint_velocity_threshold = 0.08
    waypoint_success_reward = 4.0
    reward_waypoint_distance_weight = -3.0
    reward_waypoint_tanh_weight = 1.5
    reward_waypoint_tanh_std = 0.08
    reward_orientation_weight = -0.5
    reward_action_rate_weight = -0.25
    reward_joint_velocity_weight = -0.03
    reward_joint_limit_weight = -0.5
    reward_arm_safety_weight = -3.0
    safe_arm_distance = 0.08
    critical_arm_distance = 0.03

    robot_base_pose = [-0.568, -0.858, 1.402, -2.185, -1.6060665, 1.64142667]
    robot_reset_noise_range = 0.06


@configclass
class UR5HierarchicalGrayDepthEnvCfg(ObjCameraGrayDepthPoseTrackingDirectEnvCfg):
    """High-level visual waypoint policy configuration."""

    debug_vis = True
    episode_length_s = 6.0
    decimation = 4
    action_scale = 1.0
    state_dim = 28
    camera_target_height = 120
    camera_target_width = 160
    rgbd_depth_max = 3.0

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
    state_space = 0
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

    command_resampling_time = 6.0
    curriculum_enabled = True
    curriculum_arm_speeds = [0.0, 0.05, 0.10, 0.20]
    curriculum_target_ranges = [
        {"x": (0.55, 0.68), "y": (0.40, 0.52), "z": (-0.08, 0.12)},
        {"x": (0.50, 0.75), "y": (0.35, 0.56), "z": (-0.15, 0.18)},
        {"x": (0.45, 0.82), "y": (0.25, 0.62), "z": (-0.20, 0.25)},
        {"x": (0.35, 0.90), "y": (-0.45, 0.65), "z": (-0.25, 0.35)},
    ]

    waypoint_hold_steps = 4
    waypoint_nominal_step = 0.08
    waypoint_residual_scale = (0.08, 0.08, 0.06)
    waypoint_bounds = {
        "x": (0.35, 0.90),
        "y": (-0.60, 0.65),
        "z": (-0.25, 0.35),
    }

    ik_lambda = 0.05
    ik_target_smoothing = 0.35
    max_joint_velocity = 1.2
    joint_limit_safety_margin = 0.05

    success_position_threshold = 0.025
    success_orientation_threshold = 0.08
    success_velocity_threshold = 0.08
    success_arm_clearance = 0.06
    success_reward = 8.0

    reward_distance_weight = -3.0
    reward_distance_tanh_weight = 2.0
    reward_distance_tanh_std = 0.08
    reward_progress_weight = 4.0
    reward_waypoint_feasibility_weight = 1.0
    reward_orientation_weight = -0.5
    reward_action_rate_weight = -0.15
    reward_joint_velocity_weight = -0.03
    reward_table_collision_weight = -4.0
    reward_arm_safety_weight = -4.0
    safe_arm_distance = 0.10
    critical_arm_distance = 0.03

    debug_goal_marker_radius = 0.035
    debug_waypoint_marker_radius = 0.025

    robot_base_pose = [-0.868, -0.558, 1.402, -2.185, -1.6060665, 1.64142667]
    robot_reset_noise_range = 0.06


class _UR5HierarchyMathMixin:
    """Shared helpers for waypoint tracking environments."""

    def _joint_pos_to_normalized(self, joint_pos: torch.Tensor) -> torch.Tensor:
        lower = self._robot_dof_lower_limits + self.cfg.joint_limit_safety_margin
        upper = self._robot_dof_upper_limits - self.cfg.joint_limit_safety_margin
        center = 0.5 * (upper + lower)
        half_range = torch.clamp(0.5 * (upper - lower), min=1e-6)
        return torch.clamp((joint_pos - center) / half_range, -1.0, 1.0)

    def _clamp_joint_targets(self, joint_targets: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            joint_targets,
            self._robot_dof_lower_limits + self.cfg.joint_limit_safety_margin,
            self._robot_dof_upper_limits - self.cfg.joint_limit_safety_margin,
        )

    def _limit_joint_velocity(
        self, desired_joint_pos: torch.Tensor, current_joint_pos: torch.Tensor
    ) -> torch.Tensor:
        max_delta = self.cfg.max_joint_velocity * self.physics_dt
        joint_delta = torch.clamp(
            desired_joint_pos - current_joint_pos, -max_delta, max_delta
        )
        return current_joint_pos + joint_delta

    def _get_ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat_w = self._ee_frame.data.target_quat_w[..., 0, :]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        return math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

    def _target_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        target_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, self._target_poses[:, :3]
        )
        target_quat_w = math_utils.quat_mul(robot_quat, self._target_poses[:, 3:7])
        return target_pos_w, target_quat_w

    def _arm_distance(self) -> torch.Tensor:
        arm_half_extents = torch.tensor([0.25, 0.1, 0.06], device=self.device)
        return self._point_to_box_distance(
            self._ee_frame.data.target_pos_w[..., 0, :],
            self._arm.data.root_pos_w[:, :3],
            self._arm.data.root_quat_w,
            arm_half_extents,
        )

    def _arm_safety_penalty(self, min_distances: torch.Tensor) -> torch.Tensor:
        safe_distance = float(self.cfg.safe_arm_distance)
        critical_distance = float(self.cfg.critical_arm_distance)
        safety_band = max(safe_distance - critical_distance, 1e-6)
        normalized = torch.clamp(
            (safe_distance - min_distances) / safety_band, 0.0, 1.0
        )
        return normalized * normalized

    def _joint_limit_penalty(self, joint_pos: torch.Tensor) -> torch.Tensor:
        lower_dist = joint_pos - self._robot_dof_lower_limits
        upper_dist = self._robot_dof_upper_limits - joint_pos
        margin = float(self.cfg.joint_limit_safety_margin) * 2.0
        lower_penalty = torch.clamp((margin - lower_dist) / margin, 0.0, 1.0)
        upper_penalty = torch.clamp((margin - upper_dist) / margin, 0.0, 1.0)
        return torch.mean(lower_penalty**2 + upper_penalty**2, dim=-1)


class UR5WaypointLowLevelEnv(
    _UR5HierarchyMathMixin, ObjCameraGrayDepthPoseTrackingDirectEnv
):
    """State-only waypoint tracker that outputs smooth joint deltas."""

    cfg: UR5WaypointLowLevelEnvCfg

    def __init__(
        self,
        cfg: UR5WaypointLowLevelEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros((self.num_envs, 6), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self._waypoint_pose_b = torch.zeros((self.num_envs, 7), device=self.device)

        self._episode_sums = {
            "position_error": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
            "success_count": torch.zeros(self.num_envs, device=self.device),
            "min_arm_distance": torch.ones(self.num_envs, device=self.device)
            * float("inf"),
        }

        print("[INFO] Low-level action mode: smoothed joint deltas to waypoint")

    def _setup_scene(self):
        """Set up the low-level scene without camera sensors."""
        self._robot = Articulation(self.cfg.robot_cfg)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        self._arm = RigidObject(self.cfg.arm_cfg)
        self._table = RigidObject(self.cfg.table_cfg)
        self._white_plane = RigidObject(self.cfg.white_plane_cfg)
        self._clemson_plane = RigidObject(self.cfg.clemson_plane_cfg)
        self._i2r_plane = RigidObject(self.cfg.i2r_plane_cfg)

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.rigid_objects["arm"] = self._arm
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["white_plane"] = self._white_plane
        self.scene.rigid_objects["clemson_plane"] = self._clemson_plane
        self.scene.rigid_objects["i2r_plane"] = self._i2r_plane

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=1600.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/DomeLight", light_cfg)

        dir_light_cfg = sim_utils.DistantLightCfg(
            intensity=1000.0, color=(1.0, 1.0, 0.9), angle=0.53
        )
        dir_light_cfg.func("/World/DirectionalLight", dir_light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clone().float().clamp(-1.0, 1.0) * self.cfg.action_scale

        self._command_time_left -= self.physics_dt
        expired_mask = self._command_time_left <= 0.0
        if torch.any(expired_mask):
            expired_ids = torch.nonzero(expired_mask, as_tuple=False).squeeze(-1)
            env_ids = expired_ids.cpu().tolist()
            self._sample_commands(env_ids)
            self._waypoint_pose_b[expired_ids] = self._target_poses[expired_ids]
            self._command_time_left[expired_mask] = self.cfg.command_resampling_time

        self._check_curriculum_advancement()
        self._reset_robot_when_stuck_at_table()
        self._update_arm_position()
        self._update_debug_visualization()

    def _apply_action(self) -> None:
        current_joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        desired_joint_pos = current_joint_pos + self.actions
        desired_joint_pos = self._clamp_joint_targets(desired_joint_pos)
        desired_joint_pos = self._limit_joint_velocity(
            desired_joint_pos, current_joint_pos
        )
        self._robot_dof_targets = desired_joint_pos
        self._robot.set_joint_position_target(
            self._robot_dof_targets, joint_ids=self._joint_indices
        )

    def _get_observations(self) -> dict:
        return {"policy": {"state": self._get_state_observations()}}

    def _get_state_observations(self) -> torch.Tensor:
        joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        if self.cfg.joint_pos_noise_max > 0.0:
            joint_pos = joint_pos + sample_uniform(
                self.cfg.joint_pos_noise_min,
                self.cfg.joint_pos_noise_max,
                joint_pos.shape,
                self.device,
            )
        joint_vel = self._robot.data.joint_vel[:, self._joint_indices]
        ee_pos_b, ee_quat_b = self._get_ee_pose_b()
        return torch.cat(
            [
                self._joint_pos_to_normalized(joint_pos),
                joint_vel,
                self._waypoint_pose_b,
                ee_pos_b,
                ee_quat_b,
                self.actions,
            ],
            dim=-1,
        )

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat_w = self._ee_frame.data.target_quat_w[..., 0, :]

        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        waypoint_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, self._waypoint_pose_b[:, :3]
        )
        waypoint_quat_w = math_utils.quat_mul(robot_quat, self._waypoint_pose_b[:, 3:7])

        position_error = torch.norm(ee_pos_w - waypoint_pos_w, dim=-1)
        position_reward = self.cfg.reward_waypoint_distance_weight * self._huber_loss(
            position_error, self.cfg.huber_delta
        )
        position_reward += self.cfg.reward_waypoint_tanh_weight * (
            1.0 - torch.tanh(position_error / self.cfg.reward_waypoint_tanh_std)
        )

        orientation_error = math_utils.quat_error_magnitude(ee_quat_w, waypoint_quat_w)
        orientation_reward = self.cfg.reward_orientation_weight * self._huber_loss(
            orientation_error, self.cfg.huber_delta * 0.5
        )

        joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        joint_vel = self._robot.data.joint_vel[:, self._joint_indices]
        joint_vel_norm = torch.norm(joint_vel, p=2, dim=-1)
        velocity_penalty = self.cfg.reward_joint_velocity_weight * torch.mean(
            joint_vel**2, dim=-1
        )
        action_rate_penalty = self.cfg.reward_action_rate_weight * torch.mean(
            (self.actions - self.previous_actions) ** 2, dim=-1
        )
        joint_limit_penalty = (
            self.cfg.reward_joint_limit_weight * self._joint_limit_penalty(joint_pos)
        )

        min_arm_dist = self._arm_distance()
        arm_penalty = self.cfg.reward_arm_safety_weight * self._arm_safety_penalty(
            min_arm_dist
        )
        table_penalty = torch.where(
            ee_pos_w[:, 2] < TABLE_HEIGHT + 0.05,
            torch.full_like(position_error, self.cfg.reward_table_collision_weight),
            torch.zeros_like(position_error),
        )

        success = (
            (position_error < self.cfg.waypoint_position_threshold)
            & (orientation_error < self.cfg.waypoint_orientation_threshold)
            & (joint_vel_norm < self.cfg.waypoint_velocity_threshold)
            & (min_arm_dist > self.cfg.safe_arm_distance)
        )
        success_reward = torch.where(
            success,
            torch.full_like(position_error, self.cfg.waypoint_success_reward),
            torch.zeros_like(position_error),
        )

        rewards = (
            position_reward
            + orientation_reward
            + velocity_penalty
            + action_rate_penalty
            + joint_limit_penalty
            + arm_penalty
            + table_penalty
            + success_reward
        )

        self._episode_sums["position_error"] += position_error
        self._episode_sums["total_reward"] += rewards
        self._episode_sums["success_count"] += success.float()
        self._episode_sums["min_arm_distance"] = torch.minimum(
            self._episode_sums["min_arm_distance"], min_arm_dist
        )

        if torch.any(success):
            self._success_buffer[self._success_buffer_idx] = success.float().mean()
            self._success_buffer_idx = (
                self._success_buffer_idx + 1
            ) % self.cfg.success_window_size

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        ee_pos_w = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat_w = self._ee_frame.data.target_quat_w[..., 0, :]
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        waypoint_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, self._waypoint_pose_b[:, :3]
        )
        waypoint_quat_w = math_utils.quat_mul(robot_quat, self._waypoint_pose_b[:, 3:7])
        position_error = torch.norm(ee_pos_w - waypoint_pos_w, p=2, dim=-1)
        orientation_error = math_utils.quat_error_magnitude(ee_quat_w, waypoint_quat_w)
        joint_vel = torch.norm(
            self._robot.data.joint_vel[:, self._joint_indices], p=2, dim=-1
        )
        success = (
            (position_error < self.cfg.waypoint_position_threshold)
            & (orientation_error < self.cfg.waypoint_orientation_threshold)
            & (joint_vel < self.cfg.waypoint_velocity_threshold)
        )
        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        reset_ids = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.tensor(env_ids, device=self.device, dtype=torch.long)
        )
        self._waypoint_pose_b[reset_ids] = self._target_poses[reset_ids]
        self.actions[reset_ids] = 0.0
        self.previous_actions[reset_ids] = 0.0
        self._robot_dof_targets[reset_ids] = self._robot.data.joint_pos[reset_ids][
            :, self._joint_indices
        ]


class UR5HierarchicalGrayDepthEnv(
    _UR5HierarchyMathMixin, ObjCameraGrayDepthPoseTrackingDirectEnv
):
    """High-level visual policy that commands Cartesian waypoints."""

    cfg: UR5HierarchicalGrayDepthEnvCfg

    def __init__(
        self,
        cfg: UR5HierarchicalGrayDepthEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_high_level_actions = torch.zeros_like(self.actions)
        self._effective_high_level_actions = torch.zeros_like(self.actions)
        self._waypoint_hold_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._waypoint_pose_b = torch.zeros((self.num_envs, 7), device=self.device)
        self._last_target_distance = torch.zeros(self.num_envs, device=self.device)

        self._init_differential_ik()

        self._episode_sums = {
            "position_error": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
            "success_count": torch.zeros(self.num_envs, device=self.device),
            "min_arm_distance": torch.ones(self.num_envs, device=self.device)
            * float("inf"),
        }

        print("[INFO] High-level action mode: visual Cartesian waypoint residuals")

    def _init_differential_ik(self) -> None:
        self._diff_ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": self.cfg.ik_lambda},
            ),
            num_envs=self.num_envs,
            device=self.device,
        )
        self._ee_body_idx = self._robot.find_bodies("ee_link")[0][0]
        self._ee_jacobi_idx = self._ee_body_idx - 1
        self._ee_offset_pos = torch.tensor(
            [0.1226, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().float().clamp(-1.0, 1.0)

        self._command_time_left -= self.physics_dt
        expired_mask = self._command_time_left <= 0.0
        if torch.any(expired_mask):
            expired_ids = torch.nonzero(expired_mask, as_tuple=False).squeeze(-1)
            self._sample_commands(expired_ids.cpu().tolist())
            self._command_time_left[expired_mask] = self.cfg.command_resampling_time
            self._waypoint_hold_count[expired_ids] = 0

        self._update_waypoint_commands(self.actions)
        self._check_curriculum_advancement()
        self._reset_robot_when_stuck_at_table()
        self._update_arm_position()
        self._update_debug_visualization()

    def _update_waypoint_commands(self, actions: torch.Tensor) -> None:
        self._waypoint_hold_count = torch.clamp(self._waypoint_hold_count - 1, min=0)
        update_mask = self._waypoint_hold_count <= 0
        if not torch.any(update_mask):
            return

        ee_pos_b, _ = self._get_ee_pose_b()
        target_delta = self._target_poses[:, :3] - ee_pos_b
        target_distance = torch.norm(target_delta, dim=-1, keepdim=True)
        direction = target_delta / torch.clamp(target_distance, min=1e-6)
        nominal_step = direction * torch.clamp(
            target_distance, max=self.cfg.waypoint_nominal_step
        )

        residual_scale = torch.tensor(
            self.cfg.waypoint_residual_scale, device=self.device
        ).unsqueeze(0)
        waypoint_pos = ee_pos_b + nominal_step + actions * residual_scale
        low = torch.tensor(
            [
                self.cfg.waypoint_bounds["x"][0],
                self.cfg.waypoint_bounds["y"][0],
                self.cfg.waypoint_bounds["z"][0],
            ],
            device=self.device,
        )
        high = torch.tensor(
            [
                self.cfg.waypoint_bounds["x"][1],
                self.cfg.waypoint_bounds["y"][1],
                self.cfg.waypoint_bounds["z"][1],
            ],
            device=self.device,
        )
        waypoint_pos = torch.max(torch.min(waypoint_pos, high), low)

        self._previous_high_level_actions[update_mask] = (
            self._effective_high_level_actions[update_mask]
        )
        self._effective_high_level_actions[update_mask] = actions[update_mask]
        self._waypoint_pose_b[update_mask, :3] = waypoint_pos[update_mask]
        self._waypoint_pose_b[update_mask, 3:7] = self._target_poses[update_mask, 3:7]
        self._waypoint_hold_count[update_mask] = int(self.cfg.waypoint_hold_steps)

    def _apply_action(self) -> None:
        ee_pos_b, ee_quat_b = self._get_ee_pose_b()
        joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        jacobian = self._compute_offset_jacobian_b()

        self._diff_ik.set_command(self._waypoint_pose_b)
        joint_pos_des = self._diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        joint_pos_des = torch.nan_to_num(joint_pos_des, nan=0.0, posinf=0.0, neginf=0.0)
        joint_pos_des = self._clamp_joint_targets(joint_pos_des)

        smoothed_target = self._robot_dof_targets + self.cfg.ik_target_smoothing * (
            joint_pos_des - self._robot_dof_targets
        )
        smoothed_target = self._clamp_joint_targets(smoothed_target)
        self._robot_dof_targets = self._limit_joint_velocity(smoothed_target, joint_pos)

        self._robot.set_joint_position_target(
            self._robot_dof_targets, joint_ids=self._joint_indices
        )

    def _compute_offset_jacobian_b(self) -> torch.Tensor:
        jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self._ee_jacobi_idx, :, self._joint_indices
        ].clone()
        base_rot = self._robot.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        jacobian[:, 0:3, :] += torch.bmm(
            -math_utils.skew_symmetric_matrix(self._ee_offset_pos), jacobian[:, 3:, :]
        )
        return jacobian

    def _get_observations(self) -> dict:
        return {
            "policy": {
                "image": self._get_camera_observations(),
                "state": self._get_state_observations(),
            }
        }

    def _get_state_observations(self) -> torch.Tensor:
        joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        if self.cfg.joint_pos_noise_max > 0.0:
            joint_pos = joint_pos + sample_uniform(
                self.cfg.joint_pos_noise_min,
                self.cfg.joint_pos_noise_max,
                joint_pos.shape,
                self.device,
            )
        joint_vel = self._robot.data.joint_vel[:, self._joint_indices]
        ee_pos_b, _ = self._get_ee_pose_b()
        return torch.cat(
            [
                self._joint_pos_to_normalized(joint_pos),
                joint_vel,
                self._target_poses,
                ee_pos_b,
                self._waypoint_pose_b[:, :3],
                self._effective_high_level_actions,
            ],
            dim=-1,
        )

    def _get_camera_observations(self) -> torch.Tensor:
        rgb_data = self._tiled_camera_gray.data.output["rgb"][..., :3] / 255.0
        gray_data = torch.mean(rgb_data, dim=-1, keepdim=True)

        depth_data = self._tiled_camera_depth.data.output["distance_to_image_plane"]
        max_depth = float(self.cfg.rgbd_depth_max)
        depth_data = torch.nan_to_num(
            depth_data, nan=max_depth, posinf=max_depth, neginf=0.0
        )
        depth_data = torch.clamp(depth_data, 0.0, max_depth) / max_depth

        combined_data = torch.cat([gray_data, depth_data], dim=-1)
        combined_data = combined_data - torch.mean(
            combined_data, dim=(1, 2), keepdim=True
        )
        cropped = combined_data[
            :, self.cfg.camera_crop_top : -self.cfg.camera_crop_bottom, :, :
        ].permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            cropped,
            size=(self.cfg.camera_target_height, self.cfg.camera_target_width),
            mode="bilinear",
            align_corners=False,
        )
        if (
            self.common_step_counter > 0
            and self.common_step_counter % self.cfg.visualize_camera_interval == 0
        ):
            self._visualize_camera_observation(rgb_data, resized, env_id=0)
        return resized.permute(0, 2, 3, 1).contiguous()

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat_w = self._ee_frame.data.target_quat_w[..., 0, :]
        target_pos_w, target_quat_w = self._target_pose_w()

        position_error = torch.norm(ee_pos_w - target_pos_w, dim=-1)
        position_reward = self.cfg.reward_distance_weight * self._huber_loss(
            position_error, self.cfg.huber_delta
        )
        position_reward += self.cfg.reward_distance_tanh_weight * (
            1.0 - torch.tanh(position_error / self.cfg.reward_distance_tanh_std)
        )

        progress = self._last_target_distance - position_error
        progress_reward = self.cfg.reward_progress_weight * torch.clamp(
            progress, min=-0.05, max=0.05
        )
        self._last_target_distance = position_error.detach()

        ee_pos_b, _ = self._get_ee_pose_b()
        waypoint_error = torch.norm(ee_pos_b - self._waypoint_pose_b[:, :3], dim=-1)
        waypoint_reward = self.cfg.reward_waypoint_feasibility_weight * (
            1.0 - torch.tanh(waypoint_error / 0.08)
        )

        orientation_error = math_utils.quat_error_magnitude(ee_quat_w, target_quat_w)
        orientation_reward = self.cfg.reward_orientation_weight * self._huber_loss(
            orientation_error, self.cfg.huber_delta * 0.5
        )

        joint_vel = self._robot.data.joint_vel[:, self._joint_indices]
        joint_vel_norm = torch.norm(joint_vel, p=2, dim=-1)
        velocity_penalty = self.cfg.reward_joint_velocity_weight * torch.mean(
            joint_vel**2, dim=-1
        )
        action_rate_penalty = self.cfg.reward_action_rate_weight * torch.mean(
            (self._effective_high_level_actions - self._previous_high_level_actions)
            ** 2,
            dim=-1,
        )

        min_arm_dist = self._arm_distance()
        arm_penalty = self.cfg.reward_arm_safety_weight * self._arm_safety_penalty(
            min_arm_dist
        )
        table_penalty = torch.where(
            ee_pos_w[:, 2] < TABLE_HEIGHT + 0.05,
            torch.full_like(position_error, self.cfg.reward_table_collision_weight),
            torch.zeros_like(position_error),
        )

        success = (
            (position_error < self.cfg.success_position_threshold)
            & (orientation_error < self.cfg.success_orientation_threshold)
            & (joint_vel_norm < self.cfg.success_velocity_threshold)
            & (min_arm_dist > self.cfg.success_arm_clearance)
        )
        success_reward = torch.where(
            success,
            torch.full_like(position_error, self.cfg.success_reward),
            torch.zeros_like(position_error),
        )

        rewards = (
            position_reward
            + progress_reward
            + waypoint_reward
            + orientation_reward
            + velocity_penalty
            + action_rate_penalty
            + arm_penalty
            + table_penalty
            + success_reward
        )

        self._episode_sums["position_error"] += position_error
        self._episode_sums["total_reward"] += rewards
        self._episode_sums["success_count"] += success.float()
        self._episode_sums["min_arm_distance"] = torch.minimum(
            self._episode_sums["min_arm_distance"], min_arm_dist
        )

        if torch.any(success):
            self._success_buffer[self._success_buffer_idx] = success.float().mean()
            self._success_buffer_idx = (
                self._success_buffer_idx + 1
            ) % self.cfg.success_window_size

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        ee_pos_w = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat_w = self._ee_frame.data.target_quat_w[..., 0, :]
        target_pos_w, target_quat_w = self._target_pose_w()
        min_arm_dist = self._arm_distance()
        position_error = torch.norm(ee_pos_w - target_pos_w, p=2, dim=-1)
        orientation_error = math_utils.quat_error_magnitude(ee_quat_w, target_quat_w)
        joint_vel = torch.norm(
            self._robot.data.joint_vel[:, self._joint_indices], p=2, dim=-1
        )
        success = (
            (position_error < self.cfg.success_position_threshold)
            & (orientation_error < self.cfg.success_orientation_threshold)
            & (joint_vel < self.cfg.success_velocity_threshold)
            & (min_arm_dist > self.cfg.success_arm_clearance)
        )
        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        reset_ids = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.tensor(env_ids, device=self.device, dtype=torch.long)
        )
        self._waypoint_pose_b[reset_ids, :3] = self._target_poses[reset_ids, :3]
        self._waypoint_pose_b[reset_ids, 3:7] = self._target_poses[reset_ids, 3:7]
        self.actions[reset_ids] = 0.0
        self._effective_high_level_actions[reset_ids] = 0.0
        self._previous_high_level_actions[reset_ids] = 0.0
        self._waypoint_hold_count[reset_ids] = 0
        self._robot_dof_targets[reset_ids] = self._robot.data.joint_pos[reset_ids][
            :, self._joint_indices
        ]

        target_pos_w, _ = self._target_pose_w()
        ee_pos_w = self._ee_frame.data.target_pos_w[..., 0, :]
        self._last_target_distance[reset_ids] = torch.norm(
            ee_pos_w[reset_ids] - target_pos_w[reset_ids], dim=-1
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                goal_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/hierarchical_episode_goal",
                    markers={
                        "goal": sim_utils.SphereCfg(
                            radius=self.cfg.debug_goal_marker_radius,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 0.9, 0.2), roughness=0.5
                            ),
                        )
                    },
                )
                self.goal_pos_visualizer = VisualizationMarkers(goal_marker_cfg)
            if not hasattr(self, "waypoint_pos_visualizer"):
                waypoint_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/hierarchical_waypoint",
                    markers={
                        "waypoint": sim_utils.SphereCfg(
                            radius=self.cfg.debug_waypoint_marker_radius,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.1, 0.35, 1.0), roughness=0.5
                            ),
                        )
                    },
                )
                self.waypoint_pos_visualizer = VisualizationMarkers(waypoint_marker_cfg)

            self.goal_pos_visualizer.set_visibility(True)
            self.waypoint_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "waypoint_pos_visualizer"):
                self.waypoint_pos_visualizer.set_visibility(False)

    def _update_debug_visualization(self):
        if (
            not self.cfg.debug_vis
            or not hasattr(self, "goal_pos_visualizer")
            or not hasattr(self, "waypoint_pos_visualizer")
        ):
            return
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]

        goal_pos_w, _ = self._target_pose_w()
        waypoint_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, self._waypoint_pose_b[:, :3]
        )
        self.goal_pos_visualizer.visualize(translations=goal_pos_w)
        self.waypoint_pos_visualizer.visualize(translations=waypoint_pos_w)


def create_waypoint_low_level_env(
    cfg: UR5WaypointLowLevelEnvCfg | None = None,
    render_mode: str | None = None,
    **kwargs,
) -> UR5WaypointLowLevelEnv:
    if cfg is None:
        cfg = UR5WaypointLowLevelEnvCfg()
    return UR5WaypointLowLevelEnv(cfg, render_mode=render_mode, **kwargs)


def create_hierarchical_gray_depth_env(
    cfg: UR5HierarchicalGrayDepthEnvCfg | None = None,
    render_mode: str | None = None,
    **kwargs,
) -> UR5HierarchicalGrayDepthEnv:
    if cfg is None:
        cfg = UR5HierarchicalGrayDepthEnvCfg()
    return UR5HierarchicalGrayDepthEnv(cfg, render_mode=render_mode, **kwargs)
