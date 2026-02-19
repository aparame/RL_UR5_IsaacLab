from __future__ import annotations

import torch
import numpy as np
import gymnasium as gym
import os

from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .assets. import UR5_GRIPPER_CFG


@configclass
class UR5CalibrationEnvCfg(DirectRLEnvCfg):
    # Basic settings
    episode_length_s = 1000.0  # Long episode, we just need one frame really
    decimation = 4
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))  # Dummy
    observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))  # Dummy
    state_space = 0

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
    )

    # Robot
    robot_cfg: ArticulationCfg = UR5_GRIPPER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Table
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/adi2440/Desktop/RL_UR5_IsaacLab/source/RL_UR5/RL_UR5/tasks/direct/rl_ur5/assets/table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, -0.0234), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # White plane configuration
    white_plane_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/white_plane",
        spawn=sim_utils.CuboidCfg(
            size=(0.7, 2.5, 0.01),
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
            pos=(0.15001, -0.2, 0.93), rot=(0.50000, 0.50000, 0.50000, 0.50000)
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
            pos=(0.15001, 0.2, 0.93), rot=(0.50000, 0.50000, 0.50000, 0.50000)
        ),
    )

    # Depth Camera (Same as Reference)
    tiled_camera_depth: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CameraDepth",
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.82706,
            focus_distance=30.0,
            horizontal_aperture=5.229,
            vertical_aperture=2.942,
            clipping_range=(0.3, 20.0),  # Matches reference
        ),
        width=960,
        height=540,
        offset=TiledCameraCfg.OffsetCfg(  # Matches reference
            pos=(1.27, -0.06, 1.143),
            rot=(0.59637, 0.37993, 0.37993, 0.59637),
            convention="opengl",
        ),
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,  # Only need 1 for calibration
        env_spacing=4.0,
        replicate_physics=True,
    )

    # Reset Pose (Matches Reference)
    robot_base_pose = [-0.610865, -0.858, 1.402, -2.185, -1.57, 1.57]


class UR5CalibrationEnv(DirectRLEnv):
    cfg: UR5CalibrationEnvCfg

    def __init__(
        self, cfg: UR5CalibrationEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_indices, _ = self._robot.find_joints(
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
        )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self._table = RigidObject(self.cfg.table_cfg)
        self._white_plane = RigidObject(self.cfg.white_plane_cfg)
        self._clemson_plane = RigidObject(self.cfg.clemson_plane_cfg)
        self._i2r_plane = RigidObject(self.cfg.i2r_plane_cfg)
        self._tiled_camera_depth = TiledCamera(self.cfg.tiled_camera_depth)

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera_depth"] = self._tiled_camera_depth
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["white_plane"] = self._white_plane
        self.scene.rigid_objects["clemson_plane"] = self._clemson_plane
        self.scene.rigid_objects["i2r_plane"] = self._i2r_plane

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1600.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/DomeLight", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Static robot - maintain home pose
        # We enforce this every step because gravity might pull it down otherwise
        # although checking "kinematic_enabled" usually applies to rigid bodies,
        # for articulations we assume we control positions.

        # Target = Home Pose
        targets = torch.tensor([self.cfg.robot_base_pose], device=self.device).repeat(
            self.num_envs, 1
        )
        self._robot.set_joint_position_target(targets, joint_ids=self._joint_indices)

    def _apply_action(self) -> None:
        pass  # Actions handled in pre_physics using static targets

    def _get_observations(self) -> dict:
        # Return depth map
        depth = self._tiled_camera_depth.data.output["distance_to_image_plane"]
        # Shape: (num_envs, height, width, 1) -> (num_envs, height, width)
        return {"depth": depth.squeeze(-1)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        ), torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # Reset Robot to Home Pose
        if env_ids is None:
            env_ids = slice(None)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        # Initial pose from config
        targets = torch.tensor(self.cfg.robot_base_pose, device=self.device)
        # Verify indices match (assuming default order matches UR5 definition, but safer to use joint_names if doing robustly)
        # Here we assume standard UR5 joint order for the list provided in config

        # Manually set the joints corresponding to the list
        # We need to map our list to the correct indices in default_joint_pos
        # But wait, self._robot.set_joint_position does it by index.

        # Let's just force the positions
        # Create full joint pos vector (including gripper)
        # The provided pose is 6 DOF
        # UR5 + Gripper usually has more joints. The config provided has indices for the arm.

        pos = self._robot.data.default_joint_pos[env_ids].clone()
        pos[:, self._joint_indices] = targets

        self._robot.write_joint_state_to_sim(
            pos, torch.zeros_like(pos), env_ids=env_ids
        )
