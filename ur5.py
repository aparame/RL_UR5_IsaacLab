# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UR5 robot."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


# USD path with proper resolution for cross-platform compatibility
USD_PATH = "/home/adi2440/Desktop/ur5_isaacsim/usd/ur5_moveit.usd"

# Create custom UR5 robot configuration
UR5_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
            disable_gravity=True,
        ),
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=0,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=True,  # Change to True to make sure we copy the articulation root
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),  # Position the base of the robot
        joint_pos={
            "shoulder_pan_joint": -0.568,
            "shoulder_lift_joint": -0.658,
            "elbow_joint":1.602,
            "wrist_1_joint": -2.585,
            "wrist_2_joint": -1.6060665,
            "wrist_3_joint": -1.64142667,
            "robotiq_85_left_knuckle_joint": 0.0
        }
    ),
    actuators={
        "arm_actuator": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint"
            ],
            velocity_limit=50.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "gripper_actuator": ImplicitActuatorCfg(
            joint_names_expr=["robotiq_85_left_knuckle_joint"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
)
