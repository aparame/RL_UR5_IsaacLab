# tasks/__init__.py

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="UR5-Calibration-Depth",
    entry_point=f"{__name__}.ur5_calibration_env:UR5CalibrationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_calibration_env:UR5CalibrationEnvCfg",
    },
)

gym.register(
    id="UR5-Waypoint-LowLevel-PPO",
    entry_point=f"{__name__}.huber_obj_hierarchical_gray_depth:UR5WaypointLowLevelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.huber_obj_hierarchical_gray_depth:UR5WaypointLowLevelEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:PPO_skrl_waypoint_low_level.yaml",
    },
)

gym.register(
    id="UR5-Hierarchical-Depth-PPO",
    entry_point=f"{__name__}.huber_obj_hierarchical_gray_depth:UR5HierarchicalGrayDepthEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.huber_obj_hierarchical_gray_depth:UR5HierarchicalGrayDepthEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:PPO_skrl_hierarchical_gray_depth.yaml",
    },
)
