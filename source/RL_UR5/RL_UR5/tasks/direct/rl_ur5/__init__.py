# tasks/__init__.py

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="UR5-RGB-PPO",
    entry_point=f"{__name__}.huber_obj_direct:ObjCameraPoseTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.huber_obj_direct:ObjCameraPoseTrackingDirectEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:PPO_skrl_camera.yaml",
    },
)

gym.register(
    id="UR5-Depth-PPO",
    entry_point=f"{__name__}.huber_obj_direct_gray_depth:ObjCameraGrayDepthPoseTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.huber_obj_direct_gray_depth:ObjCameraGrayDepthPoseTrackingDirectEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:PPO_skrl_camera_gray_depth.yaml",
    },
)

gym.register(
    id="UR5-Depth-SAC",
    entry_point=f"{__name__}.huber_obj_direct_gray_depth:ObjCameraGrayDepthPoseTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.huber_obj_direct_gray_depth:ObjCameraGrayDepthPoseTrackingDirectEnvCfg",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:SAC_skrl_camera_gray_depth.yaml",
    },
)

gym.register(
    id="UR5-Depth-DR-PPO",
    entry_point=f"{__name__}.huber_obj_direct_depth_dr:ObjCameraGrayDepthDRPoseTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.huber_obj_direct_depth_dr:ObjCameraGrayDepthDRPoseTrackingDirectEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:PPO_skrl_camera_gray_depth.yaml",
    },
)


gym.register(
    id="UR5-Simple-PPO",
    entry_point=f"{__name__}.simple_pose_tracking_direct:SimpleCameraPoseTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simple_pose_tracking_direct:SimpleCameraPoseTrackingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:PPO_skrl_camera.yaml",
    },
)

gym.register(
    id="UR5-Calibration-Depth",
    entry_point=f"{__name__}.ur5_calibration_env:UR5CalibrationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_calibration_env:UR5CalibrationEnvCfg",
    },
)

gym.register(
    id="UR5-AGAN-Data-Collection-PPO",
    entry_point=f"{__name__}.agan_data_collection_direct_gray_depth:AGANDataCollectionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.agan_data_collection_direct_gray_depth:AGANDataCollectionEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:PPO_skrl_camera_gray_depth.yaml",
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
