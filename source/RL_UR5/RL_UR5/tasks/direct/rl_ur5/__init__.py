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
    id="UR5-Simple-PPO",
    entry_point=f"{__name__}.simple_pose_tracking_direct:SimpleCameraPoseTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simple_pose_tracking_direct:SimpleCameraPoseTrackingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:PPO_skrl_simple.yaml",
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
