# tasks/__init__.py

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-UR5-HuberDirectObj-PPO",
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

