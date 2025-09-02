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
    id="Isaac-UR5-HuberModObj-PPO",
    entry_point=f"{__name__}.huber_obj_direct_modified:ObjCameraPoseTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.huber_obj_direct_modified:ObjCameraPoseTrackingDirectEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:PPO_skrl_camera.yaml",
    },
)


gym.register(
    id="Isaac-UR5-DirectObjCamera-DR",
    entry_point=f"{__name__}.DR_obj_camera_direct:ObjCameraPoseTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.DR_obj_camera_direct:ObjCameraPoseTrackingDirectEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:PPO_skrl_camera.yaml",
    },
)

gym.register(
    id="Isaac-UR5-DirectObjCamera-PPO",
    entry_point=f"{__name__}.obj_camera_direct:ObjCameraPoseTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obj_camera_direct:ObjCameraPoseTrackingDirectEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:PPO_skrl_camera.yaml",
    },
)

gym.register(
    id="Isaac-UR5-DirectObjCamera-SAC",
    entry_point=f"{__name__}.huber_obj_direct:ObjCameraPoseTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.huber_obj_direct:ObjCameraPoseTrackingDirectEnv",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:SAC_skrl_camera.yaml",
    },
)


gym.register(
    id="Isaac-UR5-DirectPoseTracking-PPO",
    entry_point=f"{__name__}.simple_pose_tracking_direct:SimpleCameraPoseTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simple_pose_tracking_direct:SimpleCameraPoseTrackingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera.yaml",
    },
)


gym.register(
    id="Isaac-UR5-CDF-PPO",
    entry_point=f"{__name__}.ur5_sphere_cdf_env:SphereObstacleCDFEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_sphere_cdf_env:SphereObstacleCDFEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:PPO_skrl_CDF.yaml",
    },
)