from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg  
from isaaclab_assets.robots.ur5 import UR5_GRIPPER_CFG
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from . import mdp


    
marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/FrameTransformer"


##
# Scene definition
##

@configclass
class PoseTrackingSceneCfg(InteractiveSceneCfg):
    """Configuration for a UR5 pick and place scene."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    # UR5 Robot
    robot = UR5_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


    # Key change: Move the frame transformer creation after the robot is created
    # This will be set up in the post_reset phase instead
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.12, 0.0, 0.0],
                    ),
                ),
            ],
        )


    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/adi2440/Desktop/ur5_isaacsim/usd/table.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.7, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 0.0)),
    )

    
    # # CORRECT STRUCTURE - Change to this
    # tiled_camera_left: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/camera_left",  # Move prim_path here
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=2.12,
    #         focus_distance=28.0,
    #         horizontal_aperture=5.76,
    #         vertical_aperture=3.24,
    #         clipping_range=(0.1, 1000.0)
    #     ),
    #     width=224,
    #     height=224,
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(1.27, -0.06, 1.143),
    #         rot=(0.62933, 0.32239, 0.32239, 0.62933),
    #         convention="world"
    #     )
    # )

    # tiled_camera_right: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/camera_right",  # Move prim_path here
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=2.12,
    #         focus_distance=28.0,
    #         horizontal_aperture=5.76,
    #         vertical_aperture=3.24,
    #         clipping_range=(0.1, 1000.0)
    #     ),
    #     width=224,
    #     height=224,
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(1.27, 0.06, 1.143),
    #         rot=(0.62933, 0.32239, 0.32239, 0.62933),
    #         convention="world"
    #     )
    # )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
    )

##
# MDP settings
##

## Commands class is for defining the final object location
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    tracking_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="ee_link",  
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.7), pos_y=(-0.6, 0.6), pos_z=(-0.1, 0.2), roll=(0.0, 0.0), pitch=(1.57, 1.57), yaw=(0.0, 0.0)
        ),
    )



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Joint position control for UR5 robot
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        scale=0.5,  # Scale for each joint
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Joint state observations - using joint_names instead of joint_ids
        joint_positions = ObsTerm(
            func=mdp.joint_pos,noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot",
                                                joint_names=["shoulder_pan_joint",
                                                            "shoulder_lift_joint",
                                                            "elbow_joint",
                                                            "wrist_1_joint",
                                                            "wrist_2_joint",
                                                            "wrist_3_joint"]
                                                )
                    }
        )
        
        # Joint velocity observations - also using joint_names
        joint_velocities = ObsTerm(
            func=mdp.joint_vel,noise=Unoise(n_min=-0.001, n_max=0.001),
            params={"asset_cfg": SceneEntityCfg("robot", 
                                                joint_names=["shoulder_pan_joint",
                                                            "shoulder_lift_joint",
                                                            "elbow_joint",
                                                            "wrist_1_joint",
                                                            "wrist_2_joint",
                                                            "wrist_3_joint"]
                                                )
                    }
        )
        
        # End-effector pose
        # ee_pose = ObsTerm(func=mdp.end_effector_pose)
    
        
        # Target position (for the end-effector)
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "tracking_pose"})
        print('target_position', target_position)

        # Actions
        # actions = ObsTerm(func=mdp.last_action)

        
        def __post_init__(self):
            self.concatenate_terms = True
    
    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""
                # Camera images - updated for clarity
        # camera_images_left = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("tiled_camera_left"), "data_type": "rgb"},
        # )

        # camera_images_right = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("tiled_camera_right"), "data_type": "rgb"},
        # )



    # observation groups
    policy: PolicyCfg = PolicyCfg()

    # rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset robot joints with random noise
    reset_robot = EventTerm(
        func=mdp.reset_robot_pose_with_noise,
        mode="reset",
        params={
            'base_pose': [-0.71055204, -1.3046993,  1.9, -2.23, -1.59000665,  1.76992667],
            'noise_range': 0.01,  # Start with modest noise, increase as training progresses
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

   # Distance-based reward to approach the tracking pose
    distance_to_tracking_pose = RewTerm(
        func=mdp.position_command_error,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "tracking_pose"},
    )

    distance_to_tracking_pose_tanh = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.1, "command_name": "tracking_pose"},
    )

    # In the RewardsCfg class in rl_ur5_env_cfg.py
    orientation_reward = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.5,  # Adjust weight as needed
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),"command_name": "tracking_pose",
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.00001)
    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=0.00001,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    
    # In the RewardsCfg class in rl_ur5_env_cfg.py
    # joint_torques_penalty = RewTerm(
    #     func=mdp.joint_torques_l2,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    #     weight=0.000001,  # Adjust weight as needed
    # )

    # # In RewardsCfg class
    # success_reward = RewTerm(
    #     func=mdp.pose_tracking_success,
    #     weight=10.0,  # Large positive reward for success
    #     params={
    #         "position_threshold": 0.05,
    #         "orientation_threshold": 0.1,
    #         "velocity_threshold": 1.0,
    #         "command_name": "tracking_pose",
    #     },
    # )

        # NEW: End-effector table collision prevention
    ee_frame_table_collision = RewTerm(
        func=mdp.ee_frame_table_collision,
        weight=-1.0, 
        params={
            "table_height": 0.77,  # 0.77 from your thresholds.py
            "safety_margin": 0.05,  # 5cm safety margin above table
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Task success (end-effector at target pose)
    task_success = DoneTerm(
        func=mdp.pose_tracking_success,
        params={
            "position_threshold": 0.05,
            "orientation_threshold": 0.1,
            "velocity_threshold": 0.05,
            "torque_threshold": 1.0,
            "command_name": "tracking_pose",
        },
    )

    
    # # NEW: End-effector table collision prevention
    # ee_frame_table_collision = DoneTerm(
    #     func=mdp.ee_frame_table_collision,
    #     params={
    #         "table_height": 0.77,  # 0.77 from your thresholds.py
    #         "safety_margin": 0.05,  # 5cm safety margin above table
    #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    #     },
    # )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    joint_torque = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_torques_penalty", "weight": -0.0001, "num_steps": 1000}
    )

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.001, "num_steps": 1000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 1000}

    )


##
# Environment configuration
##

@configclass
class PoseTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for UR5 pick and place task."""
    
    # Scene settings
    scene: PoseTrackingSceneCfg = PoseTrackingSceneCfg(num_envs=8, env_spacing=4.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()
    curriculum = None

    # Unused managers
    commands: CommandsCfg = CommandsCfg()  # Add the command configuration


    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 4.0  # Longer episodes for pick and place


        # make a smaller scene for play
        self.scene.num_envs = 8
        self.scene.env_spacing = 4.0


        self.sim.dt = 1 / 120   #120Hz
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
