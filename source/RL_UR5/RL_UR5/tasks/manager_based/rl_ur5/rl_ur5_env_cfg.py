import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg  
from isaaclab_assets.robots.ur5 import UR5_GRIPPER_CFG
from . import mdp
from .mdp.thresholds import *

marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/FrameTransformer"


##
# Scene definition
##

@configclass
class RlUr5SceneCfg(InteractiveSceneCfg):
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
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.1, 0.0, 0.0],
                    ),
                ),
            ],
        )
    
    # red_cube_frame: FrameTransformerCfg = FrameTransformerCfg(
    #         prim_path="{ENV_REGEX_NS}/Robot/base_link",
    #         debug_vis=True,
    #         visualizer_cfg=marker_cfg,
    #         target_frames=[
    #             FrameTransformerCfg.FrameCfg(
    #                 prim_path="{ENV_REGEX_NS}/red_cube",
    #                 name="red_cube_frames",
    #                 offset=OffsetCfg(
    #                     pos=[0.0, 0.0, 0.0],
    #                 ),
    #             ),
    #         ],
    #     )


    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/adi2440/Desktop/ur5_isaacsim/usd/table.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.7, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Red cube - plastic material
    red_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/red_cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.0635,0.0286, 0.1082),  # Dimensions in meters
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),  # 10 grams
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Red
                metallic=0.0,  # Non-metallic for plastic
                roughness=0.5  # Medium roughness for plastic
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.4, 0.77),  # Will be randomized during reset
        ),
    )

    # Green cube - plastic material
    green_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/green_cube",
        spawn=sim_utils.CuboidCfg(
            size=( 0.0635,0.0286, 0.1082),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),  # 10 grams
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),  # Green
                metallic=0.0,  # Non-metallic for plastic
                roughness=0.5  # Medium roughness for plastic
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -0.4, 0.77),  # Will be randomized during reset
        ),
    )

    # Blue cube - plastic material
    blue_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/blue_cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.0635,0.0286, 0.1082),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),  # 10 grams
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),  # Blue
                metallic=0.0,  # Non-metallic for plastic
                roughness=0.5  # Medium roughness for plastic
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7, -0.4, 0.77),  # Will be randomized during reset
        ),
    )
    
    # # CORRECT STRUCTURE - Change to this
    # tiled_camera_left: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/camera_left",  # Move prim_path here
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=2.208,
    #         focus_distance=28.0,
    #         horizontal_aperture=5.76,
    #         vertical_aperture=3.24,
    #         clipping_range=(0.1, 1000.0)
    #     ),
    #     width=256,
    #     height=144,
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(1.2, -0.06, 1.2),
    #         rot=(0.3536, 0.3536, 0.1464, 0.8536),
    #         convention="world"
    #     )
    # )

    # tiled_camera_right: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/camera_right",  # Move prim_path here
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=2.208,
    #         focus_distance=28.0,
    #         horizontal_aperture=5.76,
    #         vertical_aperture=3.24,
    #         clipping_range=(0.1, 1000.0)
    #     ),
    #     width=256,
    #     height=144,
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(1.2, 0.06, 1.2),
    #         rot=(0.3536, 0.3536, 0.1464, 0.8536),
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

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="ee_link",  
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), 
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
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
        scale=0.25,  # Scale for each joint
        use_default_offset=True,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(            
        asset_name="robot",
        joint_names=["robotiq_85_left_knuckle_joint"],
        open_command_expr={"robotiq_85_left_knuckle_joint": 0.0},
        close_command_expr={"robotiq_85_left_knuckle_joint": 45.0},
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Joint state observations - using joint_names instead of joint_ids
        joint_positions = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", 
                                                joint_names=["shoulder_pan_joint",
                                                            "shoulder_lift_joint",
                                                            "elbow_joint",
                                                            "wrist_1_joint",
                                                            "wrist_2_joint",
                                                            "wrist_3_joint",
                                                            "robotiq_85_left_knuckle_joint"]
                                                )
                    }
        )
        
        # Joint velocity observations - also using joint_names
        joint_velocities = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", 
                                                joint_names=["shoulder_pan_joint",
                                                            "shoulder_lift_joint",
                                                            "elbow_joint",
                                                            "wrist_1_joint",
                                                            "wrist_2_joint",
                                                            "wrist_3_joint",
                                                            "robotiq_85_left_knuckle_joint"]
                                                )
                    }
        )
        
        # End-effector pose
        ee_pose = ObsTerm(func=mdp.end_effector_pose)
        
        # Cube positions
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("red_cube"),  # Will be determined by task_id
            },
        )
        # cube_positions = ObsTerm(func=mdp.cube_positions)
        
        # Target position (for placing the cube)
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        
        # Task ID (which cube to pick)
        task_id = ObsTerm(func=mdp.task_id)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    # @configclass
    # class RGBCameraPolicyCfg(ObsGroup):
    #     """Observations for policy group with RGB images."""
    #             # Camera images - updated for clarity
    #     camera_images_left = ObsTerm(
    #         func=mdp.image,
    #         params={"sensor_cfg": SceneEntityCfg("tiled_camera_left"), "data_type": "rgb"},
    #     )

    #     camera_images_right = ObsTerm(
    #         func=mdp.image,
    #         params={"sensor_cfg": SceneEntityCfg("tiled_camera_right"), "data_type": "rgb"},
    #     )

    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = False


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset robot joints using the working function
    reset_robot = EventTerm(
        func=mdp.reset_robot_pose_with_noise,
        mode="reset",
        params={
            'base_pose': [-0.2321, -1.8647, 1.9495, 0.8378, 1.5097, 0.0, 0.0],
            'noise_range': 0.05,  # Start with modest noise, increase as training progresses
        },
    )

    # debug_gripper = EventTerm(
    #     func=mdp.debug_gripper_state,
    #     mode="reset",  # This makes it run during every reset
    # )

    # test_gripper = EventTerm(
    #     func=mdp.test_gripper_functionality,
    #     mode="reset",  # Run during reset
    # )

    ## Randomize cube positions
    reset_cubes = EventTerm(
        func=mdp.reset_cube_positions,
        mode="reset",
        params={
            "cube_cfgs": [
                SceneEntityCfg(name="red_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
                SceneEntityCfg(name="green_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
                SceneEntityCfg(name="blue_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
            ],
            "min_distance": 0.3,  # Minimum distance between cubes
        },
    )
    
    # Randomly select target cube and set placement position from commands
    set_task = EventTerm(
        func=mdp.set_pick_and_place_task,
        mode="reset",
        params={
            "cube_cfgs": [
                SceneEntityCfg(name="red_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
                SceneEntityCfg(name="green_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
                SceneEntityCfg(name="blue_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
            ],
        },
    )



    #Think about adding noise to joint states for better generalization

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Dense reward for approaching the cube - helpful for initial learning
    # reaching_object = RewTerm(
    #     func=mdp.distance_to_target_cube, 
    #     weight=1.5,
    #     params={ "ee_frame_cfg": SceneEntityCfg("ee_frame")}
    # )

    reaching_object_fine = RewTerm(
        func=mdp.distance_to_target_cube_tanh, 
        weight=1.5,
        params={"std": 0.1, "ee_frame_cfg": SceneEntityCfg("ee_frame")}
    )
    
    # Sparse reward for grasping the cube
    grasp_reward = RewTerm(
        func=mdp.grasp_reward,
        weight=20.0,
        params={
            "gripper_threshold": GRIPPER_CLOSED_THRESHOLD,
            "distance_threshold": POSITION_THRESHOLD,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )
    
    # Sparse reward for lifting the cube above threshold
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        weight=50.0,
        params={
            "minimal_height": 0.1,  # Minimum height to consider lifted
            "object_cfg": SceneEntityCfg("red_cube"),  # Will be determined by task_id
        },
    )

    # Goal-oriented reward for moving lifted cube to target (coarse)
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        weight=5.0,
        params={
            "std": 0.3,
            "minimal_height": 0.1,
            "command_name": "object_pose",
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("red_cube"),  # Will be determined by task_id
        },
    )

    # Fine-grained reward for precise positioning
    object_goal_tracking_fine = RewTerm(
        func=mdp.object_goal_distance,
        weight=5.0,
        params={
            "std": 0.05,
            "minimal_height": 0.1,
            "command_name": "object_pose",
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("red_cube"),  # Will be determined by task_id
        },
    )
    
    # # Orientation alignment for proper grasping
    # orientation_reward = RewTerm(
    #     func=mdp.orientation_alignment_reward,
    #     weight=0.5,
    #     params={
    #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    #     },
    # )

    # Movement penalties - small negative weights to discourage excessive motion
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Penalize excessive torques
    # joint_torques_penalty = RewTerm(
    #     func=mdp.joint_torques_l2,
    #     weight=-0.0000005,
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Object Dropping
    red_cube_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": TABLE_HEIGHT + 0.005, "asset_cfg": SceneEntityCfg("red_cube")}
    )
    green_cube_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": TABLE_HEIGHT + 0.005, "asset_cfg": SceneEntityCfg("green_cube")}
    )
    blue_cube_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": TABLE_HEIGHT + 0.005, "asset_cfg": SceneEntityCfg("blue_cube")}
    )

    robot_instability = DoneTerm(
        func=mdp.robot_instability,
        params={
            "velocity_threshold": 50.0,  # Adjust based on your robot
            "torque_threshold": 50.0,   # Adjust based on your robot
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    object_reached_goal = DoneTerm(
        func=mdp.object_reached_goal,
        params={
            "command_name": "object_pose",
            "threshold": 0.02,
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("red_cube"),  # Will be determined by task_id
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": " action_rate", "weight": -1e-1, "num_steps": 1200}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 1200}
    )

    orientation_reward_term = CurrTerm(
        func = mdp.modify_reward_weight, params={'term_name': 'orientation_reward', 'weight': 0.01, 'num_steps': 200}
    )
##
# Environment configuration
##

@configclass
class RlUr5EnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for UR5 pick and place task."""
    
    # Scene settings
    scene: RlUr5SceneCfg = RlUr5SceneCfg(num_envs=8, env_spacing=4.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands: CommandsCfg = CommandsCfg()  # Add the command configuration
    # curriculum: CurriculumCfg = CurriculumCfg()  # Add the curriculum configuration
    curriculum = None  # Disable curriculum for now

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 5  # Longer episodes for pick and place

        # make a smaller scene for play
        self.scene.num_envs = 8
        self.scene.env_spacing = 4.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # Simulation settings

        self.sim.dt = 1 / 120  # Smaller timestep for more stability (was 1/60)
        self.sim.physx.num_position_iterations = 8  # Increase from default
        self.sim.physx.num_velocity_iterations = 2  # Increase from default
        self.sim.physx.contact_offset = 0.01
        self.sim.physx.rest_offset = 0.0
        self.sim.disable_contact_processing = False  # Enable contact processing