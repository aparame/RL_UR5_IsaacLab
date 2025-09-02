from dataclasses import MISSING
import torch
from typing import TYPE_CHECKING
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
# REMOVED: from isaaclab.managers import RecorderTermCfg as RecTerm  # This doesn't exist
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


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


##
# Scene definition
##

@configclass
class ObjCameraPoseTrackingSceneCfg(InteractiveSceneCfg):
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
    
    # White mesh plane - visual only (non-interactive)
    white_plane = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/white_plane",
        spawn=sim_utils.CuboidCfg(
            size=(0.42112, 2.81, 0.01),  # Using scale as size (making it thin like a plane)
            rigid_props=None,  # No physics - visual only
            mass_props=None,   # No mass - visual only
            collision_props=None,  # No collision - visual only
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),  # White color
                metallic=0.0,   # Non-metallic
                roughness=0.1,  # Smooth surface
                opacity=1.0     # Fully opaque
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.2, 0.0, 0.85925),  # Specified position
            rot=(0.70711, 0.0, 0.70711, 0.0),  # Specified orientation (w,x,y,z)
        ),
    )

    # Red cube - plastic material
    red_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/red_cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.0572, 0.0635, 0.191),  # Dimensions in meters
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 10 grams
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Red
                metallic=0.0,  # Non-metallic for plastic
                roughness=0.5  # Medium roughness for plastic
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.87),  # Will be randomized during reset
        ),
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
    #         convention="opengl"
    #     )
    # )

    tiled_camera_right: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/camera_right",  # Move prim_path here
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.208,
            focus_distance=28.0,
            horizontal_aperture=5.76,
            vertical_aperture=3.24,
            clipping_range=(0.1, 1000.0)
        ),
        width=224,
        height=224,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.17, 0.06, 1.143),
            rot=( 0.62933,0.32239, 0.32239,0.62933),
            convention="opengl"
        )
    )

    # UPDATED LIGHTING SETUP - Multiple lights creating a default light rig
    # Option 1: Multiple directional lights (simulates a typical 3-point lighting setup)
    key_light = AssetBaseCfg(
        prim_path="/World/Lights/KeyLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=2000.0,
            color=(1.0, 1.0, 0.9),  # Slightly warm white
            angle=0.53,  # Sun-like angular diameter
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.0, 2.0, 3.0),
            rot=(0.683, 0.183, 0.183, 0.683)  # Looking down and towards scene
        ),
    )
    
    fill_light = AssetBaseCfg(
        prim_path="/World/Lights/FillLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=800.0,
            color=(0.9, 0.9, 1.0),  # Slightly cool white
            angle=0.53,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-1.5, 1.5, 2.0),
            rot=(0.5, -0.5, 0.5, 0.5)  # Different angle
        ),
    )
    
    rim_light = AssetBaseCfg(
        prim_path="/World/Lights/RimLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=600.0,
            color=(1.0, 0.95, 0.8),  # Warm accent
            angle=0.53,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -2.0, 2.5),
            rot=(0.707, 0.0, 0.707, 0.0)  # Back lighting
        ),
    )

##
# MDP settings
##

## Action Term Low-Pass Filter
@configclass 
class AdvancedFilteredJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for advanced filtered joint position action term."""
    
    cutoff_freq: float = 5.0
    filter_order: int = 2
    damping_ratio: float = 0.707  # For second-order filter (0.707 = critically damped)

class AdvancedFilteredJointPositionAction(JointPositionAction):
    """Joint position action with advanced Butterworth filtering."""
    
    cfg: AdvancedFilteredJointPositionActionCfg
    
    def __init__(self, cfg: AdvancedFilteredJointPositionActionCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        
        self._dt = env.step_dt
        self._num_joints = len(self._joint_ids)
        self._num_envs = env.num_envs
        
        # Calculate filter coefficients
        if cfg.filter_order == 1:
            # First-order low-pass
            tau = 1.0 / (2.0 * math.pi * cfg.cutoff_freq)
            self._alpha = self._dt / (tau + self._dt)
            
            self._prev_output = torch.zeros((self._num_envs, self._num_joints), device=env.device)
            
        elif cfg.filter_order == 2:
            # Second-order Butterworth
            omega = 2.0 * math.pi * cfg.cutoff_freq
            k = omega * self._dt
            a1 = k * k
            a2 = k * 2.0 * cfg.damping_ratio
            a3 = a1 + a2 + 1.0
            
            self._b0 = a1 / a3
            self._b1 = 2.0 * a1 / a3  
            self._b2 = a1 / a3
            self._a1 = (2.0 * a1 - 2.0) / a3
            self._a2 = (a1 - a2 + 1.0) / a3
            
            # Filter memory
            self._x1 = torch.zeros((self._num_envs, self._num_joints), device=env.device)
            self._x2 = torch.zeros((self._num_envs, self._num_joints), device=env.device)
            self._y1 = torch.zeros((self._num_envs, self._num_joints), device=env.device)
            self._y2 = torch.zeros((self._num_envs, self._num_joints), device=env.device)
        
        print(f"Initialized advanced filter: order={cfg.filter_order}, freq={cfg.cutoff_freq}Hz")
    
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._x1.device if hasattr(self, '_x1') else self._prev_output.device)
        
        # Reset filter states
        if self.cfg.filter_order == 1:
            self._prev_output[env_ids] = 0.0
        elif self.cfg.filter_order == 2:
            self._x1[env_ids] = 0.0
            self._x2[env_ids] = 0.0
            self._y1[env_ids] = 0.0
            self._y2[env_ids] = 0.0
    
    def apply_actions(self, actions: torch.Tensor) -> None:
        """Apply Butterworth filtered actions."""
        
        if self.cfg.filter_order == 1:
            # First-order filter
            filtered_actions = self._alpha * actions + (1.0 - self._alpha) * self._prev_output
            self._prev_output = filtered_actions.clone()
            
        elif self.cfg.filter_order == 2:
            # Second-order Butterworth filter
            # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
            filtered_actions = (self._b0 * actions + 
                              self._b1 * self._x1 + 
                              self._b2 * self._x2 -
                              self._a1 * self._y1 - 
                              self._a2 * self._y2)
            
            # Update filter memory
            self._x2 = self._x1.clone()
            self._x1 = actions.clone()
            self._y2 = self._y1.clone()
            self._y1 = filtered_actions.clone()
        else:
            filtered_actions = actions
        
        super().apply_actions(filtered_actions)

## Commands class is for defining the final object location
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    tracking_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="ee_link",  
        resampling_time_range=(3.0, 3.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.7), pos_y=(-0.5, 0.5), pos_z=(-0.3, 0.0), roll=(0.0, 0.0), pitch=(1.57, 1.57), yaw=(0.0, 0.0)
        ),
    )



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Joint position control for UR5 robot with low-pass filtering
    arm_action = AdvancedFilteredJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        scale=0.5,
        use_default_offset=True,
        # Filter parameters to reduce jittering
        cutoff_freq=2.0,    # Hz - Lower values = more smoothing, higher values = more responsive
        filter_order=2,     # 1 = simple first-order filter, 2 = second-order (more aggressive)
        damping_ratio=0.707,  # For second-order filter (0.707 = critically damped)
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
        

        camera_images_right = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera_right"), "data_type": "rgb", "normalize": True,
                    "top_crop": 50, "bottom_crop": 70, "target_size": (80, 100)},
             # Normalize images to [0, 1]
        )

        # Actions
        # actions = ObsTerm(func=mdp.last_action)

        
        def __post_init__(self):
            self.concatenate_terms = True
            self.concatenate_dim = 2
    
    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""
                # Camera images - updated for clarity
        # camera_images_left = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("tiled_camera_left"), "data_type": "rgb"},
        # )





    # observation groups
    policy: PolicyCfg = PolicyCfg()

    # policy: RGBCameraPolicyCfg = RGBCameraPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset robot joints with random noise
    reset_robot = EventTerm(
        func=mdp.reset_robot_pose_with_noise,
        mode="reset",
        params={
            'base_pose': [-0.568, -0.658,  1.602, -2.585, -1.6060665,  -1.64142667],
            'noise_range': 0.01,  # Start with modest noise, increase as training progresses
        },
    )

    # Reset dynamic obstacle position at episode start
    reset_obstacle = EventTerm(
        func=mdp.reset_dynamic_obstacle,
        mode="reset",
        params={
            "obstacle_cfg": SceneEntityCfg("red_cube"),
            "reset_bounds": {
                "x_min": 0.4, 
                "x_max": 0.7, 
                "y_min": -0.4, 
                "y_max": 0.4, 
                "z": 0.87
            },
        },
    )
    
    # NEW: Reset robot when stuck at table
    unstuck_robot = EventTerm(
        func=mdp.reset_robot_when_stuck_at_table,
        mode="interval",
        interval_range_s=(0.05, 0.05),  # Check every 0.1 seconds
        params={
            "table_height": 0.77,
            "safety_margin": 0.05,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "asset_cfg": SceneEntityCfg("robot"),
            "noise_range": 0.02,
        },
    )

    # Move dynamic obstacle during episode
    move_obstacle = EventTerm(
        func=mdp.move_dynamic_obstacle,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # Update every 0.02 seconds (50Hz)
        params={
            "obstacle_cfg": SceneEntityCfg("red_cube"),
            "movement_type": "random_smooth",
            "center_pos": [0.5, 0.0, 0.79],
            "radius": 0.75,
            "speed": 0.8,
            "height_variation": 0.0,
            "diagonal_bounds": {
                "x_min": 0.3, 
                "x_max": 0.6, 
                "y_min": -0.35, 
                "y_max": 0.35
            },
        },
    )

    start_recording = EventTerm(
        func=mdp.start_recording_on_reset,
        mode="reset",
        params={
            "recorder_config": {
                "enable_recording": True,  # Set to True to enable recording
                "output_dir": "./logs/object_camera_pose_tracking_recordings",
                "camera_name": "tiled_camera_right",
                "max_episodes": 200,
                "record_video": True,
                "record_robot_states": True,
                "record_actions": True,
            }
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

    # # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.0001,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    
    # In the RewardsCfg class in rl_ur5_env_cfg.py
    joint_torques_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-0.00001,  # Adjust weight as needed
    )

        # NEW: End-effector table collision prevention
    ee_frame_table_collision = RewTerm(
        func=mdp.ee_frame_table_collision,
        weight=-50.0, 
        params={
            "table_height": 0.77,  # 0.77 from your thresholds.py
            "safety_margin": 0.05,  # 5cm safety margin above table
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )


        # REPLACE the old obstacle avoidance with full arm version
    obstacle_avoidance_full_arm = RewTerm(
        func=mdp.obstacle_avoidance_penalty_full_arm,
        weight=1.0,
        params={
            "obstacle_cfg": SceneEntityCfg("red_cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "safe_distance": 0.25,
            "danger_distance": 0.08,
            "max_penalty": -3.0,
            # Custom body weights (optional)
            "body_weights": {
                "base_link": 0.1,
                "shoulder_link": 0.2,
                "upper_arm_link": 0.4,
                "forearm_link": 0.7,
                "wrist_1_link": 0.8,
                "wrist_2_link": 0.9,
                "wrist_3_link": 1.0,
                "ee_link": 1.0
            }
        },
    )
    
    # Alternative: Use the tanh version with minimum distance
    obstacle_avoidance_smooth_full_arm = RewTerm(
        func=mdp.obstacle_avoidance_penalty_tanh_full_arm,
        weight=2.0,
        params={
            "obstacle_cfg": SceneEntityCfg("red_cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "safe_distance": 0.3,
            "std": 0.1,
            "use_minimum_distance": True,  # Use most conservative approach
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

    # nan_observations = DoneTerm(
    #     func=mdp.nan_observation_termination,
    #     # No parameters needed
    # )
    

    # robot_instability = DoneTerm(
    #     func=mdp.robot_instability,
    #     params={
    #         "velocity_threshold": 30.0,  # Adjust based on your robot
    #         "torque_threshold": 30.0,   # Adjust based on your robot
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # joint_torque = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_torques_penalty", "weight": -0.0001, "num_steps": 1000}
    # )

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.001, "num_steps": 1000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 1000}
    # )


# NEW: Custom Recorder Configuration
@configclass  
class RecorderCfg:
    """Configuration for custom recording system."""
    
    # Enable/disable recording
    enable_recording: bool = True  # Set to True to enable automatic recording
    
    # Recording settings
    output_dir: str = "./logs/object_camera_pose_tracking_recordings"
    record_video: bool = True
    record_robot_states: bool = True  
    record_actions: bool = True
    
    # Video settings
    camera_name: str = "tiled_camera_right"
    video_fps: int = 30
    
    # Auto-recording settings
    max_episodes_to_record: int = 10  # Maximum episodes to auto-record
    record_every_n_episodes: int = 1  # Record every N episodes


##
# Environment configuration
##

@configclass
class ObjCameraPoseTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for UR5 pick and place task."""
    
    # Scene settings
    scene: ObjCameraPoseTrackingSceneCfg = ObjCameraPoseTrackingSceneCfg(num_envs=8, env_spacing=4.0)
    
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

    # NEW: Add custom recorder configuration  
    recorder: RecorderCfg = RecorderCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 6.0  # Longer episodes for pick and place

        # make a smaller scene for play
        self.scene.num_envs = 8
        self.scene.env_spacing = 4.0


        self.sim.dt = 1 / 120   #120Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.disable_contact_processing = False  # Enable contact processing