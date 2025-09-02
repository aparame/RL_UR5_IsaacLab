from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING, Dict, Any, List
import isaaclab.utils.math as math_utils
import random
from .thresholds import *  # Import all constants from thresholds


from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


"""
Observation functions
"""

def joint_positions(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get joint positions for the robot."""
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos

def joint_velocities(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get joint velocities for the robot."""
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel

def ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get end-effector position.
    
    Args:
        env: The RL environment instance
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: End-effector position [x, y, z]
    """
    # Extract the robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get end-effector body index
    ee_link_name = "ee_link"  # Using ee_link as the end-effector
    
    # Find the body index by name (compatible with IsaacLab 4.5)
    body_names = asset.body_names
    body_idx = body_names.index(ee_link_name) if ee_link_name in body_names else -1
    
    if body_idx == -1:
        raise ValueError(f"End-effector link '{ee_link_name}' not found in articulation body names")
    
    # Get body position
    ee_pos = asset.data.body_pos_w[:, body_idx, :]
    
    return ee_pos


def ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get end-effector orientation quaternion.
    
    Args:
        env: The RL environment instance
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: End-effector quaternion [qx, qy, qz, qw]
    """
    # Extract the robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get end-effector body index
    ee_link_name = "ee_link"  # Using ee_link as the end-effector
    
    # Find the body index by name (compatible with IsaacLab 4.5)
    body_names = asset.body_names
    body_idx = body_names.index(ee_link_name) if ee_link_name in body_names else -1
    
    if body_idx == -1:
        raise ValueError(f"End-effector link '{ee_link_name}' not found in articulation body names")
    
    # Get body orientation
    ee_quat = asset.data.body_quat_w[:, body_idx, :]
    
    return ee_quat


def end_effector_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get combined end-effector pose (position and orientation).
    
    Args:
        env: The RL environment instance
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: End-effector pose [x, y, z, qx, qy, qz, qw]
    """
    position = ee_pos(env, asset_cfg)
    quaternion = ee_quat(env, asset_cfg)
    
    # Combine into a single pose vector [x, y, z, qx, qy, qz, qw]
    pose = torch.cat([position, quaternion], dim=-1)
    
    return pose

"""
Updated functions for IsaacLab 4.5 compatibility
"""

def cube_positions(
    env: ManagerBasedRLEnv, 
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("green_cube"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("blue_cube"),
) -> torch.Tensor:
    """Get positions of the cubes relative to environment origins.
    
    Args:
        env: The RL environment instance
        cube_1_cfg: Configuration for the first cube
        cube_2_cfg: Configuration for the second cube
        cube_3_cfg: Configuration for the third cube
        
    Returns:
        torch.Tensor: Concatenated positions of all cubes, shape (batch_size, 9)
    """
    # Get cubes from the scene
    cube_1 = env.scene[cube_1_cfg.name]
    cube_2 = env.scene[cube_2_cfg.name]
    cube_3 = env.scene[cube_3_cfg.name]
    
    # Get positions (world frame)
    cube_1_pos_w = cube_1.data.root_pos_w
    cube_2_pos_w = cube_2.data.root_pos_w
    cube_3_pos_w = cube_3.data.root_pos_w
    
    # Adjust positions relative to environment origins
    cube_1_pos = cube_1_pos_w - env.scene.env_origins[:, 0:3]
    cube_2_pos = cube_2_pos_w - env.scene.env_origins[:, 0:3]
    cube_3_pos = cube_3_pos_w - env.scene.env_origins[:, 0:3]
    
    # Concatenate all positions
    all_positions = torch.cat((cube_1_pos, cube_2_pos, cube_3_pos), dim=1)
    
    return all_positions


def target_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the target position for placing the cube.
    
    Args:
        env: The RL environment instance
        
    Returns:
        torch.Tensor: Target positions for all environments
    """
    # Initialize tensor to hold target positions for all environments
    target_positions = torch.zeros((env.num_envs, 3), device=env.device)
    
    # Fill in target positions from task_info for each environment
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_positions[i] = env.task_info[i]["placement_position"]
    
    return target_positions

def target_cube_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the position of the target cube based on task_id.
    
    Args:
        env: The RL environment instance
        
    Returns:
        torch.Tensor: Position of the target cube for each environment
    """
    # Initialize tensor to hold target cube positions for all environments
    target_positions = torch.zeros((env.num_envs, 3), device=env.device)
    
    # Fill in target positions from task_info for each environment
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_cube_name = env.task_info[i]["target_cube"]
            cube = env.scene[target_cube_name]
            target_positions[i] = cube.data.root_pos_w[i, :3]
    
    return target_positions


def task_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the one-hot encoded task ID (which cube to pick).
    
    Args:
        env: The RL environment instance
        
    Returns:
        torch.Tensor: One-hot encoded task IDs for all environments
    """
    # Initialize tensor for task IDs
    batch_size = env.num_envs
    num_cubes = 3  # Assuming 3 cubes (red, green, blue)
    task_ids = torch.zeros((batch_size, num_cubes), device=env.device)
    
    # Fill in one-hot encoded task IDs
    for i in range(batch_size):
        if hasattr(env, "task_info") and i in env.task_info:
            cube_idx = env.task_info[i]["target_cube_idx"]
            task_ids[i, cube_idx] = 1.0
    
    return task_ids



def camera_images(env: ManagerBasedRLEnv, camera_cfg: SceneEntityCfg = SceneEntityCfg("camera")):
    """Get RGB images from the camera."""
    camera = env.scene.sensors[camera_cfg.name]
    
    # Get RGB data and normalize it
    rgb_data = camera.data.output["rgb"] / 255.0
    
    return rgb_data


"""
Reset functions
"""

@torch.no_grad()
def reset_robot_pose_with_noise(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    base_pose: list[float] | torch.Tensor | None = None,
    noise_range: float = 0.10,
    train_gripper: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Vectorised reset: sets the *same* home pose for all `env_ids` and adds
    uniform ±`noise_range` rad noise **on-device**.

    • Uses only `torch.rand` → deterministic with `torch.manual_seed`.  
    • `train_gripper=False` → 6-DoF arm only.  If `True`, the gripper joint
      is appended and driven like the others.
    """
    robot = env.scene[asset_cfg.name]

    # ------------- resolve joint list -------------
    joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint",     "wrist_2_joint",        "wrist_3_joint",
    ]
    if train_gripper:
        joint_names.append("robotiq_85_left_knuckle_joint")

    joint_ids = torch.as_tensor(
        [robot.joint_names.index(n) for n in joint_names],
        device=env.device,
        dtype=torch.long,
    )

    # ------------- base pose & noise --------------
    if base_pose is None:
        # same order as joint_names
        base_pose = [-0.2321, -2.0647, 1.9495, 0.8378, 1.5097, 0.0]
        if train_gripper:
            base_pose.append(0.0)

    base_pose = torch.tensor(base_pose, device=env.device)          #  (J,)
    noise     = (torch.rand((len(env_ids), len(base_pose)),
                            device=env.device) * 2 - 1) * noise_range

    q_pos = base_pose.unsqueeze(0) + noise                          #  (B,J)
    q_vel = torch.zeros_like(q_pos)

    # ----- 1) teleport the robot ----------------------------------------
    robot.write_joint_state_to_sim(
        q_pos, q_vel, joint_ids=joint_ids, env_ids=env_ids
    )

    # ----- 2) align controller targets ----------------------------------
    robot.set_joint_position_target(q_pos, joint_ids, env_ids)
    robot.set_joint_velocity_target(q_vel, joint_ids, env_ids)
    
    return {}

def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list

def reset_dynamic_obstacle(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    reset_bounds: dict = {"x_min": 0.4, "x_max": 0.6, "y_min": 0.0, "y_max": 0.3, "z": 0.85},
):
    """Reset the dynamic obstacle to a random position at the start of each episode.
    
    Args:
        env: The environment instance
        env_ids: Tensor of environment IDs to reset
        obstacle_cfg: Configuration for the obstacle object
        reset_bounds: Bounds for initial reset position
    """
    # Get the obstacle object
    obstacle = env.scene[obstacle_cfg.name]
    
    # Define pose range for sampling
    pose_range = {
        "x": (reset_bounds["x_min"], reset_bounds["x_max"]),
        "y": (reset_bounds["y_min"], reset_bounds["y_max"]),
        "z": (reset_bounds["z"], reset_bounds["z"]),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    
    # Reset position for each environment
    for cur_env in env_ids.tolist():
        # Sample a random pose
        pose_list = sample_object_poses(
            num_objects=1,
            min_separation=0.0,
            pose_range=pose_range,
            max_sample_tries=10
        )
        
        # Convert pose to tensor
        pose_tensor = torch.tensor([pose_list[0]], device=env.device)
        positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
        orientations = math_utils.quat_from_euler_xyz(
            pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
        )
        
        # Write pose to simulation
        obstacle.write_root_pose_to_sim(
            torch.cat([positions, orientations], dim=-1),
            env_ids=torch.tensor([cur_env], device=env.device)
        )
        
        # Reset velocities
        obstacle.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device)
        )
    
    return {}  
    

def move_dynamic_obstacle(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    movement_type: str = "circular",
    center_pos: list = [0.5, 0.25, 0.85],
    radius: float = 0.15,
    speed: float = 0.5,
    height_variation: float = 0.1,
    # New diagonal movement parameters
    diagonal_bounds: dict = {"x_min": -0.6, "x_max": -0.3, "y_min": -0.35, "y_max": 0.35},
    # Reset parameters
    reset_bounds: dict = {"x_min": 0.4, "x_max": 0.6, "y_min": 0.0, "y_max": 0.3, "z": 0.85},
):
    """Move the dynamic obstacle in a specified pattern, with embedded reset functionality.
    
    Args:
        env: The environment instance
        env_ids: Tensor of environment IDs to update
        obstacle_cfg: Configuration for the obstacle object
        movement_type: Type of movement ("circular", "linear", "diagonal", "random_smooth")
        center_pos: Center position for movement pattern
        radius: Radius for circular movements
        speed: Speed of movement
        height_variation: Vertical oscillation amplitude
        diagonal_bounds: Bounds for diagonal movement {"x_min", "x_max", "y_min", "y_max"}
        reset_bounds: Bounds for initial reset position {"x_min", "x_max", "y_min", "y_max", "z"}
    """
    # Get the obstacle object
    obstacle = env.scene[obstacle_cfg.name]
    
    # Check if this is being called during reset (episode_length_buf would be 0)
    is_reset = (env.episode_length_buf == 0).any()
    
    for i, env_id in enumerate(env_ids.tolist()):
        
        # Handle reset positioning
        if is_reset or env.episode_length_buf[env_id] == 0:
            # Random reset position within bounds
            x = random.uniform(reset_bounds["x_min"], reset_bounds["x_max"])
            y = random.uniform(reset_bounds["y_min"], reset_bounds["y_max"])
            z = reset_bounds["z"]
            
            # Create reset position
            reset_position = torch.tensor([x, y, z], device=env.device)
            reset_position_world = reset_position + env.scene.env_origins[env_id, :3]
            
            # Default orientation (no rotation)
            reset_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
            
            # Combine position and orientation
            reset_pose = torch.cat([reset_position_world, reset_quat])
            
            # Write to simulation
            obstacle.write_root_pose_to_sim(
                reset_pose.unsqueeze(0),
                env_ids=torch.tensor([env_id], device=env.device)
            )
            
            # Reset velocities
            obstacle.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([env_id], device=env.device)
            )
            
        else:
            # Handle dynamic movement during episode
            current_time = env.episode_length_buf[env_id].float() * env.step_dt
            time = current_time.item()
            
            if movement_type == "circular":
                # Circular motion in XY plane
                x = center_pos[0] + radius * math.cos(speed * time)
                y = center_pos[1] + radius * math.sin(speed * time)
                z = center_pos[2] + height_variation * math.sin(2 * speed * time)
                
            elif movement_type == "linear":
                # Linear back-and-forth motion
                x = center_pos[0] + radius * math.sin(speed * time)
                y = center_pos[1]
                z = center_pos[2] + height_variation * math.sin(2 * speed * time)
                
            elif movement_type == "diagonal":
                # Diagonal movement within specified bounds
                x_range = diagonal_bounds["x_max"] - diagonal_bounds["x_min"]
                y_range = diagonal_bounds["y_max"] - diagonal_bounds["y_min"]
                
                # Create diagonal movement using sine wave (0 to 1)
                t_normalized = (math.sin(speed * time) + 1) / 2  # Normalize to 0-1
                
                # Map to bounds
                x = diagonal_bounds["x_min"] + t_normalized * x_range
                y = diagonal_bounds["y_min"] + t_normalized * y_range
                z = center_pos[2] + height_variation * math.sin(2 * speed * time)
                
            elif movement_type == "random_smooth":
                # Smooth random movement using multiple sine waves
                x = center_pos[0] + 0.1 * (math.sin(speed * time) + 0.5 * math.sin(1.7 * speed * time))
                y = center_pos[1] + 0.1 * (math.cos(0.8 * speed * time) + 0.3 * math.cos(2.3 * speed * time))
                z = center_pos[2] + height_variation * math.sin(1.5 * speed * time)
            
            else:
                # Default to circular if unknown movement type
                x = center_pos[0] + radius * math.cos(speed * time)
                y = center_pos[1] + radius * math.sin(speed * time)
                z = center_pos[2] + height_variation * math.sin(2 * speed * time)
            
            # Create new position tensor
            new_position = torch.tensor([x, y, z], device=env.device)
            
            # Apply environment offset
            new_position_world = new_position + env.scene.env_origins[env_id, :3]
            
            # Keep current orientation (no rotation)
            current_quat = obstacle.data.root_quat_w[env_id, :]
            
            # Combine position and orientation
            new_pose = torch.cat([new_position_world, current_quat])
            
            # Write to simulation
            obstacle.write_root_pose_to_sim(
                new_pose.unsqueeze(0),
                env_ids=torch.tensor([env_id], device=env.device)
            )
            
            # Reset velocities to maintain smooth motion
            obstacle.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([env_id], device=env.device)
            )
    
    return {}


def set_pick_and_place_task(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_cfgs: list[SceneEntityCfg] = None,
):
    """Set up pick and place tasks for the environment.
    
    Args:
        env: The environment instance
        env_ids: Tensor of environment IDs to set tasks for
        command_cfg: Configuration for commands (target positions)
        cube_cfgs: List of configurations for the cubes
    """
    # Set default cube configurations if not provided
    if cube_cfgs is None:
        cube_cfgs = [
            SceneEntityCfg("red_cube"),
            SceneEntityCfg("green_cube"),
            SceneEntityCfg("blue_cube")
        ]
    
    # Initialize task info dictionary if it doesn't exist
    if not hasattr(env, "task_info"):
        env.task_info = {}
    
    # Get target positions from command_cfg if available
    target_positions = None
    if hasattr(env, "command_data") and "object_pose" in env.command_data:
        target_positions = env.command_data["object_pose"]
    
    # For each environment, assign a random task
    for cur_env in env_ids.tolist():
        # Randomly select which cube to pick
        target_cube_idx = random.randint(0, len(cube_cfgs) - 1)
        target_cube_name = cube_cfgs[target_cube_idx].name
        
        # Get placement position from target positions if available
        if target_positions is not None:
            placement_position = target_positions[cur_env, :3]
        else:
            # Sample a placement position as fallback
            pose_range = {
                "x": (0.3, 0.6),
                "y": (0.0, 0.2),
                "z": (0.78, 0.85),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
            placement_pose = sample_object_poses(
                num_objects=1,
                pose_range=pose_range
            )[0]
            placement_position = torch.tensor(
                placement_pose[:3], 
                device=env.device
            )
        
        # Store task information
        env.task_info[cur_env] = {
            "target_cube": target_cube_name,
            "target_cube_idx": target_cube_idx,
            "placement_position": placement_position,
        }
    
    return {}


def initialize_task_stages(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):
    """Initialize task stages for curriculum learning.
    
    Args:
        env: The environment instance
        env_ids: Tensor of environment IDs to initialize
    """
    # Initialize task stages dictionary if it doesn't exist
    if not hasattr(env, "task_stages"):
        env.task_stages = {}
    
    # Set initial stage for each environment
    for cur_env in env_ids.tolist():
        env.task_stages[cur_env] = 0  # Stage 0: Alignment above cube
    
    return {}


def update_task_stages(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """Update task stages based on subtask completion.
    
    Args:
        env: The environment instance
        env_ids: Tensor of environment IDs to update
    """
    # Skip if task stages not initialized
    if not hasattr(env, "task_stages") or not hasattr(env, "obs_data") or "subtask_terms" not in env.obs_data:
        return {}
    
    # Process each environment
    for i in env_ids.tolist():
        if i not in env.task_stages:
            continue
            
        current_stage = env.task_stages[i]
        
        # Check for stage transitions based on subtask completion
        if current_stage == 0:  # Alignment stage
            if env.obs_data["subtask_terms"]["alignment_complete"][i]:
                env.task_stages[i] = 1  # Move to grasp stage
                
        elif current_stage == 1:  # Grasp stage
            if env.obs_data["subtask_terms"]["cube_grasped"][i]:
                env.task_stages[i] = 2  # Move to placement stage
                
        elif current_stage == 2:  # Placement stage
            if env.obs_data["subtask_terms"]["cube_placed"][i]:
                env.task_stages[i] = 3  # Task complete
    
    return {}

"""
Subtask Cfg Functions

"""

def alignment_above_cube_complete(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    height_threshold: float = 0.8,
    alignment_threshold: float = 0.9,
) -> torch.Tensor:
    """Check if end-effector is aligned above the target cube.
    
    Args:
        env: The RL environment instance
        robot_cfg: Configuration for the robot
        ee_frame_cfg: Configuration for the end-effector frame
        height_threshold: Desired height above the cube
        alignment_threshold: Threshold for orientation alignment (0-1)
        
    Returns:
        torch.Tensor: Boolean tensor indicating alignment success
    """
    # Initialize result tensor
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Get end-effector position and orientation
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    
    for i in range(env.num_envs):
        # Skip if no task info
        if not hasattr(env, "task_info") or i not in env.task_info:
            continue
            
        target_info = env.task_info[i]
        target_cube_name = target_info["target_cube"]
        cube = env.scene[target_cube_name]
        cube_position = cube.data.root_pos_w[i, :3]
        cube_quat = cube.data.root_quat_w[i, :]
        
        # Calculate desired hover position
        target_position = cube_position.clone()
        target_position[2] += height_threshold
        
        # Check position
        hover_distance = torch.norm(ee_position[i] - target_position, p=2)
        
        # Check orientation alignment
        ee_rot_mat = math_utils.matrix_from_quat(ee_quat[i].unsqueeze(0)).squeeze(0)
        cube_rot_mat = math_utils.matrix_from_quat(cube_quat.unsqueeze(0)).squeeze(0)
        
        ee_x_axis = ee_rot_mat[:, 0]  # X-axis is first column
        ee_y_axis = ee_rot_mat[:, 1]  # Y-axis is second column
        
        # For cube
        cube_x_axis = cube_rot_mat[:, 0]  # Y-axis is second column
        cube_z_axis = cube_rot_mat[:, 2]  # Z-axis is third column
        
        # We want negative ee_x_axis to align with positive cube_z_axis
        # And negative ee_y_axis to align with positive cube_x_axis
        x_z_alignment = torch.abs(torch.dot(-ee_x_axis, cube_z_axis))
        y_y_alignment = torch.abs(torch.dot(-ee_y_axis, cube_x_axis))
        alignment = torch.sqrt(x_z_alignment * y_y_alignment)
        
        # Set result if both position and orientation conditions are met
        if hover_distance < 0.05 and alignment > alignment_threshold:
            result[i] = True
            
    return result

def cube_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    distance_threshold: float = 0.05,
    gripper_threshold: float = 25.0,
) -> torch.Tensor:
    """Check if the target cube is grasped.
    
    Args:
        env: The RL environment instance
        robot_cfg: Configuration for the robot
        ee_frame_cfg: Configuration for the end-effector frame
        distance_threshold: Maximum distance to consider "close"
        gripper_threshold: Minimum gripper position to consider "closed"
        
    Returns:
        torch.Tensor: Boolean tensor indicating grasp success
    """
    # Initialize result tensor
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Get end-effector position
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get robot
    robot = env.scene[robot_cfg.name]
    
    # Find gripper joint index
    joint_names = robot.joint_names
    gripper_joint_name = "robotiq_85_left_knuckle_joint"
    gripper_joint_idx = joint_names.index(gripper_joint_name) if gripper_joint_name in joint_names else -1
    
    if gripper_joint_idx == -1:
        return result
    
    # Get gripper position
    gripper_position = robot.data.joint_pos[:, gripper_joint_idx]
    
    for i in range(env.num_envs):
        # Skip if no task info
        if not hasattr(env, "task_info") or i not in env.task_info:
            continue
            
        target_info = env.task_info[i]
        target_cube_name = target_info["target_cube"]
        cube = env.scene[target_cube_name]
        cube_position = cube.data.root_pos_w[i, :3]
        
        # Calculate distance to cube
        distance = torch.norm(ee_position[i] - cube_position, p=2)
        
        # Check if close to cube and gripper is closed
        if distance < distance_threshold and gripper_position[i] > gripper_threshold:
            result[i] = True
            
    return result

def cube_placed_at_target(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    distance_threshold: float = 0.05,
    gripper_threshold: float = 5.0,
) -> torch.Tensor:
    """Check if the cube has been placed at the target location.
    
    Args:
        env: The RL environment instance
        robot_cfg: Configuration for the robot
        distance_threshold: Maximum distance to consider "at target"
        gripper_threshold: Maximum gripper position to consider "open"
        
    Returns:
        torch.Tensor: Boolean tensor indicating placement success
    """
    # Initialize result tensor
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Get robot
    robot = env.scene[robot_cfg.name]
    
    # Find gripper joint index
    joint_names = robot.joint_names
    gripper_joint_name = "robotiq_85_left_knuckle_joint"
    gripper_joint_idx = joint_names.index(gripper_joint_name) if gripper_joint_name in joint_names else -1
    
    if gripper_joint_idx == -1:
        return result
    
    # Get gripper position
    gripper_position = robot.data.joint_pos[:, gripper_joint_idx]
    
    for i in range(env.num_envs):
        # Skip if no task info
        if not hasattr(env, "task_info") or i not in env.task_info:
            continue
            
        target_info = env.task_info[i]
        target_cube_name = target_info["target_cube"]
        target_position = target_info.get("placement_position", None)
        
        # Skip if no placement position is defined
        if target_position is None:
            continue
            
        # Get cube position
        cube = env.scene[target_cube_name]
        cube_position = cube.data.root_pos_w[i, :3]
        
        # Calculate distance between cube and target
        distance = torch.norm(cube_position - target_position, p=2)
        
        # Check if cube is at target and gripper is open
        if distance < distance_threshold and gripper_position[i] < gripper_threshold:
            result[i] = True
            
    return result



def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("red_cube")
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    # Get robot for coordinate transformation
    robot = env.scene[robot_cfg.name]
    
    # Initialize output tensor
    object_pos_b = torch.zeros((env.num_envs, 3), device=env.device)
    
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            
            # Get cube position in world frame
            cube_pos_w = cube.data.root_pos_w[i, :3]
            
            # Transform to robot root frame
            pos_b, _ = math_utils.subtract_frame_transforms(
                robot.data.root_state_w[i, :3].unsqueeze(0),
                robot.data.root_state_w[i, 3:7].unsqueeze(0),
                cube_pos_w.unsqueeze(0)
            )
            object_pos_b[i] = pos_b.squeeze(0)
    
    return object_pos_b


def reset_robot_when_stuck_at_table(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    table_height: float = TABLE_HEIGHT,
    safety_margin: float = 0.05,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safe_poses: list = None,
    noise_range: float = 0.02,
):
    """Reset robot to a safe position when it gets stuck at the table.
    
    Args:
        env: The environment instance
        env_ids: Tensor of environment IDs to reset
        table_height: Height of the table surface
        safety_margin: Safety margin above table surface
        ee_frame_cfg: Configuration for the end-effector frame
        asset_cfg: Configuration for the robot
        safe_poses: List of safe joint poses to choose from
        noise_range: Random noise to add to joint positions
    """
    # Default safe poses (well above table)
    if safe_poses is None:
        safe_poses = [
            [-0.71055204, -1.3046993, 1.9, -2.23, -1.59000665, 1.76992667],  # Original safe pose
        ]
    
    # Get the robot and end-effector frame
    robot = env.scene[asset_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Check which environments have robots stuck at table
    ee_height = ee_position[:, 2]
    stuck_at_table = ee_height < (table_height + safety_margin)
    
    # Get environment IDs that are stuck
    stuck_env_ids = []
    for i, env_id in enumerate(env_ids.tolist()):
        if stuck_at_table[env_id]:
            stuck_env_ids.append(env_id)
    
    if len(stuck_env_ids) == 0:
        return {}
    
    # print(f"Resetting stuck robots in environments: {stuck_env_ids}")
    
    # Reset stuck robots to safe positions
    for env_id in stuck_env_ids:
        # Choose a random safe pose
        base_pose = random.choice(safe_poses)
        
        # Convert pose to tensor
        joint_pos_base = torch.tensor(base_pose, device=env.device)
        
        # Add random noise
        num_joints = len(base_pose)
        noise = torch.rand(num_joints, device=env.device) * 2 * noise_range - noise_range
        joint_pos = joint_pos_base + noise
        joint_vel = torch.zeros_like(joint_pos)
        
        # Get actuatable joint indices (exclude gripper if present)
        actuatable_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint", 
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        
        joint_indices = []
        for name in actuatable_joint_names:
            if name in robot.joint_names:
                joint_indices.append(robot.joint_names.index(name))
        
        # Reset robot to safe position
        robot.set_joint_position_target(
            joint_pos.unsqueeze(0), 
            joint_ids=joint_indices, 
            env_ids=torch.tensor([env_id], device=env.device)
        )
        robot.set_joint_velocity_target(
            joint_vel.unsqueeze(0), 
            joint_ids=joint_indices, 
            env_ids=torch.tensor([env_id], device=env.device)
        )
        robot.write_joint_state_to_sim(
            joint_pos.unsqueeze(0), 
            joint_vel.unsqueeze(0), 
            joint_ids=joint_indices, 
            env_ids=torch.tensor([env_id], device=env.device)
        )
    
    return {}


def start_recording_on_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    recorder_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Event function to initialize and manage recording on episode reset.
    
    Args:
        env: The RL environment instance
        env_ids: Environment IDs that are resetting
        recorder_config: Optional recorder configuration dictionary
        
    Returns:
        Empty dictionary
    """
    # Default recorder config
    default_config = {
        "enable_recording": False,
        "output_dir": "./logs/recordings",
        "camera_name": "tiled_camera_right",
        "max_episodes": 5,
        "record_video": True,
        "record_robot_states": True,
        "record_actions": True,
    }
    
    # Merge with provided config
    if recorder_config is not None:
        config = {**default_config, **recorder_config}
    else:
        config = default_config
    
    # Skip if recording is disabled
    if not config.get("enable_recording", False):
        return {}
    
    try:
        # Import recorder here to avoid circular imports
        from .recorder import EnvironmentRecorder, RecordingEventManager
        
        # Initialize recorder if not exists
        if not hasattr(env, '_recorder'):
            env._recorder = EnvironmentRecorder(
                env=env,
                output_dir=config["output_dir"],
                record_video=config["record_video"],
                record_robot_states=config["record_robot_states"],
                record_actions=config["record_actions"],
                camera_name=config["camera_name"]
            )
            env._recording_manager = RecordingEventManager(env._recorder)
            env._recording_manager.max_episodes = config["max_episodes"]
            print(f"[INFO] Initialized recorder - will record up to {config['max_episodes']} episodes")
            print(f"[INFO] Output directory: {config['output_dir']}")
        
        # Handle recording on reset
        if hasattr(env, '_recording_manager'):
            env._recording_manager.on_reset(env, env_ids)
        
    except ImportError as e:
        print(f"[WARNING] Could not initialize recorder: {e}")
        print("[WARNING] Continuing without recording...")
    except Exception as e:
        print(f"[ERROR] Recording initialization failed: {e}")
        print("[ERROR] Continuing without recording...")
    
    return {}


def record_step_data_event(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> Dict[str, Any]:
    """
    Event function to record step data during episodes.
    
    Args:
        env: The RL environment instance
        env_ids: Environment IDs 
        
    Returns:
        Empty dictionary
    """
    # This would be called as an interval event if needed
    # For now, we handle recording in the main step loop
    if hasattr(env, '_recording_manager') and hasattr(env, '_last_actions'):
        try:
            env._recording_manager.on_step(env, env._last_actions)
        except Exception as e:
            # Silently continue if recording fails
            pass
    
    return {}