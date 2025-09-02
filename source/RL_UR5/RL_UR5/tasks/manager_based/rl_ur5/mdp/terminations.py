from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING
import numpy as np
import random
from .observations import *
from .thresholds import *  # Import all constants from thresholds


import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# tasks/manager_based/rl_ur5/mdp/terminations.py

def pose_tracking_success(
    env: ManagerBasedRLEnv,
    position_threshold: float = 0.05,
    orientation_threshold: float = 0.1,
    velocity_threshold: float = 0.05,
    torque_threshold: float = 1.0,
    command_name: str = "tracking_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Determine if the pose tracking task has been successfully completed.
    
    Args:
        env: The RL environment instance
        position_threshold: Maximum position error to consider successful
        orientation_threshold: Maximum orientation error to consider successful
        velocity_threshold: Maximum joint velocity magnitude to consider stable
        torque_threshold: Maximum joint torque magnitude to consider stable
        command_name: Name of the command containing the target pose
        asset_cfg: Configuration for the robot asset
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Boolean tensor indicating task success
    """
    # Get end-effector position and orientation using ee_frame
    asset = env.scene[asset_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    
    # Get the desired position and orientation from the command
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    
    # Transform to world frame
    des_pos_w, _ = math_utils.combine_frame_transforms(
        asset.data.root_state_w[:, :3], 
        asset.data.root_state_w[:, 3:7], 
        des_pos_b
    )
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    
    # Calculate position error
    position_error = torch.norm(ee_position - des_pos_w, p=2, dim=-1)
    
    # Calculate orientation error
    orientation_error = math_utils.quat_error_magnitude(ee_quat, des_quat_w)
    
    # Get joint velocities and torques
    joint_velocities = torch.norm(asset.data.joint_vel, p=2, dim=-1)


    
    # Check if all conditions are met
    position_success = position_error < position_threshold
    orientation_success = orientation_error < orientation_threshold
    velocity_success = joint_velocities < velocity_threshold

    
    # Success is when all criteria are met (position, orientation, velocity, and torque)
    success = torch.logical_and(
        torch.logical_and(position_success, orientation_success),velocity_success
    )
    
    return success


def alignment_success(
    env: ManagerBasedRLEnv,
    position_threshold: float = 0.05,
    orientation_threshold: float = 0.9,
    velocity_threshold: float = 0.05,
    torque_threshold: float = 1.0,
    height_offset: float = 0.3,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Determine if the end-effector is successfully aligned above the target cube.
    
    Args:
        env: The RL environment instance
        position_threshold: Maximum position error to consider successful
        orientation_threshold: Minimum orientation alignment to consider successful (0-1)
        velocity_threshold: Maximum joint velocity magnitude to consider stable
        torque_threshold: Maximum joint torque magnitude to consider stable
        height_offset: Target height above the cube
        ee_frame_cfg: Configuration for the end-effector frame
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Boolean tensor indicating alignment success
    """
    # Get end-effector position and orientation
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    
    # Get robot asset for joint velocities and torques
    asset = env.scene[asset_cfg.name]
    joint_velocities = torch.norm(asset.data.joint_vel, p=2, dim=-1)
    
    # Check if joint torques are available
    has_torques = hasattr(asset.data, 'joint_effort') and asset.data.joint_effort is not None
    if has_torques:
        joint_torques = torch.norm(asset.data.joint_effort, p=2, dim=-1)
    else:
        # If torques aren't available, just use a tensor of zeros
        joint_torques = torch.zeros_like(joint_velocities)
    
    # Initialize success tensor
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            cube_position = cube.data.root_pos_w[i, :3]
            cube_quat = cube.data.root_quat_w[i, :]
            
            # Calculate target position (above cube)
            target_position = cube_position.clone()
            target_position[2] += height_offset
            
            # Check position
            position_error = torch.norm(ee_position[i] - target_position, p=2)
            position_success = position_error < position_threshold
            
            # Check orientation alignment
            ee_rot_mat = math_utils.matrix_from_quat(ee_quat[i].unsqueeze(0)).squeeze(0)
            cube_rot_mat = math_utils.matrix_from_quat(cube_quat.unsqueeze(0)).squeeze(0)
            
            # Extract axes
            ee_x_axis = ee_rot_mat[:, 0]
            ee_y_axis = ee_rot_mat[:, 1]
            cube_y_axis = cube_rot_mat[:, 1]
            cube_z_axis = cube_rot_mat[:, 2]
            
            # Calculate alignment
            x_z_alignment = torch.abs(torch.dot(-ee_x_axis, cube_z_axis))
            y_y_alignment = torch.abs(torch.dot(-ee_y_axis, cube_y_axis))
            combined_alignment = torch.sqrt(x_z_alignment * y_y_alignment)
            
            orientation_success = combined_alignment > orientation_threshold
            
            # Check velocity and torque stability
            velocity_success = joint_velocities[i] < velocity_threshold
            torque_success = joint_torques[i] < torque_threshold
            
            # Set success if all criteria are met
            if position_success and orientation_success and velocity_success and torque_success:
                success[i] = True
    
    return success


def task_success(
    env: ManagerBasedRLEnv,
    gripper_threshold: float = 5.0,
    distance_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Determine if the pick and place task has been successfully completed.
    
    The task is considered successful when:
    1. The target cube is at the desired placement position
    2. The gripper is open (cube has been released)
    
    Args:
        env: The RL environment instance
        gripper_threshold: Maximum gripper position to consider "open"
        distance_threshold: Maximum distance to consider "at target"
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Boolean tensor indicating task success
    """
    # Initialize success tensor
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Get robot
    robot = env.scene[asset_cfg.name]
    
    # Find joint index by name
    joint_names = robot.joint_names
    gripper_joint_name = "robotiq_85_left_knuckle_joint"
    gripper_joint_idx = joint_names.index(gripper_joint_name) if gripper_joint_name in joint_names else -1
    
    if gripper_joint_idx == -1:
        return success
    
    # Get gripper position
    gripper_position = robot.data.joint_pos[:, gripper_joint_idx]
    
    for i in range(env.num_envs):
        # Skip if no task info
        if not hasattr(env, "task_info") or i not in env.task_info:
            continue
            
        target_info = env.task_info[i]
        target_cube_name = target_info["target_cube"]
        placement_position = target_info.get("placement_position", None)
        
        # Skip if no placement position is defined
        if placement_position is None:
            continue
            
        # Get cube position
        cube = env.scene[target_cube_name]
        cube_position = cube.data.root_pos_w[i, :3]
        
        # Calculate distance between cube and target
        distance = torch.norm(cube_position - placement_position, p=2)
        
        # Check if cube is at target and gripper is open
        is_at_target = distance < distance_threshold
        is_gripper_open = gripper_position[i] < gripper_threshold
        
        if is_at_target and is_gripper_open:
            success[i] = True
    
    return success




def robot_instability(
    env: ManagerBasedRLEnv,
    velocity_threshold: float = 50.0,
    torque_threshold: float = 50.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Termination condition for detecting robot instability.
    
    Args:
        env: The RL environment instance
        velocity_threshold: Maximum allowable joint velocity magnitude
        torque_threshold: Maximum allowable joint torque magnitude
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Boolean tensor indicating unstable environments
    """
    # Get robot asset
    robot = env.scene[asset_cfg.name]
    
    # Initialize result tensor
    is_unstable = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Check joint velocities
    if hasattr(robot.data, 'joint_vel') and robot.data.joint_vel is not None:
        max_joint_vel = torch.max(torch.abs(robot.data.joint_vel), dim=-1)[0]
        high_velocity = max_joint_vel > velocity_threshold
        is_unstable = torch.logical_or(is_unstable, high_velocity)
    
    # Check joint torques/efforts if available
    if hasattr(robot.data, 'joint_effort') and robot.data.joint_effort is not None:
        max_joint_torque = torch.max(torch.abs(robot.data.joint_effort), dim=-1)[0]
        high_torque = max_joint_torque > torque_threshold
        is_unstable = torch.logical_or(is_unstable, high_torque)
    
    return is_unstable





def ee_frame_table_collision(
    env: ManagerBasedRLEnv,
    table_height: float = TABLE_HEIGHT,
    safety_margin: float = 0.05,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Determine if the end-effector frame is too close to or below the table surface.
    
    Args:
        env: The RL environment instance
        table_height: Height of the table surface
        safety_margin: Safety margin above table surface
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Boolean tensor indicating if end-effector is too close to table
    """
    # Get end-effector position from the frame transformer
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get z-coordinate (height) of end-effector
    ee_height = ee_position[:, 2]
    
    # Check if height is below table + safety margin
    too_close_to_table = ee_height < (table_height + safety_margin)
    
    return too_close_to_table


def nan_observation_termination(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Termination condition for NaN values in observations.
    
    Args:
        env: The RL environment instance
        
    Returns:
        torch.Tensor: Boolean tensor indicating environments with NaN observations
    """
    # Initialize result tensor
    has_nan = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    try:
        # Method 1: Check if we have cached observation data
        if hasattr(env, '_obs_buf') and env._obs_buf is not None:
            for group_name, obs_data in env._obs_buf.items():
                if torch.is_tensor(obs_data):
                    group_has_nan = torch.isnan(obs_data).any(dim=-1)
                    has_nan = torch.logical_or(has_nan, group_has_nan)
                elif isinstance(obs_data, dict):
                    for key, value in obs_data.items():
                        if torch.is_tensor(value):
                            if value.ndim > 1:
                                term_has_nan = torch.isnan(value).any(dim=tuple(range(1, value.ndim)))
                            else:
                                term_has_nan = torch.isnan(value)
                            has_nan = torch.logical_or(has_nan, term_has_nan)
        
        # Method 2: Check observation manager directly
        elif hasattr(env, 'observation_manager') and env.observation_manager is not None:
            obs_manager = env.observation_manager
            
            # Try to get available groups
            if hasattr(obs_manager, '_group_obs_terms'):
                for group_name, terms in obs_manager._group_obs_terms.items():
                    try:
                        # Compute observations for this group
                        group_obs = obs_manager.compute_group(group_name)
                        if torch.is_tensor(group_obs):
                            group_has_nan = torch.isnan(group_obs).any(dim=-1)
                            has_nan = torch.logical_or(has_nan, group_has_nan)
                    except Exception as e:
                        print(f"Error computing observations for group {group_name}: {e}")
                        continue
            
            # Alternative: Check individual terms
            elif hasattr(obs_manager, '_terms'):
                for term_name, term in obs_manager._terms.items():
                    try:
                        obs_data = term.func(env, **term.params)
                        if torch.is_tensor(obs_data):
                            if obs_data.ndim > 1:
                                term_has_nan = torch.isnan(obs_data).any(dim=tuple(range(1, obs_data.ndim)))
                            else:
                                term_has_nan = torch.isnan(obs_data)
                            has_nan = torch.logical_or(has_nan, term_has_nan)
                    except Exception as e:
                        print(f"Error checking term {term_name}: {e}")
                        continue
        
        # Method 3: Direct check of critical scene data (most reliable)
        else:
            # Check robot joint states
            if hasattr(env.scene, 'robot'):
                robot = env.scene['robot']
                if hasattr(robot.data, 'joint_pos'):
                    joint_pos_nan = torch.isnan(robot.data.joint_pos).any(dim=-1)
                    has_nan = torch.logical_or(has_nan, joint_pos_nan)
                
                if hasattr(robot.data, 'joint_vel'):
                    joint_vel_nan = torch.isnan(robot.data.joint_vel).any(dim=-1)
                    has_nan = torch.logical_or(has_nan, joint_vel_nan)
            
            # Check end-effector frame
            if hasattr(env.scene, 'ee_frame'):
                ee_frame = env.scene['ee_frame']
                if hasattr(ee_frame.data, 'target_pos_w'):
                    ee_pos_nan = torch.isnan(ee_frame.data.target_pos_w).any(dim=-1).any(dim=-1)
                    has_nan = torch.logical_or(has_nan, ee_pos_nan)
                
                if hasattr(ee_frame.data, 'target_quat_w'):
                    ee_quat_nan = torch.isnan(ee_frame.data.target_quat_w).any(dim=-1).any(dim=-1)
                    has_nan = torch.logical_or(has_nan, ee_quat_nan)
        
        # Log NaN occurrences
        if has_nan.any():
            nan_env_ids = torch.where(has_nan)[0].tolist()
            print(f"WARNING: NaN observations detected, terminating environments: {nan_env_ids}")
    
    except Exception as e:
        print(f"Error checking for NaN observations: {e}")
        # Return no termination in case of error to avoid cascading failures
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    return has_nan

