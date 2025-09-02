# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from .thresholds import *  # Import all constants from thresholds


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# tasks/manager_based/rl_ur5/mdp/rewards.py

import numpy as np
import random
from .observations import *

from isaaclab.managers import SceneEntityCfg


# @configclass
# class JointPositionActionCfg:
#     """Configuration for joint position control."""
    
#     # Type annotations with explicit MISSING markers to indicate required fields
#     asset_name: str = dataclasses.MISSING
#     joint_names: List[str] = dataclasses.MISSING
#     scale: List[float] = dataclasses.MISSING






"""
Reward functions
"""
def position_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward based on distance between end-effector and target cube.
    
    Args:
        env: The RL environment instance
        asset_cfg: Configuration for the robot asset
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Negative distance between end-effector and target cube
    """
    # Get end-effector position using ee_frame
    asset: RigidObject = env.scene[asset_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get the desired position from the command
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    
    # Calculate distance
    distance = torch.norm(ee_position - des_pos_w, dim=-1)
    
    return distance  # Negative because smaller distance is better


def orientation_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward based on distance between end-effector and target cube.
    
    Args:
        env: The RL environment instance
        asset_cfg: Configuration for the robot asset
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Negative distance between end-effector and target cube
    """
    # Get end-effector position using ee_frame
    asset: RigidObject = env.scene[asset_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]  # Get orientation quaternion
    
    # Get the desired position from the command
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)

   
    
    return math_utils.quat_error_magnitude(ee_quat,des_quat_w)  # Quaternion error between orientations

def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward based on distance between end-effector and target cube.
    
    Args:
        env: The RL environment instance
        asset_cfg: Configuration for the robot asset
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Negative distance between end-effector and target cube
    """
    # Get end-effector position using ee_frame
    asset: RigidObject = env.scene[asset_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get the desired position from the command
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    
    # Calculate distance
    distance = torch.norm(ee_position - des_pos_w, p=2, dim=-1)
    
    return 1-torch.tanh(distance/std)  # Tanh kernel mapped



def distance_to_target_cube(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward based on distance between end-effector and target cube.
    
    Args:
        env: The RL environment instance
        asset_cfg: Configuration for the robot asset
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Negative distance between end-effector and target cube
    """
    # Get end-effector position using ee_frame
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get target cube positions for each environment
    cube_positions_tensor = torch.zeros((env.num_envs, 3), device=env.device)
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            cube_positions_tensor[i] = cube.data.root_pos_w[i, :3]
    
    # Calculate distance
    distance = torch.norm(ee_position - cube_positions_tensor, p=2, dim=-1)
    
    return -distance  # Negative because smaller distance is better




def distance_to_target_cube_tanh(
    env: ManagerBasedRLEnv,
    std:float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward based on distance between end-effector and target cube.
    
    Args:
        env: The RL environment instance
        std: Standard deviation for the tanh function
        asset_cfg: Configuration for the robot asset
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Tanh-transformed negative distance between end-effector and target cube
    """
    # Get end-effector position using ee_frame
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get target cube positions for each environment
    cube_positions_tensor = torch.zeros((env.num_envs, 3), device=env.device)
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            cube_positions_tensor[i] = cube.data.root_pos_w[i, :3]
    
    # Calculate distance
    distance = torch.norm(ee_position - cube_positions_tensor, p=2, dim=-1)
    
    return 1 - torch.tanh(distance/std)  # Tanh-transformed reward (1 when close, 0 when far)

def alignment_success_reward(
    env: ManagerBasedRLEnv,
    position_threshold: float = 0.05,
    orientation_threshold: float = 0.9,
    height_offset: float = 0.1,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for successful alignment above the cube (smoother than termination condition).
    
    Args:
        env: The RL environment instance
        position_threshold: Maximum position error to consider successful
        orientation_threshold: Minimum orientation alignment to consider successful (0-1)
        height_offset: Target height above the cube
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Binary reward (1.0 if aligned, 0.0 otherwise)
    """
    # Get end-effector position and orientation
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    
    # Initialize reward tensor
    rewards = torch.zeros(env.num_envs, device=env.device)
    
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
            
            # Smoother reward for near-success cases
            position_factor = torch.exp(-(position_error**2)/(2*(position_threshold**2)))
            if combined_alignment > orientation_threshold and position_error < position_threshold:
                rewards[i] = 1.0
            else:
                # Partial reward for being close
                rewards[i] = 0.5 * position_factor * combined_alignment
    
    return rewards



def orientation_alignment_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for aligning the end-effector orientation with the cube for proper grasping.
    
    Specific alignment requirements:
    - Negative x-axis of ee_frame should align with positive z-axis of cube
    - Negative y-axis of ee_frame should align with positive y-axis of cube
    
    Args:
        env: The RL environment instance
        ee_frame_cfg: Configuration for the end-effector frame
        align_weight: Weight for the alignment reward
        
    Returns:
        torch.Tensor: Reward based on orientation alignment
    """
    # Get end-effector orientation
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]  # Get orientation quaternion
    
    # Initialize reward tensor
    rewards = torch.zeros(env.num_envs, device=env.device)
    
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            cube_quat = cube.data.root_quat_w[i, :]  # Get cube orientation quaternion
            
            # Convert quaternions to rotation matrices
            ee_rot_mat = math_utils.matrix_from_quat(ee_quat[i].unsqueeze(0)).squeeze(0)
            cube_rot_mat = math_utils.matrix_from_quat(cube_quat.unsqueeze(0)).squeeze(0)
            
            # Extract axes from rotation matrices
            # For ee_frame
            ee_x_axis = ee_rot_mat[:, 0]  # X-axis is first column
            ee_y_axis = ee_rot_mat[:, 1]  # Y-axis is second column
            
            # For cube
            cube_x_axis = cube_rot_mat[:, 0]  # Y-axis is second column
            cube_z_axis = cube_rot_mat[:, 2]  # Z-axis is third column
            
            # We want negative ee_x_axis to align with positive cube_z_axis
            # And negative ee_y_axis to align with positive cube_x_axis
            x_z_alignment = torch.abs(torch.dot(-ee_x_axis, cube_z_axis))
            y_y_alignment = torch.abs(torch.dot(-ee_y_axis, cube_x_axis))
            
            # Combine alignments (geometric mean works well for this)
            combined_alignment = torch.sqrt(x_z_alignment * y_y_alignment)
            
            # Apply reward
            rewards[i] = combined_alignment
    
    return rewards



def pose_tracking_success_reward(
    env: ManagerBasedRLEnv,
    position_threshold: float = 0.05,
    orientation_threshold: float = 0.1,
    command_name: str = "tracking_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Determine if the pose tracking task has been successfully completed.
    
    Args:
        env: The RL environment instance
        position_threshold: Maximum position error to consider successful
        orientation_threshold: Maximum orientation error to consider successful
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
    
    # Check if both position and orientation are within thresholds
    position_success = position_error < position_threshold
    orientation_success = orientation_error < orientation_threshold
    
    # Success is when both position and orientation criteria are met
    success = torch.logical_and(position_success, orientation_success)
    
    return success


def approach_reward(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for being close to the target cube.
    
    Args:
        env: The RL environment instance
        distance_threshold: Maximum distance to consider "close"
        asset_cfg: Configuration for the robot asset
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Binary reward (1.0 if close, 0.0 otherwise)
    """
    # Get end-effector position
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get target cube positions for each environment
    cube_positions_tensor = torch.zeros((env.num_envs, 3), device=env.device)
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            cube_positions_tensor[i] = cube.data.root_pos_w[i, :3]
    
    # Calculate distance
    distance = torch.norm(ee_position - cube_positions_tensor, p=2, dim=-1)
    
    # Calculate reward (1.0 if close, 0.0 otherwise)
    reward = torch.where(distance < distance_threshold, 
                        torch.ones_like(distance), 
                        torch.zeros_like(distance))
    
    return reward


def grasp_reward(
    env: ManagerBasedRLEnv,
    gripper_threshold: float = 0.4,
    distance_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for grasping the cube.
    
    Args:
        env: The RL environment instance
        gripper_threshold: Minimum gripper position to consider "closed"
        distance_threshold: Maximum distance to consider "close"
        asset_cfg: Configuration for the robot asset
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Binary reward (2.0 if grasping, 0.0 otherwise)
    """
    # Get end-effector position
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get target cube positions for each environment
    cube_positions_tensor = torch.zeros((env.num_envs, 3), device=env.device)
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            cube_positions_tensor[i] = cube.data.root_pos_w[i, :3]
    
    # Calculate distance
    distance = torch.norm(ee_position - cube_positions_tensor, p=2, dim=-1)
    is_close = distance < distance_threshold
    
    # Get gripper position
    robot = env.scene[asset_cfg.name]
    
    # Find joint index by name
    joint_names = robot.joint_names
    gripper_joint_name = "robotiq_85_left_knuckle_joint"
    gripper_joint_idx = joint_names.index(gripper_joint_name) if gripper_joint_name in joint_names else -1
    
    if gripper_joint_idx == -1:
        raise ValueError(f"Gripper joint '{gripper_joint_name}' not found in joint names")
    
    gripper_position = robot.data.joint_pos[:, gripper_joint_idx]
    
    # Check if gripper is closed enough
    is_closed = gripper_position > gripper_threshold
    
    # Calculate reward (2.0 if close and gripper closed, 0.0 otherwise)
    reward = torch.where(torch.logical_and(is_close, is_closed), 
                        2.0 * torch.ones_like(distance), 
                        torch.zeros_like(distance))
    
    return reward


def placement_reward(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.05,
    gripper_threshold: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for placing the cube at the target position.
    
    Args:
        env: The RL environment instance
        distance_threshold: Maximum distance to consider "at target"
        gripper_threshold: Minimum gripper position to consider "grasping"
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Reward for placing the cube at the target (5.0 if successful)
    """
    rewards = torch.zeros(env.num_envs, device=env.device)
    
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            # Use placement_position from task_info
            target_position = target_info.get("placement_position", None)
            
            # Skip if no placement position is defined
            if target_position is None:
                continue
            
            # Get cube position
            cube = env.scene[target_cube_name]
            cube_position = cube.data.root_pos_w[i, :3]
            
            # Calculate distance between cube and target
            distance = torch.norm(cube_position - target_position, p=2)
            
            # Check if cube is at target
            is_at_target = distance < distance_threshold
            
            # Get gripper position
            robot = env.scene[asset_cfg.name]
            
            # Find joint index by name
            joint_names = robot.joint_names
            gripper_joint_name = "robotiq_85_left_knuckle_joint"
            gripper_joint_idx = joint_names.index(gripper_joint_name) if gripper_joint_name in joint_names else -1
            
            if gripper_joint_idx == -1:
                raise ValueError(f"Gripper joint '{gripper_joint_name}' not found in joint names")
            
            gripper_position = robot.data.joint_pos[i, gripper_joint_idx]
            is_grasping = gripper_position > gripper_threshold
            
            # Calculate reward (5.0 if cube at target and grasping, 0.0 otherwise)
            if is_at_target and is_grasping:
                rewards[i] = 5.0
    
    return rewards


def success_reward(
    env: ManagerBasedRLEnv,
    gripper_threshold: float = 0.2,
    distance_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for successfully completing the task.
    
    Args:
        env: The RL environment instance
        gripper_threshold: Maximum gripper position to consider "open"
        distance_threshold: Maximum distance to consider "at target"
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Reward for successfully completing the task (10.0 if successful)
    """
    rewards = torch.zeros(env.num_envs, device=env.device)
    
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            # Use placement_position from task_info
            target_position = target_info.get("placement_position", None)
            
            # Skip if no placement position is defined
            if target_position is None:
                continue
            
            # Get cube position
            cube = env.scene[target_cube_name]
            cube_position = cube.data.root_pos_w[i, :3]
            
            # Calculate distance between cube and target
            distance = torch.norm(cube_position - target_position, p=2)
            
            # Check if cube is at target
            is_at_target = distance < distance_threshold
            
            # Get gripper position
            robot = env.scene[asset_cfg.name]
            
            # Find joint index by name
            joint_names = robot.joint_names
            gripper_joint_name = "robotiq_85_left_knuckle_joint"
            gripper_joint_idx = joint_names.index(gripper_joint_name) if gripper_joint_name in joint_names else -1
            
            if gripper_joint_idx == -1:
                raise ValueError(f"Gripper joint '{gripper_joint_name}' not found in joint names")
            
            gripper_position = robot.data.joint_pos[i, gripper_joint_idx]
            is_open = gripper_position < gripper_threshold
            
            # Calculate reward (10.0 if cube at target and gripper open, 0.0 otherwise)
            if is_at_target and is_open:
                rewards[i] = 10.0
    
    return rewards


def movement_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for excessive movement.
    
    Args:
        env: The RL environment instance
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Penalty proportional to joint velocities
    """
    # Get joint velocities
    robot = env.scene[asset_cfg.name]
    joint_vels = robot.data.joint_vel
    
    # Calculate penalty (sum of absolute velocities)
    penalty = -0.01 * torch.sum(torch.abs(joint_vels), dim=-1)
    
    return penalty


def curriculum_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg(name="ee_frame", joint_ids=[], fixed_tendon_ids=[], body_ids=[], object_collection_ids=[]),
    height_threshold: float = 0.8,
    distance_threshold: float = 0.05,
    gripper_threshold: float = 5.0,
    alignment_weight: float = 1.0,
) -> torch.Tensor:
    """Reward function based on the current task stage.
    
    Stages:
    0 - Alignment: Hover above cube at specified height and align gripper
    1 - Grasp: Move down and grasp the cube
    2 - Placement: Move to target location and place the cube
    3 - Complete: Task successfully completed
    
    Args:
        env: The RL environment instance
        ee_frame_cfg: Configuration for the end-effector frame
        height_threshold: Height to maintain in stage 0
        distance_threshold: Distance threshold for success conditions
        gripper_threshold: Gripper position threshold for grasping/releasing
        alignment_weight: Weight for orientation alignment reward
        
    Returns:
        torch.Tensor: Stage-appropriate reward
    """
    # Initialize rewards
    rewards = torch.zeros(env.num_envs, device=env.device)
    
    # Skip if task stages not initialized
    if not hasattr(env, "task_stages"):
        return rewards
    
    # Get end-effector position and orientation
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    
    # Get robot
    robot = env.scene["robot"]
    
    # Find joint index for gripper
    joint_names = robot.joint_names
    gripper_joint_name = "robotiq_85_left_knuckle_joint"
    gripper_joint_idx = joint_names.index(gripper_joint_name) if gripper_joint_name in joint_names else -1
    
    if gripper_joint_idx == -1:
        return rewards
    
    # Get gripper position
    gripper_position = robot.data.joint_pos[:, gripper_joint_idx]
    
    # Process each environment
    for i in range(env.num_envs):
        if i not in env.task_stages:
            continue
            
        current_stage = env.task_stages[i]
        
        # Skip environments missing task info
        if not hasattr(env, "task_info") or i not in env.task_info:
            continue
            
        target_info = env.task_info[i]
        target_cube_name = target_info["target_cube"]
        placement_position = target_info.get("placement_position", None)
        
        # Get cube position
        cube = env.scene[target_cube_name]
        cube_position = cube.data.root_pos_w[i, :3]
        cube_quat = cube.data.root_quat_w[i, :]
        
        # Stage 0: Alignment above cube
        if current_stage == 0:
            # Calculate desired position (above cube)
            target_position = cube_position.clone()
            target_position[2] += height_threshold
            
            # Calculate distance from desired hover position
            hover_distance = torch.norm(ee_position[i] - target_position, p=2)
            position_reward = torch.exp(-hover_distance * 5.0)
            
            # Calculate orientation alignment
            ee_rot_mat = math_utils.matrix_from_quat(ee_quat[i].unsqueeze(0)).squeeze(0)
            cube_rot_mat = math_utils.matrix_from_quat(cube_quat.unsqueeze(0)).squeeze(0)
            
            # Extract axes
            ee_x_axis = ee_rot_mat[:, 0]
            ee_y_axis = ee_rot_mat[:, 1]
            cube_x_axis = cube_rot_mat[:, 0]
            cube_z_axis = cube_rot_mat[:, 2]
            
            # Calculate alignment (negative x-axis of ee to positive z-axis of cube,
            # negative y-axis of ee to positive y-axis of cube)
            x_z_alignment = torch.abs(torch.dot(-ee_x_axis, cube_z_axis))
            y_y_alignment = torch.abs(torch.dot(-ee_y_axis, cube_x_axis))
            alignment_reward = torch.sqrt(x_z_alignment * y_y_alignment)
            
            # Combine rewards for stage 0
            rewards[i] = 0.5 * position_reward + alignment_weight * alignment_reward
            
        # Stage 1: Grasp
        elif current_stage == 1:
            # Calculate horizontal distance to cube
            horizontal_distance = torch.norm(ee_position[i, :2] - cube_position[:2], p=2)
            
            # Reward for moving down while staying centered
            height_diff = ee_position[i, 2] - cube_position[2]
            centered_descent_reward = torch.exp(-horizontal_distance * 10.0) * (1.0 - torch.clamp(height_diff / height_threshold, 0.0, 1.0))
            
            # Calculate distance to cube
            distance = torch.norm(ee_position[i] - cube_position, p=2)
            
            # Add grasp reward
            grasp_reward = 0.0
            if distance < distance_threshold and gripper_position[i] > gripper_threshold:
                grasp_reward = 2.0
            
            # Combine rewards for stage 1
            rewards[i] = 0.5 * centered_descent_reward + grasp_reward
            
        # Stage 2: Placement
        elif current_stage == 2:
            if placement_position is not None:
                # Reward for bringing cube close to target position
                target_distance = torch.norm(cube_position - placement_position, p=2)
                transport_reward = torch.exp(-target_distance * 5.0)
                
                # Add placement reward
                placement_reward = 0.0
                if target_distance < distance_threshold and gripper_position[i] < gripper_threshold:
                    placement_reward = 5.0
                
                # Combine rewards for stage 2
                rewards[i] = transport_reward + placement_reward
                
        # Stage 3: Complete
        elif current_stage == 3:
            # Success reward
            rewards[i] = 10.0
    
    return rewards



## Position above cube rewards
def position_above_cube_reward(
    env: ManagerBasedRLEnv,
    height_offset: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for positioning the end-effector above the target cube.
    
    Args:
        env: The RL environment instance
        height_offset: Height above the cube to target
        asset_cfg: Configuration for the robot asset
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Negative distance between end-effector and target position above cube
    """
    # Get end-effector position using ee_frame
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get target cube positions for each environment
    cube_positions_tensor = torch.zeros((env.num_envs, 3), device=env.device)
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            cube_positions_tensor[i] = cube.data.root_pos_w[i, :3]
    
    # Calculate target positions (above cube)
    target_positions = cube_positions_tensor.clone()
    target_positions[:, 2] += height_offset  # Add height offset to z-coordinate
    
    # Calculate distance to target position above cube
    distance = torch.norm(ee_position - target_positions, p=2, dim=-1)
    
    return distance  # Negative because smaller distance is better


## Position above cube rewards
def position_above_cube_tanh(
    env: ManagerBasedRLEnv,
    height_offset: float = 0.3,
    std: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for positioning the end-effector above the target cube.
    
    Args:
        env: The RL environment instance
        height_offset: Height above the cube to target
        asset_cfg: Configuration for the robot asset
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Negative distance between end-effector and target position above cube
    """
    # Get end-effector position using ee_frame
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get target cube positions for each environment
    cube_positions_tensor = torch.zeros((env.num_envs, 3), device=env.device)
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            cube_positions_tensor[i] = cube.data.root_pos_w[i, :3]
    
    # Calculate target positions (above cube)
    target_positions = cube_positions_tensor.clone()
    target_positions[:, 2] += height_offset  # Add height offset to z-coordinate
    
    # Calculate distance to target position above cube
    distance = torch.norm(ee_position - target_positions, p=2, dim=-1)
    
    return 1-torch.tanh(distance/std)  # Negative because smaller distance is better

def cube_height_reward(
    env: ManagerBasedRLEnv,
    base_height: float = CUBE_START_HEIGHT,  # Updated to use cube start height
    max_height: float = CUBE_MAX_HEIGHT,    # Height for maximum reward
) -> torch.Tensor:
    """Reward based on the height of the target cube above its starting position.
    Higher cube means better grasp.
    
    Args:
        env: The RL environment instance
        base_height: The base height (cube starting height) with no reward
        max_height: The height at which maximum reward is given
        
    Returns:
        torch.Tensor: Reward based on cube height
    """
    # Initialize rewards
    rewards = torch.zeros(env.num_envs, device=env.device)
    
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            
            # Get cube position
            cube = env.scene[target_cube_name]
            cube_height = cube.data.root_pos_w[i, 2]  # Z-coordinate
            
            # Calculate reward based on height
            # No reward if at or below base height, max reward at max_height
            if cube_height > base_height:
                # Normalized height between 0 and 1
                norm_height = min(1.0, (cube_height - base_height) / (max_height - base_height))
                rewards[i] = norm_height
    
    return rewards



# Add to mdp/rewards.py

def object_is_lifted(
    env: ManagerBasedRLEnv, 
    minimal_height: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("red_cube")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    # Get target object based on task_id for each environment
    rewards = torch.zeros(env.num_envs, device=env.device)
    
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            cube_height = cube.data.root_pos_w[i, 2]
            
            # Reward if cube is lifted above the minimal height
            if cube_height > minimal_height + TABLE_HEIGHT:
                rewards[i] = 1.0
    
    return rewards


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("red_cube")
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # Get command for goal position
    command = env.command_manager.get_command(command_name)
    rewards = torch.zeros(env.num_envs, device=env.device)
    
    # Get robot for coordinate transformation
    robot = env.scene[robot_cfg.name]
    
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            
            # Get cube position
            cube_pos_w = cube.data.root_pos_w[i, :3]
            
            # Transform goal position to world frame
            des_pos_b = command[i, :3]
            des_pos_w, _ = math_utils.combine_frame_transforms(
                robot.data.root_state_w[i, :3].unsqueeze(0), 
                robot.data.root_state_w[i, 3:7].unsqueeze(0), 
                des_pos_b.unsqueeze(0)
            )
            des_pos_w = des_pos_w.squeeze(0)
            
            # Calculate distance
            distance = torch.norm(des_pos_w - cube_pos_w, p=2)
            
            # Reward if cube is lifted and close to goal
            if cube_pos_w[2] > minimal_height + TABLE_HEIGHT:
                rewards[i] = 1.0 - torch.tanh(distance / std)
    
    return rewards


# Add to mdp/terminations.py

def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("red_cube")
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position."""
    # Get command for goal position
    command = env.command_manager.get_command(command_name)
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Get robot for coordinate transformation
    robot = env.scene[robot_cfg.name]
    
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            
            # Get cube position
            cube_pos_w = cube.data.root_pos_w[i, :3]
            
            # Transform goal position to world frame
            des_pos_b = command[i, :3]
            des_pos_w, _ = math_utils.combine_frame_transforms(
                robot.data.root_state_w[i, :3].unsqueeze(0), 
                robot.data.root_state_w[i, 3:7].unsqueeze(0), 
                des_pos_b.unsqueeze(0)
            )
            des_pos_w = des_pos_w.squeeze(0)
            
            # Calculate distance
            distance = torch.norm(des_pos_w - cube_pos_w, p=2)
            
            # Check if cube reached goal
            if distance < threshold:
                result[i] = True
    
    return result


def obstacle_avoidance_penalty(
    env: ManagerBasedRLEnv,
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    safe_distance: float = 0.2,
    danger_distance: float = 0.1,
    max_penalty: float = -5.0,
) -> torch.Tensor:
    """Penalty for getting too close to the dynamic obstacle.
    
    Args:
        env: The RL environment instance
        obstacle_cfg: Configuration for the obstacle object
        ee_frame_cfg: Configuration for the end-effector frame
        safe_distance: Distance at which penalty starts
        danger_distance: Distance at which maximum penalty is applied
        max_penalty: Maximum penalty value (should be negative)
        
    Returns:
        torch.Tensor: Penalty based on proximity to obstacle
    """
    # Get end-effector position
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get obstacle position
    obstacle = env.scene[obstacle_cfg.name]
    obstacle_position = obstacle.data.root_pos_w[:, :3]
    
    # Calculate distance between end-effector and obstacle
    distance = torch.norm(ee_position - obstacle_position, p=2, dim=-1)
    
    # Calculate penalty using exponential decay
    penalty = torch.zeros_like(distance)
    
    # Apply penalty only when within safe distance
    within_safe_distance = distance < safe_distance
    
    # Exponential penalty that increases as distance decreases
    normalized_distance = torch.clamp((distance - danger_distance) / (safe_distance - danger_distance), 0.0, 1.0)
    penalty_magnitude = max_penalty * (1.0 - normalized_distance) ** 2
    
    penalty = torch.where(within_safe_distance, penalty_magnitude, penalty)
    
    return penalty

def obstacle_avoidance_penalty_tanh(
    env: ManagerBasedRLEnv,
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    safe_distance: float = 0.2,
    std: float = 0.1,
) -> torch.Tensor:
    """Smooth tanh-based penalty for obstacle proximity.
    
    Args:
        env: The RL environment instance
        obstacle_cfg: Configuration for the obstacle object
        ee_frame_cfg: Configuration for the end-effector frame
        safe_distance: Reference distance for penalty calculation
        std: Standard deviation for tanh function
        
    Returns:
        torch.Tensor: Smooth penalty based on proximity to obstacle
    """
    # Get end-effector position
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get obstacle position
    obstacle = env.scene[obstacle_cfg.name]
    obstacle_position = obstacle.data.root_pos_w[:, :3]
    
    # Calculate distance between end-effector and obstacle
    distance = torch.norm(ee_position - obstacle_position, p=2, dim=-1)
    
    # Tanh-based penalty (negative reward that increases as distance decreases)
    penalty = -torch.tanh((safe_distance - distance) / std)
    
    # Only apply penalty when distance is less than safe_distance
    penalty = torch.where(distance < safe_distance, penalty, torch.zeros_like(penalty))
    
    return penalty


def obstacle_avoidance_penalty_full_arm(
    env: ManagerBasedRLEnv,
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safe_distance: float = 0.2,
    danger_distance: float = 0.1,
    max_penalty: float = -5.0,
    arm_body_names: list = None,
    body_weights: dict = None,
) -> torch.Tensor:
    """Penalty for getting too close to the dynamic obstacle (considers entire arm).
    
    Args:
        env: The RL environment instance
        obstacle_cfg: Configuration for the obstacle object
        robot_cfg: Configuration for the robot
        safe_distance: Distance at which penalty starts
        danger_distance: Distance at which maximum penalty is applied
        max_penalty: Maximum penalty value (should be negative)
        arm_body_names: List of robot body names to check for collision
        body_weights: Dictionary of weights for different body parts
        
    Returns:
        torch.Tensor: Penalty based on proximity to obstacle
    """
    # Default arm body names for UR5
    if arm_body_names is None:
        arm_body_names = [
            "base_link",
            "shoulder_link", 
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link", 
            "wrist_3_link",
            "ee_link"
        ]
    
    # Default weights (higher weight = more important to avoid collision)
    if body_weights is None:
        body_weights = {
            "base_link": 0.1,      # Base is less likely to hit obstacle
            "shoulder_link": 0.3,
            "upper_arm_link": 0.5,
            "forearm_link": 0.7,
            "wrist_1_link": 0.8,
            "wrist_2_link": 0.9,
            "wrist_3_link": 0.9,
            "ee_link": 1.0         # End-effector most important
        }
    
    # Get robot and obstacle
    robot = env.scene[robot_cfg.name]
    obstacle = env.scene[obstacle_cfg.name]
    obstacle_position = obstacle.data.root_pos_w[:, :3]
    
    # Get body names from robot
    robot_body_names = robot.body_names
    
    # Initialize penalty tensor
    total_penalty = torch.zeros(env.num_envs, device=env.device)
    
    # Check each body part
    for body_name in arm_body_names:
        if body_name in robot_body_names:
            # Get body index
            body_idx = robot_body_names.index(body_name)
            
            # Get body position
            body_position = robot.data.body_pos_w[:, body_idx, :]
            
            # Calculate distance between this body part and obstacle
            distance = torch.norm(body_position - obstacle_position, p=2, dim=-1)
            
            # Calculate penalty for this body part
            body_penalty = torch.zeros_like(distance)
            
            # Apply penalty only when within safe distance
            within_safe_distance = distance < safe_distance
            
            # Exponential penalty that increases as distance decreases
            normalized_distance = torch.clamp(
                (distance - danger_distance) / (safe_distance - danger_distance), 
                0.0, 1.0
            )
            penalty_magnitude = max_penalty * (1.0 - normalized_distance) ** 2
            
            body_penalty = torch.where(within_safe_distance, penalty_magnitude, body_penalty)
            
            # Apply body weight and accumulate
            body_weight = body_weights.get(body_name, 1.0)
            total_penalty += body_weight * body_penalty
    
    return total_penalty

def obstacle_avoidance_penalty_tanh_full_arm(
    env: ManagerBasedRLEnv,
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safe_distance: float = 0.2,
    std: float = 0.1,
    arm_body_names: list = None,
    body_weights: dict = None,
    use_minimum_distance: bool = True,
) -> torch.Tensor:
    """Smooth tanh-based penalty for obstacle proximity (considers entire arm).
    
    Args:
        env: The RL environment instance
        obstacle_cfg: Configuration for the obstacle object
        robot_cfg: Configuration for the robot
        safe_distance: Reference distance for penalty calculation
        std: Standard deviation for tanh function
        arm_body_names: List of robot body names to check for collision
        body_weights: Dictionary of weights for different body parts
        use_minimum_distance: If True, use minimum distance; if False, use weighted sum
        
    Returns:
        torch.Tensor: Smooth penalty based on proximity to obstacle
    """
    # Default arm body names for UR5
    if arm_body_names is None:
        arm_body_names = [
            "base_link",
            "shoulder_link", 
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link", 
            "wrist_3_link",
            "ee_link"
        ]
    
    # Default weights
    if body_weights is None:
        body_weights = {
            "base_link": 0.1,
            "shoulder_link": 0.3,
            "upper_arm_link": 0.5,
            "forearm_link": 0.7,
            "wrist_1_link": 0.8,
            "wrist_2_link": 0.9,
            "wrist_3_link": 0.9,
            "ee_link": 1.0
        }
    
    # Get robot and obstacle
    robot = env.scene[robot_cfg.name]
    obstacle = env.scene[obstacle_cfg.name]
    obstacle_position = obstacle.data.root_pos_w[:, :3]
    
    # Get body names from robot
    robot_body_names = robot.body_names
    
    if use_minimum_distance:
        # Use minimum distance approach (most conservative)
        min_distances = torch.full((env.num_envs,), float('inf'), device=env.device)
        
        for body_name in arm_body_names:
            if body_name in robot_body_names:
                body_idx = robot_body_names.index(body_name)
                body_position = robot.data.body_pos_w[:, body_idx, :]
                distance = torch.norm(body_position - obstacle_position, p=2, dim=-1)
                min_distances = torch.min(min_distances, distance)
        
        # Apply tanh penalty based on minimum distance
        penalty = -torch.tanh((safe_distance - min_distances) / std)
        penalty = torch.where(min_distances < safe_distance, penalty, torch.zeros_like(penalty))
        
    else:
        # Use weighted sum approach
        total_penalty = torch.zeros(env.num_envs, device=env.device)
        total_weight = 0.0
        
        for body_name in arm_body_names:
            if body_name in robot_body_names:
                body_idx = robot_body_names.index(body_name)
                body_position = robot.data.body_pos_w[:, body_idx, :]
                distance = torch.norm(body_position - obstacle_position, p=2, dim=-1)
                
                # Tanh-based penalty for this body part
                body_penalty = -torch.tanh((safe_distance - distance) / std)
                body_penalty = torch.where(distance < safe_distance, body_penalty, torch.zeros_like(body_penalty))
                
                # Apply body weight
                body_weight = body_weights.get(body_name, 1.0)
                total_penalty += body_weight * body_penalty
                total_weight += body_weight
        
        # Normalize by total weight
        penalty = total_penalty / total_weight if total_weight > 0 else total_penalty
    
    return penalty

def obstacle_avoidance_penalty_adaptive(
    env: ManagerBasedRLEnv,
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safe_distance: float = 0.2,
    std: float = 0.1,
    critical_bodies: list = None,
    penalty_scale: float = 2.0,
) -> torch.Tensor:
    """Adaptive penalty that focuses on the closest body parts to the obstacle.
    
    Args:
        env: The RL environment instance
        obstacle_cfg: Configuration for the obstacle object
        robot_cfg: Configuration for the robot
        safe_distance: Reference distance for penalty calculation
        std: Standard deviation for tanh function
        critical_bodies: Most important body parts to protect
        penalty_scale: Scaling factor for penalty
        
    Returns:
        torch.Tensor: Adaptive penalty based on closest body parts
    """
    # Critical body parts that should avoid collision at all costs
    if critical_bodies is None:
        critical_bodies = ["forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link", "ee_link"]
    
    # All body parts to consider
    all_bodies = ["base_link", "shoulder_link", "upper_arm_link", "forearm_link", 
                  "wrist_1_link", "wrist_2_link", "wrist_3_link", "ee_link"]
    
    # Get robot and obstacle
    robot = env.scene[robot_cfg.name]
    obstacle = env.scene[obstacle_cfg.name]
    obstacle_position = obstacle.data.root_pos_w[:, :3]
    robot_body_names = robot.body_names
    
    # Calculate distances for all body parts
    body_distances = {}
    for body_name in all_bodies:
        if body_name in robot_body_names:
            body_idx = robot_body_names.index(body_name)
            body_position = robot.data.body_pos_w[:, body_idx, :]
            distance = torch.norm(body_position - obstacle_position, p=2, dim=-1)
            body_distances[body_name] = distance
    
    # Find the closest body part for each environment
    penalty = torch.zeros(env.num_envs, device=env.device)
    
    for env_id in range(env.num_envs):
        min_distance = float('inf')
        closest_body = None
        
        # Find closest body part
        for body_name, distances in body_distances.items():
            if distances[env_id] < min_distance:
                min_distance = distances[env_id].item()
                closest_body = body_name
        
        # Apply penalty based on closest body
        if closest_body and min_distance < safe_distance:
            # Higher penalty if critical body is closest
            scale = penalty_scale if closest_body in critical_bodies else 1.0
            penalty[env_id] = -scale * torch.tanh((safe_distance - min_distance) / std)
    
    return penalty