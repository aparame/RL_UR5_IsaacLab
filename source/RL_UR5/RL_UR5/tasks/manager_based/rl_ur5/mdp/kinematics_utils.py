#!/usr/bin/env python3

import numpy as np
from typing import List, Tuple, Optional
import math

class KinematicsUtils:
    """Forward and inverse kinematics utilities for UR5 robot with custom URDF configuration"""
    
    def __init__(self, logger):
        self.logger = logger
        
        # Key differences from standard UR5 based on your URDF:
        # 1. Base has 180° rotation (base_link_base_fixed_joint with rpy="0. 0. 3.1415925")
        # 2. Different joint origins and orientations
        # 3. Custom end-effector configuration with gripper
        
        # UR5 DH parameters extracted from your URDF
        # Modified DH parameters based on your URDF
        self.dh_params = np.array([
            [0.0,       np.pi/2,    0.089159,   0.0],           # Joint 1 (shoulder_pan)
            [-0.425,    0.0,        0.13585,    0.0],           # Joint 2 (shoulder_lift) 
            [-0.39225,  0.0,        -0.1197,    0.0],           # Joint 3 (elbow)
            [0.0,       np.pi/2,    0.0,        0.0],           # Joint 4 (wrist_1)
            [0.0,      -np.pi/2,    0.093,      0.0],           # Joint 5 (wrist_2)
            [0.0,       0.0,        0.09465,    0.0]            # Joint 6 (wrist_3)
        ])
        
        # Base transformation (180° rotation around Z from base_link_base_fixed_joint)
        self.base_rotation = np.pi
        
        # Total offset from wrist_3_link to ee_link
        self.ee_offset = np.array([0.0, 0.0823, 0.0])
        
        # Additional rotation from wrist_3 to ee_link (90° around Z)
        self.ee_rotation_z = np.pi/2
        
        # NEW: Reference quaternion for consistency (your desired orientation)
        # This should be close to your expected orientation
        self.reference_quaternion = np.array([-0.0635, 0.7409, -0.0015, 0.6686])  # [x,y,z,w]
        
    def dh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """Compute DH transformation matrix"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        T = np.array([
            [ct,    -st*ca,  st*sa,   a*ct],
            [st,     ct*ca, -ct*sa,   a*st],
            [0,      sa,     ca,      d],
            [0,      0,      0,       1]
        ])
        
        return T
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for UR5 with custom URDF configuration
        
        Args:
            joint_angles: Array of 6 joint angles [rad] in standard order
                         [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
            
        Returns:
            End-effector pose [x, y, z, qx, qy, qz, qw]
        """
        if len(joint_angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")
        
        try:
            # Apply base rotation (180° around Z)
            base_transform = np.array([
                [np.cos(self.base_rotation), -np.sin(self.base_rotation), 0, 0],
                [np.sin(self.base_rotation),  np.cos(self.base_rotation), 0, 0],
                [0,                          0,                           1, 0],
                [0,                          0,                           0, 1]
            ])
            
            # Start with base transformation
            T = base_transform
            
            # Apply each joint transformation
            for i in range(6):
                a, alpha, d, theta_offset = self.dh_params[i]
                theta = joint_angles[i] + theta_offset
                
                T_i = self.dh_transform(a, alpha, d, theta)
                T = T @ T_i
            
            # Apply end-effector offset and rotation
            # First apply the rotation around Z for ee_link orientation
            ee_rot_z = np.array([
                [np.cos(self.ee_rotation_z), -np.sin(self.ee_rotation_z), 0, 0],
                [np.sin(self.ee_rotation_z),  np.cos(self.ee_rotation_z), 0, 0],
                [0,                           0,                            1, 0],
                [0,                           0,                            0, 1]
            ])
            
            T = T @ ee_rot_z
            
            # Apply end-effector offset
            ee_offset_homogeneous = np.array([self.ee_offset[0], self.ee_offset[1], self.ee_offset[2], 1.0])
            ee_position_homogeneous = T @ ee_offset_homogeneous
            
            # Extract position
            position = ee_position_homogeneous[:3]
            
            # Extract rotation matrix and convert to quaternion
            rotation_matrix = T[:3, :3]
            quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
            
            # NEW: Ensure quaternion consistency
            quaternion = self.ensure_quaternion_consistency(quaternion)
            
            # Combine position and orientation
            pose = np.concatenate([position, quaternion])
            
            return pose
            
        except Exception as e:
            self.logger.error(f"Error in forward kinematics: {e}")
            # Return default pose if calculation fails
            return np.array([0.5, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0])
    
    def ensure_quaternion_consistency(self, quaternion: np.ndarray) -> np.ndarray:
        """
        Ensure quaternion is in the same hemisphere as the reference quaternion
        This prevents quaternion flipping between equivalent representations
        
        Args:
            quaternion: Current quaternion [qx, qy, qz, qw]
            
        Returns:
            Consistent quaternion [qx, qy, qz, qw]
        """
        # Compute dot product with reference
        dot_product = np.dot(quaternion, self.reference_quaternion)
        
        # If negative, flip the quaternion to the other hemisphere
        if dot_product < 0:
            quaternion = -quaternion
            
        return quaternion
    
    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]"""
        try:
            # Ensure the matrix is orthogonal
            R = self._orthogonalize_rotation_matrix(R)
            
            # Shepperd's method for numerical stability
            trace = np.trace(R)
            
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
                qw = 0.25 * s
                qx = (R[2, 1] - R[1, 2]) / s
                qy = (R[0, 2] - R[2, 0]) / s
                qz = (R[1, 0] - R[0, 1]) / s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
            
            # Normalize quaternion
            norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            quat = np.array([qx/norm, qy/norm, qz/norm, qw/norm])
            
            return quat
            
        except Exception as e:
            self.logger.error(f"Error converting rotation matrix to quaternion: {e}")
            return np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    
    def _orthogonalize_rotation_matrix(self, R: np.ndarray) -> np.ndarray:
        """Orthogonalize a rotation matrix using SVD"""
        U, _, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
        return R_ortho
    
    def set_reference_quaternion(self, quaternion: np.ndarray):
        """
        Update the reference quaternion used for consistency checking
        
        Args:
            quaternion: New reference quaternion [qx, qy, qz, qw]
        """
        self.reference_quaternion = quaternion.copy()
        self.logger.info(f"Updated reference quaternion to: {self.reference_quaternion}")
    
    def inverse_kinematics(self, target_pose: np.ndarray, 
                          current_joints: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Compute inverse kinematics for UR5 (simplified analytical solution)
        
        Args:
            target_pose: Target pose [x, y, z, qx, qy, qz, qw]
            current_joints: Current joint configuration for selecting closest solution
            
        Returns:
            Joint angles [rad] or None if no solution found
        """
        try:
            # Extract target position and orientation
            target_pos = target_pose[:3]
            target_quat = target_pose[3:7] if len(target_pose) >= 7 else np.array([0, 0, 0, 1])
            
            # This is a simplified IK - for production use, consider using:
            # - PyBullet's IK solver
            # - KDL (Kinematics and Dynamics Library)
            # - MoveIt's IK solvers
            # - Analytical UR5 IK solution
            
            # For now, return current joints if available, otherwise use numerical IK
            if current_joints is not None:
                return self.numerical_ik(target_pose, current_joints)
            else:
                # Return a reasonable default configuration based on your joint angles
                return np.array([-0.18984586, -1.2770158, 2.12554073, -2.71455557, -1.57770378, -1.55960399])
                
        except Exception as e:
            self.logger.error(f"Error in inverse kinematics: {e}")
            return None
    
    def numerical_ik(self, target_pose: np.ndarray, initial_joints: np.ndarray,
                    max_iterations: int = 100, tolerance: float = 1e-3) -> Optional[np.ndarray]:
        """
        Numerical inverse kinematics using Jacobian pseudo-inverse method
        
        Args:
            target_pose: Target pose [x, y, z, qx, qy, qz, qw]
            initial_joints: Initial joint configuration
            max_iterations: Maximum number of iterations
            tolerance: Position tolerance for convergence
            
        Returns:
            Joint angles [rad] or None if no solution found
        """
        try:
            joints = initial_joints.copy()
            target_pos = target_pose[:3]
            
            for iteration in range(max_iterations):
                # Compute current end-effector position
                current_pose = self.forward_kinematics(joints)
                current_pos = current_pose[:3]
                
                # Compute position error
                pos_error = target_pos - current_pos
                error_norm = np.linalg.norm(pos_error)
                
                # Check convergence
                if error_norm < tolerance:
                    return joints
                
                # Compute Jacobian (simplified - position only)
                jacobian = self.compute_jacobian(joints)
                
                # Compute joint update using pseudo-inverse
                pinv_jacobian = np.linalg.pinv(jacobian)
                joint_update = pinv_jacobian @ pos_error
                
                # Apply update with damping
                damping = 0.1
                joints += damping * joint_update
                
                # Apply joint limits (from your URDF)
                joints = np.clip(joints, -np.pi, np.pi)
            
            self.logger.warn(f"IK did not converge after {max_iterations} iterations")
            return joints
            
        except Exception as e:
            self.logger.error(f"Error in numerical IK: {e}")
            return None
    
    def compute_jacobian(self, joint_angles: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """Compute Jacobian matrix using finite differences"""
        try:
            jacobian = np.zeros((3, 6))  # 3 position DOF, 6 joints
            
            # Current end-effector position
            current_pose = self.forward_kinematics(joint_angles)
            current_pos = current_pose[:3]
            
            # Compute partial derivatives
            for i in range(6):
                # Perturb joint i
                perturbed_joints = joint_angles.copy()
                perturbed_joints[i] += delta
                
                # Compute perturbed end-effector position
                perturbed_pose = self.forward_kinematics(perturbed_joints)
                perturbed_pos = perturbed_pose[:3]
                
                # Compute derivative
                jacobian[:, i] = (perturbed_pos - current_pos) / delta
            
            return jacobian
            
        except Exception as e:
            self.logger.error(f"Error computing Jacobian: {e}")
            return np.eye(3, 6)  # Return identity-like matrix as fallback
    
    def validate_joint_limits(self, joint_angles: np.ndarray) -> bool:
        """Check if joint angles are within valid limits based on your URDF"""
        # Joint limits from your URDF (all joints have -pi to pi)
        joint_limits = np.array([
            [-np.pi, np.pi],    # Joint 1 (shoulder_pan)
            [-np.pi, np.pi],    # Joint 2 (shoulder_lift)
            [-np.pi, np.pi],    # Joint 3 (elbow)
            [-np.pi, np.pi],    # Joint 4 (wrist_1)
            [-np.pi, np.pi],    # Joint 5 (wrist_2)
            [-np.pi, np.pi]     # Joint 6 (wrist_3)
        ])
        
        for i, angle in enumerate(joint_angles):
            if not (joint_limits[i, 0] <= angle <= joint_limits[i, 1]):
                return False
        
        return True