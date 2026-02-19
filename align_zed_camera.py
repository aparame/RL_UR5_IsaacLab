#!/usr/bin/env python3.10
"""
Script to align real ZED2 camera with simulation depth data.
Usage:
    source /opt/ros/humble/setup.bash
    python3.10 align_zed_camera.py

Controls:
    'q': Quit
    'm': Toggle mode (Overlay -> Edges -> Difference)
    't': Toggle transparency/mix (in overlay mode)
"""

import sys
import os
import time
import numpy as np
import cv2
import threading

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
    from rclpy.qos import qos_profile_sensor_data
except ImportError:
    print("Error: ROS2 modules not found. Make sure to source your ROS2 setup script.")
    sys.exit(1)

# Constants
SIM_CSV_PATH = "/home/adi2440/Desktop/RL_UR5_IsaacLab/sim_raw_depth.csv"
CAMERA_TOPIC = "/zed/zed_node/depth/depth_registered"
MAX_DEPTH_VIS = 3.0  # Meters, for visualization scaling


class DepthAlignmentNode(Node):
    def __init__(self):
        super().__init__("depth_alignment_node")
        self.bridge = CvBridge()

        # 1. Load Simulation Depth
        self.get_logger().info(f"Loading simulation depth map from {SIM_CSV_PATH}...")
        try:
            # Use numpy to load CSV (handles 'inf' automatically)
            self.sim_depth_raw = np.loadtxt(
                SIM_CSV_PATH, delimiter=",", dtype=np.float32
            )
            self.get_logger().info(f"Loaded Sim Depth: {self.sim_depth_raw.shape}")
        except Exception as e:
            self.get_logger().error(f"Failed to load CSV: {e}")
            sys.exit(1)

        # Process Sim Depth
        # Mask infs
        self.sim_valid_mask = np.isfinite(self.sim_depth_raw)
        self.sim_depth_clean = np.nan_to_num(
            self.sim_depth_raw, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Create Sim Visualization
        self.sim_norm = self._normalize_depth(self.sim_depth_clean)
        self.sim_edges = cv2.Canny(self.sim_norm, 50, 150)
        self.sim_colormap = cv2.applyColorMap(self.sim_norm, cv2.COLORMAP_JET)

        # 2. Subscribe to Real Camera
        self.latest_real_depth = None
        self.lock = threading.Lock()

        self.sub = self.create_subscription(
            Image, CAMERA_TOPIC, self.depth_callback, qos_profile_sensor_data
        )
        self.get_logger().info(f"Subscribed to {CAMERA_TOPIC}")

        # State
        self.alpha = 0.5
        self.display_mode = 0  # 0: Overlay, 1: Edges, 2: Diff

    def _normalize_depth(self, depth_map):
        """Normalize depth map to 0-255 uint8 for visualization."""
        # Clip to useful range
        d = np.clip(depth_map, 0, MAX_DEPTH_VIS)
        # Normalize
        d_norm = (d / MAX_DEPTH_VIS * 255).astype(np.uint8)
        return d_norm

    def depth_callback(self, msg):
        try:
            # Convert ROS to OpenCV
            # Handle float32 depth
            if msg.encoding == "32FC1":
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding="passthrough"
                )

            with self.lock:
                self.latest_real_depth = cv_image.copy()

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Callback Error: {e}")

    def _get_center_of_mass(self, depth_map):
        """Calculate center of mass for 'close' objects (low depth values)."""
        # Invert depth so close objects have high weight
        # Only consider valid range (e.g. < 2.0m)
        d = depth_map.copy()
        mask = (d > 0.1) & (d < 2.0)
        if not np.any(mask):
            return 0, 0

        weights = np.zeros_like(d)
        weights[mask] = 2.0 - d[mask]  # Higher weight for closer objects

        total_weight = np.sum(weights)
        if total_weight == 0:
            return 0, 0

        # Grid of coordinates
        h, w = d.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))

        cx = np.sum(X * weights) / total_weight
        cy = np.sum(Y * weights) / total_weight

        return cx, cy

    def run_visualization(self):
        cv2.namedWindow("Alignment Tool", cv2.WINDOW_NORMAL)
        self.get_logger().info("Starting visualization... Press 'q' to quit.")

        target_h, target_w = self.sim_depth_clean.shape
        sim_cx, sim_cy = self._get_center_of_mass(self.sim_depth_clean)

        # UI Constants
        PANEL_HEIGHT = 150
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        while rclpy.ok():
            real_depth = None
            with self.lock:
                if self.latest_real_depth is not None:
                    real_depth = self.latest_real_depth.copy()

            if real_depth is None:
                # Waiting screen
                waiting = np.zeros((400, 600, 3), dtype=np.uint8)
                cv2.putText(
                    waiting,
                    "Waiting for ZED2 stream...",
                    (50, 200),
                    FONT,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Alignment Tool", waiting)
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break
                continue

            # Process Real Depth
            real_resized = cv2.resize(
                real_depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )
            real_resized = np.nan_to_num(real_resized, nan=0.0, posinf=0.0, neginf=0.0)

            # --- Analysis for Guidance ---
            # 1. Distance (Global Z bias)
            # Compare mean depth of "valid" pixels (e.g. < 2.0m)
            valid_mask = (
                (self.sim_depth_clean > 0.1)
                & (self.sim_depth_clean < 2.0)
                & (real_resized > 0.1)
                & (real_resized < 2.0)
            )

            if np.any(valid_mask):
                mean_sim = np.mean(self.sim_depth_clean[valid_mask])
                mean_real = np.mean(real_resized[valid_mask])
                diff_z = mean_real - mean_sim
            else:
                diff_z = 0.0

            # 2. Position (Center of Mass shift)
            real_cx, real_cy = self._get_center_of_mass(real_resized)
            diff_x = real_cx - sim_cx
            diff_y = real_cy - sim_cy

            # Formulate Guidance
            guidance = []

            # Z-Axis Advice
            if abs(diff_z) > 0.05:  # 5cm tolerance
                if diff_z < 0:
                    guidance.append(f"MOVE BACK ({abs(diff_z):.2f}m too close)")
                else:
                    guidance.append(f"MOVE FORWARD ({abs(diff_z):.2f}m too far)")
            else:
                guidance.append("DISTANCE: GOOD")

            # X-Axis Advice (Pan)
            # If real center is to the right (>0), camera needs to move Right to shift image Left?
            # Wait, coordinate logic:
            # If Camera moves Right, Scene moves Left.
            # If Real Scene is to the Right (cx > sim_cx), we need it to move Left.
            # So Camera must move Right.
            if abs(diff_x) > 10:  # Pixels tolerance
                if diff_x > 0:
                    guidance.append(f"PAN RIGHT (Offset: {diff_x:.0f}px)")
                else:
                    guidance.append(f"PAN LEFT (Offset: {abs(diff_x):.0f}px)")
            else:
                guidance.append("HORIZONTAL: ALIGNED")

            # Y-Axis Advice (Tilt/Height)
            # If Real Scene is Lower (cy > sim_cy [y positive down]), we need it to move Up.
            # Moving Camera Down shifts Scene Up.
            # So if cy > sim_cy (Real is lower), Move Camera Down.
            if abs(diff_y) > 10:
                if diff_y > 0:
                    guidance.append(f"MOVE/TILT DOWN (Offset: {diff_y:.0f}px)")
                else:
                    guidance.append(f"MOVE/TILT UP (Offset: {abs(diff_y):.0f}px)")
            else:
                guidance.append("VERTICAL: ALIGNED")

            # --- Visualization ---
            real_norm = self._normalize_depth(real_resized)
            real_colormap = cv2.applyColorMap(real_norm, cv2.COLORMAP_JET)

            img_disp = None
            if self.display_mode == 0:  # Overlay
                img_disp = cv2.addWeighted(
                    self.sim_colormap, self.alpha, real_colormap, 1.0 - self.alpha, 0
                )
                mode_text = f"Mode: Overlay (Alpha: {self.alpha:.1f}) | 't' to change"
            elif self.display_mode == 1:  # Edges
                real_edges = cv2.Canny(real_norm, 50, 150)
                sim_edge_color = np.zeros_like(real_colormap)
                sim_edge_color[self.sim_edges > 0] = [0, 255, 0]  # Green
                real_edge_color = np.zeros_like(real_colormap)
                real_edge_color[real_edges > 0] = [0, 0, 255]  # Red
                img_disp = cv2.addWeighted(sim_edge_color, 1.0, real_edge_color, 0.7, 0)
                mode_text = "Mode: Edges (Green=Sim, Red=Real)"
            elif self.display_mode == 2:  # Difference
                diff = cv2.absdiff(self.sim_norm, real_norm)
                img_disp = cv2.applyColorMap(diff, cv2.COLORMAP_TURBO)
                mode_text = "Mode: Difference Heatmap"

            # Create Panel
            panel = np.zeros((PANEL_HEIGHT, target_w, 3), dtype=np.uint8)

            # Line 1: Mode & Keybinds
            cv2.putText(
                panel,
                f"{mode_text} | 'm': Cycle Mode | 'q': Quit",
                (20, 30),
                FONT,
                0.7,
                (200, 200, 200),
                1,
            )

            # Line 2: Guidance Header
            color_res = (
                (0, 255, 0)
                if "GOOD" in guidance[0] and "ALIGNED" in guidance[1]
                else (0, 255, 255)
            )
            cv2.putText(
                panel, "ALIGNMENT GUIDANCE:", (20, 70), FONT, 0.7, (255, 255, 255), 2
            )

            # Line 3: Actual Instructions
            x_start = 20
            for i, txt in enumerate(guidance):
                # Color code
                c = (0, 255, 0) if "GOOD" in txt or "ALIGNED" in txt else (0, 0, 255)
                cv2.putText(panel, f"{txt}", (x_start, 110), FONT, 0.8, c, 2)
                x_start += 350  # Spacing

            # Combine
            final_frame = np.vstack([img_disp, panel])

            cv2.imshow("Alignment Tool", final_frame)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                self.display_mode = (self.display_mode + 1) % 3
            elif key == ord("t"):
                self.alpha += 0.1
                if self.alpha > 1.0:
                    self.alpha = 0.0

        cv2.destroyAllWindows()


def main(args=None):
    # Check for ROS environment
    if "ROS_DISTRO" not in os.environ:
        print(
            "Warning: ROS environment not detected. Did you 'source /opt/ros/humble/setup.bash'?"
        )

    rclpy.init(args=args)
    node = DepthAlignmentNode()

    # Spin in thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run_visualization()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
