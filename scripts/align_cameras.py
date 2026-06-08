#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import threading
import argparse


class AlignmentTool(Node):
    def __init__(self, ref_image_path):
        super().__init__("camera_align_tool")
        self.bridge = CvBridge()

        # Load reference
        ref_image = cv2.imread(ref_image_path)
        if ref_image is None:
            self.get_logger().error(
                f"Could not load reference image from {ref_image_path}"
            )
            raise Exception("Failed to load reference image")

        # Target shape is from policy (120x160)
        self.target_h, self.target_w = 120, 160

        # Extract the middle Gray panel from the reference plot (which is 3 side-by-side plots)
        # We'll just display the full reference image in a separate window to avoid parsing matplotlib layout
        self.ref_image = cv2.resize(ref_image, (1200, 400))

        self.latest_rgb = None
        self.latest_depth = None
        self.lock = threading.Lock()

        self.rgb_sub = self.create_subscription(
            CompressedImage,
            "/zed/zed_node/rgb/color/rect/image/compressed",
            self.rgb_callback,
            qos_profile_sensor_data,
        )

        self.depth_sub = self.create_subscription(
            Image,
            "/zed/zed_node/depth/depth_registered",
            self.depth_callback,
            qos_profile_sensor_data,
        )

    def rgb_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            with self.lock:
                self.latest_rgb = cv_img
        except Exception as e:
            self.get_logger().error(f"RGB Error: {e}")

    def depth_callback(self, msg):
        try:
            if msg.encoding == "32FC1":
                depth_img = np.frombuffer(msg.data, dtype=np.float32).reshape(
                    (msg.height, msg.width)
                )
                depth_img = np.nan_to_num(depth_img, nan=20.0, posinf=20.0, neginf=0.3)

                # Normalize exactly like the Sim [0.3, 20.0] -> [0, 1]
                depth_img = np.clip(depth_img, 0.3, 20.0)
                norm_depth = (depth_img - 0.3) / (20.0 - 0.3)

                with self.lock:
                    self.latest_depth = norm_depth
        except Exception as e:
            self.get_logger().error(f"Depth Error: {e}")

    def get_processed_frames(self):
        with self.lock:
            rgb = self.latest_rgb.copy() if self.latest_rgb is not None else None
            depth = self.latest_depth.copy() if self.latest_depth is not None else None

        if rgb is None:
            return None, None

        # Matching process logic from ur5_gym_env.py
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray_norm = gray.astype(np.float32) / 255.0

        if depth is None:
            depth = np.zeros_like(gray_norm)
        elif depth.shape != gray_norm.shape:
            depth = cv2.resize(
                depth,
                (gray_norm.shape[1], gray_norm.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Combine [2, H, W]
        combined = np.stack([gray_norm, depth], axis=0)

        # Interpolate to 640x480
        # (Opencv expects HWC for operations, so we manually do it per channel)
        gray_640 = cv2.resize(combined[0], (640, 480), interpolation=cv2.INTER_LINEAR)
        depth_640 = cv2.resize(combined[1], (640, 480), interpolation=cv2.INTER_LINEAR)

        # Crop Top 60, Bottom 20
        gray_crop = gray_640[60:-20, :]
        depth_crop = depth_640[60:-20, :]

        # Target Resize
        final_gray = cv2.resize(
            gray_crop, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR
        )
        final_depth = cv2.resize(
            depth_crop, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR
        )

        # Mean subtraction per channel as executed by tensor
        final_gray = final_gray - np.mean(final_gray)
        final_depth = final_depth - np.mean(final_depth)

        return final_gray, final_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref", type=str, required=True, help="Path to Isaac Lab reference image"
    )
    args = parser.parse_args()

    rclpy.init()
    node = AlignmentTool(args.ref)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    print("Starting Alignment Interface...")
    print("Press 'q' or 'ESC' to exit")

    cv2.namedWindow("Isaac Reference", cv2.WINDOW_NORMAL)
    cv2.imshow("Isaac Reference", node.ref_image)

    while True:
        gray, depth = node.get_processed_frames()
        if gray is not None:
            # Shift back for visualization
            viz_gray = np.clip(gray + 0.5, 0, 1)
            viz_gray = (viz_gray * 255).astype(np.uint8)

            # Map depth to color map
            viz_depth = np.clip(depth + 0.5, 0, 1)
            viz_depth_color = cv2.applyColorMap(
                (viz_depth * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
            )

            # Upscale for better viewing
            display_scale = 4
            d_gray = cv2.resize(viz_gray, (0, 0), fx=display_scale, fy=display_scale)
            d_depth = cv2.resize(
                viz_depth_color, (0, 0), fx=display_scale, fy=display_scale
            )

            cv2.imshow("ZED2 Processed Gray (Match to center pic)", d_gray)
            cv2.imshow("ZED2 Processed Depth (Match to right pic)", d_depth)

        key = cv2.waitKey(30)
        if key in [27, ord("q")]:
            break

    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
