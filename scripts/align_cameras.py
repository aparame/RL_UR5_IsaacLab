#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import cv2.aruco as aruco
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import sys

class CharucoAlignmentNode(Node):
    def __init__(self):
        super().__init__('charuco_alignment_node')
        
        # --- Configuration (Matches create_charuco.py) ---
        self.squares_x = 5
        self.squares_y = 7
        self.square_length = 0.035  # meters
        self.marker_length = 0.026  # meters
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        
        # --- Compatibility Setup ---
        self.use_new_api = False
        if hasattr(aruco, 'CharucoDetector'):
            self.use_new_api = True
            self.get_logger().info("Using OpenCV 4.7+ CharucoDetector API")
        else:
            self.get_logger().info("Using Legacy OpenCV Aruco API")

        # --- Detector Parameters for Difficult Angles ---
        self.det_params = aruco.DetectorParameters() if hasattr(aruco, 'DetectorParameters') else aruco.DetectorParameters_create()
        self.det_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.det_params.minMarkerPerimeterRate = 0.01  # Detect smaller markers (due to perspective foreshortening)
        self.det_params.adaptiveThreshWinSizeStep = 5
        self.det_params.adaptiveThreshWinSizeMin = 3
        self.det_params.adaptiveThreshWinSizeMax = 23
        
        # Create Charuco Board
        if self.use_new_api:
            self.board = aruco.CharucoBoard((self.squares_x, self.squares_y), 
                                            self.square_length, 
                                            self.marker_length, 
                                            self.dictionary)
            self.charuco_detector = aruco.CharucoDetector(self.board, detectorParams=self.det_params)
        else:
            try:
                self.board = aruco.CharucoBoard_create(self.squares_x, self.squares_y, 
                                                       self.square_length, 
                                                       self.marker_length, 
                                                       self.dictionary)
            except AttributeError:
                # Fallback
                self.board = aruco.CharucoBoard((self.squares_x, self.squares_y), 
                                                self.square_length, 
                                                self.marker_length, 
                                                self.dictionary)

        # Utils
        self.bridge = CvBridge()
        
        # State
        self.real_image = None
        self.sim_image = None
        self.real_cam_info = None
        self.sim_cam_info = None
        
        # Topics
        self.real_image_topic = '/zed/zed_node/rgb/color/rect/image'
        self.real_info_topic = '/zed/zed_node/rgb/color/rect/camera_info'
        self.sim_image_topic = '/rgb_left_node/rgb_left'
        self.sim_info_topic = '/rgb_left_node/camera_info' 
        
        # Subscribers
        self.create_subscription(Image, self.real_image_topic, self.real_image_callback, 10)
        self.create_subscription(CameraInfo, self.real_info_topic, self.real_info_callback, 10)
        self.create_subscription(Image, self.sim_image_topic, self.sim_image_callback, 10)
        self.create_subscription(CameraInfo, self.sim_info_topic, self.sim_info_callback, 10)
        
        print(f"Waiting for images on:\n REAL: {self.real_image_topic}\n SIM:  {self.sim_image_topic}")

        # Timer for processing and display
        self.create_timer(0.1, self.process_and_display) # 10 Hz

    def real_image_callback(self, msg):
        try:
            self.real_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge (Real) Error: {e}")

    def sim_image_callback(self, msg):
        try:
            # Isaac Sim sometimes sends rgb8 or rgba8
            if msg.encoding == 'rgba8':
                image = self.bridge.imgmsg_to_cv2(msg, "rgba8")
                self.sim_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                self.sim_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge (Sim) Error: {e}")

    def real_info_callback(self, msg):
        self.real_cam_info = msg

    def sim_info_callback(self, msg):
        self.sim_cam_info = msg
        
    def get_cam_matrix_dist(self, cam_info_msg):
        if cam_info_msg is None:
            return None, None
            
        # Use Projection matrix 'P' if available and looks valid (not all zeros)
        # P is 3x4 (12 elements). We want the 3x3 intrinsic part.
        p_mat = np.array(cam_info_msg.p).reshape((3, 4))
        k_mat = np.array(cam_info_msg.k).reshape((3, 3))
        
        # Check if P is populated (usually P[0,0] > 0)
        if p_mat[0,0] > 0:
            mtx = p_mat[:, :3]
            dist = np.zeros(5) # Rectified images have no distortion
        else:
            mtx = k_mat
            dist = np.array(cam_info_msg.d)
            
        return mtx, dist

    def detect_and_estimate(self, image, cam_info, label):
        if image is None:
            return None, None, image

        display_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        charuco_corners = None
        charuco_ids = None
        
        # Detection
        if self.use_new_api:
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(image)
        else:
            try:
                corners, ids, rejected = aruco.detectMarkers(gray, self.dictionary, parameters=self.det_params)
                if len(corners) > 0:
                    retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                        corners, ids, gray, self.board)
            except AttributeError:
                if hasattr(aruco, "ArucoDetector"):
                     detector = aruco.ArucoDetector(self.dictionary, self.det_params)
                     corners, ids, rejected = detector.detectMarkers(gray)
                     if len(corners) > 0:
                         retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                            corners, ids, gray, self.board)
                else: 
                     return None, None, display_image

        # Drawing - customized per user request
        # For SIM: we can keep standard drawing or minimal. 
        # For REAL: User explicitly asked to NOT show all ids/markers.
        if label == "SIM":
            if charuco_corners is not None and len(charuco_corners) > 0:
                aruco.drawDetectedCornersCharuco(display_image, charuco_corners, charuco_ids)

        rvec = None
        tvec = None

        if charuco_corners is not None and len(charuco_corners) > 0:
             mtx, dist = self.get_cam_matrix_dist(cam_info)
             
             if mtx is not None:
                 valid = False
                 if self.use_new_api:
                     try:
                         obj_points, img_points = self.board.matchImagePoints(charuco_corners, charuco_ids)
                         if len(obj_points) >= 4:
                             valid, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)
                     except Exception:
                         pass
                 else:
                     try:
                         valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
                             charuco_corners, charuco_ids, self.board, mtx, dist, None, None)
                     except AttributeError:
                         pass
                 
                 # Draw Axes for SIM only in its own view
                 if valid and label == "SIM":
                     cv2.drawFrameAxes(display_image, mtx, dist, rvec, tvec, 0.1)

        return rvec, tvec, display_image

    def process_and_display(self):
        r_rvec, r_tvec, r_disp = self.detect_and_estimate(self.real_image, self.real_cam_info, "REAL")
        s_rvec, s_tvec, s_disp = self.detect_and_estimate(self.sim_image, self.sim_cam_info, "SIM")
        
        aligned = False
        dist_diff = 1000.0 # Default high
        ang_diff = 1000.0
        
        # --- Check Alignment & Stats ---
        if r_tvec is not None and s_tvec is not None:
            # Calculate difference
            t_diff = r_tvec.flatten() - s_tvec.flatten()
            dist_diff = np.linalg.norm(t_diff)
            
            r_diff = r_rvec.flatten() - s_rvec.flatten()
            ang_diff = np.linalg.norm(r_diff)
            
            # Thresholds for "Green" dot
            pos_thresh = 0.02 # 2cm
            rot_thresh = 0.1  # ~5.7 degrees
            
            if dist_diff < pos_thresh and ang_diff < rot_thresh:
                aligned = True
            
            # Print debug for user
            print(f"REAL: {r_tvec.flatten()} | SIM: {s_tvec.flatten()} | DIFF: {dist_diff:.3f}", end='\r')

        # --- Draw REAL Overlays (User Requests) ---
        if r_disp is not None:
            r_mtx, r_dist = self.get_cam_matrix_dist(self.real_cam_info)
            
            # 1. Overlay SIM Frame onto REAL Image (2D to 2D Projection)
            # Instead of using Real Intrinsics (which might differ), we take the 2D pixels 
            # of the board in the Sim View and overlay them on the Real View.
            # This creates a visual "Ghost" target that is robust to intrinsic differences.
            if s_tvec is not None and self.sim_image is not None:
                s_h, s_w = self.sim_image.shape[:2]
                r_h, r_w = r_disp.shape[:2]
                
                # Get Sim Intrinsics
                s_mtx, s_dist = self.get_cam_matrix_dist(self.sim_cam_info)
                
                if s_mtx is not None:
                    try:
                        # Define Board Outline in Board Frame
                        w = self.squares_x * self.square_length
                        h = self.squares_y * self.square_length
                        board_corners_3d = np.array([
                            [0, 0, 0],
                            [w, 0, 0],
                            [w, h, 0],
                            [0, h, 0]
                        ], dtype=np.float32)
                        
                        # Project to Sim Image (UV)
                        s_img_pts, _ = cv2.projectPoints(board_corners_3d, s_rvec, s_tvec, s_mtx, s_dist)
                        
                        # Scale UVs to Real Image Resolution
                        # This assumes we want to match the "relative screen position"
                        scale_x = r_w / float(s_w)
                        scale_y = r_h / float(s_h)
                        
                        s_img_pts_scaled = s_img_pts.copy()
                        s_img_pts_scaled[:, :, 0] *= scale_x
                        s_img_pts_scaled[:, :, 1] *= scale_y
                        
                        # Draw Cyan Ghost on Real Image
                        cv2.polylines(r_disp, [s_img_pts_scaled.astype(int)], True, (0, 255, 255), 2)
                        cv2.putText(r_disp, "TARGET (Sim Ghost)", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                    except Exception as e:
                       print(f"Overlay Error: {e}")

            # 2. Draw Single Dot for Real Board
            if r_tvec is not None and r_mtx is not None:
                # Calculate center of board
                # Board Origin is usually bottom-left corner. Center is roughly half widths.
                # X axis is horizontal, Y is vertical on board.
                center_local_3d = np.array([
                    (self.squares_x * self.square_length) / 2.0,
                    (self.squares_y * self.square_length) / 2.0,
                    0.0
                ])
                
                try:
                    # Project this point to image
                    img_pts, _ = cv2.projectPoints(center_local_3d.reshape(1,3), r_rvec, r_tvec, r_mtx, r_dist)
                    center_px = tuple(img_pts[0][0].astype(int))
                    
                    # Color: Green if aligned, Red otherwise
                    color = (0, 255, 0) if aligned else (0, 0, 255)
                    
                    # Draw Dot
                    cv2.circle(r_disp, center_px, 8, color, -1) # Filled circle
                    cv2.circle(r_disp, center_px, 10, (255, 255, 255), 2) # White outline for visibility
                except Exception:
                    pass
        
        # --- Visualization Layout ---
        if r_disp is not None and s_disp is not None:
            # 1. Image Row
            h1, w1 = r_disp.shape[:2]
            h2, w2 = s_disp.shape[:2]
            
            if h1 != h2 and h2 > 0:
                scale = h1 / float(h2)
                w2_new = int(w2 * scale)
                s_disp = cv2.resize(s_disp, (w2_new, h1))
            
            cv2.putText(r_disp, "REAL CAMERA", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(s_disp, "SIMULATION", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            images_row = np.hstack((r_disp, s_disp))
            total_w = images_row.shape[1]
            
            # Reduced Stats Panel
            panel_h = 100
            stats_panel = np.zeros((panel_h, total_w, 3), dtype=np.uint8)
            
            info_text = "Align the Red Dot to the Target Frame"
            if aligned:
                info_text = "ALIGNED! (Green)"
                
            cv2.putText(stats_panel, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if aligned else (0,0,255), 2)
            
            if r_tvec is not None and s_tvec is not None:
                err_text = f"Err: {dist_diff*100:.1f} cm, {np.degrees(ang_diff):.1f} deg"
                cv2.putText(stats_panel, err_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            else:
                cv2.putText(stats_panel, "Detecting...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

            final_img = np.vstack((images_row, stats_panel))
            cv2.imshow("Charuco Alignment Tool", final_img)
            cv2.waitKey(1)
            
        elif r_disp is not None:
            cv2.putText(r_disp, "Waiting for Sim...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Charuco Alignment Tool", r_disp)
            cv2.waitKey(1)
        elif s_disp is not None:
             cv2.putText(s_disp, "Waiting for Real...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
             cv2.imshow("Charuco Alignment Tool", s_disp)
             cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CharucoAlignmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Robust shutdown
        try:
            if rclpy.ok():
                node.destroy_node()
                rclpy.shutdown()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
