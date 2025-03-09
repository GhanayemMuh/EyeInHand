import pyrealsense2 as rs
import cv2
import json
import numpy as np
import os
from xarm.wrapper import XArmAPI

def euler_to_rot_matrix(roll, pitch, yaw):
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx

def pose_to_transformation(pose_list):
    x, y, z, roll, pitch, yaw = pose_list
    roll_rad, pitch_rad, yaw_rad = np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)
    R = euler_to_rot_matrix(roll_rad, pitch_rad, yaw_rad)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3], T[:3, 3] = R, [x, y, z]
    return T

image_dir, pose_dir = "Cal2/RGBImgs/", "Cal2/T_base2ee/"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(pose_dir, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

robot_ip = '192.168.1.170'
arm = XArmAPI(robot_ip)
arm.connect()
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(0)

profile = pipeline.start(config)


# Load the JSON preset
preset_file = r"C:\Users\USER\Desktop\Project\v1.json"

# Get the device from the profile
device = profile.get_device()

# Enable and apply Advanced Mode for D435
try:
    # Check if the device supports Advanced Mode
    if not rs.rs400_advanced_mode.is_supported(device):
        print("Advanced Mode is not supported on this device.")
    else:
        advanced_mode = rs.rs400_advanced_mode(device)
        if not advanced_mode.is_enabled():
            advanced_mode.toggle_advanced_mode(True)
            print("Advanced Mode enabled.")

        # Load and apply the JSON preset
        with open(preset_file, 'r') as file:
            preset = file.read()  # Read the JSON as a string
            advanced_mode.load_json(preset)  # Apply the preset
        print("Preset loaded successfully!")
except Exception as e:
    print(f"Failed to apply JSON preset: {e}")

capture_count = 1

# Parameters for the ChArUco board
charuco_board_dims = (11, 8)  # Rows and columns
square_length = 0.015  # Square size in meters
marker_length = 0.011  # Marker size in meters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
board = cv2.aruco.CharucoBoard(charuco_board_dims, square_length, marker_length, dictionary)
board.setLegacyPattern(True)

try:
    print("Press 's' to capture an image and record the robot pose, or 'q' to quit.")
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Create a copy of the original frame for ChArUco detection
        charuco_display = color_image.copy()

        # Detect ArUco markers and the ChArUco board
        gray_image = cv2.cvtColor(charuco_display, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray_image, dictionary)

        # Draw detected markers and ChArUco board only on the second window
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(charuco_display, corners, ids)
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray_image, board
            )
            if retval > 0:
                cv2.aruco.drawDetectedCornersCharuco(charuco_display, charuco_corners, charuco_ids)

        # Display the live video stream and the ChArUco detection window
        cv2.imshow('RealSense', color_image)
        cv2.imshow('ChArUco Detection', charuco_display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            img_filename = os.path.join(image_dir, f"img_{capture_count:04d}.png")
            if cv2.imwrite(img_filename, color_image):
                print(f"Image saved: {img_filename}")
            else:
                print("Error saving image.")

            pose = arm.get_position()
            print("Raw pose returned:", pose)
            if isinstance(pose, tuple) and len(pose) == 2 and isinstance(pose[1], list):
                pose_data = pose[1]
                print("Parsed pose data:", pose_data)
                try:
                    T = pose_to_transformation(pose_data)
                    print("Computed transformation matrix:\n", T)
                    pose_filename = os.path.join(pose_dir, f"pose_{capture_count:04d}.npz")
                    np.savez(pose_filename, arr_0=T)
                    print(f"Pose saved: {pose_filename}")
                except Exception as e:
                    print("Error converting pose to transformation matrix:", e)
            else:
                print("Error: Unexpected pose format from the robot.")

            capture_count += 1

        elif key == ord('q'):
            print("Quitting capture mode.")
            break

except Exception as e:
    print("An error occurred during capture:", e)

finally:
    pipeline.stop()
    arm.disconnect()
    cv2.destroyAllWindows()
    print("Capture session ended.")
