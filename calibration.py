import os
import cv2
import numpy as np
import glob
import argparse
import pickle
from scipy.spatial.transform import Rotation

def visualize_and_save_corners(image_files, all_corners, all_ids, output_dir):
    """ Visualize detected ArUco markers and save images with overlay. """
    os.makedirs(output_dir, exist_ok=True)

    for i, (image_file, corners, ids) in enumerate(zip(image_files, all_corners, all_ids)):
        image = cv2.imread(image_file)
        if ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
        output_path = os.path.join(output_dir, f"corners_{i:03d}.png")
        cv2.imwrite(output_path, image)

def compute_reprojection_error(T_cam2ee, camera_matrix, dist_coeffs, robot_poses, camera_poses, board_corners, detected_data):
    """
    Compute reprojection error for the calibration.
    :param T_cam2ee: Transformation from camera to end-effector.
    :param camera_matrix: Intrinsic camera matrix.
    :param dist_coeffs: Camera distortion coefficients.
    :param robot_poses: List of T_base2ee (robot poses in base frame).
    :param camera_poses: List of T_cam (camera poses relative to the board).
    :param board_corners: 3D positions of ChArUco board corners in the board's coordinate frame.
    :param detected_data: Detected corners and IDs from validation images.
    """
    reprojection_errors = []

    for i, (T_base2ee, T_cam, detection) in enumerate(zip(robot_poses, camera_poses, detected_data)):
        detected_corners = detection["corners"]  # Detected 2D corners
        detected_ids = detection["ids"]  # Detected corner IDs

        if detected_corners is None or detected_ids is None or len(detected_ids) == 0:
            print(f"Image {i}: No detected corners. Skipping.")
            continue

        # Filter board corners based on detected IDs
        detected_ids = detected_ids.flatten()
        valid_board_corners = board_corners[detected_ids]

        # Compute T_board2base
        T_board2base = T_base2ee @ T_cam2ee @ np.linalg.inv(T_cam)

        # Transform valid board corners to the base frame
        valid_board_corners_h = np.hstack((valid_board_corners, np.ones((valid_board_corners.shape[0], 1))))  # Homogeneous coords
        projected_points_h = (T_board2base @ valid_board_corners_h.T).T
        projected_points = projected_points_h[:, :3] / projected_points_h[:, 3:]  # Normalize to 3D

        # Project points to image plane
        image_points, _ = cv2.projectPoints(
            projected_points,
            np.zeros(3), np.zeros(3),
            camera_matrix, dist_coeffs
        )

        # Compute reprojection error only for the valid points
        detected_corners = detected_corners.squeeze()
        image_points = image_points.squeeze()
        error = np.linalg.norm(image_points - detected_corners, axis=1).mean()
        reprojection_errors.append(error)

    # Compute average reprojection error
    avg_error = np.mean(reprojection_errors) if reprojection_errors else float("inf")
    return reprojection_errors, avg_error

def calibrate_camera_to_ee():
    # Parameters for the ChArUco board
    charuco_board_dims = (11, 8)  # Rows and columns
    square_length = 0.015  # Square size in meters
    marker_length = 0.011  # Marker size in meters
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

    board = cv2.aruco.CharucoBoard(charuco_board_dims, square_length, marker_length, dictionary)
    board.setLegacyPattern(True)

    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # Example refinement
    aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Paths
    image_folder = "Cal2/RGBImgs"
    poses_folder = "Cal2/T_base2ee"

    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    pose_files = sorted(glob.glob(os.path.join(poses_folder, "*.npz")))

    assert len(image_files) == len(pose_files), "Number of images and poses must match."

    # Intrinsic calibration
    all_corners, all_ids, charuco_corners, charuco_ids, robot_poses = [], [], [], [], []

    for image_file, pose_file in zip(image_files, pose_files):
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco_detector.detectMarkers(gray)

        if ids is not None and len(corners) > 0:
            retval, charuco_corners_img, charuco_ids_img = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners, markerIds=ids, image=gray, board=board)

            if retval > 0:
                charuco_corners.append(charuco_corners_img)
                charuco_ids.append(charuco_ids_img)

                with np.load(pose_file) as pose_data:
                    T_base2ee = pose_data["arr_0"]
                    T_base2ee[:3, 3] /= 1000.0  # Convert translation from mm to meters
                    robot_poses.append(T_base2ee)

    assert len(charuco_corners) > 10, "Not enough valid detections for calibration."

    visualize_and_save_corners(image_files, all_corners, all_ids, output_dir="corners")

    # Perform intrinsic calibration
    retval, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
        charuco_corners, charuco_ids, board, gray.shape[::-1], None, None)

    print("Intrinsic Calibration Results:")
    print(f"Camera Matrix:\n{camera_matrix}")
    print(f"Distortion Coefficients:\n{dist_coeffs}")

    camera_poses = []
    for charuco_corners_img, charuco_ids_img in zip(charuco_corners, charuco_ids):
        if charuco_ids_img is None or len(charuco_ids_img) == 0:
            print("No valid ChArUco corners detected for pose estimation. Skipping this image.")
            continue

        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charucoCorners=charuco_corners_img,
            charucoIds=charuco_ids_img,
            board=board,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.zeros((3, 1), dtype=np.float64)
        )

        if retval > 0:  # Pose estimation succeeded
            R, _ = cv2.Rodrigues(rvec)
            T_cam = np.eye(4)
            T_cam[:3, :3] = R
            T_cam[:3, 3] = tvec.ravel()
            camera_poses.append(T_cam)
        else:
            print("Pose estimation failed for this image. Skipping.")

    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
        np.array(robot_poses)[:, :3, :3], np.array(robot_poses)[:, :3, 3],
        np.array(camera_poses)[:, :3, :3], np.array(camera_poses)[:, :3, 3])

    # Define ZYX Euler angles for a 90-degree rotation around the Z-axis (yaw = 90, pitch = 0, roll = 0)
    euler_angles_zyx = [90, 0, 0]  # in degrees

    # Create the rotation matrix
    rotation = Rotation.from_euler('zyx', euler_angles_zyx, degrees=True)
    rotation_matrix = rotation.as_matrix()
    T_cam2ee = np.eye(4)
    #T_cam2ee[:3, :3] = R_cam2ee
    T_cam2ee[:3, :3] = rotation_matrix
    T_cam2ee[:3, 3] = t_cam2ee.ravel()
    T_cam2ee[2,3] = -T_cam2ee[2,3]

    # Save calibration results
    np.savez("hand_eye_calibration_results.npz", T_cam2ee=T_cam2ee, camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)
    np.save("camera_poses.npy", camera_poses)
    np.save("robot_poses.npy", robot_poses)

    # Save detected corners and IDs
    charuco_data = [{"corners": corners, "ids": ids} for corners, ids in zip(charuco_corners, charuco_ids)]
    with open("detected_charuco_corners.pkl", "wb") as f:
        pickle.dump(charuco_data, f)

    print("Calibration completed and saved.")