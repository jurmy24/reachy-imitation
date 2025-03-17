# This script extracts 3D points from a human using a depth camera (Intel RealSense D435i)
from typing import Literal
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np


def _setup() -> tuple[mp.solutions.hands, mp.solutions.pose, rs.align, rs.pipeline]:
    # Initialize MediaPipe for hand and body point map detection
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    # Configure intel RealSense camera (color and depth streams)
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Align depth frame to color frame # TODO: check what this means
    align = rs.align(rs.stream.color)
    pipeline = rs.pipeline()
    pipeline.start(config)

    return (mp_hands, mp_pose, align, pipeline)


def _get_3D_coordinates(landmark, depth_frame, w, h, intrinsics):
    """Convert a landmark to 3D coordinates using depth information.

    Transforms from camera coordinates to robot coordinates:
    - Robot x = -Camera depth
    - Robot y = -Camera x
    - Robot z = -Camera y
    """
    cx, cy = int(landmark.x * w), int(landmark.y * h)

    # Check if pixel coordinates are within image bounds
    if 0 <= cx < w and 0 <= cy < h:
        depth = depth_frame.get_distance(cx, cy)
        if depth > 0:  # Ensure depth is valid
            # Get camera intrinsic parameters
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy

            # Convert to camera 3D coordinates
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy

            # Transform to robot coordinate system
            return np.array([-depth, -x, -y])

    # Default return if coordinates are invalid
    return np.array([0, 0, 0])


def _get_right_arm_coordinates(
    mp_pose, mp_hands, pose_landmark, hand_results, depth_frame, w, h, intrinsics
):
    right_arm_points = {
        "shoulder_right": _get_3D_coordinates(
            pose_landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            depth_frame,
            w,
            h,
            intrinsics,
        ),
        "elbow_right": _get_3D_coordinates(
            pose_landmark[mp_pose.PoseLandmark.RIGHT_ELBOW],
            depth_frame,
            w,
            h,
            intrinsics,
        ),
        "wrist_right": _get_3D_coordinates(
            pose_landmark[mp_pose.PoseLandmark.RIGHT_WRIST],
            depth_frame,
            w,
            h,
            intrinsics,
        ),
    }

    # Get relative coordinates of the right arm with shoulder as origin
    right_arm_relative_coordinates = {}
    for name, coord in right_arm_points.items():
        right_arm_relative_coordinates[name] = (
            coord - right_arm_points["shoulder_right"]
        )

    # Filter for the right hand among detected hands
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for i, hand_landmark in enumerate(hand_results.multi_hand_landmarks):
            # Check if this is the right hand
            hand_type = hand_results.multi_handedness[i].classification[0].label
            if hand_type == "Right":  # Only process if it's the right hand
                index_mcp = _get_3D_coordinates(
                    hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    depth_frame,
                    w,
                    h,
                    intrinsics,
                )
                pinky_mcp = _get_3D_coordinates(
                    hand_landmark.landmark[mp_hands.HandLandmark.PINKY_MCP],
                    depth_frame,
                    w,
                    h,
                    intrinsics,
                )

                right_arm_relative_coordinates["index_mcp"] = (
                    index_mcp - right_arm_points["shoulder_right"]
                )
                right_arm_relative_coordinates["pinky_mcp"] = (
                    pinky_mcp - right_arm_points["shoulder_right"]
                )
                break  # Found the right hand, no need to continue

    return right_arm_relative_coordinates


def _get_left_arm_coordinates(
    mp_pose, mp_hands, pose_landmark, hand_results, depth_frame, w, h, intrinsics
):
    left_arm_points = {
        "shoulder_left": _get_3D_coordinates(
            pose_landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
            depth_frame,
            w,
            h,
            intrinsics,
        ),
        "elbow_left": _get_3D_coordinates(
            pose_landmark[mp_pose.PoseLandmark.LEFT_ELBOW],
            depth_frame,
            w,
            h,
            intrinsics,
        ),
        "wrist_left": _get_3D_coordinates(
            pose_landmark[mp_pose.PoseLandmark.LEFT_WRIST],
            depth_frame,
            w,
            h,
            intrinsics,
        ),
    }

    # Get relative coordinates of the left arm with shoulder as origin
    left_arm_relative_coordinates = {}
    for name, coord in left_arm_points.items():
        left_arm_relative_coordinates[name] = coord - left_arm_points["shoulder_left"]

    # Filter for the left hand among detected hands
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for i, hand_landmark in enumerate(hand_results.multi_hand_landmarks):
            # Check if this is the left hand
            hand_type = hand_results.multi_handedness[i].classification[0].label
            if hand_type == "Left":  # Only process if it's the left hand
                index_mcp = _get_3D_coordinates(
                    hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    depth_frame,
                    w,
                    h,
                    intrinsics,
                )
                pinky_mcp = _get_3D_coordinates(
                    hand_landmark.landmark[mp_hands.HandLandmark.PINKY_MCP],
                    depth_frame,
                    w,
                    h,
                    intrinsics,
                )

                left_arm_relative_coordinates["index_mcp"] = (
                    index_mcp - left_arm_points["shoulder_left"]
                )
                left_arm_relative_coordinates["pinky_mcp"] = (
                    pinky_mcp - left_arm_points["shoulder_left"]
                )
                break  # Found the left hand, no need to continue

    return left_arm_relative_coordinates


def run(arm: Literal["right", "left", "both"] = "right"):
    # Initialize MediaPipe and RealSense camera for hand and body point map detection
    mp_hands, mp_pose, align, pipeline = _setup()

    try:
        while True:
            # Get frames from RealSense camera
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            # Check if frames are valid, if not, skip
            if not color_frame or not depth_frame:
                continue

            # OpenCV uses BGR format, MediaPipe uses RGB format, so we convert the color image
            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Get intrinsics of the RealSense camera, including fx, fy (focal lengths in pixels), ppx, ppy (optical center in pixels)
            # These are used to convert pixel coordinates to 3D real-world coordinates
            intrinsics = (
                pipeline.get_active_profile()
                .get_stream(rs.stream.depth)
                .as_video_stream_profile()
                .get_intrinsics()
            )

            # Get height and width of the color image
            h, w, _ = color_image.shape

            # Run computer vision pose and hand detection
            pose_results = mp_pose.Pose().process(rgb_image)
            hand_results = mp_hands.Hands().process(rgb_image)

            # Initialize arm coordinates
            right_arm_coordinates = {}
            left_arm_coordinates = {}

            if pose_results.pose_landmarks and hand_results.multi_hand_landmarks:
                pose_landmark = pose_results.pose_landmarks.landmark

                if arm == "right" or arm == "both":
                    right_arm_coordinates = _get_right_arm_coordinates(
                        mp_pose,
                        mp_hands,
                        pose_landmark,
                        hand_results,
                        depth_frame,
                        w,
                        h,
                        intrinsics,
                    )

                if arm == "left" or arm == "both":
                    left_arm_coordinates = _get_left_arm_coordinates(
                        mp_pose,
                        mp_hands,
                        pose_landmark,
                        hand_results,
                        depth_frame,
                        w,
                        h,
                        intrinsics,
                    )

            # Display 3D coordinates on the image
            y_offset = 30
            for name, coord in right_arm_coordinates.items():
                x, y, z = coord
                cv2.putText(
                    color_image,
                    f"R_{name}: ({x:.2f}, {y:.2f}, {z:.2f})m",
                    (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),  # Green for right arm
                    2,
                )
                y_offset += 20

            for name, coord in left_arm_coordinates.items():
                x, y, z = coord
                cv2.putText(
                    color_image,
                    f"L_{name}: ({x:.2f}, {y:.2f}, {z:.2f})m",
                    (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),  # Blue for left arm
                    2,
                )
                y_offset += 20

            # Set window title based on which arm(s) is being tracked
            window_title = "RealSense "
            if arm == "right":
                window_title += "Right Arm"
            elif arm == "left":
                window_title += "Left Arm"
            elif arm == "both":
                window_title += "Both Arms"
            window_title += " 3D Coordinates"

            # Display the image
            cv2.imshow(window_title, color_image)

            # Quit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(arm="right")
