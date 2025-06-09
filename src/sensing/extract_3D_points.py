# This script extracts 3D points from a human using a depth camera (Intel RealSense D435i)
import numpy as np
from src.utils.three_d import get_3D_coordinates_reachy_perspective, get_3D_coordinates


def calculate_arm_coordinates(
    pose, hands, mp_pose, mp_hands, intrinsics, rgb_image, depth_frame, w, h, arm
):
    """
    Extracts 3D coordinates for arm joints and hand landmarks relative to shoulder position.
    Returns dictionaries containing coordinates for both arms if requested.
    """
    # Initialize arm coordinates for this frame
    right_arm_coordinates = {}
    left_arm_coordinates = {}

    # Run computer vision pose and hand detection
    pose_results = pose.process(rgb_image)
    hand_results = hands.process(rgb_image)

    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark

        # Process right arm if requested
        if arm == "right" or arm == "both":
            # Get right arm joint positions
            right_shoulder = get_3D_coordinates(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                depth_frame,
                w,
                h,
                intrinsics,
            )
            right_elbow = get_3D_coordinates(
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                depth_frame,
                w,
                h,
                intrinsics,
            )
            right_wrist = get_3D_coordinates(
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
                depth_frame,
                w,
                h,
                intrinsics,
            )

            # Store points relative to shoulder
            right_arm_coordinates["shoulder_right"] = np.array([0, 0, 0])  # Origin
            right_arm_coordinates["elbow_right"] = right_elbow - right_shoulder
            right_arm_coordinates["wrist_right"] = right_wrist - right_shoulder

        # Process left arm if requested
        if arm == "left" or arm == "both":
            # Get left arm joint positions
            left_shoulder = get_3D_coordinates(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                depth_frame,
                w,
                h,
                intrinsics,
            )
            left_elbow = get_3D_coordinates(
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                depth_frame,
                w,
                h,
                intrinsics,
            )
            left_wrist = get_3D_coordinates(
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
                depth_frame,
                w,
                h,
                intrinsics,
            )

            # Store points relative to shoulder
            left_arm_coordinates["shoulder_left"] = np.array([0, 0, 0])  # Origin
            left_arm_coordinates["elbow_left"] = left_elbow - left_shoulder
            left_arm_coordinates["wrist_left"] = left_wrist - left_shoulder

        # Process hands if detected
        if hand_results.multi_hand_landmarks:
            # Process hands when we have both hand and body landmarks
            right_hand_idx = -1
            left_hand_idx = -1

            # Find indices of right and left hands if handedness is available
            if hand_results.multi_handedness:
                for i, handedness in enumerate(hand_results.multi_handedness):
                    hand_type = handedness.classification[0].label
                    if hand_type == "Right":
                        left_hand_idx = i
                    elif hand_type == "Left":
                        right_hand_idx = i

            # If no handedness info, make assumptions based on available hands
            # This is a fallback but not as reliable as the handedness detection
            if right_hand_idx == -1 and left_hand_idx == -1:
                if len(hand_results.multi_hand_landmarks) >= 1:
                    right_hand_idx = 0
                if len(hand_results.multi_hand_landmarks) >= 2:
                    left_hand_idx = 1

            # Process right hand if available and requested
            if (arm == "right" or arm == "both") and right_hand_idx >= 0:
                right_hand = hand_results.multi_hand_landmarks[right_hand_idx]

                # Get finger MCP points (Metacarpophalangeal joints)
                index_mcp = get_3D_coordinates(
                    right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    depth_frame,
                    w,
                    h,
                    intrinsics,
                )
                pinky_mcp = get_3D_coordinates(
                    right_hand.landmark[mp_hands.HandLandmark.PINKY_MCP],
                    depth_frame,
                    w,
                    h,
                    intrinsics,
                )

                # Store relative to right shoulder
                right_arm_coordinates["index_mcp"] = index_mcp - right_shoulder
                right_arm_coordinates["pinky_mcp"] = pinky_mcp - right_shoulder

            # Process left hand if available and requested
            if (arm == "left" or arm == "both") and left_hand_idx >= 0:
                left_hand = hand_results.multi_hand_landmarks[left_hand_idx]

                # Get finger MCP points (Metacarpophalangeal joints)
                index_mcp = get_3D_coordinates(
                    left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    depth_frame,
                    w,
                    h,
                    intrinsics,
                )
                pinky_mcp = get_3D_coordinates(
                    left_hand.landmark[mp_hands.HandLandmark.PINKY_MCP],
                    depth_frame,
                    w,
                    h,
                    intrinsics,
                )

                # Store relative to left shoulder
                left_arm_coordinates["index_mcp"] = index_mcp - left_shoulder
                left_arm_coordinates["pinky_mcp"] = pinky_mcp - left_shoulder

    return right_arm_coordinates, left_arm_coordinates


def get_head_coordinates(pose, mp_pose, intrinsics, rgb_image, depth_frame, w, h):
    """
    Extract the 3D coordinates of the human head from a frame.
    Returns a tuple (x, y, z) representing the head position in meters.
    Returns None if no head is detected.
    """
    # Run pose detection
    pose_results = pose.process(rgb_image)

    if not pose_results.pose_landmarks:
        return None

    landmarks = pose_results.pose_landmarks.landmark

    # Use nose as the primary head point
    head_position = get_3D_coordinates_reachy_perspective(
        landmarks[mp_pose.PoseLandmark.NOSE], depth_frame, w, h, intrinsics
    )

    return head_position
