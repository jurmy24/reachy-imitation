from src.utils.three_d import get_3D_coordinates
import numpy as np
from config.CONSTANTS import HUMAN_ELBOW_TO_HAND_DEFAULT, HUMAN_UPPERARM_DEFAULT


def get_arm_lengths(pose_landmarks, mp_pose, depth_frame, w, h, intrinsics):
    # Note: This function only calculates for the right arm (assuming arms match in length)
    required_landmarks = [
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_WRIST,
    ]
    for landmark in required_landmarks:
        if not (0.5 < pose_landmarks.landmark[landmark].visibility <= 1):
            return None, None

    # r_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    # r_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    # r_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    # r_elbow_3d = get_3D_coordinates(r_elbow, depth_frame, w, h, intrinsics)
    # r_shoulder_3d = get_3D_coordinates(r_shoulder, depth_frame, w, h, intrinsics)
    # r_wrist_3d = get_3D_coordinates(r_wrist, depth_frame, w, h, intrinsics)

    # upper_arm_length = np.linalg.norm(r_elbow_3d - r_shoulder_3d)
    # forearm_length = np.linalg.norm(r_elbow_3d - r_wrist_3d)
    elbow_to_hand = HUMAN_ELBOW_TO_HAND_DEFAULT
    upperarm = HUMAN_UPPERARM_DEFAULT
    return elbow_to_hand, upperarm
