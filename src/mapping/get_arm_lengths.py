from src.utils.three_d import get_3D_coordinates
import numpy as np


def get_arm_lengths(pose_landmarks, mp_pose, depth_frame, w, h, intrinsics):
    required_landmarks = [
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_WRIST,
    ]
    for landmark in required_landmarks:
        if not (0.5 < pose_landmarks.landmark[landmark].visibility <= 1):
            return None, None

    r_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    r_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    r_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    r_elbow_3d = get_3D_coordinates(r_elbow, depth_frame, w, h, intrinsics)
    r_shoulder_3d = get_3D_coordinates(r_shoulder, depth_frame, w, h, intrinsics)
    r_wrist_3d = get_3D_coordinates(r_wrist, depth_frame, w, h, intrinsics)

    forearm_length = np.linalg.norm(r_elbow_3d - r_shoulder_3d)
    upper_length = np.linalg.norm(r_elbow_3d - r_wrist_3d)
    return forearm_length, upper_length
