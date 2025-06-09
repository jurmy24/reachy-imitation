from config.CONSTANTS import HUMAN_ELBOW_TO_HAND_DEFAULT, HUMAN_UPPERARM_DEFAULT


def get_arm_lengths(pose_landmarks, mp_pose, depth_frame, w, h, intrinsics):
    """
    Calculates the arm lengths from the pose landmarks (assumed to be the same for both arms)
    """
    required_landmarks = [
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_WRIST,
    ]
    for landmark in required_landmarks:
        if not (0.5 < pose_landmarks.landmark[landmark].visibility <= 1):
            return None, None

    elbow_to_hand = HUMAN_ELBOW_TO_HAND_DEFAULT
    upperarm = HUMAN_UPPERARM_DEFAULT
    return elbow_to_hand, upperarm
