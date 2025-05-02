import numpy as np

# Reachy Robot Physical Dimensions Configuration
LEN_REACHY_UPPERARM = 0.28
LEN_REACHY_FOREARM = 0.25
LEN_REACHY_WRIST = 0.0325
LEN_REACHY_HAND = 0.075
LEN_REACHY_HAND_OFFSET = 0.01
LEN_REACHY_ELBOW_TO_END_EFFECTOR = 0.3575
LEN_REACHY_ARM = LEN_REACHY_ELBOW_TO_END_EFFECTOR + LEN_REACHY_UPPERARM
REACHY_R_SHOULDER_COORDINATES = np.array([0, -0.19, 0])
REACHY_L_SHOULDER_COORDINATES = np.array([0, 0.19, 0])

# Default values for human arm lengths
HUMAN_ELBOW_TO_HAND_DEFAULT = 0.38
HUMAN_UPPERARM_DEFAULT = 0.3

# Camera to Reachy Coordinates Transformation
CAMERA_TO_REACHY_Z = 0.35  # camera is 35cm above Reachy's origin
CAMERA_TO_REACHY_X = -0.18  # camera is 18cm behind Reachy's origin


def get_zero_pos(reachy, arm="both"):
    """Generate the zero position dictionary for the specified arm based on a reachy instance."""
    if arm not in ["right", "left", "both"]:
        raise ValueError("Invalid arm")

    zero_pos = {}

    if arm == "right" or arm == "both":
        zero_pos.update(
            {
                reachy.r_arm.r_shoulder_pitch: 0,
                reachy.r_arm.r_shoulder_roll: 0,
                reachy.r_arm.r_arm_yaw: 0,
                reachy.r_arm.r_elbow_pitch: 0,
                reachy.r_arm.r_forearm_yaw: 0,
                reachy.r_arm.r_wrist_pitch: 0,
                reachy.r_arm.r_wrist_roll: 0,
                reachy.r_arm.r_gripper: 10,
            }
        )

    if arm == "left" or arm == "both":
        zero_pos.update(
            {
                reachy.l_arm.l_shoulder_pitch: 0,
                reachy.l_arm.l_shoulder_roll: 0,
                reachy.l_arm.l_arm_yaw: 0,
                reachy.l_arm.l_elbow_pitch: 0,
                reachy.l_arm.l_forearm_yaw: 0,
                reachy.l_arm.l_wrist_pitch: 0,
                reachy.l_arm.l_wrist_roll: 0,
                reachy.l_arm.l_wrist_roll: -10,
            }
        )

    return zero_pos


def get_finger_tips(mp_hands):
    """Get the finger tip landmarks from the MediaPipe hands module."""
    return [
        mp_hands.HandLandmark.THUMB_TIP.value,
        mp_hands.HandLandmark.INDEX_FINGER_TIP.value,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value,
        mp_hands.HandLandmark.RING_FINGER_TIP.value,
        mp_hands.HandLandmark.PINKY_TIP.value,
    ]
