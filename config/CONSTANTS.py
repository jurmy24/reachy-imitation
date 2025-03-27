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

# Camera to Reachy Coordinates Transformation
CAMERA_TO_REACHY_Z = 0.35  # camera is 35cm above Reachy's origin
CAMERA_TO_REACHY_X = -0.18  # camera is 18cm behind Reachy's origin

JOINT_NAMES = [
    "shoulder pitch",
    "shoulder roll",
    "arm yaw",
    "elbow pitch",
    "forearm yaw",
    "wrist pitch",
    "wrist roll",
]


# Functions to generate reachy-dependent constants
def get_zero_right_pos(reachy):
    """Generate the zero position dictionary for the right arm based on a reachy instance."""
    return {
        reachy.r_arm.r_shoulder_pitch: 0,
        reachy.r_arm.r_shoulder_roll: 0,
        reachy.r_arm.r_arm_yaw: 0,
        reachy.r_arm.r_elbow_pitch: 0,
        reachy.r_arm.r_forearm_yaw: 0,
        reachy.r_arm.r_wrist_pitch: 0,
        reachy.r_arm.r_wrist_roll: 0,
    }


def get_zero_left_pos(reachy):
    """Generate the zero position dictionary for the left arm based on a reachy instance."""
    return {
        reachy.l_arm.l_shoulder_pitch: 0,
        reachy.l_arm.l_shoulder_roll: 0,
        reachy.l_arm.l_arm_yaw: 0,
        reachy.l_arm.l_elbow_pitch: 0,
        reachy.l_arm.l_forearm_yaw: 0,
        reachy.l_arm.l_wrist_pitch: 0,
        reachy.l_arm.l_wrist_roll: 0,
    }


def get_zero_pos(reachy):
    """Generate the zero position dictionary for both arms based on a reachy instance."""
    return {
        **get_zero_right_pos(reachy),
        **get_zero_left_pos(reachy),
    }


# TODO: check if this one is necessary
def get_ordered_joint_names(reachy, arm="right"):
    """Get the ordered list of joint names from a reachy instance."""
    if arm == "right":
        return [
            reachy.r_arm.r_shoulder_pitch,
            reachy.r_arm.r_shoulder_roll,
            reachy.r_arm.r_arm_yaw,
            reachy.r_arm.r_elbow_pitch,
            reachy.r_arm.r_forearm_yaw,
            reachy.r_arm.r_wrist_pitch,
            reachy.r_arm.r_wrist_roll,
        ]
    elif arm == "left":
        return [
            reachy.l_arm.l_shoulder_pitch,
            reachy.l_arm.l_shoulder_roll,
            reachy.l_arm.l_arm_yaw,
            reachy.l_arm.l_elbow_pitch,
            reachy.l_arm.l_forearm_yaw,
            reachy.l_arm.l_wrist_pitch,
            reachy.l_arm.l_wrist_roll,
        ]
    else:
        raise ValueError("Invalid arm")
