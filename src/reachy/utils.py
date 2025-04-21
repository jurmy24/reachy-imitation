from typing import Literal, Dict, Any, List

# List of joint names for each arm
RIGHT_JOINT_NAMES = [
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_arm_yaw",
    "r_elbow_pitch",
    "r_forearm_yaw",
    "r_wrist_pitch",
    "r_wrist_roll",
    "r_gripper",
]

LEFT_JOINT_NAMES = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_arm_yaw",
    "l_elbow_pitch",
    "l_forearm_yaw",
    "l_wrist_pitch",
    "l_wrist_roll",
    "l_gripper",
]


def setup_torque_limits(
    reachy, torque_limit: float, arm: Literal["right", "left", "both"] = "right"
):
    """
    Set torque limits for all joints in the specified arm(s).

    Args:
        reachy: The Reachy robot instance
        torque_limit: The torque limit to set in percentage of the maximum torque
        arm: Which arm to set limits for ("right", "left", or "both")
    """
    if arm == "right" or arm == "both":
        for joint_name in RIGHT_JOINT_NAMES:
            setattr(getattr(reachy.r_arm, joint_name), "torque_limit", torque_limit)

    if arm == "left" or arm == "both":
        for joint_name in LEFT_JOINT_NAMES:
            setattr(getattr(reachy.l_arm, joint_name), "torque_limit", torque_limit)


def get_joint_positions(
    reachy, arm: Literal["right", "left", "both"] = "right"
) -> tuple[dict, dict]:
    """
    Get the current positions of all joints for the specified arm(s).

    Args:
        reachy: The Reachy robot instance
        arm: Which arm to get positions for ("right", "left", or "both")

    Returns:
        A tuple of (right_positions, left_positions) dictionaries.
        If an arm is not selected, its dictionary will be empty.
    """
    right_positions = {}
    left_positions = {}

    if arm == "right" or arm == "both":
        for joint_name in RIGHT_JOINT_NAMES:
            right_positions[joint_name] = getattr(
                getattr(reachy.r_arm, joint_name), "present_position"
            )

    if arm == "left" or arm == "both":
        for joint_name in LEFT_JOINT_NAMES:
            left_positions[joint_name] = getattr(
                getattr(reachy.l_arm, joint_name), "present_position"
            )

    return right_positions, left_positions
