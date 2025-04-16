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
        torque_limit: The torque limit to set
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
) -> Dict[str, Dict[str, float]]:
    """
    Get the current positions of all joints for the specified arm(s).

    Args:
        reachy: The Reachy robot instance
        arm: Which arm to get positions for ("right", "left", or "both")

    Returns:
        A dictionary containing joint positions for each specified arm
    """
    result = {}

    if arm == "right" or arm == "both":
        right_positions = {}
        for joint_name in RIGHT_JOINT_NAMES:
            right_positions[joint_name] = getattr(
                getattr(reachy.r_arm, joint_name), "present_position"
            )
        result["right"] = right_positions

    if arm == "left" or arm == "both":
        left_positions = {}
        for joint_name in LEFT_JOINT_NAMES:
            left_positions[joint_name] = getattr(
                getattr(reachy.l_arm, joint_name), "present_position"
            )
        result["left"] = left_positions

    return result
