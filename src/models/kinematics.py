# This file contains the kinematics code for the robot and the human
import numpy as np


def dh_transformation_matrix(
    theta: float, alpha: float, r: float, d: float
) -> np.ndarray:
    """
    Calculate the homogeneous transformation matrix (for a joint) using DH parameters.

    Parameters:
    -----------
    theta : float
        Joint angle in radians
    d : float
        Link offset in joint axis direction
    r : float
        Link length perpendicular to joint axis
    alpha : float
        Link twist angle in radians

    Returns:
    --------
    T : numpy.ndarray
        4x4 homogeneous transformation matrix
    """
    # Calculate trigonometric values once to improve efficiency
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    # Create the transformation matrix directly without intermediate steps
    T = np.array(
        [
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, r * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, r * sin_theta],
            [0, sin_alpha, cos_alpha, d],
            [0, 0, 0, 1],
        ]
    )

    return T


def forward_kinematics(
    dh_params: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """
    Calculate the complete forward kinematics transformation using DH parameters.

    Parameters:
    -----------
    dh_params : list of tuples
        List of (theta, alpha, r, d) DH parameters for each joint

    Returns:
    --------
    T : numpy.ndarray
        4x4 homogeneous transformation matrix representing the end-effector pose
    """
    # Start with identity matrix
    T = np.eye(4)

    # Multiply transformations for each joint
    for params in dh_params:
        T_i = dh_transformation_matrix(*params)
        T = T @ T_i

    return T


def get_human_arm_dh_params(joint_angles: list[float], arm="right"):
    if arm == "right":
        return [
            (joint_angles[0], np.pi / 2, 0, 0),  # Joint 1
            (joint_angles[1], np.pi / 2, 0, 0),  # Joint 2
            (joint_angles[2], -np.pi / 2, 0, d_3),  # Joint 3
            (joint_angles[3], -np.pi / 2, 0, 0),  # Joint 4
            (joint_angles[4], np.pi / 2, 0, d_5),  # Joint 5
            (joint_angles[5], np.pi / 2, 0, 0),  # Joint 6
            (joint_angles[6], 0, 0, 0),  # Joint 7
        ]
    else:
        raise ValueError("Only right arm is supported for now")


def calculate_human_joint_positions(joint_angles: list[float], arm="right"):
    """
    Calculate positions of all joints, including the base and end effector.

    Parameters:
    -----------
    dh_params : List of tuples
        List of (theta, alpha, r, d) DH parameters for each joint

    Returns:
    --------
    joint_positions : np.ndarray
        Array of shape (n+1, 3) containing 3D positions of all joints
    """
    dh_params = get_human_arm_dh_params(joint_angles, arm)
    # Start with the base at origin
    T_current = np.eye(4)
    positions = [T_current[:3, 3]]  # Base position

    # Calculate position after each transformation
    for params in dh_params:
        T_i = dh_transformation_matrix(*params)
        T_current = T_current @ T_i
        positions.append(T_current[:3, 3])

    return np.array(positions)


import numpy as np
from reachy_sdk import ReachySDK


def calculate_reachy_joint_positions(reachy, joint_angles=None, side="right"):
    """
    Calculate the positions of all joints in coordinate space using the Reachy SDK.

    Parameters:
    -----------
    reachy : ReachySDK
        Instance of the Reachy SDK
    joint_angles : list of float, optional
        Joint angles in degrees for the 7 joints in the kinematic chain.
        If None, uses the current joint positions.
    side : str
        'left' or 'right' arm

    Returns:
    --------
    joint_positions : list of np.ndarray
        List of 3D positions (x, y, z) for each joint in the kinematic chain
    """
    arm = reachy.r_arm if side == "right" else reachy.l_arm

    # If joint angles are provided, temporarily move the arm to those positions
    # in the internal model (without actually moving the physical robot)
    if joint_angles is not None:
        # Save current positions
        original_positions = {}
        for i, joint_name in enumerate(arm.joints.keys()):
            if i < len(joint_angles) and joint_name in arm._kinematics_chain:
                joint = getattr(arm, joint_name)
                original_positions[joint_name] = joint.present_position

    # Calculate forward kinematics using the SDK
    # This will give us the end-effector position
    end_effector_matrix = arm.forward_kinematics(joint_angles)

    # To get intermediate joint positions, we need to calculate FK for each subset of joints
    joint_positions = []

    # Start with the base position (shoulder)
    base_position = np.array([0.0, 0.0, 0.0])  # Assuming shoulder at origin
    joint_positions.append(base_position)

    # For each joint in the chain, calculate its position
    chain = list(arm._kinematics_chain)
    for i in range(1, len(chain) + 1):
        # Calculate FK up to this joint
        subset_angles = joint_angles[:i] if joint_angles is not None else None
        partial_matrix = arm.forward_kinematics(subset_angles)

        # Extract position from the transformation matrix
        position = partial_matrix[:3, 3]
        joint_positions.append(position)

    return joint_positions


# Example usage
if __name__ == "__main__":
    # Connect to the robot
    reachy = ReachySDK(host="localhost")

    # Example DH parameters for a 7-DOF robot ar
    q_1 = np.pi / 4
    q_2 = np.pi / 6
    q_3 = np.pi / 3
    q_4 = np.pi / 4
    q_5 = np.pi / 3
    q_6 = np.pi / 4
    q_7 = np.pi / 3
    d_3 = 0.3  # length of shoulder -> elbow joint (upper arm)
    d_5 = 0.4  # length of elbow -> wrist joint (forearm)

    dh_params = [
        (q_1, np.pi / 2, 0, 0),  # Joint 1
        (q_2, np.pi / 2, 0, 0),  # Joint 2
        (q_3, -np.pi / 2, 0, d_3),  # Joint 3
        (q_4, -np.pi / 2, 0, 0),  # Joint 4
        (q_5, np.pi / 2, 0, d_5),  # Joint 5
        (q_6, np.pi / 2, 0, 0),  # Joint 6
        (q_7, 0, 0, 0),  # Joint 7
    ]

    # Example joint angles (in degrees)
    joint_angles = [0, 0, 0, 0, 0, 0, 0]  # Neutral position

    # Calculate end-effector position using forward kinematics
    end_effector_matrix = reachy.r_arm.forward_kinematics(joint_angles)
    print("\nEnd-effector transformation matrix:")
    print(end_effector_matrix)

# Example usage
if __name__ == "__main__":
    # TODO: add possibility of of selecting an arm
    # Example DH parameters for a 6-DOF robot arm
    # Format: (theta, alpha, r, d) for each joint
    # Values for the joint angles (might be useful to add a range)
    q_1 = np.pi / 4
    q_2 = np.pi / 6
    q_3 = np.pi / 3
    q_4 = np.pi / 4
    q_5 = np.pi / 3
    q_6 = np.pi / 4
    q_7 = np.pi / 3
    d_3 = 0.3  # length of shoulder -> elbow joint (upper arm)
    d_5 = 0.4  # length of elbow -> wrist joint (forearm)

    # dh_params = [
    #     (q_1, np.pi / 2, 0, 0),  # Joint 1
    #     (q_2, np.pi / 2, 0, 0),  # Joint 2
    #     (q_3, -np.pi / 2, 0, d_3),  # Joint 3
    #     (q_4, -np.pi / 2, 0, 0),  # Joint 4
    #     (q_5, np.pi / 2, 0, d_5),  # Joint 5
    #     (q_6, np.pi / 2, 0, 0),  # Joint 6
    #     (q_7, 0, 0, 0),  # Joint 7
    # ]

    # # Calculate the end-effector transformation
    # T_end_effector = forward_kinematics(dh_params)

    # Print the result with clean formatting
    np.set_printoptions(precision=4, suppress=True)
    print("End-effector transformation matrix:")
    # print(T_end_effector)
