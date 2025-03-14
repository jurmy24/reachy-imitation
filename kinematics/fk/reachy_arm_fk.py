import numpy as np
from reachy_sdk import ReachySDK


def calculate_joint_positions(reachy, joint_angles=None, side="right"):
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
