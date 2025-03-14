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


def get_joint_positions(dh_params):
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
    # Start with the base at origin
    T_current = np.eye(4)
    positions = [T_current[:3, 3]]  # Base position

    # Calculate position after each transformation
    for params in dh_params:
        T_i = dh_transformation_matrix(*params)
        T_current = T_current @ T_i
        positions.append(T_current[:3, 3])

    return np.array(positions)


# Example usage
if __name__ == "__main__":
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

    dh_params = [
        (q_1, np.pi / 2, 0, 0),  # Joint 1
        (q_2, np.pi / 2, 0, 0),  # Joint 2
        (q_3, -np.pi / 2, 0, d_3),  # Joint 3
        (q_4, -np.pi / 2, 0, 0),  # Joint 4
        (q_5, np.pi / 2, 0, d_5),  # Joint 5
        (q_6, np.pi / 2, 0, 0),  # Joint 6
        (q_7, 0, 0, 0),  # Joint 7
    ]

    # Calculate the end-effector transformation
    T_end_effector = forward_kinematics(dh_params)

    # Print the result with clean formatting
    np.set_printoptions(precision=4, suppress=True)
    print("End-effector transformation matrix:")
    print(T_end_effector)
