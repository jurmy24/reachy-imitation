import sympy as sp
import numpy as np


def mattransfo(alpha, d, theta, r):
    """
    Compute the transformation matrix using DH parameters.
    """
    ct = sp.cos(theta)
    st = sp.sin(theta)
    ca = sp.cos(alpha)
    sa = sp.sin(alpha)

    return sp.Matrix(
        [
            [ct, -st, 0, d],
            [ca * st, ca * ct, -sa, -r * sa],
            [sa * st, sa * ct, ca, r * ca],
            [0, 0, 0, 1],
        ]
    )


def compute_transformation_matrices(th, L1, L2):
    """
    Compute the transformation matrices for the robotic arm.
    """
    pi = sp.pi
    alpha = [0, -pi / 2, -pi / 2, -pi / 2]
    d = [0, 0, 0, 0]
    r = [0, 0, -L2, 0]

    Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, L1)
    T01 = Tbase0 * mattransfo(alpha[0], d[0], th[0], r[0])
    T12 = mattransfo(alpha[1], d[1], th[1] - pi / 2, r[1])
    T23 = mattransfo(alpha[2], d[2], th[2] - pi / 2, r[2])
    T34 = mattransfo(alpha[3], d[3], th[3], r[3])

    T02 = T01 * T12
    T03 = T02 * T23
    T04 = T03 * T34

    return T04


def forward_kinematics(th, L1, L2):
    """
    Calculate the end-effector position using forward kinematics.
    """
    T04 = compute_transformation_matrices(th, L1, L2)
    position = T04[0:3, 3]
    return np.array(position, dtype=np.float64).flatten()


def jacobian(th, L1, L2):
    """
    Compute the Jacobian matrix of the forward kinematics.
    """
    pos = forward_kinematics(th, L1, L2)
    J = np.zeros((3, len(th)))

    epsilon = 1e-6
    for i in range(len(th)):
        th_epsilon = np.copy(th)
        th_epsilon[i] += epsilon
        pos_epsilon = forward_kinematics(th_epsilon, L1, L2)
        J[:, i] = (pos_epsilon - pos) / epsilon

    return J


def inverse_kinematics(
    desired_position, initial_guess, L1, L2, tolerance=1e-6, max_iterations=100
):
    """
    Implement the Newton-Raphson method for inverse kinematics.
    """
    th = initial_guess
    for _ in range(max_iterations):
        current_position = forward_kinematics(th, L1, L2)
        error = desired_position - current_position
        if np.linalg.norm(error) < tolerance:
            break
        J = jacobian(th, L1, L2)
        th = th + np.linalg.pinv(J) @ error
    return th


# Desired end-effector position
desired_position = np.array([0.3, 0.2, 0.1])

# Initial guess for joint angles
initial_guess = np.array([0.1, 0.1, 0.1, 0.1])

# Link lengths
L1 = 0.19
L2 = 0.28

# Compute inverse kinematics
joint_angles = inverse_kinematics(desired_position, initial_guess, L1, L2)
print("Joint Angles:", joint_angles)
