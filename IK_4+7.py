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

    return sp.Matrix([
        [ct, -st, 0, d],
        [ca*st, ca*ct, -sa, -r*sa],
        [sa*st, sa*ct, ca, r*ca],
        [0, 0, 0, 1]
    ])


def compute_transformation_matrices(th, L):
    """
    Compute the transformation matrices for the robotic arm.
    """

    pi = sp.pi
    alpha = [0, -pi/2, -pi/2, -pi/2, +pi/2, -pi/2, -pi/2, - pi/2]
    d = [0, 0, 0, 0, 0, 0, -0.325, -0.01]
    r = np.array([0, 0, -L[1], 0, -0.25, 0, 0, -0.075])

    Tbase0 = mattransfo(-pi/2, 0, -pi/2, L[0])
    T01 = Tbase0 * mattransfo(alpha[0], d[0], th[0], r[0])
    T12 = mattransfo(alpha[1], d[1], th[1] - pi/2, r[1])
    T23 = mattransfo(alpha[2], d[2], th[2] - pi/2, r[2])
    T34 = mattransfo(alpha[3], d[3], th[3], r[3])
    T45 = mattransfo(alpha[4], d[4], th[4], r[4])
    T56 = mattransfo(alpha[5], d[5], th[5] - pi/2, r[5])
    T67 = mattransfo(alpha[6], d[6], th[6] - pi/2, r[6])
    T78 = mattransfo(alpha[7], d[7], - pi/2, r[7])

    T02 = T01 * T12
    T03 = T02 * T23
    T04 = T03 * T34
    T05 = T04 * T45
    T06 = T05 * T56
    T07 = T06 * T67
    T08 = T07 * T78
    return T04,T08


def forward_kinematics(th, L):
    """
    Calculate the end-effector position using forward kinematics.
    """

    T04,T08 = compute_transformation_matrices(th, L)
    position_e=T04[0:3, 3]
    position = T08[0:3, 3]
    return  np.array(position_e, dtype=np.float64).flatten(), np.array(position, dtype=np.float64).flatten()


def jacobian(th, L):
    """
    Compute the Jacobian matrix of the forward kinematics.
    """
    _, pos = forward_kinematics(th, L)
    J = np.zeros((3, len(th)))

    epsilon = 1e-6
    for i in range(len(th)):
        th_epsilon = np.copy(th)
        th_epsilon[i] += epsilon
        _, pos_epsilon = forward_kinematics(th_epsilon, L)
        J[:, i] = (pos_epsilon - pos) / epsilon
        

    return J


def inverse_kinematics(desired_position, initial_guess, L, tolerance=1e-6, max_iterations=100):
    """
    Implement the Newton-Raphson method for inverse kinematics.
    """
    th = initial_guess
    for _ in range(max_iterations):
        current_position = forward_kinematics(th, L)[1]
        error = desired_position - current_position
        if np.linalg.norm(error) < tolerance:
            break
        J = jacobian(th, L)
        th = th + np.linalg.pinv(J) @ error
    return th


# Desired end-effector position
desired_position = np.array([0.3, 0.2, 0.1])

# Initial guess for joint angles
initial_guess = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Link lengths
L = [0.19, 0.28]

# Compute inverse kinematics
joint_angles = inverse_kinematics(desired_position, initial_guess, L)
print("Joint Angles:", joint_angles)
print(forward_kinematics(joint_angles, L))
