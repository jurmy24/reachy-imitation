import numpy as np
from scipy.optimize import minimize


def mattransfo(alpha, d, theta, r):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st, 0, d],
        [ca*st, ca*ct, -sa, -r*sa],
        [sa*st, sa*ct, ca, r*ca],
        [0, 0, 0, 1]
    ], dtype=np.float64)

# who= "reachy" or "human"
# correspond to reachy or human
# L = lenght parameters


def compute_transformation_matrices(joint_angles, who, length, side):
    """
    Compute the transformation matrices for the robotic arm.
    """
    pi = np.pi
    alpha = [0, -pi/2, -pi/2, -pi/2, +pi/2, -pi/2, -pi/2, - pi/2]
    r = np.array([0, 0, -length[0], 0, -length[1], 0, 0, -length[2]])
    th = [joint_angles[0], joint_angles[1]-pi/2, joint_angles[2] - pi/2, joint_angles[3],
          joint_angles[4], joint_angles[5] - pi/2, joint_angles[6] - pi/2, - pi/2]

    if who == "human":
        d = [0, 0, 0, 0, 0, 0, 0, 0]
        Tbase0 = mattransfo(-pi/2, 0, -pi/2, 0)
    if who == "reachy" and side == 'right':
        d = [0, 0, 0, 0, 0, 0, -0.325, -0.01]
        Tbase0 = mattransfo(-pi/2, 0, -pi/2, -0.19)

    if who == "reachy" and side == 'left':
        d = [0, 0, 0, 0, 0, 0, -0.325, 0.01]
        Tbase0 = mattransfo(-pi/2, 0, -pi/2, 0.19)

    T01 = Tbase0 @ mattransfo(alpha[0], d[0], th[0], r[0])
    T12 = mattransfo(alpha[1], d[1], th[1], r[1])
    T23 = mattransfo(alpha[2], d[2], th[2], r[2])
    T34 = mattransfo(alpha[3], d[3], th[3], r[3])
    T45 = mattransfo(alpha[4], d[4], th[4], r[4])
    T56 = mattransfo(alpha[5], d[5], th[5], r[5])
    T67 = mattransfo(alpha[6], d[6], th[6], r[6])
    T78 = mattransfo(alpha[7], d[7], th[7], r[7])

    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T05 = T04 @ T45
    T06 = T05 @ T56
    T07 = T06 @ T67
    T08 = T07 @ T78
    return T04, T08


def forward_kinematics(joint_angle, who="reachy", length=[0.28, 0.25, 0.075], side='right'):
    """
    Calculate the hand-effector position using forward kinematics.
    """
    T04, T08 = compute_transformation_matrices(joint_angle, who, length, side)
    position_e = np.array(T04[0:3, 3], dtype=np.float64).flatten()
    position = np.array(T08[0:3, 3], dtype=np.float64).flatten()
    return position_e, position


def cost_function(joint_angles, hand_position, elbow_position, elbow_weight, who, length, side):
    """
    Compute the cost function that includes hand-effector and elbow position errors.
    """
    # Compute the hand position
    _, current_position = forward_kinematics(joint_angles, who, length, side)
    hand_effector_error = np.linalg.norm(hand_position - current_position)

    # Compute the elbow position
    T04, _ = compute_transformation_matrices(joint_angles, who, length, side)
    elbow_position_actual = np.array(
        T04[0:3, 3], dtype=np.float64).flatten()
    elbow_error = np.linalg.norm(
        elbow_position_actual - elbow_position)

    # Compute the total cost
    total_cost = hand_effector_error + elbow_weight * elbow_error
    return total_cost


def inverse_kinematics(hand_position, elbow_position, initial_guess, elbow_weight=0.1,
                       who="reachy", length=[0.28, 0.25, 0.075], side='right'):
    pi = np.pi
    joint_limits = [(-1.0 * pi,  0.5 * pi), (-1.0 * pi,  10/180 * pi),
                    (-0.5 * pi,  0.5 * pi), (-125/180 * pi,  0), (0, 0), (0, 0), (0, 0)]

    if side == "both":
        results = {}
        for s in ['left', 'right']:
            result = minimize(cost_function, initial_guess[s], args=(
                hand_position[s], elbow_position[s], elbow_weight, who, length, s),
                method='SLSQP', bounds=joint_limits)
            results[s] = result.x
        return results
    else:
        result = minimize(cost_function, initial_guess, args=(
            hand_position, elbow_position, elbow_weight, who, length, side),
            method='SLSQP', bounds=joint_limits)
        return result.x


if __name__ == "__main__":
    # Example for using both arms
    side = "both"
    who = "reachy"
    length = [0.28, 0.25, 0.075]
    elbow_weight = 0

    hand_position = {
        "right": np.array([0.3, -0.2, 0.1]),
        "left": np.array([0.3, 0.2, 0.1])
    }

    elbow_position = {
        "right": np.array([0.1, -0.1, -0.3]),
        "left": np.array([0.1, 0.1, -0.3])
    }

    initial_guess = {
        "right": np.array([0.1, 0.3, 0.1, 0.1, 0, 0, 0]),
        "left": np.array([0.1, 0.3, 0.1, 0.1, 0, 0, 0])
    }

    joint_angles = inverse_kinematics(
        hand_position, elbow_position, initial_guess, elbow_weight, who, length, side)

    print("Joint Angles Right:", joint_angles["right"])
    print("Right Arm Elbow-Effector Position:",
          forward_kinematics(joint_angles["right"], who, length, "right")[0])
    print("Right Arm Hand-Effector Position:",
          forward_kinematics(joint_angles["right"], who, length, "right")[1])

    print("Joint Angles Left:", joint_angles["left"])
    print("Left Arm Elbow-Effector Position:",
          forward_kinematics(joint_angles["left"], who, length, "left")[0])
    print("Left Arm Hand-Effector Position:",
          forward_kinematics(joint_angles["left"], who, length, "left")[1])
