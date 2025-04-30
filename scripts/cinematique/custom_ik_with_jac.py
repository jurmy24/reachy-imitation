import numpy as np
from scipy.optimize import minimize


def mattransfo(alpha, d, theta, r):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array(
        [
            [ct, -st, 0, d],
            [ca * st, ca * ct, -sa, -r * sa],
            [sa * st, sa * ct, ca, r * ca],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


# who= "reachy" or "human"
# correspond to reachy or human
# L = lenght parameters


def compute_transformation_matrices(joint_angles, who, length, side):
    """
    Compute the transformation matrices for the robotic arm.
    """
    pi = np.pi
    alpha = [0, -pi / 2, -pi / 2, -pi / 2, +pi / 2, -pi / 2, -pi / 2, -pi / 2]
    r = np.array([0, 0, -length[0], 0, -length[1], 0, 0, -length[2]])
    th = [
        joint_angles[0],
        joint_angles[1] - pi / 2,
        joint_angles[2] - pi / 2,
        joint_angles[3],
        joint_angles[4],
        joint_angles[5] - pi / 2,
        joint_angles[6] - pi / 2,
        # 0,
        # -pi / 2, - this is an alternative to setting joint limits on the wrist to fixed.
        # -pi / 2,
        -pi / 2,
    ]

    if who == "human":
        d = [0, 0, 0, 0, 0, 0, 0, 0]
        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, 0)
    if who == "reachy" and side == "right":
        d = [0, 0, 0, 0, 0, 0, -0.0325, -0.01]
        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, -0.19)
    if who == "reachy" and side == "left":
        d = [0, 0, 0, 0, 0, 0, -0.0325, 0.01]
        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, 0.19)

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


# ! THIS IS THE NEW ONE
# NOTE: We should return the transformation matrices, not the position
def compute_transformation_matrices_and_jacobian(joint_angles, who, length, side):
    """
    Compute the transformation matrices for the robotic arm.
    """
    pi = np.pi
    alpha = [0, -pi / 2, -pi / 2, -pi / 2, +pi / 2, -pi / 2, -pi / 2, -pi / 2]
    r = np.array([0, 0, -length[0], 0, -length[1], 0, 0, -length[2]])
    th = [
        joint_angles[0],
        joint_angles[1] - pi / 2,
        joint_angles[2] - pi / 2,
        joint_angles[3],
        joint_angles[4],
        joint_angles[5] - pi / 2,
        joint_angles[6] - pi / 2,
        # 0,
        # -pi / 2, - this is an alternative to setting joint limits on the wrist to fixed.
        # -pi / 2,
        -pi / 2,
    ]

    if who == "human":
        d = [0, 0, 0, 0, 0, 0, 0, 0]
        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, 0)
    if who == "reachy" and side == "right":
        d = [0, 0, 0, 0, 0, 0, -0.0325, -0.01]
        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, -0.19)
    if who == "reachy" and side == "left":
        d = [0, 0, 0, 0, 0, 0, -0.0325, 0.01]
        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, 0.19)

    T01 = Tbase0 @ mattransfo(alpha[0], d[0], th[0], r[0])

    Ts_gi_i1 = [T01]
    for i in range(1, 8):
        Ts_gi_i1.append(mattransfo(alpha[i], d[i], th[i], r[i]))

    Ts_g_0i = []
    T = np.eye(4, 4)
    for i in range(8):
        T = T @ Ts_gi_i1[i]
        Ts_g_0i.append(T.copy())

    T04 = Ts_g_0i[3]
    T08 = Ts_g_0i[-1]
    position_e = np.array(T04[0:3, 3], dtype=np.float64).flatten()
    position_hand = np.array(T08[0:3, 3], dtype=np.float64).flatten()

    # Yeah what is the 8th column representing?
    jacobian = np.zeros(
        (6, 8)
    )  # 6 rows (3 linear + 3 angular).  technically this should be (6,7)
    elbow_jacobian = np.zeros((6, 8))  # technically this should be (6,4)

    for i in range(8):
        g_0i = Ts_g_0i[i]  # Transform from base to joint i
        R_0i = g_0i[:3, :3]
        p_i = g_0i[:3, 3]  # Position of joint i in base frame
        Roz_i = R_0i[:, 2]
        jacobian[0:3, i] = np.cross(
            Roz_i, position_hand - p_i
        )  # Linear part - from lecture slides
        # jacobian[3:6, i] = 0                                 # Angular part is being NEGLECTED
        # elbow_jacobian[3:6, i] = 0
        if i < 4:  # successive joint params have no affect on the elbow position
            elbow_jacobian[0:3, i] = np.cross(Roz_i, position_e - p_i)
        # else:
        #     elbow_jacobian[0:3, i] = 0

    return (
        position_e,
        position_hand,
        jacobian[:, :-1],
        elbow_jacobian[:, :-1],
    )  # T04, T08, jacobian (ignoring angular twist), elbow_jacobian (ignoring angular twist)

    # T01 = Tbase0 @ mattransfo(alpha[0], d[0], th[0], r[0])
    # T12 = mattransfo(alpha[1], d[1], th[1], r[1])
    # T23 = mattransfo(alpha[2], d[2], th[2], r[2])
    # T34 = mattransfo(alpha[3], d[3], th[3], r[3])
    # T45 = mattransfo(alpha[4], d[4], th[4], r[4])
    # T56 = mattransfo(alpha[5], d[5], th[5], r[5])
    # T67 = mattransfo(alpha[6], d[6], th[6], r[6])
    # T78 = mattransfo(alpha[7], d[7], th[7], r[7])

    # T02 = T01 @ T12
    # T03 = T02 @ T23
    # T04 = T03 @ T34
    # T05 = T04 @ T45
    # T06 = T05 @ T56
    # T07 = T06 @ T67
    # T08 = T07 @ T78
    # return T04, T08


# ! THIS IS THE OLD ONE
def compute_transformation_matrices_and_jacobian_old(joint_angles, L):
    """
    Compute the transformation matrices for the robotic arm.
    """
    pi = np.pi
    if L[0] == "reachy":
        alpha = [0, -pi / 2, -pi / 2, -pi / 2, +pi / 2, -pi / 2, -pi / 2, -pi / 2]
        d = [0, 0, 0, 0, 0, 0, -0.0325, -0.01]
        r = np.array([0, 0, -L[1], 0, -L[2], 0, 0, -L[3]])
        th = [
            joint_angles[0],
            joint_angles[1] - pi / 2,
            joint_angles[2] - pi / 2,
            joint_angles[3],
            joint_angles[4],
            joint_angles[5] - pi / 2,
            joint_angles[6] - pi / 2,
            # 0,
            # -pi / 2, - this is an alternative to setting joint limits on the wrist to fixed.
            # -pi / 2,
            -pi / 2,
        ]
        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, -0.19)
    if L[1] == "human":
        alpha = [0, -pi / 2, -pi / 2, -pi / 2, +pi / 2, -pi / 2, -pi / 2, -pi / 2]
        d = [0, 0, 0, 0, 0, 0, 0, 0]
        r = np.array([0, 0, -L[1], 0, -L[2], 0, 0, -L[3]])
        th = [
            joint_angles[0],
            joint_angles[1] - pi / 2,
            joint_angles[2] - pi / 2,
            joint_angles[3],
            joint_angles[4],
            joint_angles[5] - pi / 2,
            joint_angles[6] - pi / 2,
            -pi / 2,
        ]

        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, 0)

    T01 = Tbase0 @ mattransfo(alpha[0], d[0], th[0], r[0])
    Ts_gi_i1 = [T01]
    for i in range(1, 8):
        Ts_gi_i1.append(mattransfo(alpha[i], d[i], th[i], r[i]))

    Ts_g_0i = []
    T = np.eye(4, 4)
    for i in range(8):
        T = T @ Ts_gi_i1[i]
        Ts_g_0i.append(T.copy())

    T04 = Ts_g_0i[3]
    T08 = Ts_g_0i[-1]
    position_e = np.array(T04[0:3, 3], dtype=np.float64).flatten()
    position_hand = np.array(T08[0:3, 3], dtype=np.float64).flatten()

    jacobian = np.zeros(
        (6, 8)
    )  # 6 rows (3 linear + 3 angular).  technically this should be (6,7)
    elbow_jacobian = np.zeros((6, 8))  # technically this should be (6,4)

    for i in range(8):
        g_0i = Ts_g_0i[i]  # Transform from base to joint i
        R_0i = g_0i[:3, :3]
        p_i = g_0i[:3, 3]  # Position of joint i in base frame
        Roz_i = R_0i[:, 2]
        jacobian[0:3, i] = np.cross(
            Roz_i, position_hand - p_i
        )  # Linear part - from lecture slides
        # jacobian[3:6, i] = 0                                 # Angular part is being NEGLECTED
        # elbow_jacobian[3:6, i] = 0
        if i < 4:  # successive joint params have no affect on the elbow position
            elbow_jacobian[0:3, i] = np.cross(Roz_i, position_e - p_i)
        # else:
        #     elbow_jacobian[0:3, i] = 0

    return (
        position_e,
        position_hand,
        jacobian[:, :-1],
        elbow_jacobian[:, :-1],
    )  # T04, T08, jacobian (ignoring angular twist), elbow_jacobian (ignoring angular twist)

    # Question for Marin, I'm ignoring the last column of the jacbobian because this doesn't correspond to a motor joint.
    # HOWEVER, does that mean my position_hand variable should change to be the translation of T07


def forward_kinematics(
    joint_angles, who="reachy", length=[0.28, 0.25, 0.075], side="right"
):
    """
    Calculate the hand-effector position using forward kinematics.
    """
    T04, T08 = compute_transformation_matrices(joint_angles, who, length, side)
    position_e = np.array(T04[0:3, 3], dtype=np.float64).flatten()
    position = np.array(T08[0:3, 3], dtype=np.float64).flatten()
    return position_e, position


def forward_kinematics_with_jacobian(
    joint_angles, who="reachy", length=[0.28, 0.25, 0.075], side="right"
):
    """
    Calculate the hand-effector position using forward kinematics.
    """
    position_e, position, jacobian, elbow_jacobian = (
        compute_transformation_matrices_and_jacobian(joint_angles, who, length, side)
    )

    return position_e, position, jacobian, elbow_jacobian


def cost_function(
    joint_angles, target_ee_coords, target_elbow_coords, elbow_weight, who, length, side
):
    """
    Compute the cost function that includes hand-effector and elbow position errors.
    This calculates the sum of squared errors between the current and target positions. L2 norm.
    """
    # Compute the
    current_elbow_coords, current_ee_coords = forward_kinematics(
        joint_angles, who, length, side
    )
    ee_error = np.linalg.norm(current_ee_coords - target_ee_coords)
    elbow_error = np.linalg.norm(current_elbow_coords - target_elbow_coords)

    # Compute the total cost
    total_cost = ee_error + elbow_weight * elbow_error
    return total_cost


# Finite difference jacobian of the cost function (gradient vector)
def jac_fd(
    joint_angles, target_ee_coords, target_elbow_coords, elbow_weight, who, length, side
):
    eps = 1e-3
    jac = np.zeros((len(joint_angles)))
    for i in range(len(joint_angles)):
        Eps = np.zeros(len(joint_angles))
        Eps[i] = eps
        jac[i] = (
            cost_function(
                joint_angles + Eps,
                target_ee_coords,
                target_elbow_coords,
                elbow_weight,
                who,
                length,
                side,
            )
            - cost_function(
                joint_angles,
                target_ee_coords,
                target_elbow_coords,
                elbow_weight,
                who,
                length,
                side,
            )
        ) / eps
    return jac


# Analytical jacobian of the cost function (gradient vector)
def jac_analytical_fixed_wrist(
    joint_angles, target_ee_coords, target_elbow_coords, elbow_weight, who, length, side
):

    number_joints = len(joint_angles)
    jac = np.zeros(number_joints)
    current_elbow_coords, current_ee_coords, fk_hand_jacob, fk_elbow_jacob = (
        forward_kinematics_with_jacobian(joint_angles, who, length, side)
    )

    hand_error = np.zeros(6)
    elbow_error = np.zeros(6)
    # Only filling the first 3 elements of the error vectors because we don't care about the orientation error
    hand_error[:3] = target_ee_coords - current_ee_coords
    elbow_error[:3] = target_elbow_coords - current_elbow_coords
    hand_norm = np.linalg.norm(hand_error)
    elbow_norm = np.linalg.norm(elbow_error)

    if hand_norm == 0:
        hand_term = np.zeros(6)
    else:
        hand_term = hand_error / hand_norm

    if elbow_norm == 0:
        elbow_term = np.zeros(6)
    else:
        elbow_term = elbow_error / elbow_norm

    for i in range(number_joints):
        fk_hand_jacob_qi = fk_hand_jacob[:, i]
        fk_elbow_jacob_qi = fk_elbow_jacob[:, i]

        jac[i] = -np.dot(hand_term, fk_hand_jacob_qi) - elbow_weight * np.dot(
            elbow_term, fk_elbow_jacob_qi
        )

    return jac


def inverse_kinematics_fixed_wrist(
    ee_coords,
    elbow_coords,
    initial_guess,
    elbow_weight=0.1,
    who="reachy",
    length=[0.28, 0.25, 0.075],
    side="right",
):
    """
    Implement the inverse kinematics with a fixed wrist.
    Input initial_guess is in degrees. Output is in degrees.
    """
    pi = np.pi
    # Convert initial guess from degrees to radians
    initial_guess_rad = np.deg2rad(initial_guess)
    joint_limits_fixed_wrist = [
        (-1.0 * pi, 0.5 * pi),
        (-1.0 * pi, 10 / 180 * pi),
        (-0.5 * pi, 0.5 * pi),
        (-125 / 180 * pi, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ]
    result = minimize(
        cost_function,
        initial_guess_rad,  # Use radian value for optimization
        jac=jac_analytical_fixed_wrist,
        args=(ee_coords, elbow_coords, elbow_weight, who, length, side),
        method="SLSQP",
        bounds=joint_limits_fixed_wrist,
        tol=1e-2,  # Higher tolerance
        options={"maxiter": 20},
    )

    # Convert result from radians to degrees
    return np.rad2deg(result.x)


def inverse_kinematics_fixed_wrist(
    hand_position,
    elbow_position,
    initial_guess,
    elbow_weight=0.1,
    L=["reachy", 0.28, 0.25, 0.075],
):
    """
    Implement the inverse kinematics.
    """
    pi = np.pi
    # joint_limits = [
    #     (-1.0 * pi, 0.5 * pi),
    #     (-1.0 * pi, 10 / 180 * pi),
    #     (-0.5 * pi, 0.5 * pi),
    #     (-125 / 180 * pi, 0),
    #     (-100 / 180 * pi, 100 / 180 * pi),
    #     (-0.25 * pi, 0.25 * pi),
    #     (-0.25 * pi, 0.25 * pi),
    # ]

    joint_limits_fixed_wrist = [
        (-1.0 * pi, 0.5 * pi),
        (-1.0 * pi, 10 / 180 * pi),
        (-0.5 * pi, 0.5 * pi),
        (-125 / 180 * pi, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ]

    result = minimize(
        cost_function,
        initial_guess,
        jac=jac_analytical_fixed_wrist,
        args=(hand_position, elbow_position, elbow_weight, L),
        method="SLSQP",
        bounds=joint_limits_fixed_wrist,
    )

    return result.x


if __name__ == "__main__":

    # Desired hand position
    hand_position = np.array([0.3, 0.2, 0.1])

    # Elbow position and weight
    elbow_position = np.array([0.1, 0.1, 0.1])
    elbow_weight = 0.1

    # Initial guess for joint angles
    # you should use previous position if you have one
    initial_guess = np.array([0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1])

    # If you use reachy or human
    # L = ['who', lenght between shoulder and elbow, length between elbow and wrist, lenght between wrist and center of the hand]
    L = ["reachy", 0.28, 0.25, 0.075]

    joint_angles = inverse_kinematics_fixed_wrist(
        hand_position, elbow_position, initial_guess, elbow_weight, L
    )

    print("Joint Angles:", joint_angles)
    print("Elbow-Effector Position:", forward_kinematics(joint_angles, L)[0])
    print("Hand-Effector Position:", forward_kinematics(joint_angles, L)[1])
