import numpy as np
from scipy.optimize import minimize
import time
from collections import deque


class MinimizeTimer:
    """Class to track timing statistics for the minimize function."""

    def __init__(self, max_samples=1000):
        self.times = deque(maxlen=max_samples)
        self.total_time = 0
        self.calls = 0

    def start(self):
        """Start timing a minimize call."""
        self.start_time = time.time()

    def stop(self):
        """Stop timing a minimize call and record the result."""
        elapsed = time.time() - self.start_time
        self.times.append(elapsed)
        self.total_time += elapsed
        self.calls += 1

    def get_stats(self):
        """Get statistics about the minimize function calls."""
        if not self.times:
            return {
                "avg_time": 0,
                "min_time": 0,
                "max_time": 0,
                "total_time": 0,
                "calls": 0,
            }

        return {
            "avg_time": sum(self.times) / len(self.times),
            "min_time": min(self.times),
            "max_time": max(self.times),
            "total_time": self.total_time,
            "calls": self.calls,
        }


# Create a global timer instance
minimize_timer = MinimizeTimer()


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
# length = lenght parameters


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
    position_elbow = np.array(T04[0:3, 3], dtype=np.float64).flatten()
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
            elbow_jacobian[0:3, i] = np.cross(Roz_i, position_elbow - p_i)
        # else:
        #     elbow_jacobian[0:3, i] = 0

    return (
        T04,
        T08,
        elbow_jacobian[:, :-1],
        jacobian[:, :-1],
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


def forward_kinematics(
    joint_angle, who="reachy", length=[0.28, 0.25, 0.075], side="right"
):
    """
    Calculate the hand-effector position using forward kinematics.
    """
    T04, T08 = compute_transformation_matrices(joint_angle, who, length, side)
    position_elbow = np.array(T04[0:3, 3], dtype=np.float64).flatten()
    position_hand = np.array(T08[0:3, 3], dtype=np.float64).flatten()
    return position_elbow, position_hand


def forward_kinematics_with_jacobian(
    joint_angles, who="reachy", length=[0.28, 0.25, 0.075], side="right"
):
    """
    Calculate the hand-effector position using forward kinematics.

    Args:
        joint_angles: The joint angles of the robotic arm.
        who: The type of robotic arm.
        length: The length of the robotic arm.
        side: The side of the robotic arm.

    Returns:
        T04: The transformation matrix of the elbow.
        T08: The transformation matrix of the hand.
        elbow_jacobian: The Jacobian matrix of the elbow.
        jacobian: The Jacobian matrix for the hand.
    """
    T04, T08, elbow_jacobian, jacobian = compute_transformation_matrices_and_jacobian(
        joint_angles, who, length, side
    )
    position_elbow = np.array(T04[0:3, 3], dtype=np.float64).flatten()
    position_hand = np.array(T08[0:3, 3], dtype=np.float64).flatten()
    return position_elbow, position_hand, elbow_jacobian, jacobian


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


# ! This is actually the gradient of the cost function, not the jacobian
def jac_fd(
    joint_angles, hand_position, elbow_position, elbow_weight, who, length, side
):
    eps = 1e-3
    jac = np.zeros((len(joint_angles)))
    for i in range(len(joint_angles)):
        Eps = np.zeros(len(joint_angles))
        Eps[i] = eps
        jac[i] = (
            cost_function(
                joint_angles + Eps,
                hand_position,
                elbow_position,
                elbow_weight,
                who,
                length,
                side,
            )
            - cost_function(
                joint_angles,
                hand_position,
                elbow_position,
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
    current_elbow_coords, current_ee_coords, fk_elbow_jacob, fk_ee_jacob = (
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
        fk_hand_jacob_qi = fk_ee_jacob[:, i]
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
    # joint_limits = [
    #     (-1.0 * pi, 0.5 * pi),
    #     (-1.0 * pi, 10 / 180 * pi),
    #     (-0.5 * pi, 0.5 * pi),
    #     (-125 / 180 * pi, 0),
    #     (-100 / 180 * pi, 100 / 180 * pi),
    #     (-0.25 * pi, 0.25 * pi),
    #     (-0.25 * pi, 0.25 * pi),
    # ]
    # Start timing
    minimize_timer.start()
    result = minimize(
        cost_function,
        initial_guess_rad,  # Use radian value for optimization
        jac=jac_fd,
        args=(ee_coords, elbow_coords, elbow_weight, who, length, side),
        method="SLSQP",
        bounds=joint_limits_fixed_wrist,
        tol=1e-2,  # Higher tolerance
        options={"maxiter": 20},
    )
    # Stop timing
    minimize_timer.stop()

    # Convert result from radians to degrees
    return np.rad2deg(result.x)


if __name__ == "__main__":

    # Desired hand position
    hand_position = np.array([0.3, 0.2, 0.1])

    # Elbow position and weight
    elbow_position = np.array([0.1, 0, -0.3])
    elbow_weight = 0

    # Initial guess for joint angles
    # you should use previous position if you have one
    initial_guess = np.array([0.1, 0.3, 0.1, 0.1, 0, 0, 0])

    # If you use reachy or human
    # length=[length between shoulder and elbow, length between elbow and wrist, length between wrist and center of the hand]
    who = "reachy"
    length = [0.28, 0.25, 0.075]
    side = "right"

    # Run a test and print the timing statistics
    result = inverse_kinematics_fixed_wrist(
        hand_position, elbow_position, initial_guess, elbow_weight, who, length, side
    )

    # Print timing statistics
    stats = minimize_timer.get_stats()
    print("\n===== MINIMIZE FUNCTION TIMING STATISTICS =====")
    print(f"Total calls: {stats['calls']}")
    print(f"Average time: {stats['avg_time']*1000:.2f} ms")
    print(f"Min time: {stats['min_time']*1000:.2f} ms")
    print(f"Max time: {stats['max_time']*1000:.2f} ms")
    print(f"Total time: {stats['total_time']*1000:.2f} ms")
