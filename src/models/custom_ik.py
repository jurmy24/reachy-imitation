import numpy as np
from scipy.optimize import minimize
import time
from collections import deque


class MinimizeTimer:
    """Tracks timing statistics for optimization function calls."""

    def __init__(self, max_samples=1000):
        self.times = deque(maxlen=max_samples)
        self.total_time = 0
        self.calls = 0
        self.iterations = deque(maxlen=max_samples)
        self.total_iterations = 0

    def start(self):
        """Start timing a minimize call."""
        self.start_time = time.time()

    def stop(self, iterations=None):
        """Stop timing and record statistics."""
        elapsed = time.time() - self.start_time
        self.times.append(elapsed)
        self.total_time += elapsed
        self.calls += 1
        if iterations is not None:
            self.iterations.append(iterations)
            self.total_iterations += iterations

    def get_stats(self):
        """Return timing and iteration statistics."""
        if not self.times:
            return {
                "avg_time": 0,
                "min_time": 0,
                "max_time": 0,
                "total_time": 0,
                "calls": 0,
                "avg_iterations": 0,
                "min_iterations": 0,
                "max_iterations": 0,
                "total_iterations": 0,
            }

        return {
            "avg_time": sum(self.times) / len(self.times),
            "min_time": min(self.times),
            "max_time": max(self.times),
            "total_time": self.total_time,
            "calls": self.calls,
            "avg_iterations": (
                sum(self.iterations) / len(self.iterations) if self.iterations else 0
            ),
            "min_iterations": min(self.iterations) if self.iterations else 0,
            "max_iterations": max(self.iterations) if self.iterations else 0,
            "total_iterations": self.total_iterations,
        }


# Global timer instance for tracking optimization performance
minimize_timer = MinimizeTimer()


def mattransfo(alpha, d, theta, r):
    """Compute homogeneous transformation matrix for a single joint."""
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


def compute_transformation_matrices(joint_angles, who, length, side):
    """Compute transformation matrices for the entire robotic arm chain."""
    pi = np.pi
    # DH parameters for the arm
    alpha = [0, -pi / 2, -pi / 2, -pi / 2, pi / 2, -pi / 2, -pi / 2, -pi / 2]
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

    # Set up parameters based on robot type and side
    if who == "human":
        d = [0, 0, 0, 0, 0, 0, 0, 0]
        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, 0)
    if who == "reachy" and side == "right":
        d = [0, 0, 0, 0, 0, 0, -0.0325, -0.01]
        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, -0.19)
    if who == "reachy" and side == "left":
        d = [0, 0, 0, 0, 0, 0, -0.0325, 0.01]
        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, 0.19)

    # Compute transformation matrices for each joint
    T01 = Tbase0 @ mattransfo(alpha[0], d[0], th[0], r[0])
    T12 = mattransfo(alpha[1], d[1], th[1], r[1])
    T23 = mattransfo(alpha[2], d[2], th[2], r[2])
    T34 = mattransfo(alpha[3], d[3], th[3], r[3])
    T45 = mattransfo(alpha[4], d[4], th[4], r[4])
    T56 = mattransfo(alpha[5], d[5], th[5], r[5])
    T67 = mattransfo(alpha[6], d[6], th[6], r[6])
    T78 = mattransfo(alpha[7], d[7], th[7], r[7])

    # Compute cumulative transformations
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T05 = T04 @ T45
    T06 = T05 @ T56
    T07 = T06 @ T67
    T08 = T07 @ T78
    return T04, T08


def fixed_wrist_forward_kinematics_with_jacobian(joint_angles, who, length, side):
    """Compute forward kinematics and Jacobian matrices for the arm with fixed wrist."""
    pi = np.pi
    # DH parameters
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

    # Set up parameters based on robot type and side
    if who == "human":
        d = [0, 0, 0, 0, 0, 0, 0, 0]
        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, 0)
    if who == "reachy" and side == "right":
        d = [0, 0, 0, 0, 0, 0, -0.0325, -0.01]
        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, -0.19)
    if who == "reachy" and side == "left":
        d = [0, 0, 0, 0, 0, 0, -0.0325, 0.01]
        Tbase0 = mattransfo(-pi / 2, 0, -pi / 2, 0.19)

    # Compute transformation matrices
    T01 = Tbase0 @ mattransfo(alpha[0], d[0], th[0], r[0])
    Ts_gi_i1 = [T01]
    for i in range(1, 8):
        Ts_gi_i1.append(mattransfo(alpha[i], d[i], th[i], r[i]))

    # Compute cumulative transformations
    Ts_g_0i = []
    T = np.eye(4, 4)
    for i in range(8):
        T = T @ Ts_gi_i1[i]
        Ts_g_0i.append(T.copy())

    T04 = Ts_g_0i[3]
    T08 = Ts_g_0i[-1]

    # Extract positions
    position_elbow = np.array(T04[0:3, 3], dtype=np.float64).flatten()
    position_hand = np.array(T08[0:3, 3], dtype=np.float64).flatten()

    # Compute Jacobians
    jacobian = np.zeros((6, 8))
    elbow_jacobian = np.zeros((6, 8))

    for i in range(8):
        g_0i = Ts_g_0i[i]
        R_0i = g_0i[:3, :3]
        p_i = g_0i[:3, 3]
        Roz_i = R_0i[:, 2]
        jacobian[0:3, i] = np.cross(Roz_i, position_hand - p_i)
        if i < 4:
            elbow_jacobian[0:3, i] = np.cross(Roz_i, position_elbow - p_i)

    return position_elbow, position_hand, elbow_jacobian[:, :-1], jacobian[:, :-1]


def forward_kinematics(
    joint_angle, who="reachy", length=[0.28, 0.25, 0.075], side="right"
):
    """Calculate end-effector and elbow positions using forward kinematics."""
    T04, T08 = compute_transformation_matrices(joint_angle, who, length, side)
    position_elbow = np.array(T04[0:3, 3], dtype=np.float64).flatten()
    position_hand = np.array(T08[0:3, 3], dtype=np.float64).flatten()
    return position_elbow, position_hand


def cost_function(
    joint_angles, target_ee_coords, target_elbow_coords, elbow_weight, who, length, side
):
    """Compute cost function for optimization, combining hand and elbow position errors."""
    current_elbow_coords, current_ee_coords = forward_kinematics(
        joint_angles, who, length, side
    )
    ee_error = np.linalg.norm(current_ee_coords - target_ee_coords)
    elbow_error = np.linalg.norm(current_elbow_coords - target_elbow_coords)
    return ee_error + elbow_weight * elbow_error


def jac_fd(
    joint_angles, hand_position, elbow_position, elbow_weight, who, length, side
):
    """Compute Jacobian using finite differences."""
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


def jac_analytical_fixed_wrist(
    joint_angles, target_ee_coords, target_elbow_coords, elbow_weight, who, length, side
):
    """Compute analytical Jacobian for the cost function with fixed wrist."""
    number_joints = len(joint_angles)
    jac = np.zeros(number_joints)
    current_elbow_coords, current_ee_coords, fk_elbow_jacob, fk_ee_jacob = (
        fixed_wrist_forward_kinematics_with_jacobian(joint_angles, who, length, side)
    )

    # Compute error vectors
    hand_error = np.zeros(6)
    elbow_error = np.zeros(6)
    hand_error[:3] = target_ee_coords - current_ee_coords
    elbow_error[:3] = target_elbow_coords - current_elbow_coords
    hand_norm = np.linalg.norm(hand_error)
    elbow_norm = np.linalg.norm(elbow_error)

    # Normalize error vectors
    hand_term = hand_error / hand_norm if hand_norm != 0 else np.zeros(6)
    elbow_term = elbow_error / elbow_norm if elbow_norm != 0 else np.zeros(6)

    jac = -hand_term @ fk_ee_jacob - elbow_weight * elbow_term @ fk_elbow_jacob
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
    """Solve inverse kinematics with fixed wrist using optimization."""
    pi = np.pi
    initial_guess_rad = np.deg2rad(initial_guess)

    # Define joint limits based on robot side
    if side == "right":
        joint_limits_fixed_wrist = [
            (-1.0 * pi, 0.5 * pi),
            (-1.0 * pi, 10 / 180 * pi),
            (-0.5 * pi, 0.5 * pi),
            (-125 / 180 * pi, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ]
    elif side == "left":
        joint_limits_fixed_wrist = [
            (-1.0 * pi, 0.5 * pi),
            (-10 / 180 * pi, 1.0 * pi),
            (-0.5 * pi, 0.5 * pi),
            (-125 / 180 * pi, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ]

    # Run optimization
    minimize_timer.start()
    result = minimize(
        cost_function,
        initial_guess_rad,
        jac=jac_analytical_fixed_wrist,
        args=(ee_coords, elbow_coords, elbow_weight, who, length, side),
        method="SLSQP",
        bounds=joint_limits_fixed_wrist,
        tol=1e-3,
        options={"maxiter": 20},
    )
    minimize_timer.stop(result.nit)

    return np.rad2deg(result.x)


if __name__ == "__main__":
    # Test parameters
    hand_position = np.array([0.3, 0.2, 0.1])
    elbow_position = np.array([0.1, 0, -0.3])
    elbow_weight = 0
    initial_guess = np.array([0.1, 0.3, 0.1, 0.1, 0, 0, 0])
    who = "reachy"
    length = [0.28, 0.25, 0.075]
    side = "right"

    # Run test and print statistics
    result = inverse_kinematics_fixed_wrist(
        hand_position, elbow_position, initial_guess, elbow_weight, who, length, side
    )
    stats = minimize_timer.get_stats()

    print("\n===== MINIMIZE FUNCTION TIMING STATISTICS =====")
    print(f"Total calls: {stats['calls']}")
    print(f"Average time: {stats['avg_time']*1000:.2f} ms")
    print(f"Min time: {stats['min_time']*1000:.2f} ms")
    print(f"Max time: {stats['max_time']*1000:.2f} ms")
    print(f"Total time: {stats['total_time']*1000:.2f} ms")
    print(f"\nIteration Statistics:")
    print(f"Average iterations: {stats['avg_iterations']:.1f}")
    print(f"Min iterations: {stats['min_iterations']}")
    print(f"Max iterations: {stats['max_iterations']}")
    print(f"Total iterations: {stats['total_iterations']}")
