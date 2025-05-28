import numpy as np
from scipy.optimize import minimize
import time
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=6, suppress=True)

class MinimizeTimer:
    """Class to track timing statistics for the minimize function."""

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
        """Stop timing a minimize call and record the result."""
        elapsed = time.time() - self.start_time
        self.times.append(elapsed)
        self.total_time += elapsed
        self.calls += 1
        if iterations is not None:
            self.iterations.append(iterations)
            self.total_iterations += iterations

    def get_stats(self):
        """Get statistics about the minimize function calls."""
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
            "avg_iterations": sum(self.iterations) / len(self.iterations) if self.iterations else 0,
            "min_iterations": min(self.iterations) if self.iterations else 0,
            "max_iterations": max(self.iterations) if self.iterations else 0,
            "total_iterations": self.total_iterations,
        }

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

def compute_transformation_matrices(joint_angles, who, length, side):
    """Compute the transformation matrices for the robotic arm."""
    pi = np.pi
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

def fixed_wrist_forward_kinematics_with_jacobian(joint_angles, who, length, side):
    """Compute the transformation matrices and jacobians for the robotic arm."""
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

    return (
        position_elbow,
        position_hand,
        elbow_jacobian[:, :-1],
        jacobian[:, :-1],
    )

def forward_kinematics(joint_angle, who="reachy", length=[0.28, 0.25, 0.075], side="right"):
    """Calculate the hand-effector position using forward kinematics."""
    T04, T08 = compute_transformation_matrices(joint_angle, who, length, side)
    position_elbow = np.array(T04[0:3, 3], dtype=np.float64).flatten()
    position_hand = np.array(T08[0:3, 3], dtype=np.float64).flatten()
    return position_elbow, position_hand

def cost_function(joint_angles, target_ee_coords, target_elbow_coords, elbow_weight, who, length, side):
    """Compute the cost function that includes hand-effector and elbow position errors."""
    current_elbow_coords, current_ee_coords = forward_kinematics(joint_angles, who, length, side)
    ee_error = np.linalg.norm(current_ee_coords - target_ee_coords)
    elbow_error = np.linalg.norm(current_elbow_coords - target_elbow_coords)
    total_cost = ee_error + elbow_weight * elbow_error
    return total_cost

def jac_fd(joint_angles, hand_position, elbow_position, elbow_weight, who, length, side):
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

def jac_analytical_fixed_wrist(joint_angles, target_ee_coords, target_elbow_coords, elbow_weight, who, length, side):
    number_joints = len(joint_angles)
    jac = np.zeros(number_joints)
    current_elbow_coords, current_ee_coords, fk_elbow_jacob, fk_ee_jacob = (
        fixed_wrist_forward_kinematics_with_jacobian(joint_angles, who, length, side)
    )

    hand_error = np.zeros(6)
    elbow_error = np.zeros(6)
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

    jac = -hand_term @ fk_ee_jacob - elbow_weight * elbow_term @ fk_elbow_jacob
    return jac

def inverse_kinematics_fixed_wrist(
    ee_coords,
    elbow_coords,
    initial_guess,
    minimize_timer,
    elbow_weight=0.1,
    who="reachy",
    length=[0.28, 0.25, 0.075],
    side="right",
    jacobian=jac_analytical_fixed_wrist
):
    """Implement the inverse kinematics with a fixed wrist."""
    pi = np.pi
    initial_guess_rad = np.deg2rad(initial_guess)
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
            (-10 / 180 * pi,  1.0 * pi),
            (-0.5 * pi, 0.5 * pi),
            (-125 / 180 * pi, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ]

    minimize_timer.start()
    result = minimize(
        cost_function,
        initial_guess_rad,
        jac=jacobian,
        args=(ee_coords, elbow_coords, elbow_weight, who, length, side),
        method="SLSQP",
        bounds=joint_limits_fixed_wrist,
        tol=1e-3,
        options={"maxiter": 20},
    )
    minimize_timer.stop(result.nit)

    return np.rad2deg(result.x), result.success

def test_solver_single(jacobian, hand_position, elbow_position, initial_guess, elbow_weight, who, length, side):
    """Test a single solve and return metrics."""
    minimize_timer = MinimizeTimer()
    
    result_angles, success = inverse_kinematics_fixed_wrist(
        hand_position, elbow_position, initial_guess, minimize_timer, elbow_weight, who, length, side, jacobian
    )
    
    if not success:
        return None  # Failed solve
    
    # Calculate actual positions
    position_elbow, position_hand = forward_kinematics(
        np.deg2rad(result_angles), who=who, length=length, side=side
    )
    
    # Calculate errors
    elbow_error = np.linalg.norm(position_elbow - elbow_position)
    hand_error = np.linalg.norm(position_hand - hand_position)
    
    stats = minimize_timer.get_stats()
    
    return {
        'iterations': stats['total_iterations'],
        'time_ms': stats['total_time'] * 1000,
        'elbow_error': elbow_error,
        'hand_error': hand_error
    }

def test_reachy_sdk_single(reachy, hand_position, elbow_position):
    """Test a single solve with Reachy SDK and return metrics."""
    minimize_timer = MinimizeTimer()
    
    try:
        # Create transformation matrix for target hand position
        transform_matrix = reachy.r_arm.forward_kinematics()
        transform_matrix[:3, 3] = hand_position
        
        minimize_timer.start()
        # Compute IK with Reachy SDK
        joint_pos = reachy.r_arm.inverse_kinematics(transform_matrix)
        minimize_timer.stop()
        
        if joint_pos is None:
            return None  # Failed solve
        
        # Calculate actual positions using our forward kinematics
        position_elbow, position_hand = forward_kinematics(
            np.deg2rad(joint_pos), who="reachy", length=[0.28, 0.25, 0.075], side="right"
        )
        
        # Calculate errors
        elbow_error = np.linalg.norm(position_elbow - elbow_position)
        hand_error = np.linalg.norm(position_hand - hand_position)
        
        stats = minimize_timer.get_stats()
        
        return {
            'iterations': None,  # Reachy SDK doesn't provide iteration count
            'time_ms': stats['total_time'] * 1000,
            'elbow_error': elbow_error,
            'hand_error': hand_error
        }
        
    except Exception as e:
        print(f"Reachy SDK error: {e}")
        return None

def generate_test_positions(base_hand_pos, base_elbow_pos, n_tests=20):
    """Generate n_tests different target positions by varying base positions by +/- 0.05."""
    np.random.seed(42)  # For reproducible results
    
    test_positions = []
    for _ in range(n_tests):
        # Add random variation of +/- 0.05 to each coordinate
        hand_variation = np.random.uniform(-0.05, 0.05, 3)
        elbow_variation = np.random.uniform(-0.05, 0.05, 3)
        
        new_hand_pos = base_hand_pos + hand_variation
        new_elbow_pos = base_elbow_pos + elbow_variation
        
        test_positions.append((new_hand_pos, new_elbow_pos))
    
    return test_positions

def benchmark_solvers():
    """Benchmark all solver types with multiple test positions."""
    
    # Base test configuration
    base_hand_position = np.array([0.3, -0.2, 0.1])
    base_elbow_position = np.array([0.1, 0, -0.3])
    elbow_weight = 0
    initial_guess = np.array([0.1, 0.3, 0.1, 0.1, 0, 0, 0])
    who = "reachy"
    length = [0.28, 0.25, 0.075]
    side = "right"
    
    # Generate test positions
    test_positions = generate_test_positions(base_hand_position, base_elbow_position, 20)
    
    # Define solvers
    solvers = {
        'Analytical Jacobian': jac_analytical_fixed_wrist,
        'Finite Difference': jac_fd,
        'Built-in Jacobian': None,
        'Reachy SDK': 'reachy_sdk'  # Special case
    }
    
    results = {}
    
    # Try to initialize Reachy SDK
    reachy = None
    try:
        from reachy_sdk import ReachySDK        
        reachy = ReachySDK(host="192.168.100.2")
        print("Reachy SDK connected successfully!")
    except Exception as e:
        print(f"Reachy SDK not available: {e}")
        reachy = None
    
    for solver_name, jacobian_func in solvers.items():
        print(f"Testing {solver_name}...")
        
        if solver_name == 'Reachy SDK':
            if reachy is None:
                print(f"Skipping {solver_name} - SDK not available")
                results[solver_name] = {
                    'iterations': np.nan,
                    'time_ms': np.nan,
                    'elbow_error': np.nan,
                    'hand_error': np.nan,
                    'success_rate': 0.0
                }
                continue
            
            # Test Reachy SDK
            solver_results = []
            for i, (hand_pos, elbow_pos) in enumerate(test_positions):
                result = test_reachy_sdk_single(reachy, hand_pos, elbow_pos)
                
                if result is not None:
                    solver_results.append(result)
                else:
                    print(f"  Test {i+1} failed for {solver_name}")
        else:
            # Test other solvers
            solver_results = []
            for i, (hand_pos, elbow_pos) in enumerate(test_positions):
                result = test_solver_single(
                    jacobian_func, hand_pos, elbow_pos, initial_guess, 
                    elbow_weight, who, length, side
                )
                
                if result is not None:
                    solver_results.append(result)
                else:
                    print(f"  Test {i+1} failed for {solver_name}")
        
        if solver_results:
            # Calculate averages
            iterations_list = [r['iterations'] for r in solver_results if r['iterations'] is not None]
            avg_iterations = np.mean(iterations_list) if iterations_list else np.nan
            
            avg_time = np.mean([r['time_ms'] for r in solver_results])
            avg_elbow_error = np.mean([r['elbow_error'] for r in solver_results])
            avg_hand_error = np.mean([r['hand_error'] for r in solver_results])
            
            results[solver_name] = {
                'iterations': avg_iterations,
                'time_ms': avg_time,
                'elbow_error': avg_elbow_error,
                'hand_error': avg_hand_error,
                'success_rate': len(solver_results) / len(test_positions)
            }
        else:
            results[solver_name] = {
                'iterations': np.nan,
                'time_ms': np.nan,
                'elbow_error': np.nan,
                'hand_error': np.nan,
                'success_rate': 0.0
            }
    
    return results

def create_comparison_table(results):
    """Create a formatted comparison table with color coding."""
    
    # Create DataFrame
    df_data = []
    for solver, metrics in results.items():
        iterations_str = f"{metrics['iterations']:.1f}" if not np.isnan(metrics['iterations']) else "N/A"
        df_data.append([
            solver,
            iterations_str,
            f"{metrics['time_ms']:.2f}" if not np.isnan(metrics['time_ms']) else "N/A",
            f"{metrics['elbow_error']*1000:.2f}" if not np.isnan(metrics['elbow_error']) else "N/A",
            f"{metrics['hand_error']*1000:.2f}" if not np.isnan(metrics['hand_error']) else "N/A"
        ])
    
    df = pd.DataFrame(df_data, columns=[
        'Solver', 'Avg Iterations', 'Avg Latency (ms)', 
        'Avg Elbow Error (mm)', 'Avg Hand Error (mm)'
    ])
    
    print("\n" + "="*80)
    print("INVERSE KINEMATICS SOLVER COMPARISON")
    print("="*80)
    print("(Based on 20 test positions with ±0.05m variation)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Create color-coded visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    solvers = list(results.keys())
    
    # Latency comparison
    latencies = [results[s]['time_ms'] for s in solvers if not np.isnan(results[s]['time_ms'])]
    valid_solvers_lat = [s for s in solvers if not np.isnan(results[s]['time_ms'])]
    
    colors_lat = ['green' if x < 5 else 'orange' if x < 20 else 'red' for x in latencies]
    bars1 = ax1.bar(valid_solvers_lat, latencies, color=colors_lat)
    ax1.set_title('Average Latency (ms)')
    ax1.set_ylabel('Time (ms)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars1, latencies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', va='bottom')
    
    # Iterations comparison (only for solvers that provide iteration count)
    iterations = [results[s]['iterations'] for s in solvers if not np.isnan(results[s]['iterations'])]
    valid_solvers_iter = [s for s in solvers if not np.isnan(results[s]['iterations'])]
    
    if iterations:  # Only create plot if we have iteration data
        colors_iter = ['green' if x < 5 else 'orange' if x < 15 else 'red' for x in iterations]
        bars2 = ax2.bar(valid_solvers_iter, iterations, color=colors_iter)
        ax2.set_title('Average Iterations')
        ax2.set_ylabel('Iterations')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars2, iterations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{val:.1f}', ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, 'No iteration data available\n(Reachy SDK does not provide)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Average Iterations')
    
    # Elbow error comparison (in mm)
    elbow_errors = [results[s]['elbow_error']*1000 for s in solvers if not np.isnan(results[s]['elbow_error'])]
    valid_solvers_elbow = [s for s in solvers if not np.isnan(results[s]['elbow_error'])]
    
    colors_elbow = ['green' if x < 1 else 'orange' if x < 5 else 'red' for x in elbow_errors]
    bars3 = ax3.bar(valid_solvers_elbow, elbow_errors, color=colors_elbow)
    ax3.set_title('Average Elbow Error (mm)')
    ax3.set_ylabel('Error (mm)')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars3, elbow_errors):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}', ha='center', va='bottom')
    
    # Hand error comparison (in mm)
    hand_errors = [results[s]['hand_error']*1000 for s in solvers if not np.isnan(results[s]['hand_error'])]
    valid_solvers_hand = [s for s in solvers if not np.isnan(results[s]['hand_error'])]
    
    colors_hand = ['green' if x < 1 else 'orange' if x < 5 else 'red' for x in hand_errors]
    bars4 = ax4.bar(valid_solvers_hand, hand_errors, color=colors_hand)
    ax4.set_title('Average Hand Error (mm)')
    ax4.set_ylabel('Error (mm)')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars4, hand_errors):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Create summary analysis
    print("\nPERFORMANCE ANALYSIS:")
    print("-" * 40)
    
    # Find best performer in each category (excluding NaN values)
    valid_results = {k: v for k, v in results.items() if not np.isnan(v['time_ms'])}
    
    if valid_results:
        fastest = min(valid_results.keys(), key=lambda x: valid_results[x]['time_ms'])
        most_accurate_hand = min(valid_results.keys(), key=lambda x: valid_results[x]['hand_error'])
        most_accurate_elbow = min(valid_results.keys(), key=lambda x: valid_results[x]['elbow_error'])
        
        # Only analyze iterations for solvers that provide iteration count
        iter_results = {k: v for k, v in valid_results.items() if not np.isnan(v['iterations'])}
        if iter_results:
            least_iterations = min(iter_results.keys(), key=lambda x: iter_results[x]['iterations'])
            print(f"⚡ Fewest iterations: {least_iterations} ({iter_results[least_iterations]['iterations']:.1f})")
        
        print(f"🏃 Fastest: {fastest} ({valid_results[fastest]['time_ms']:.2f} ms)")
        print(f"🎯 Most accurate (hand): {most_accurate_hand} ({valid_results[most_accurate_hand]['hand_error']*1000:.2f} mm)")
        print(f"🎯 Most accurate (elbow): {most_accurate_elbow} ({valid_results[most_accurate_elbow]['elbow_error']*1000:.2f} mm)")
        
        # Special note about Reachy SDK
        if 'Reachy SDK' in valid_results:
            print(f"📡 Reachy SDK: {valid_results['Reachy SDK']['time_ms']:.2f} ms (iterations not reported)")
    
    return df

if __name__ == "__main__":
    # Run the benchmark
    results = benchmark_solvers()
    
    # Create and display the comparison table
    comparison_df = create_comparison_table(results)