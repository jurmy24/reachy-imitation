import numpy as np
from scipy.optimize import minimize
import cvxpy as cp


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


def compute_transformation_matrices(joint_angles, L):
    """
    Compute the transformation matrices for the robotic arm.
    """
    pi = np.pi
    if L[0] == "reachy":
        alpha = [0, -pi / 2, -pi / 2, -pi / 2, +pi / 2, -pi / 2, -pi / 2, -pi / 2]
        d = [0, 0, 0, 0, 0, 0, -0.325, -0.01]
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

    # Compute all transforms from base to each joint
    T_list = [T01]
    T = T01
    for i in range(7):
        T_next = mattransfo(alpha[i], d[i], th[i], r[i])
        T = T @ T_next
        T_list.append(T.copy())

    T_end = T @ mattransfo(alpha[7], d[7], th[7], r[7])  # T08
    o_n = T_end[:3, 3]

    # Build Jacobian
    J = np.zeros((3, 7)) # TO DO: DON"T INCLUDE ANGULAR, MAKE 3,7
    for i in range(7):
        T_i = T_list[i]
        z_i = T_i[:3, 2]
        o_i = T_i[:3, 3]
        J[:3, i] = np.cross(z_i, o_n - o_i)  # linear
        #J[3:, i] = z_i                      # angular

    return T_list[3], T_end, J, J[:,:4]  # elbow position, hand position, J_hand, J_elbow


def forward_kinematics(joint_angle, L=["reachy", 0.28, 0.25, 0.075]):
    """
    Calculate the hand-effector position using forward kinematics.
    """
    T04, T08, J_hand, J_elbow = compute_transformation_matrices(joint_angle, L)
    position_e = np.array(T04[0:3, 3], dtype=np.float64).flatten()
    position = np.array(T08[0:3, 3], dtype=np.float64).flatten()
    return position_e, position, J_hand, J_elbow 





def cost_function(joint_angles, hand_position, elbow_position, elbow_weight, L):
    """
    Compute the cost function that includes hand-effector and elbow position errors.
    """
    # Compute the hand position
    elbow_position_actual, current_position = forward_kinematics(joint_angles, L)
    hand_effector_error = np.linalg.norm(hand_position - current_position)
    elbow_error = np.linalg.norm(elbow_position_actual - elbow_position)

    # Compute the total cost
    total_cost = hand_effector_error + elbow_weight * elbow_error
    return total_cost


def ik_qp_solver(q_current, x_hand_desired, x_elbow_desired, elbow_weight, L):
    """
    Solves inverse kinematics using quadratic programming (QP) with a secondary task.
    """

    # Compute forward kinematics and Jacobians
    x_elbow_actual, x_hand_actual, J_hand, J_elbow = forward_kinematics(q_current, L)
    #J_hand, J_elbow = compute_jacobians(q_current, L)

    # Compute position errors
    e_hand = x_hand_desired - x_hand_actual
    e_elbow = x_elbow_desired - x_elbow_actual

    # Setup QP problem
    n_joints = len(q_current)
    dq = cp.Variable(n_joints)  # delta joint angles
    #dq_elb = cp.Variable(4)
    # Build cost
    print("J_hand @ dq:", (J_hand @ dq).shape)
    print("e_hand:", e_hand.shape)

    
    #cost = cp.norm2(J_hand @ dq - e_hand)**2 #+ elbow_weight**2 * cp.norm2(J_elbow @ dq_elb - e_elbow)**2
    cost = cp.quad_form(J_hand @ dq - e_hand, np.eye(3))
    problem = cp.Problem(cp.Minimize(0.5 * cost))

    print("The problem is QP: ", problem.is_qp())

    # Solve the QP
    problem.solve(solver=cp.OSQP)

    # Update joint angles
    if dq.value is not None:
        q_next = q_current + dq.value
    else:
        q_next = q_current  # fallback if solver fails

    return q_next

def newton_raphson_ik(q_current, x_hand_desired, x_elbow_desired, elbow_weight, L, max_iter=50, tol=1e-6):
    """
    Solve inverse kinematics using the Newton-Raphson method.
    """
    q = q_current
    for iteration in range(max_iter):
    
        # Compute forward kinematics and Jacobians
        x_elbow_actual, x_hand_actual, J_hand, J_elb = forward_kinematics(q, L)
        
        # Compute the position errors
        e_hand = x_hand_desired - x_hand_actual  # (3,) vector
        #e_elbow = x_elbow_desired - x_elbow_actual  # (3,) vector
        
        # Compute the cost (error) and the Jacobian
        J_hand_position = J_hand[:3, :]  # Take only the linear part (3, 7)
        #J_elbow_position = J_elbow[:3, :]  # Take only the linear part (3, 4)

        # Combine the errors into a single residual vector
        #residual = np.concatenate([e_hand, elbow_weight * e_elbow])

        # Concatenate the Jacobian matrices (size: 6x7)
        #J = np.vstack([J_hand_position, elbow_weight * J_elbow_position])

        # Compute the update using the Newton-Raphson method
        # Update the joint angles: q = q - J_inv * residual
        try:
            # Solve for the joint angle changes
            dq = np.linalg.pinv(J_hand_position) @ e_hand  # Pseudo-inverse for solving least squares
            q_next = q + dq  # Update the joint angles
        except np.linalg.LinAlgError:
            print("Singular matrix encountered during inversion. Skipping this iteration.")
            break
        
        # Check for convergence
        if np.linalg.norm(dq) < tol:
            print(f"Converged in {iteration+1} iterations")
            break
        
        q = q_next  # Update the joint angles for the next iteration
    
    return q

def inverse_kinematics(
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
    joint_limits = [
        (-1.0 * pi, 0.5 * pi),
        (-1.0 * pi, 10 / 180 * pi),
        (-0.5 * pi, 0.5 * pi),
        (-125 / 180 * pi, 0),
        (-100 / 180 * pi, 100 / 180 * pi),
        (-0.25 * pi, 0.25 * pi),
        (-0.25 * pi, 0.25 * pi),
    ]

    result = minimize(
        cost_function,
        initial_guess,
        args=(hand_position, elbow_position, elbow_weight, L),
        method="SLSQP",
        bounds=joint_limits,
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

    joint_angles = inverse_kinematics(
        hand_position, elbow_position, initial_guess, elbow_weight, L
    )
    # joint_angles = ik_qp_solver(
    #     initial_guess, hand_position, elbow_position, elbow_weight, L
    # )



    #joint_angles = newton_raphson_ik(initial_guess, hand_position, elbow_position, elbow_weight, L)


    print("Joint Angles:", joint_angles)
    print("Elbow-Effector Position:", forward_kinematics(joint_angles, L)[0])
    print("Hand-Effector Position:", forward_kinematics(joint_angles, L)[1])
