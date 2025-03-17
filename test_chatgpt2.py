import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


# Clear all variables and close all plots
plt.close('all')

# Modeling axes 1 to 4, right arm
select_mode = 1  # 0 = numeric for display, 1 = symbolic for calculations

if select_mode == 1:
    # Symbolic values
    th = sp.symbols('th1:9')
    L1, L2 = sp.symbols('L1 L2', real=True)
    pi = sp.pi
elif select_mode == 0:
    # Numeric values
    L1 = 0.19
    L2 = 0.28
    th = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # Arm down
    # th = np.array([np.pi/2, 0, 0, 0])  # Arm horizontal backward
    # th = np.array([-np.pi/2, 0, 0, 0])  # Arm horizontal forward
    # th = np.array([0, -np.pi/2, 0, 0])  # Arm horizontal to the right

# DHM parameters for axes 1 to 4
alpha = np.array([0, -np.pi/2, -np.pi/2, -np.pi/2, +
                 np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2])
r = np.array([0, 0, 0, -L2, 0, -0.25, -0.0325, -0.075])
d = np.zeros(8)


# Transformation matrices
Tbase0 = mattransfo(-np.pi/2, 0, -np.pi/2, L1)
T01 = Tbase0 * mattransfo(alpha[0], d[0], th[0], r[0])
T12 = mattransfo(alpha[1], d[1], th[1] - np.pi/2, r[1])
T23 = mattransfo(alpha[2], d[2], th[2] - np.pi/2, r[2])
T34 = mattransfo(alpha[3], d[3], th[3], r[3])
T45 = mattransfo(alpha[4], d[4], th[4], r[4])
T56 = mattransfo(alpha[5], d[5], th[5], r[5])
T67 = mattransfo(alpha[6], d[6], th[6], r[6])
T78 = mattransfo(alpha[7], d[7], th[7], r[7])


# Homogeneous transformation matrices
T02 = T01 * T12
T03 = T02 * T23
T04 = T03 * T34
T05 = T04 * T45
T06 = T05 * T56
T07 = T06 * T67
T08 = T07 * T78

if select_mode == 0:
    scaleparam = 0.1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([0], [0], [0], 'r*')

    # Plotting the arm
    ax.plot([0, T01[0, 3], T02[0, 3], T03[0, 3], T04[0, 3]],
            [0, T01[1, 3], T02[1, 3], T03[1, 3], T04[1, 3]],
            [0, T01[2, 3], T02[2, 3], T03[2, 3], T04[2, 3]], 'k-', linewidth=2)

    # Plotting base frame vectors
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', linestyle='--', linewidth=2)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', linestyle='--', linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', linestyle='--', linewidth=2)

    # Plotting frame 0 vectors
    ax.quiver(Tbase0[0, 3], Tbase0[1, 3], Tbase0[2, 3], Tbase0[0, 0],
              Tbase0[1, 0], Tbase0[2, 0], color='r', linewidth=2)
    ax.quiver(Tbase0[0, 3], Tbase0[1, 3], Tbase0[2, 3], Tbase0[0, 1],
              Tbase0[1, 1], Tbase0[2, 1], color='g', linewidth=2)
    ax.quiver(Tbase0[0, 3], Tbase0[1, 3], Tbase0[2, 3], Tbase0[0, 2],
              Tbase0[1, 2], Tbase0[2, 2], color='b', linewidth=2)

    # Plotting frame 4 vectors
    ax.quiver(T04[0, 3], T04[1, 3], T04[2, 3], T04[0, 0],
              T04[1, 0], T04[2, 0], color='r', linewidth=2)
    ax.quiver(T04[0, 3], T04[1, 3], T04[2, 3], T04[0, 1],
              T04[1, 1], T04[2, 1], color='g', linewidth=2)
    ax.quiver(T04[0, 3], T04[1, 3], T04[2, 3], T04[0, 2],
              T04[1, 2], T04[2, 2], color='b', linewidth=2)

    ax.set_xlabel('xb')
    ax.set_ylabel('yb')
    ax.set_zlabel('zb')
    ax.set_aspect('equal')
    plt.show()

elif select_mode == 1:
    Obase = sp.Matrix([0, 0, 0])
    OED = Tbase0[0:3, 3]
    OCD = T04[0:3, 3]

    OED = sp.simplify(OED)
    OCD = sp.simplify(OCD)

    print("OED:", OED)
    print("OCD:", OCD)
