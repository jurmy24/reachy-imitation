import sympy as sp


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


def compute_transformation_matrices(th, L1, L2):
    """
    Compute the transformation matrices for the robotic arm.
    """
    pi = sp.pi
    alpha = [0, -pi/2, -pi/2, -pi/2]
    d = [0, 0, 0, 0]
    r = [0, 0, -L2, 0]

    Tbase0 = mattransfo(-pi/2, 0, -pi/2, L1)
    T01 = Tbase0 * mattransfo(alpha[0], d[0], th[0], r[0])
    T12 = mattransfo(alpha[1], d[1], th[1] - pi/2, r[1])
    T23 = mattransfo(alpha[2], d[2], th[2] - pi/2, r[2])
    T34 = mattransfo(alpha[3], d[3], th[3], r[3])

    T02 = T01 * T12
    T03 = T02 * T23
    T04 = T03 * T34

    return Tbase0, T01, T02, T03, T04


def calculate_positions(Tbase0, T04):
    """
    Calculate the positions of the end-effector and another point.
    """
    OED = Tbase0[0:3, 3]
    OCD = T04[0:3, 3]

    OED = sp.simplify(OED)
    OCD = sp.simplify(OCD)

    return OED, OCD





if __name__ == "__main__":
    # Symbolic values
    th = sp.symbols('th1:5')
    L1, L2 = sp.symbols('L1 L2', real=True)

    # Compute transformation matrices
    Tbase0, T01, T02, T03, T04 = compute_transformation_matrices(th, L1, L2)

    # Calculate positions
    OED, OCD = calculate_positions(Tbase0, T04)

    print("OED:", OED)
    print("OCD:", OCD)
