import numpy as np
import matplotlib.pyplot as plt
from kinematics.fk.human_arm_fk import forward_kinematics
from visualization.fk.human_arm_visualizer import (
    visualize_robot,
    create_interactive_robot,
    animate_robot,
)


def main():
    """Run example robot visualization demos"""
    # Values for the joint angles
    q_1 = np.pi / 4
    q_2 = np.pi / 6
    q_3 = np.pi / 3
    q_4 = np.pi / 4
    q_5 = np.pi / 3
    q_6 = np.pi / 4
    q_7 = np.pi / 3
    d_3 = 0.35  # length of shoulder -> elbow joint (upper arm)
    d_5 = 0.4  # length of elbow -> wrist joint (forearm)

    # Example DH parameters for a 7-DOF robot arm
    # Format: (theta, alpha, r, d) for each joint
    dh_params = [
        (q_1, np.pi / 2, 0, 0),  # Joint 1
        (q_2, np.pi / 2, 0, 0),  # Joint 2
        (q_3, -np.pi / 2, 0, d_3),  # Joint 3
        (q_4, -np.pi / 2, 0, 0),  # Joint 4
        (q_5, np.pi / 2, 0, d_5),  # Joint 5
        (q_6, np.pi / 2, 0, 0),  # Joint 6
        (q_7, 0, 0, 0),  # Joint 7
    ]

    # Joint names for better visualization
    joint_names = [
        "Shoulder Flexion (forward/backward)",
        "Shoulder Abduction/Adduction (left/right)",
        "Shoulder Rotation (cw/ccw)",
        "Elbow Flexion",
        "Elbow Pronation/Supination",
        "Wrist Abduction",
        "Wrist Flexion",
    ]

    # Calculate end-effector transformation
    T_end_effector = forward_kinematics(dh_params)
    print("End-effector transformation matrix:")
    np.set_printoptions(precision=4, suppress=True)
    print(T_end_effector)

    # Choose which visualization to run
    run_static = True
    run_interactive = True
    run_animation = True

    # Run static visualization
    if run_static:
        print("Creating static visualization...")
        fig, ax = visualize_robot(dh_params, joint_names)
        # plt.savefig("robot_visualization.png")
        plt.show(block=False)

    # Run interactive visualization
    if run_interactive:
        print("Creating interactive visualization...")
        fig_interactive, _, _ = create_interactive_robot(dh_params, joint_names)
        plt.show(block=False)

    # Run animated visualization
    if run_animation:
        print("Creating animation...")
        animation, fig_anim = animate_robot(dh_params)

        # Uncomment to save animation
        # animation.save('robot_animation.mp4', writer='ffmpeg', dpi=100)

        plt.show()

    print("Done! Close the plot windows to exit.")


if __name__ == "__main__":
    main()
