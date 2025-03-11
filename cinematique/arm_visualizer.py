import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from human_arm_kinematics import dh_transformation_matrix, get_joint_positions


def plot_coordinate_frame(ax, T, scale=0.1, label=None):
    """Plot coordinate frame axes at the given transformation."""
    origin = T[:3, 3]
    x_axis = origin + scale * T[:3, 0]
    y_axis = origin + scale * T[:3, 1]
    z_axis = origin + scale * T[:3, 2]

    # X axis in red
    ax.plot(
        [origin[0], x_axis[0]],
        [origin[1], x_axis[1]],
        [origin[2], x_axis[2]],
        "r-",
        linewidth=2,
    )
    # Y axis in green
    ax.plot(
        [origin[0], y_axis[0]],
        [origin[1], y_axis[1]],
        [origin[2], y_axis[2]],
        "g-",
        linewidth=2,
    )
    # Z axis in blue
    ax.plot(
        [origin[0], z_axis[0]],
        [origin[1], z_axis[1]],
        [origin[2], z_axis[2]],
        "b-",
        linewidth=2,
    )

    if label:
        ax.text(origin[0], origin[1], origin[2], label)


def visualize_robot(dh_params, joint_names=None, fig_size=(10, 8)):
    """
    Visualize the robot arm in 3D.

    Parameters:
    -----------
    dh_params : List of tuples
        List of (theta, alpha, r, d) DH parameters for each joint
    joint_names : List of strings, optional
        Names for each joint
    fig_size : tuple, optional
        Figure size
    """
    # Get joint positions
    joint_positions = get_joint_positions(dh_params)

    # Setup figure and 3D axis
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")

    # Extract x, y, z coordinates for all joints
    x = joint_positions[:, 0]
    y = joint_positions[:, 1]
    z = joint_positions[:, 2]

    # Plot robot arm segments
    ax.plot(x, y, z, "ko-", linewidth=3, markersize=6)

    # Plot coordinate frames at each joint
    T_current = np.eye(4)
    for i, params in enumerate(dh_params):
        T_i = dh_transformation_matrix(*params)
        T_current = T_current @ T_i
        label = (
            joint_names[i] if joint_names and i < len(joint_names) else f"Joint {i+1}"
        )
        plot_coordinate_frame(ax, T_current, scale=0.05, label=label)

    # Add base frame
    plot_coordinate_frame(ax, np.eye(4), scale=0.1, label="Base")

    # Add end-effector annotation
    ax.text(x[-1], y[-1], z[-1], "End Effector", color="red")

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Robot Arm Visualization")

    # Equal aspect ratio
    max_range = np.max([np.ptp(x), np.ptp(y), np.ptp(z)])
    mid_x = float(np.mean([np.min(x), np.max(x)]))
    mid_y = float(np.mean([np.min(y), np.max(y)]))
    mid_z = float(np.mean([np.min(z), np.max(z)]))
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    plt.tight_layout()
    return fig, ax


def create_interactive_robot(base_dh_params):
    """Create an interactive visualization with sliders for joint angles."""
    # Initial parameters
    num_joints = len(base_dh_params)
    current_params = base_dh_params.copy()

    # Create figure and initial plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Create initial robot visualization
    joint_positions = get_joint_positions(current_params)
    (robot_line,) = ax.plot(
        joint_positions[:, 0],
        joint_positions[:, 1],
        joint_positions[:, 2],
        "ko-",
        linewidth=3,
        markersize=6,
    )

    # Set axis limits and labels
    max_reach = max(
        sum(param[3] for param in base_dh_params), 1.0
    )  # Sum of all d values
    ax.set_xlim(-max_reach, max_reach)
    ax.set_ylim(-max_reach, max_reach)
    ax.set_zlim(-max_reach, max_reach)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Interactive Robot Arm")

    # Add sliders for joint angles
    slider_axes = []
    sliders = []
    for i in range(num_joints):
        ax_slider = plt.axes([0.2, 0.05 + i * 0.03, 0.65, 0.02])
        slider = Slider(
            ax=ax_slider,
            label=f"Joint {i+1} θ",
            valmin=-np.pi,
            valmax=np.pi,
            valinit=current_params[i][0],
        )
        slider_axes.append(ax_slider)
        sliders.append(slider)

    # Function to update plot when sliders change
    def update(_):
        # Update DH parameters with new joint angles
        for i, slider in enumerate(sliders):
            current_params[i] = (slider.val, *current_params[i][1:])

        # Recalculate joint positions
        new_positions = get_joint_positions(current_params)

        # Update robot visualization
        robot_line.set_data_3d(
            new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]
        )

        # Optional: Display end-effector position
        T_end = np.eye(4)
        for params in current_params:
            T_end = T_end @ dh_transformation_matrix(*params)
        ax.set_title(
            f"End-effector position: [{T_end[0, 3]:.2f}, {T_end[1, 3]:.2f}, {T_end[2, 3]:.2f}]"
        )

        fig.canvas.draw_idle()

    # Connect update function to sliders
    for slider in sliders:
        slider.on_changed(update)

    plt.tight_layout()
    return fig, ax, sliders


def animate_robot(dh_params, duration=5, fps=30):
    """
    Create an animation of the robot moving through a sequence.

    Parameters:
    -----------
    dh_params : List of tuples
        Base DH parameters for the robot
    duration : float
        Duration of animation in seconds
    fps : int
        Frames per second
    """
    # Setup figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Get initial joint positions
    joint_positions = get_joint_positions(dh_params)

    # Extract x, y, z coordinates
    x = joint_positions[:, 0]
    y = joint_positions[:, 1]
    z = joint_positions[:, 2]

    # Plot robot arm segments
    (line,) = ax.plot(x, y, z, "ko-", linewidth=3, markersize=6)

    # Set axis properties
    max_reach = max(sum(param[3] for param in dh_params), 1.0)  # Sum of all d values
    ax.set_xlim(-max_reach, max_reach)
    ax.set_ylim(-max_reach, max_reach)
    ax.set_zlim(-max_reach, max_reach)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Robot Arm Animation")

    # Animation function
    def update_animation(frame):
        # Calculate progress (0 to 1)
        progress = frame / (duration * fps)

        # Update joint angles based on progress
        current_params = []
        for i, (theta, alpha, r, d) in enumerate(dh_params):
            # Create some interesting motion patterns
            if i % 3 == 0:
                new_theta = theta + np.sin(progress * 2 * np.pi) * np.pi / 2
            elif i % 3 == 1:
                new_theta = theta + np.cos(progress * 2 * np.pi) * np.pi / 3
            else:
                new_theta = theta + np.sin(progress * 4 * np.pi) * np.pi / 4

            current_params.append((new_theta, alpha, r, d))

        # Recalculate joint positions
        new_positions = get_joint_positions(current_params)

        # Update the line data
        line.set_data_3d(new_positions[:, 0], new_positions[:, 1], new_positions[:, 2])

        return (line,)

    # Create animation
    frames = int(duration * fps)
    animation = FuncAnimation(
        fig=fig, func=update_animation, frames=frames, interval=1000 / fps, blit=True
    )

    plt.tight_layout()
    return animation, fig
