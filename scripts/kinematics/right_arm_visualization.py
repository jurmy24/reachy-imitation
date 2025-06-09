import numpy as np
import matplotlib.pyplot as plt
from human_arm_kinematics import forward_kinematics
from arm_visualizer import visualize_robot, create_interactive_robot, animate_robot
import numpy as np
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# go one directory up and then into 'other_folder/utils'
# module_path = os.path.abspath(os.path.join('..', "src"))
# if module_path not in sys.path:
#     sys.path.append(module_path)
from filterpy.kalman import KalmanFilter

class KalmanFilter3D:
    def __init__(self, threshold = 4):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        dt = 1 / 30.0  # Time step # 15 represents the approximate frame rate - Verify this 

        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        # Measurement function
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        self.kf.R *= 10  # Measurement noise
        self.kf.P *= 100.  # Initial covariance
        self.kf.Q *= 1e-2  # Process noise

        self.kf.x[:6] = np.zeros((6, 1))  # Initial state
        self.last_output = np.zeros(3)
        self.threshold = threshold  # meters

    # def update(self, measurement):
    #     self.kf.predict()
    #     self.kf.update(measurement)
    #     return self.kf.x[:3].reshape(-1)  # Return x, y, z
    
    def update(self, measurement):
        measurement = np.array(measurement)

        #Reject if too far from last prediction
        if np.linalg.norm(measurement - self.last_output) > self.threshold:
            print("Measurement rejected due to threshold limit.")
            return self.last_output

        self.kf.predict()
        self.kf.update(measurement)
        self.last_output = self.kf.x[:3].reshape(-1)
        return self.last_output


def project_joint_to_length(origin, joint, target_length, alpha=0.8):
    """Project joint to a fixed distance from the origin, with interpolation for smoothing."""
    direction = joint - origin
    norm = np.linalg.norm(direction)
    if norm < 1e-5:
        return joint  # Avoid division by zero if direction is very small

    corrected = origin + direction / norm * target_length
    # Interpolate for smooth transition (alpha is "trust" in correction)
    return alpha * corrected + (1 - alpha) * joint


# Helper function to convert a landmark to 3D coordinates from the camera's perspective using the human coordinate system
def get_3D_coordinates(landmark, depth_frame, w, h, intrinsics):
    """Convert a landmark to 3D coordinates using depth information.

    Note that this function converts the camera frame to the human frame whereas the origin remains the same.

    Transforms from camera coordinates to robot coordinates:
    - Human x = -Camera depth
    - Human y = -Camera x
    - Human z = -Camera y

    Args:
        landmark: Either a landmark object with x, y, z attributes or
                 a numpy array/list with [x, y, z] coordinates (normalized)
        depth_frame: The depth frame from the camera
        w: Image width
        h: Image height
        intrinsics: Camera intrinsic parameters
    """
    # Handle both landmark objects and numpy arrays/lists
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        cx, cy = int(landmark.x * w), int(landmark.y * h)
    else:
        # Assume it's a numpy array or list with [x, y, z]
        cx, cy = int(landmark[0] * w), int(landmark[1] * h)

    # Check if pixel coordinates are within image bounds
    if 0 <= cx < w and 0 <= cy < h:
        depth = depth_frame.get_distance(cx, cy)
        if depth > 0:  # Ensure depth is valid
            # Get camera intrinsic parameters
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy

            # Convert to camera 3D coordinates
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy

            # Transform to robot coordinate system
            # TODO: based on the camera's system, it should really be -z, -x, -y
            return np.array([-depth, x, -y])

    # Default return if coordinates are invalid
    return np.array([0, 0, 0])

def record_arm():

    import cv2
    import pyrealsense2 as rs
    # import utils.angles as utils


    
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    # Create model instances with optimized parameters
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
    )

    pose = mp_pose.Pose(
        static_image_mode=False,
        smooth_landmarks=True,
    )
    required_landmarks = [
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_WRIST,
    ]
    # Configure intel RealSense camera (color and depth streams)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Align depth frame to color frame
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)

    # Cache intrinsics for repeated use
    intrinsics = (
        profile.get_stream(rs.stream.depth)
        .as_video_stream_profile()
        .get_intrinsics()
    )

    mp_draw = mp.solutions.drawing_utils
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # Start streaming
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    kalman_trajectory = {}
    kalman_trajectory[mp_pose.PoseLandmark.RIGHT_ELBOW] = []
    kalman_trajectory[mp_pose.PoseLandmark.RIGHT_SHOULDER] = []
    kalman_trajectory[mp_pose.PoseLandmark.RIGHT_WRIST] = []
    raw_trajectory = {}
    raw_trajectory[mp_pose.PoseLandmark.RIGHT_ELBOW] = []
    raw_trajectory[mp_pose.PoseLandmark.RIGHT_SHOULDER] = []
    raw_trajectory[mp_pose.PoseLandmark.RIGHT_WRIST] = []
    ema_trajectory = {}
    ema_trajectory[mp_pose.PoseLandmark.RIGHT_ELBOW] = []
    ema_trajectory[mp_pose.PoseLandmark.RIGHT_SHOULDER] = []
    ema_trajectory[mp_pose.PoseLandmark.RIGHT_WRIST] = []
    
    # Initialize Kalman filters for each joint
    kf_shoulder = KalmanFilter3D()
    kf_elbow = KalmanFilter3D()
    kf_wrist = KalmanFilter3D()
    its = 0
    ema_alpha = 0.4
    try:
        while its < FRAME_CAP:
            its += 1
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            h, w, _ = color_image.shape

            pose_results = pose.process(rgb_image)
            
            if pose_results.pose_landmarks:

                mp_draw.draw_landmarks(
                    color_image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                for landmark in required_landmarks:
                        coord = get_3D_coordinates(
                            pose_results.pose_landmarks.landmark[landmark],
                            depth_frame,
                            w,
                            h,
                            intrinsics,
                        )

                        # Apply the corresponding Kalman filter
                        if landmark == mp_pose.PoseLandmark.RIGHT_SHOULDER:
                            kalman_coord = kf_shoulder.update(coord)
                        elif landmark == mp_pose.PoseLandmark.RIGHT_ELBOW:
                            kalman_coord = kf_elbow.update(coord)
                        elif landmark == mp_pose.PoseLandmark.RIGHT_WRIST:
                            kalman_coord = kf_wrist.update(coord)

                        if len(ema_trajectory[landmark]) != 0:
                            ema_coord = ema_alpha * coord + (1 - ema_alpha) * ema_trajectory[landmark][-1]
                        else:
                            ema_coord = coord

                        kalman_trajectory[landmark].append(kalman_coord)
                        ema_trajectory[landmark].append(ema_coord)
                        raw_trajectory[landmark].append(coord)

                        """
                        trajectory[landmark].append(
                            get_3D_coordinates(
                                pose_results.pose_landmarks.landmark[landmark],
                                depth_frame,
                                w,
                                h,
                                intrinsics,
                            )
                        )
                        """
                    # else:
                    #     kalman_trajectory[landmark].append(
                    #         np.array([0, 0, 0])
                    #     )
                    #     raw_trajectory[landmark].append([0,0,0])
                    #     ema_trajectory[landmark].append([0,0,0])
            
            if len(kalman_trajectory[mp_pose.PoseLandmark.RIGHT_ELBOW]) > 0 and len(kalman_trajectory[mp_pose.PoseLandmark.RIGHT_WRIST]) > 0:
                # Get the last coordinates
            
                shoulder_pos = kf_shoulder.last_output
                elbow_pos = kf_elbow.last_output
                wrist_pos = kf_wrist.last_output

                # # Project elbow and wrist to fixed segment lengths (soft constraint)
                corrected_elbow = project_joint_to_length(shoulder_pos, elbow_pos, UPPER_ARM_LENGTH)
                corrected_wrist = project_joint_to_length(corrected_elbow, wrist_pos, FOREARM_LENGTH)

                # Overwrite Kalman estimates (soft-corrected)
                kf_elbow.last_output = corrected_elbow
                kf_wrist.last_output = corrected_wrist

                # Save corrected results to trajectory
                kalman_trajectory[mp_pose.PoseLandmark.RIGHT_ELBOW][-1] = corrected_elbow
                kalman_trajectory[mp_pose.PoseLandmark.RIGHT_WRIST][-1] = corrected_wrist

            cv2.imshow("RealSense Right Arm Lengths", color_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"I am a failure because : {e}")
    finally:
        print("Trajectory recording complete.")
        pipeline.stop()
        cv2.destroyAllWindows()
        return kalman_trajectory, ema_trajectory, raw_trajectory



def set_bounds(ax):
    ax.set_xlim(-MAX_REACH, MAX_REACH)
    ax.set_ylim(-MAX_REACH, MAX_REACH)
    ax.set_zlim(-MAX_REACH, MAX_REACH)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Robot Arm Animation")

def visualize_human_arm(trajectories, fps = 30):

    mp_pose = mp.solutions.pose

    #Setup figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    set_bounds(ax)
    # Get initial joint positions
    # Extract x, y, z coordinates. # 3 joints 
    x = np.zeros(2)
    y = np.zeros(2)
    z = np.zeros(2)

    # Plot robot arm segments
    #(line,) = ax.plot(x, y, z, "ko-", linewidth=3, markersize=6)
    (upperarm,) = ax.plot(x, y, z, "ro-", linewidth=3, markersize=6)
    (forearm,) = ax.plot(x, y, z, "bo-", linewidth=3, markersize=6)

    shoulder_label = ax.text(0, 0, 0, "R_shoulder", color="black", fontsize=6) # rotation=90)
    elbow_label = ax.text(0, 0, 0, "R_elbow", color="red", fontsize=6)
    wrist_label = ax.text(0, 0, 0, "R_wrist", color="blue", fontsize=6)
    ax.set_title(f"Frame Number: {0}", fontsize=14)

    # Animation function
    def update_animation(frame):
        # Calculate progress (0 to 1)
        #progress = frame / (duration * fps)
        ax.set_title(f"Frame Number: {frame}", fontsize=14)
        shoulder = trajectories[mp_pose.PoseLandmark.RIGHT_SHOULDER][frame]
        elbow = trajectories[mp_pose.PoseLandmark.RIGHT_ELBOW][frame]
        wrist = trajectories[mp_pose.PoseLandmark.RIGHT_WRIST][frame]
        elbow_ref_shoulder = elbow - shoulder
        wrist_ref_shoulder = wrist - shoulder

        x_upper = np.array([0, elbow_ref_shoulder[0]])
        y_upper = np.array([0, elbow_ref_shoulder[1]])
        z_upper = np.array([0, elbow_ref_shoulder[2]])

        x_fore = np.array([elbow_ref_shoulder[0], wrist_ref_shoulder[0]])
        y_fore = np.array([elbow_ref_shoulder[1], wrist_ref_shoulder[1]])
        z_fore = np.array([elbow_ref_shoulder[2], wrist_ref_shoulder[2]])

        forearm.set_data_3d(x_fore, y_fore, z_fore)
        upperarm.set_data_3d(x_upper, y_upper, z_upper)

        shoulder_label.set_position((0, 0))
        shoulder_label.set_3d_properties(0)
        
        elbow_label.set_position((x_upper[1], y_upper[1]))
        elbow_label.set_3d_properties(z_upper[1])
        
        wrist_label.set_position((x_fore[1], y_fore[1]))
        wrist_label.set_3d_properties(z_fore[1])

        return upperarm, forearm, shoulder_label, elbow_label, wrist_label


    ax.view_init(elev=10, azim=10, roll=0)  # z-up view, y across 

    # Create animation
    frames = len(trajectories[mp_pose.PoseLandmark.RIGHT_ELBOW])
    animation = FuncAnimation(
        fig=fig, func=update_animation, frames=frames, interval = 1000 / fps, blit=False
    )   

    plt.tight_layout()
    plt.show()
    return animation, fig

def plot_wrist_trajectories(kalman_trajectories, ema_trajectories, raw_trajectories):
    import matplotlib.pyplot as plt
    mp_pose = mp.solutions.pose
    wrist = mp_pose.PoseLandmark.RIGHT_WRIST

    # Convert the trajectory lists into numpy arrays for easier indexing
    kalman = np.array(kalman_trajectories[wrist])
    ema = np.array(ema_trajectories[wrist])
    raw = np.array(raw_trajectories[wrist])

    axis_labels = ['X', 'Y', 'Z']

    for i in range(3):  # x=0, y=1, z=2
        plt.figure(figsize=(10, 4))
        plt.plot(raw[:, i], label='Raw', linestyle='--', alpha=0.7)
        plt.plot(ema[:, i], label='EMA', linestyle='-.', alpha=0.8)
        plt.plot(kalman[:, i], label='Kalman', linewidth=2)
        plt.xlabel("Frame")
        plt.ylabel(f"{axis_labels[i]} Position (m)")
        plt.title(f"Right Wrist {axis_labels[i]} Position Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_wrist_trajectories_comparison(kalman_trajectories, ema_trajectories, raw_trajectories):
    mp_pose = mp.solutions.pose
    wrist = mp_pose.PoseLandmark.RIGHT_WRIST
    shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER
    kalman = np.array(kalman_trajectories[wrist]) - np.array(kalman_trajectories[shoulder])
    ema = np.array(ema_trajectories[wrist]) - np.array(ema_trajectories[shoulder])
    raw = np.array(raw_trajectories[wrist]) - np.array(raw_trajectories[shoulder])

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 8))
    axes = axes.flatten()  # Flatten for easy indexing

    #axes[0].axis('off')
    #axes[1].axis('off')
    #axes[2].axis('off')

    titles = ['X Position', 'Y Position', 'Z Position']
    components = [0, 1, 2]  # Index for x, y, z

    for comp, i in enumerate(components):
        axes[i].plot(raw[:, comp], label='Raw', linestyle='--', alpha=0.6)
        axes[i].plot(ema[:, comp], label='EMA', linestyle='-.', alpha=0.8)
        axes[i].plot(kalman[:, comp], label='Kalman', linewidth=2)
        axes[i].set_title(titles[comp])
        axes[i].set_xlabel('Frame')
        axes[i].set_ylabel('Position (m)')
        axes[i].legend()
        axes[i].grid(True)

    # Optional: Hide the 4th (unused) subplot
    #axes[3].axis('off')
    fig.suptitle("Right Wrist Position (in shoulder frame) Over Time", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    """Run right arm robot visualization demo witqh mediapipe"""
    # Run animated visualization
    kalman_trajectories, ema_trajectories, raw_trajectories = record_arm()
    animation, fig_anim = visualize_human_arm(kalman_trajectories)
    #plt.show()

    #plot_wrist_trajectories_comparison(kalman_trajectories, ema_trajectories, raw_trajectories)
    
    # Uncomment to save animation
    # animation.save('robot_animation.mp4', writer='ffmpeg', dpi=100)

def create_hand_claw_rawr():
    """
    Return claw geometry defined in local coordinates (origin at wrist).
    Could be a triangle, cross, or set of lines.
    """
    # Example: 3 fingers extending out from wrist
    points = np.array([
        [0.0, 0.0, 0.0],   # base (wrist)
        [0.05, 0.02, 0.0],  # finger 1
        [0.05, -0.02, 0.0], # finger 2
        [0.05, 0.0, 0.03],  # finger 3
    ])
    return points



if __name__ == "__main__":
    import mediapipe as mp
    MAX_REACH = 1 #m
    FRAME_CAP = 120
    FOREARM_LENGTH = 0.3
    UPPER_ARM_LENGTH = 0.3
    main()
