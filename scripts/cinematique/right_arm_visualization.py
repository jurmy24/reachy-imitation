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
    def __init__(self, threshold = 0.5):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        dt = 0.1  # Time step

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

        self.kf.R *= 0.1  # Measurement noise
        self.kf.P *= 1000.  # Initial covariance
        self.kf.Q *= 1e-4  # Process noise

        self.kf.x[:6] = np.zeros((6, 1))  # Initial state
        self.last_output = np.zeros(3)
        self.threshold = threshold  # meters

    # def update(self, measurement):
    #     self.kf.predict()
    #     self.kf.update(measurement)
    #     return self.kf.x[:3].reshape(-1)  # Return x, y, z
    
    def update(self, measurement):
        measurement = np.array(measurement)

        # Reject if too far from last prediction
        if np.linalg.norm(measurement - self.last_output) > self.threshold:
            # Skip update
            return self.last_output

        self.kf.predict()
        self.kf.update(measurement)
        self.last_output = self.kf.x[:3].reshape(-1)
        return self.last_output


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

    trajectory = {}
    trajectory[mp_pose.PoseLandmark.RIGHT_ELBOW] = []
    trajectory[mp_pose.PoseLandmark.RIGHT_SHOULDER] = []
    trajectory[mp_pose.PoseLandmark.RIGHT_WRIST] = []
    
    # Initialize Kalman filters for each joint
    kf_shoulder = KalmanFilter3D()
    kf_elbow = KalmanFilter3D()
    kf_wrist = KalmanFilter3D()
    its = 0
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
                    if (0.3 < pose_results.pose_landmarks.landmark[landmark].visibility <= 1):

                        coord = get_3D_coordinates(
                            pose_results.pose_landmarks.landmark[landmark],
                            depth_frame,
                            w,
                            h,
                            intrinsics,
                        )

                        # Apply the corresponding Kalman filter
                        if landmark == mp_pose.PoseLandmark.RIGHT_SHOULDER:
                            coord = kf_shoulder.update(coord)
                        elif landmark == mp_pose.PoseLandmark.RIGHT_ELBOW:
                            coord = kf_elbow.update(coord)
                        elif landmark == mp_pose.PoseLandmark.RIGHT_WRIST:
                            coord = kf_wrist.update(coord)

                        trajectory[landmark].append(coord)
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
                    else:
                        trajectory[landmark].append(
                            np.array([0, 0, 0])
                        )

            cv2.imshow("RealSense Right Arm Lengths", color_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"I am a failure because : {e}")
    finally:
        print("Trajectory recording complete.")
        pipeline.stop()
        cv2.destroyAllWindows()
        return trajectory



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
    return animation, fig


def main():
    """Run right arm robot visualization demo with mediapipe"""
    # Run animated visualization
    trajectories = record_arm()
    animation, fig_anim = visualize_human_arm(trajectories)
    # Uncomment to save animation
    # animation.save('robot_animation.mp4', writer='ffmpeg', dpi=100)
    plt.show()


if __name__ == "__main__":
    import mediapipe as mp
    MAX_REACH = 1 #m
    FRAME_CAP = 400
    main()
