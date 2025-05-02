from abc import ABC, abstractmethod
from typing import Literal
import mediapipe as mp
import pyrealsense2 as rs
import cv2
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import numpy as np
import time

from config.CONSTANTS import get_ordered_joint_names, get_zero_pos
from src.mapping.get_arm_lengths import get_arm_lengths
from src.mapping.map_to_robot_coordinates import get_scale_factors
from src.sensing.extract_3D_points import get_head_coordinates
from config.CONSTANTS import get_zero_pos


class Pipeline(ABC):
    """Base class for all imitation approaches"""

    def __init__(self, reachy: ReachySDK = None):
        # Load configs, initialize components (lightweight)
        self.reachy = reachy
        self.mp_hands = None
        self.mp_pose = None
        self.hands = None
        self.pose = None
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.mp_draw = None
        # scale factor for shoulder to hand length ratio between robot and human (i.e. robot/human)
        self.hand_sf = None
        # scale factor for shoulder to elbow length ratio between robot and human (i.e. robot/human)
        self.elbow_sf = None
        self.zero_arm_position = get_zero_pos(self.reachy)

        self.initialize()

    def initialize(self):
        """Setup components required for this imitation approach"""
        # Initialize MediaPipe for hand and body point map detection
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

        # Create model instances with optimized parameters
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            smooth_landmarks=True,
        )

        # Configure intel RealSense camera (color and depth streams)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)

        # Align depth frame to color frame
        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(config)

        # Cache intrinsics for repeated use
        self.intrinsics = (
            profile.get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        self.mp_draw = mp.solutions.drawing_utils

        # NOTE: This turns on the entire Reachy robot (i.e. head and both arms on stiff mode)
        if self.reachy is not None:
            self.reachy.turn_on("reachy")

    def _watch_human(self):
        """
        Make self.reachy's head track a human's head using the Realsense camera.
        """
        print("Starting head tracking. Press 'q' to continue when in position.")

        try:
            while True:
                # Get frames from RealSense camera
                camera_data = self._get_camera_data()
                if camera_data is None:
                    continue

                color_frame, depth_frame, color_image, rgb_image, h, w = camera_data

                # Get the head coordinates
                head_position = get_head_coordinates(
                    self.pose,
                    self.mp_pose,
                    self.intrinsics,
                    rgb_image,
                    depth_frame,
                    w,
                    h,
                )

                # If head is detected, make self.reachy look at it
                if head_position is not None and not np.isnan(head_position).any():
                    x, y, z = head_position
                    # Display head position
                    cv2.putText(
                        color_image,
                        f"Head: ({x:.2f}, {y:.2f}, {z:.2f})m",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    # Make self.reachy look at the person, gracefully handle unreachable positions
                    try:
                        self.reachy.head.look_at(x=x, y=y, z=z, duration=0.2)
                    except Exception as e:
                        # Log the error but continue tracking
                        print(
                            f"Warning: Could not track to position ({x:.2f}, {y:.2f}, {z:.2f}): {e}"
                        )
                        # Optional: add visual feedback about unreachable position
                        cv2.putText(
                            color_image,
                            "Position unreachable",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),  # Red color for warning
                            2,
                        )

                # Draw pose landmarks if available
                pose_results = self.pose.process(rgb_image)

                if pose_results.pose_landmarks:
                    self.mp_draw.draw_landmarks(
                        color_image,
                        pose_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                    )

                # Display the image
                cv2.imshow("Head Tracking", color_image)

                # Check for key press to exit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyWindow("Head Tracking")
                    break

        except Exception as e:
            print(f"Head tracking error: {e}")
        finally:
            cv2.destroyAllWindows()
            if self.reachy:
                self.reachy.head.look_at(0.5, 0, 0, duration=1)

    def _demonstrate_stretching(self):
        print("Initiate demonstration of strong man pose")

        # Define the strong man pose
        strong_man_pose = {
            self.reachy.l_arm.l_shoulder_pitch: 0,
            self.reachy.l_arm.l_shoulder_roll: 90,
            self.reachy.l_arm.l_arm_yaw: 90,
            self.reachy.l_arm.l_elbow_pitch: -90,
            self.reachy.l_arm.l_forearm_yaw: 0,
            self.reachy.l_arm.l_wrist_pitch: 0,
            self.reachy.l_arm.l_wrist_roll: 0,
            self.reachy.r_arm.r_shoulder_pitch: 0,
            self.reachy.r_arm.r_shoulder_roll: -90,
            self.reachy.r_arm.r_arm_yaw: -90,
            self.reachy.r_arm.r_elbow_pitch: -90,
            self.reachy.r_arm.r_forearm_yaw: 0,
            self.reachy.r_arm.r_wrist_pitch: 0,
            self.reachy.r_arm.r_wrist_roll: 0,
        }
        zero_arm_position = get_zero_pos(self.reachy)

        try:
            goto(
                goal_positions=strong_man_pose,
                duration=2.0,
                interpolation_mode=InterpolationMode.MINIMUM_JERK,
            )
            time.sleep(5.0)
            goto(
                goal_positions=zero_arm_position,
                duration=2.0,
                interpolation_mode=InterpolationMode.MINIMUM_JERK,
            )
        except Exception as e:
            print(f"Failed to demonstrate stretching: {e}")

    def _calculate_scale_factors(self):
        """
        Calculate the scale factors for the robot's arm length to human arm length.
        """
        forearm_lengths = np.array([])
        upper_arm_lengths = np.array([])

        try:
            time.sleep(2.0)

            while True:
                # Get frames from RealSense camera
                camera_data = self._get_camera_data()
                if camera_data is None:
                    continue

                color_frame, depth_frame, color_image, rgb_image, h, w = camera_data

                pose_results = self.pose.process(rgb_image)
                if pose_results.pose_landmarks:

                    self.mp_draw.draw_landmarks(
                        color_image,
                        pose_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                    )

                    forearm_length, upper_arm_length = get_arm_lengths(
                        pose_results.pose_landmarks,
                        self.mp_pose,
                        depth_frame,
                        w,
                        h,
                        self.intrinsics,
                    )
                    if forearm_length is not None and upper_arm_length is not None:
                        forearm_lengths = np.append(forearm_lengths, forearm_length)
                        upper_arm_lengths = np.append(
                            upper_arm_lengths, upper_arm_length
                        )
                        if len(forearm_lengths) > 100:
                            forearm_length = np.median(forearm_lengths)
                            upper_arm_length = np.median(upper_arm_lengths)
                            hand_sf, elbow_sf = get_scale_factors(
                                forearm_length, upper_arm_length
                            )
                            break
                        cv2.putText(
                            color_image,
                            f"Forearm length: {forearm_length:.2f} m",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        cv2.putText(
                            color_image,
                            f"Lower arm length: {upper_arm_length:.2f} m",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                cv2.imshow("RealSense Right Arm Lengths", color_image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            print(f"Failed to run the recognize human pipeline: {e}")
        finally:
            print("Measurement complete.")
            cv2.destroyAllWindows()

            self.hand_sf = hand_sf
            self.elbow_sf = elbow_sf
            return hand_sf, elbow_sf

    def initiation_protocol(self):
        """Recognize the human in the frame and calculate the scale factors

        STEPS:
        1. reachy watches human entering the frame
        2. we press q to continue once the human is in position
        3. reachy demonstrates stretching out arms in front of the human
        4. human repeats the action and we calculate the scale factors
        """
        self._watch_human()
        self._demonstrate_stretching()
        self._calculate_scale_factors()

    @abstractmethod
    def shadow(
        self, arm: Literal["right", "left", "both"] = "right", display: bool = True
    ):
        pass

    def cleanup(self):
        """Clean up resources - subclasses can override if needed"""
        self.pipeline.stop()
        cv2.destroyAllWindows()

        if self.reachy is not None:
            try:
                goto(
                    goal_positions=self.zero_arm_position,
                    duration=2.0,
                    interpolation_mode=InterpolationMode.MINIMUM_JERK,
                )
                self.reachy.head.look_at(0.5, 0, 0, duration=2.0)
            finally:
                self.reachy.turn_off_smoothly("r_arm")
                self.reachy.turn_off_smoothly("l_arm")
                self.reachy.turn_off_smoothly("head")
        else:
            print("What the fudge")
