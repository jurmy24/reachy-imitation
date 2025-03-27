import time
from typing import Literal
from src.pipelines.Pipeline import Pipeline
from src.mapping.get_arm_lengths import get_arm_lengths
from src.mapping.map_to_robot_coordinates import get_scale_factors
from src.sensing.extract_3D_points import (
    calculate_arm_coordinates,
    get_head_coordinates,
)
import mediapipe as mp
import numpy as np
import cv2
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode


class Pipeline_one(Pipeline):
    """Approach 1: Uses robot arm model with IK

    Inherits from Pipeline with the following attributes:
    - self.reachy (ReachySDK)
    - mp_hands (mediapipe.solutions.hands)
    - mp_pose (mediapipe.solutions.pose)
    - hands (mediapipe.solutions.hands.Hands)
    - pose (mediapipe.solutions.pose.Pose)
    - pipeline (pyrealsense2.pipeline)
    - intrinsics (pyrealsense2.intrinsics)
    """

    def _watch_human(self):
        """
        Make self.reachy's head track a human's head using the Realsense camera.
        This continues until a key is pressed (typically 'q') to signal the human
        is in position and ready for the next step.
        """
        print("Starting head tracking. Press 'q' to continue when in position.")

        try:
            while True:
                # Get frames from RealSense camera
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                # Check if frames are valid
                if not color_frame or not depth_frame:
                    continue

                # Process the image
                color_image = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                h, w, _ = color_image.shape

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

                # NOTE: The first move of the head might be very quick since it may move far (see example code in their docs)
                # time.sleep(0.1)

        except Exception as e:
            print(f"Head tracking error: {e}")
            cv2.destroyAllWindows()
        finally:
            # TODO: Consider running this program in parallel with the arm tracking later
            if self.reachy:
                self.reachy.head.look_at(0.5, 0, 0, duration=1)
                self.reachy.turn_off_smoothly("head")

    def _demonstrate_stretching(self):
        print("Reachy, please demonstrate stretching your arms out.")

        # Define the stretched arm positions (arms extended forward)
        angled_position = {
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
        zero_arm_position  = {
            self.reachy.l_arm.l_shoulder_pitch: 0,
            self.reachy.l_arm.l_shoulder_roll: 0,
            self.reachy.l_arm.l_arm_yaw: 0,
            self.reachy.l_arm.l_elbow_pitch: 0,
            self.reachy.l_arm.l_forearm_yaw: 0,
            self.reachy.l_arm.l_wrist_pitch: 0,
            self.reachy.l_arm.l_wrist_roll: 0,
            self.reachy.r_arm.r_shoulder_pitch: 0,
            self.reachy.r_arm.r_shoulder_roll: 0,
            self.reachy.r_arm.r_arm_yaw: 0,
            self.reachy.r_arm.r_elbow_pitch: 0,
            self.reachy.r_arm.r_forearm_yaw: 0,
            self.reachy.r_arm.r_wrist_pitch: 0,
            self.reachy.r_arm.r_wrist_roll: 0,
        }
        goto(
            goal_positions=angled_position,
            duration=2.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )
        time.sleep(5.0)
        goto(
            goal_positions=zero_arm_position,
            duration=2.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )
        # Move both arms to the stretched position simultaneously
        # print("Stretching both arms simultaneously...")
        # # Combine both position dictionaries to move both arms at once
        # stretch_position_both = {**stretch_position_right, **stretch_position_left}
        # goto(
        #     goal_positions=stretch_position_both,
        #     duration=2.0,
        #     interpolation_mode=InterpolationMode.MINIMUM_JERK,
        # )

        # # Hold the position for a few seconds
        # print("Holding stretched position...")
        # time.sleep(3.0)

        self.reachy.turn_off_smoothly("r_arm")
        self.reachy.turn_off_smoothly("l_arm")

    def _calculate_scale_factors(self):
        # Initialize arrays to store arm length measurements
        forearm_lengths = np.array([])
        upper_arm_lengths = np.array([])

        try:
            print("Human, please stretch your arms out in front of you.")
            time.sleep(2.0)

            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                h, w, _ = color_image.shape

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
            print("Measurement complete. Returning to rest position.")
            self.cleanup()
            return hand_sf, elbow_sf

    def initiation_protocol(self):
        """Recognize the human in the frame and calculate the scale factors

        STEPS:
        1. self.reachy watches human entering the frame
        2. self.reachy's head is looking at the human until they get into position (we press a button on the keyboard to continue)
        3. self.reachy demonstrates stretching out arms in front of the human
        4. Human repeats the action and we calculate the scale factors
        """
        self.mp_draw = mp.solutions.drawing_utils

        # if self.reachy:
        #     # Step 1: Track the human's head until they're in position
        #     self._watch_human()

        #     # # Step 2: self.reachy demonstrates stretching out arms in front of the human
        #     self._demonstrate_stretching()

        # # Step 3: Human repeats the action and we calculate scale factors
        hand_sf, elbow_sf = self._calculate_scale_factors()

        return hand_sf, elbow_sf

    def process_frame(self, rgb_image, depth_frame, w, h, arm):
        """Process a single frame and extract arm coordinates

        Args:
            rgb_image: RGB image from the RealSense camera
            depth_frame: Depth frame from the RealSense camera
            w: Width of the image
            h: Height of the image
            arm: Which arm to track ("right", "left", or "both")

        Returns:
            Tuple of (right_arm_coordinates, left_arm_coordinates)
        """
        return calculate_arm_coordinates(
            self.pose,
            self.hands,
            self.mp_pose,
            self.mp_hands,
            self.intrinsics,
            rgb_image,
            depth_frame,
            w,
            h,
            arm,
        )

    def display_frame(
        self, arm, color_image, right_arm_coordinates, left_arm_coordinates
    ):
        # Display 3D coordinates on the image
        y_offset = 30

        # Display right arm coordinates
        for name, coord in right_arm_coordinates.items():
            x, y, z = coord
            cv2.putText(
                color_image,
                f"R_{name}: ({x:.2f}, {y:.2f}, {z:.2f})m",
                (50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),  # Green for right arm
                2,
            )
            y_offset += 20

        # Display left arm coordinates
        for name, coord in left_arm_coordinates.items():
            x, y, z = coord
            cv2.putText(
                color_image,
                f"L_{name}: ({x:.2f}, {y:.2f}, {z:.2f})m",
                (50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),  # Blue for left arm
                2,
            )
            y_offset += 20

        # Set window title based on which arm(s) is being tracked
        window_title = "RealSense "
        if arm == "right":
            window_title += "Right Arm"
        elif arm == "left":
            window_title += "Left Arm"
        elif arm == "both":
            window_title += "Both Arms"
        window_title += " 3D Coordinates"

        # Display the image
        cv2.imshow(window_title, color_image)

    def run(
        self, arm: Literal["right", "left", "both"] = "right", display: bool = True
    ):
        """Main processing loop - may be overridden by subclasses if needed"""
        self.initialize()
        try:
            while True:
                # Get frames from RealSense camera
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                # Check if frames are valid, if not, skip
                if not color_frame or not depth_frame:
                    continue

                # OpenCV uses BGR format, MediaPipe uses RGB format, so we convert the color image
                color_image = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                # Get height and width of the color image
                h, w, _ = color_image.shape
                right_arm_coordinates, left_arm_coordinates = self.process_frame(
                    rgb_image, depth_frame, w, h, arm
                )

                if display:
                    # TODO: This display can run in a different thread
                    self.display_frame(
                        arm, color_image, right_arm_coordinates, left_arm_coordinates
                    )
                # Quit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.cleanup()
