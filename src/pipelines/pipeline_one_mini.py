import time
from typing import Literal
from src.models.ik_mini import (
    scale_point,
    transform_to_shoulder_origin,
    translate_to_reachy_origin,
    within_reachys_reach,
)
from src.pipelines.Pipeline import Pipeline
from src.mapping.get_arm_lengths import get_arm_lengths
from src.mapping.map_to_robot_coordinates import get_scale_factors
from src.sensing.extract_3D_points import get_head_coordinates
import numpy as np
import cv2
from reachy_sdk.trajectory import goto, goto_async
from reachy_sdk.trajectory.interpolation import InterpolationMode
from config.CONSTANTS import get_zero_pos
from src.utils.three_d import get_3D_coordinates, get_3D_coordinates_of_hand
import asyncio


class Pipeline_one_mini(Pipeline):
    """Approach 1: Uses robot arm model with IK

    Inherits from Pipeline with the following attributes:
    - self.reachy (ReachySDK)
    - mp_hands (mediapipe.solutions.hands)
    - mp_pose (mediapipe.solutions.pose)
    - hands (mediapipe.solutions.hands.Hands)
    - pose (mediapipe.solutions.pose.Pose)
    - pipeline (pyrealsense2.pipeline)
    - intrinsics (pyrealsense2.intrinsics)
    # Plus a few more we just added
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

        except Exception as e:
            print(f"Head tracking error: {e}")
        finally:
            cv2.destroyAllWindows()
            # TODO: Consider running this program in parallel with the arm tracking later
            if self.reachy:
                self.reachy.head.look_at(0.5, 0, 0, duration=1)

    def _demonstrate_stretching(self):
        print("Reachy, please demonstrate stretching your arms out.")

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
            print("Measurement complete.")
            self.hand_sf = hand_sf
            self.elbow_sf = elbow_sf
            return hand_sf, elbow_sf

    def initiation_protocol(self):
        """Recognize the human in the frame and calculate the scale factors

        STEPS:
        1. self.reachy watches human entering the frame
        2. self.reachy's head is looking at the human until they get into position (we press a button on the keyboard to continue)
        3. self.reachy demonstrates stretching out arms in front of the human
        4. Human repeats the action and we calculate the scale factors
        """

        if self.reachy:
            # Step 1: Track the human's head until they're in position
            self._watch_human()

            # Step 2: self.reachy demonstrates stretching out arms in front of the human
            self._demonstrate_stretching()

        # Step 3: Human repeats the action and we calculate scale factors
        self._calculate_scale_factors()

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

    async def process_frame(
        self,
        rgb_image,
        color_image,
        depth_frame,
        w,
        h,
        prev_right_joint_positions,
        prev_left_joint_positions,
        arm,
    ):
        """Process a single frame and extract arm coordinates

        Args:
            rgb_image: RGB image from the RealSense camera
            color_image: Color image for display
            depth_frame: Depth frame from the RealSense camera
            w: Width of the image
            h: Height of the image
            prev_right_joint_positions: Previous joint positions for right arm (used as seed for IK)
            prev_left_joint_positions: Previous joint positions for left arm (used as seed for IK)
            arm: Which arm to track ("right", "left", or "both")

        Returns:
            Tuple of (right_arm_coordinates, left_arm_coordinates, reachy_joint_vector, updated_right_joint_positions, updated_left_joint_positions)
        """
        pass

    async def shadow(
        self, arm: Literal["right", "left", "both"] = "right", display: bool = True
    ):
        """
        Control Reachy to shadow human arm movements in real-time.

        Args:
            arm: Which arm to track ("right", "left", or "both")
            display: Whether to display the video window with augmented visualization
        """
        try:
            # Initialize previous joint positions with None
            # These will be populated on the first successful frame processing
            prev_reachy_hand_right = self.reachy.r_arm.forward_kinematics()[0:3, 3]
            prev_reachy_hand_left = self.reachy.l_arm.forward_kinematics()[0:3, 3]
            i = 0
            goto_new_position = False
            pause_update = True

            while True:
                i += 1
                if i == 10:
                    pause_update = False
                    i = 0
                else:
                    pause_update = True

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

                # Process the frame to get arm coordinates and joint positions
                right_arm_coordinates = {}
                left_arm_coordinates = {}
                reachy_joint_vector = {}

                pose_results = self.pose.process(rgb_image)

                if not pose_results.pose_landmarks:
                    print("No pose landmarks found")
                    continue

                # Get landmarks
                landmarks = pose_results.pose_landmarks.landmark
                self.mp_draw.draw_landmarks(
                    color_image,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                )

                # Initialize joint position dictionaries for both arms
                right_joint_dict = {}
                left_joint_dict = {}

                # Define arms to process based on the 'arm' parameter
                arms_to_process = []
                if arm == "right" or arm == "both":
                    arms_to_process.append("right")
                if arm == "left" or arm == "both":
                    arms_to_process.append("left")

                for current_arm in arms_to_process:
                    # Select landmarks based on arm
                    if current_arm == "right":
                        shoulder_landmark = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
                        index_landmark = self.mp_pose.PoseLandmark.RIGHT_INDEX
                        pinky_landmark = self.mp_pose.PoseLandmark.RIGHT_PINKY
                        arm_coordinates = right_arm_coordinates
                    else:  # left arm
                        shoulder_landmark = self.mp_pose.PoseLandmark.LEFT_SHOULDER
                        index_landmark = self.mp_pose.PoseLandmark.LEFT_INDEX
                        pinky_landmark = self.mp_pose.PoseLandmark.LEFT_PINKY
                        arm_coordinates = left_arm_coordinates

                    # Get 3D coordinates (shared steps for both arms)
                    shoulder = get_3D_coordinates(
                        landmarks[shoulder_landmark],
                        depth_frame,
                        w,
                        h,
                        self.intrinsics,  # Todo: double check if intrinsics should be calculated each time
                    )

                    # Get hand coordinates by averaging relevant finger landmarks
                    hand = get_3D_coordinates_of_hand(
                        landmarks[index_landmark],
                        landmarks[pinky_landmark],
                        depth_frame,
                        w,
                        h,
                        self.intrinsics,
                    )

                    # Transform coordinates
                    hand_rel_shoulder = transform_to_shoulder_origin(hand, shoulder)
                    scaled_hand = scale_point(self.hand_sf, hand_rel_shoulder)
                    reachy_hand = translate_to_reachy_origin(scaled_hand, current_arm)

                    # Check if the hand is within reachy's reach (from the shoulder origin)
                    # if not within_reachys_reach(scaled_hand):
                    #     print(
                    #         f"WARNING: Reachy can't reach that position {scaled_hand}"
                    #     )
                    #     continue

                    # # Store coordinates for display
                    # arm_coordinates["shoulder"] = shoulder
                    # arm_coordinates["hand"] = hand
                    # arm_coordinates["hand_rel_shoulder"] = hand_rel_shoulder
                    # arm_coordinates["reachy_hand"] = reachy_hand

                    # Update the robot if needed
                    if self.reachy:
                        # Get current arm kinematics
                        if current_arm == "right":
                            a = self.reachy.r_arm.forward_kinematics()
                            if not (
                                np.allclose(
                                    prev_reachy_hand_right, reachy_hand, atol=0.05
                                )
                            ):
                                prev_reachy_hand_right = reachy_hand
                                a[0, 3] = reachy_hand[0]
                                a[1, 3] = reachy_hand[1]
                                a[2, 3] = reachy_hand[2]
                                # goto_new_position = True
                            # else:
                            # goto_new_position = False

                            joint_pos = self.reachy.r_arm.inverse_kinematics(a)

                            # Store joint positions in right arm dictionary
                            for joint, pos in zip(
                                self.ordered_joint_names_right, joint_pos
                            ):
                                right_joint_dict[joint] = pos

                        else:  # left arm
                            a = self.reachy.l_arm.forward_kinematics()
                            if not (
                                np.allclose(
                                    prev_reachy_hand_left, reachy_hand, atol=0.05
                                )
                            ):
                                prev_reachy_hand_left = reachy_hand
                                a[0, 3] = reachy_hand[0]
                                a[1, 3] = reachy_hand[1]
                                a[2, 3] = reachy_hand[2]
                                # goto_new_position = True
                            # else:
                            # goto_new_position = False

                            joint_pos = self.reachy.l_arm.inverse_kinematics(a)

                            # Store joint positions in left arm dictionary
                            for joint, pos in zip(
                                self.ordered_joint_names_left, joint_pos
                            ):
                                left_joint_dict[joint] = pos

                # Combine joint dictionaries for both arms
                combined_joint_dict = {**right_joint_dict, **left_joint_dict}

                if not pause_update and combined_joint_dict:
                    await goto_async(
                        combined_joint_dict,
                        duration=1,
                        interpolation_mode=InterpolationMode.MINIMUM_JERK,
                    )
                    # Alternatively, use shorter duration and don't await completion
                    # duration = 0.1  # 100ms movements
                    # asyncio.create_task(goto_async(
                    #     combined_joint_dict, 
                    #     duration=duration,
                    #     sampling_freq=100,  # Higher sampling frequency
                    #     interpolation_mode=InterpolationMode.MINIMUM_JERK
                    # ))
                if display:
                    # Display the tracking information
                    self.display_frame(
                        arm, color_image, right_arm_coordinates, left_arm_coordinates
                    )

                # Quit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            print(f"Failed to run the shadow pipeline: {e}")
