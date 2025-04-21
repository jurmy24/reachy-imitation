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
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
from config.CONSTANTS import get_zero_pos
from src.utils.three_d import get_3D_coordinates, get_3D_coordinates_of_hand
from src.reachy.utils import setup_torque_limits, get_joint_positions
import asyncio


class Pipeline_one_mini(Pipeline):
    """Approach 1: Uses robot arm model with IK

    Inherits from Pipeline:
        self.reachy = reachy
        self.mp_hands = None
        self.mp_pose = None
        self.hands = None
        self.pose = None
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.mp_draw = None
        self.hand_sf = None  # scale factor for shoulder to hand length ratio between robot and human (i.e. robot/human)
        self.elbow_sf = None  # scale factor for shoulder to elbow length ratio between robot and human (i.e. robot/human)
        self.zero_arm_position = get_zero_pos(self.reachy)
        self.ordered_joint_names_right = get_ordered_joint_names(self.reachy, "right")
        self.ordered_joint_names_left = get_ordered_joint_names(self.reachy, "left")
    """

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

    def display_frame(
        self,
        arm,
        color_image,
        right_arm_coordinates,
        left_arm_coordinates,
        pose_landmarks=None,
    ):
        # Display landmarks on the image
        if pose_landmarks:
            self.mp_draw.draw_landmarks(
                color_image,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )

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

    def _get_camera_data(self):
        """Get frames from RealSense camera with validity check.

        Returns:
            tuple: (color_frame, depth_frame, color_image, rgb_image, h, w) if valid
                   None if frames are invalid
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # Check if frames are valid
        if not color_frame or not depth_frame:
            return None

        # Process the color image
        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        h, w, _ = color_image.shape

        return color_frame, depth_frame, color_image, rgb_image, h, w

    async def shadow(
        self, arm: Literal["right", "left", "both"] = "right", display: bool = True
    ):
        """
        Control Reachy to shadow human arm movements in real-time.

        Args:
            arm: Which arm to track ("right", "left", or "both")
            display: Whether to display the video window in the computer's monitor
        """
        ############### Parameters ###############
        smoothing_buffer_size = 5
        right_position_history = []
        left_position_history = []
        position_alpha = 0.4  # For EMA position smoothing
        movement_interval = 0.03  # Send commands at ~30Hz
        ########################################

        ############### FLAGS ##################
        cleanup_requested = False
        ########################################

        ############### VARIABLES ##############
        right_joint_dict = {}
        left_joint_dict = {}
        ########################################

        try:
            # Set torque limits for all motor joints for safety
            setup_torque_limits(self.reachy, 80.0, arm)

            # Get initial coordinates and joint positions for the specified arm(s)
            prev_reachy_hand_right = (
                self.reachy.r_arm.forward_kinematics()[0:3, 3]
                if arm in ["right", "both"]
                else None
            )
            prev_reachy_hand_left = (
                self.reachy.l_arm.forward_kinematics()[0:3, 3]
                if arm in ["left", "both"]
                else None
            )
            prev_right_joint_pos, prev_left_joint_pos = get_joint_positions(
                self.reachy, arm
            )

            # For frame rate and movement control
            last_movement_time_right = time.time()
            last_movement_time_left = time.time()

            print("Starting shadowing. Press 'q' to exit safely.")

            while not cleanup_requested:
                loop_start_time = time.time()

                # 1. get data from RealSense camera
                camera_data = self._get_camera_data()
                if camera_data is None:
                    await asyncio.sleep(0.01)
                    continue
                color_frame, depth_frame, color_image, rgb_image, h, w = camera_data

                # Coordinate storage for visualization
                right_arm_coordinates = {}
                left_arm_coordinates = {}

                # 2. get pose landmarks from the image using mediapipe
                pose_results = self.pose.process(rgb_image)
                if not pose_results.pose_landmarks:
                    await asyncio.sleep(0.01)
                    continue
                landmarks = pose_results.pose_landmarks.landmark

                # 3. Process the arm(s)
                arms_to_process = ["right", "left"] if arm == "both" else [arm]
                for current_arm in arms_to_process:
                    if current_arm == "right":
                        shoulder_landmark = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
                        index_landmark = self.mp_pose.PoseLandmark.RIGHT_INDEX
                        pinky_landmark = self.mp_pose.PoseLandmark.RIGHT_PINKY
                        arm_coordinates = right_arm_coordinates
                    else:
                        shoulder_landmark = self.mp_pose.PoseLandmark.LEFT_SHOULDER
                        index_landmark = self.mp_pose.PoseLandmark.LEFT_INDEX
                        pinky_landmark = self.mp_pose.PoseLandmark.LEFT_PINKY
                        arm_coordinates = left_arm_coordinates

                    # Get 3D coordinates
                    shoulder = get_3D_coordinates(
                        landmarks[shoulder_landmark],
                        depth_frame,
                        w,
                        h,
                        self.intrinsics,
                    )
                    hand = get_3D_coordinates_of_hand(
                        landmarks[index_landmark],
                        landmarks[pinky_landmark],
                        depth_frame,
                        w,
                        h,
                        self.intrinsics,
                    )

                    # Skip if any coordinate is invalid
                    if shoulder is None or hand is None:
                        await asyncio.sleep(0.01)  # Short sleep to prevent CPU hogging
                        continue

                    # Transform coordinates to robot space
                    hand_rel_shoulder = transform_to_shoulder_origin(hand, shoulder)
                    scaled_hand = scale_point(self.hand_sf, hand_rel_shoulder)
                    reachy_hand = translate_to_reachy_origin(scaled_hand, current_arm)

                    # Store coordinates for display
                    arm_coordinates["shoulder"] = shoulder
                    arm_coordinates["hand"] = hand
                    arm_coordinates["hand_rel_shoulder"] = hand_rel_shoulder
                    arm_coordinates["reachy_hand"] = reachy_hand

                    # # Check if the hand is within reachy's reach
                    # if not within_reachys_reach(scaled_hand):
                    #     print(f"WARNING: Position {scaled_hand} is beyond reach")
                    #     continue

                    # Update robot control based on arm
                    if current_arm == "right":
                        # Apply position smoothing
                        right_position_history.append(reachy_hand)
                        if len(right_position_history) > smoothing_buffer_size:
                            right_position_history.pop(0)

                        # Compute EMA for smoother position
                        smoothed_position = (
                            position_alpha * reachy_hand
                            + (1 - position_alpha) * prev_reachy_hand_right
                        )

                        # Check if the desired position is different to where it's currently at
                        a = self.reachy.r_arm.forward_kinematics()
                        current_pos = np.array([a[0, 3], a[1, 3], a[2, 3]])
                        if np.allclose(current_pos, smoothed_position, atol=0.03):
                            right_already_there = True
                        else:
                            right_already_there = False

                        # Only update if position changed significantly from the previous count
                        if (
                            not np.allclose(
                                prev_reachy_hand_right, smoothed_position, atol=0.02
                            )
                            and not right_already_there
                        ):
                            # Update previous position
                            prev_reachy_hand_right = smoothed_position

                            # Compute IK
                            try:
                                # a = self.reachy.r_arm.forward_kinematics()
                                a[0, 3] = smoothed_position[0]
                                a[1, 3] = smoothed_position[1]
                                a[2, 3] = smoothed_position[2]
                                joint_pos = self.reachy.r_arm.inverse_kinematics(a)

                                # Directly map joint positions instead of using loops
                                # The order of joint_pos corresponds to: shoulder_pitch, shoulder_roll, arm_yaw,
                                # elbow_pitch, forearm_yaw, wrist_pitch, wrist_roll, gripper

                                # Set right arm joint positions directly without loops
                                r_shoulder_pitch_pos = joint_pos[0]
                                r_shoulder_roll_pos = joint_pos[1]
                                r_arm_yaw_pos = joint_pos[2]
                                r_elbow_pitch_pos = joint_pos[3]
                                r_forearm_yaw_pos = joint_pos[4]
                                r_wrist_pitch_pos = joint_pos[5]
                                r_wrist_roll_pos = joint_pos[6]
                                r_gripper_pos = (
                                    joint_pos[7]
                                    if len(joint_pos) > 7
                                    else prev_right_joint_pos.get("r_gripper", 0)
                                )

                                # Apply velocity limiting individually for each joint
                                # Calculate maximum allowed change per update
                                max_change = 3.0  # degrees per update

                                # Shoulder pitch
                                current_pos = prev_right_joint_pos.get(
                                    "r_shoulder_pitch", r_shoulder_pitch_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    r_shoulder_pitch_pos - current_pos,
                                    -max_change,
                                    max_change,
                                )
                                right_joint_dict["r_shoulder_pitch"] = limited_pos
                                prev_right_joint_pos["r_shoulder_pitch"] = limited_pos

                                # Shoulder roll
                                current_pos = prev_right_joint_pos.get(
                                    "r_shoulder_roll", r_shoulder_roll_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    r_shoulder_roll_pos - current_pos,
                                    -max_change,
                                    max_change,
                                )
                                right_joint_dict["r_shoulder_roll"] = limited_pos
                                prev_right_joint_pos["r_shoulder_roll"] = limited_pos

                                # Arm yaw
                                current_pos = prev_right_joint_pos.get(
                                    "r_arm_yaw", r_arm_yaw_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    r_arm_yaw_pos - current_pos, -max_change, max_change
                                )
                                right_joint_dict["r_arm_yaw"] = limited_pos
                                prev_right_joint_pos["r_arm_yaw"] = limited_pos

                                # Elbow pitch
                                current_pos = prev_right_joint_pos.get(
                                    "r_elbow_pitch", r_elbow_pitch_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    r_elbow_pitch_pos - current_pos,
                                    -max_change,
                                    max_change,
                                )
                                right_joint_dict["r_elbow_pitch"] = limited_pos
                                prev_right_joint_pos["r_elbow_pitch"] = limited_pos

                                # Forearm yaw
                                current_pos = prev_right_joint_pos.get(
                                    "r_forearm_yaw", r_forearm_yaw_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    r_forearm_yaw_pos - current_pos,
                                    -max_change,
                                    max_change,
                                )
                                right_joint_dict["r_forearm_yaw"] = limited_pos
                                prev_right_joint_pos["r_forearm_yaw"] = limited_pos

                                # Wrist pitch
                                current_pos = prev_right_joint_pos.get(
                                    "r_wrist_pitch", r_wrist_pitch_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    r_wrist_pitch_pos - current_pos,
                                    -max_change,
                                    max_change,
                                )
                                right_joint_dict["r_wrist_pitch"] = limited_pos
                                prev_right_joint_pos["r_wrist_pitch"] = limited_pos

                                # Wrist roll
                                current_pos = prev_right_joint_pos.get(
                                    "r_wrist_roll", r_wrist_roll_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    r_wrist_roll_pos - current_pos,
                                    -max_change,
                                    max_change,
                                )
                                right_joint_dict["r_wrist_roll"] = limited_pos
                                prev_right_joint_pos["r_wrist_roll"] = limited_pos

                                # Gripper
                                current_pos = prev_right_joint_pos.get(
                                    "r_gripper", r_gripper_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    r_gripper_pos - current_pos, -max_change, max_change
                                )
                                right_joint_dict["r_gripper"] = limited_pos
                                prev_right_joint_pos["r_gripper"] = limited_pos

                            except Exception as e:
                                print(f"Right arm IK calculation error: {e}")
                    else:  # left arm
                        # Apply position smoothing
                        left_position_history.append(reachy_hand)
                        if len(left_position_history) > smoothing_buffer_size:
                            left_position_history.pop(0)

                        # Compute EMA for smoother position
                        smoothed_position = (
                            position_alpha * reachy_hand
                            + (1 - position_alpha) * prev_reachy_hand_left
                        )

                        # Check if the desired position is different to where it's currently at
                        a = self.reachy.r_arm.forward_kinematics()
                        current_pos = np.array([a[0, 3], a[1, 3], a[2, 3]])
                        if np.allclose(current_pos, smoothed_position, atol=0.03):
                            left_already_there = True
                        else:
                            left_already_there = False

                        # Only update if position changed significantly
                        if (
                            not np.allclose(
                                prev_reachy_hand_left, smoothed_position, atol=0.02
                            )
                            and not left_already_there
                        ):
                            # Update previous position
                            prev_reachy_hand_left = smoothed_position

                            # Compute IK
                            try:
                                # a = self.reachy.l_arm.forward_kinematics()
                                a[0, 3] = smoothed_position[0]
                                a[1, 3] = smoothed_position[1]
                                a[2, 3] = smoothed_position[2]
                                joint_pos = self.reachy.l_arm.inverse_kinematics(a)

                                # Directly map joint positions instead of using loops
                                # The order of joint_pos corresponds to: shoulder_pitch, shoulder_roll, arm_yaw,
                                # elbow_pitch, forearm_yaw, wrist_pitch, wrist_roll, gripper

                                # Set left arm joint positions directly without loops
                                l_shoulder_pitch_pos = joint_pos[0]
                                l_shoulder_roll_pos = joint_pos[1]
                                l_arm_yaw_pos = joint_pos[2]
                                l_elbow_pitch_pos = joint_pos[3]
                                l_forearm_yaw_pos = joint_pos[4]
                                l_wrist_pitch_pos = joint_pos[5]
                                l_wrist_roll_pos = joint_pos[6]
                                l_gripper_pos = (
                                    joint_pos[7]
                                    if len(joint_pos) > 7
                                    else prev_left_joint_pos.get("l_gripper", 0)
                                )

                                # Apply velocity limiting individually for each joint
                                # Calculate maximum allowed change per update
                                max_change = 3.0  # degrees per update

                                # Shoulder pitch
                                current_pos = prev_left_joint_pos.get(
                                    "l_shoulder_pitch", l_shoulder_pitch_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    l_shoulder_pitch_pos - current_pos,
                                    -max_change,
                                    max_change,
                                )
                                left_joint_dict["l_shoulder_pitch"] = limited_pos
                                prev_left_joint_pos["l_shoulder_pitch"] = limited_pos

                                # Shoulder roll
                                current_pos = prev_left_joint_pos.get(
                                    "l_shoulder_roll", l_shoulder_roll_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    l_shoulder_roll_pos - current_pos,
                                    -max_change,
                                    max_change,
                                )
                                left_joint_dict["l_shoulder_roll"] = limited_pos
                                prev_left_joint_pos["l_shoulder_roll"] = limited_pos

                                # Arm yaw
                                current_pos = prev_left_joint_pos.get(
                                    "l_arm_yaw", l_arm_yaw_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    l_arm_yaw_pos - current_pos, -max_change, max_change
                                )
                                left_joint_dict["l_arm_yaw"] = limited_pos
                                prev_left_joint_pos["l_arm_yaw"] = limited_pos

                                # Elbow pitch
                                current_pos = prev_left_joint_pos.get(
                                    "l_elbow_pitch", l_elbow_pitch_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    l_elbow_pitch_pos - current_pos,
                                    -max_change,
                                    max_change,
                                )
                                left_joint_dict["l_elbow_pitch"] = limited_pos
                                prev_left_joint_pos["l_elbow_pitch"] = limited_pos

                                # Forearm yaw
                                current_pos = prev_left_joint_pos.get(
                                    "l_forearm_yaw", l_forearm_yaw_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    l_forearm_yaw_pos - current_pos,
                                    -max_change,
                                    max_change,
                                )
                                left_joint_dict["l_forearm_yaw"] = limited_pos
                                prev_left_joint_pos["l_forearm_yaw"] = limited_pos

                                # Wrist pitch
                                current_pos = prev_left_joint_pos.get(
                                    "l_wrist_pitch", l_wrist_pitch_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    l_wrist_pitch_pos - current_pos,
                                    -max_change,
                                    max_change,
                                )
                                left_joint_dict["l_wrist_pitch"] = limited_pos
                                prev_left_joint_pos["l_wrist_pitch"] = limited_pos

                                # Wrist roll
                                current_pos = prev_left_joint_pos.get(
                                    "l_wrist_roll", l_wrist_roll_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    l_wrist_roll_pos - current_pos,
                                    -max_change,
                                    max_change,
                                )
                                left_joint_dict["l_wrist_roll"] = limited_pos
                                prev_left_joint_pos["l_wrist_roll"] = limited_pos

                                # Gripper
                                current_pos = prev_left_joint_pos.get(
                                    "l_gripper", l_gripper_pos
                                )
                                limited_pos = current_pos + np.clip(
                                    l_gripper_pos - current_pos, -max_change, max_change
                                )
                                left_joint_dict["l_gripper"] = limited_pos
                                prev_left_joint_pos["l_gripper"] = limited_pos

                            except Exception as e:
                                print(f"Left arm IK calculation error: {e}")

                # Apply goal positions directly at controlled rate
                current_time = time.time()
                if (
                    current_time - last_movement_time_right >= movement_interval
                    and not right_already_there
                ):
                    last_movement_right_time = current_time

                    # Apply right arm joint positions if any
                    for joint_name, position in right_joint_dict.items():
                        try:
                            # Apply position directly to the right joint using its name
                            if joint_name == "r_shoulder_pitch":
                                self.reachy.r_arm.r_shoulder_pitch.goal_position = (
                                    position
                                )
                            elif joint_name == "r_shoulder_roll":
                                self.reachy.r_arm.r_shoulder_roll.goal_position = (
                                    position
                                )
                            elif joint_name == "r_arm_yaw":
                                self.reachy.r_arm.r_arm_yaw.goal_position = position
                            elif joint_name == "r_elbow_pitch":
                                self.reachy.r_arm.r_elbow_pitch.goal_position = position
                            elif joint_name == "r_forearm_yaw":
                                self.reachy.r_arm.r_forearm_yaw.goal_position = position
                            elif joint_name == "r_wrist_pitch":
                                self.reachy.r_arm.r_wrist_pitch.goal_position = position
                            elif joint_name == "r_wrist_roll":
                                self.reachy.r_arm.r_wrist_roll.goal_position = position
                            elif joint_name == "r_gripper":
                                self.reachy.r_arm.r_gripper.goal_position = position
                        except Exception as e:
                            print(f"Error setting position for {joint_name}: {e}")

                if (
                    current_time - last_movement_time_left >= movement_interval
                    and not left_already_there
                ):
                    last_movement_left_time = current_time
                    # Apply left arm joint positions if any
                    for joint_name, position in left_joint_dict.items():
                        try:
                            # Apply position directly to the left joint using its name
                            if joint_name == "l_shoulder_pitch":
                                self.reachy.l_arm.l_shoulder_pitch.goal_position = (
                                    position
                                )
                            elif joint_name == "l_shoulder_roll":
                                self.reachy.l_arm.l_shoulder_roll.goal_position = (
                                    position
                                )
                            elif joint_name == "l_arm_yaw":
                                self.reachy.l_arm.l_arm_yaw.goal_position = position
                            elif joint_name == "l_elbow_pitch":
                                self.reachy.l_arm.l_elbow_pitch.goal_position = position
                            elif joint_name == "l_forearm_yaw":
                                self.reachy.l_arm.l_forearm_yaw.goal_position = position
                            elif joint_name == "l_wrist_pitch":
                                self.reachy.l_arm.l_wrist_pitch.goal_position = position
                            elif joint_name == "l_wrist_roll":
                                self.reachy.l_arm.l_wrist_roll.goal_position = position
                            elif joint_name == "l_gripper":
                                self.reachy.l_arm.l_gripper.goal_position = position
                        except Exception as e:
                            print(f"Error setting position for {joint_name}: {e}")

                # Display tracking data if enabled
                if display:
                    self.display_frame(
                        arm, color_image, right_arm_coordinates, left_arm_coordinates
                    )

                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cleanup_requested = True

                # Ensure we don't hog the CPU
                elapsed = time.time() - loop_start_time
                if elapsed < 0.01:  # Try to maintain reasonable loop time
                    await asyncio.sleep(0.01 - elapsed)
                else:
                    await asyncio.sleep(0.001)  # Minimal yield to event loop

        except Exception as e:
            print(f"Failed to run the shadow pipeline: {e}")
            # Safety: make sure to set arms to compliant mode on error
            try:
                if arm == "right" or arm == "both":
                    self.reachy.turn_off_smoothly("r_arm")
                if arm == "left" or arm == "both":
                    self.reachy.turn_off_smoothly("l_arm")
            except:
                print("Error during emergency shutdown")
            cv2.destroyAllWindows()
        finally:

            # Perform graceful shutdown
            print("Exiting control loop, performing gradual shutdown...")

            # Gradually reduce torque to prevent sudden drops - using direct joint access
            for torque in range(70, 20, -10):
                # Right arm
                if arm == "right" or arm == "both":
                    if hasattr(self.reachy.r_arm, "r_shoulder_pitch"):
                        self.reachy.r_arm.r_shoulder_pitch.torque_limit = torque
                    if hasattr(self.reachy.r_arm, "r_shoulder_roll"):
                        self.reachy.r_arm.r_shoulder_roll.torque_limit = torque
                    if hasattr(self.reachy.r_arm, "r_arm_yaw"):
                        self.reachy.r_arm.r_arm_yaw.torque_limit = torque
                    if hasattr(self.reachy.r_arm, "r_elbow_pitch"):
                        self.reachy.r_arm.r_elbow_pitch.torque_limit = torque
                    if hasattr(self.reachy.r_arm, "r_forearm_yaw"):
                        self.reachy.r_arm.r_forearm_yaw.torque_limit = torque
                    if hasattr(self.reachy.r_arm, "r_wrist_pitch"):
                        self.reachy.r_arm.r_wrist_pitch.torque_limit = torque
                    if hasattr(self.reachy.r_arm, "r_wrist_roll"):
                        self.reachy.r_arm.r_wrist_roll.torque_limit = torque
                    if hasattr(self.reachy.r_arm, "r_gripper"):
                        self.reachy.r_arm.r_gripper.torque_limit = torque

                # Left arm
                if arm == "left" or arm == "both":
                    if hasattr(self.reachy.l_arm, "l_shoulder_pitch"):
                        self.reachy.l_arm.l_shoulder_pitch.torque_limit = torque
                    if hasattr(self.reachy.l_arm, "l_shoulder_roll"):
                        self.reachy.l_arm.l_shoulder_roll.torque_limit = torque
                    if hasattr(self.reachy.l_arm, "l_arm_yaw"):
                        self.reachy.l_arm.l_arm_yaw.torque_limit = torque
                    if hasattr(self.reachy.l_arm, "l_elbow_pitch"):
                        self.reachy.l_arm.l_elbow_pitch.torque_limit = torque
                    if hasattr(self.reachy.l_arm, "l_forearm_yaw"):
                        self.reachy.l_arm.l_forearm_yaw.torque_limit = torque
                    if hasattr(self.reachy.l_arm, "l_wrist_pitch"):
                        self.reachy.l_arm.l_wrist_pitch.torque_limit = torque
                    if hasattr(self.reachy.l_arm, "l_wrist_roll"):
                        self.reachy.l_arm.l_wrist_roll.torque_limit = torque
                    if hasattr(self.reachy.l_arm, "l_gripper"):
                        self.reachy.l_arm.l_gripper.torque_limit = torque

                await asyncio.sleep(0.2)

            # Finally turn motors to compliant mode
            if arm == "right" or arm == "both":
                self.reachy.turn_off_smoothly("r_arm")
            if arm == "left" or arm == "both":
                self.reachy.turn_off_smoothly("l_arm")

            cv2.destroyAllWindows()
