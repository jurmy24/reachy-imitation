import time
from typing import List, Literal
import numpy as np
import cv2
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
from collections import defaultdict

from src.utils.three_d import get_reachy_coordinates
from src.pipelines.Pipeline import Pipeline
from src.mapping.get_arm_lengths import get_arm_lengths
from src.mapping.map_to_robot_coordinates import get_scale_factors
from src.sensing.extract_3D_points import get_head_coordinates
from config.CONSTANTS import get_zero_pos
from src.reachy.utils import setup_torque_limits
from src.models.shadow_arms import ShadowArm
from src.models.custom_ik import minimize_timer  # Import the timer


class Pipeline_custom_ik(Pipeline):
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

    def display_frame(
        self,
        arm,
        color_image,
        pose_landmarks=None,
    ):
        # Display landmarks on the image
        if pose_landmarks:
            self.mp_draw.draw_landmarks(
                color_image,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )

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

    def shadow(
        self, side: Literal["right", "left", "both"] = "right", display: bool = True
    ):
        """
        Control Reachy to shadow human arm movements in real-time.

        Args:
            side: Which arm to track ("right", "left", or "both")
            display: Whether to display the video window with tracking information
        """
        ############### Parameters ###############
        smoothing_buffer_size = 5  # ! Not actually used rn i believe
        position_alpha = 0.4  # For EMA position smoothing
        movement_interval = 0.0333  # Send commands at ~30Hz maximum
        max_change = 5.0  # maximum change in degrees per joint per update
        elbow_weight = 0.5
        target_pos_tolerance = (
            0.03  # update current position if it is more than this far from the target
        )
        movement_min_tolerance = (
            0.005  # update current position if it has moved more than this
        )
        torque_limit = 100.0  # as a percentage of maximum
        ########################################

        ############### FLAGS ##################
        cleanup_requested = False
        successful_update = False
        ########################################

        # Performance metrics
        timings = defaultdict(list)
        update_count = 0
        arm_count = 0  # Track number of arms for per-arm averages

        try:
            # Set torque limits for all motor joints for safety
            setup_torque_limits(self.reachy, torque_limit, side)

            # Initialize the arm(s) for shadowing
            arms_to_process: List[ShadowArm] = []
            if side in ["right", "both"]:
                right_arm = ShadowArm(
                    self.reachy.r_arm,
                    "right",
                    smoothing_buffer_size,
                    position_alpha,
                    max_change,
                    self.mp_pose,
                )
                arms_to_process.append(right_arm)
            if side in ["left", "both"]:
                left_arm = ShadowArm(
                    self.reachy.l_arm,
                    "left",
                    smoothing_buffer_size,
                    position_alpha,
                    max_change,
                    self.mp_pose,
                )
                arms_to_process.append(left_arm)

            arm_count = len(arms_to_process)

            # For frame rate and movement control
            last_movement_time = time.time()

            print("Starting shadowing. Press 'q' to exit safely.")

            while not cleanup_requested:
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cleanup_requested = True

                loop_start_time = time.time()

                # 1. get data from RealSense camera
                camera_start = time.time()
                camera_data = self._get_camera_data()
                camera_end = time.time()
                timings["camera_acquisition"].append(camera_end - camera_start)

                if camera_data is None:
                    continue
                color_frame, depth_frame, color_image, rgb_image, h, w = camera_data

                # 2. get pose landmarks from the image using mediapipe
                pose_start = time.time()
                pose_results = self.pose.process(rgb_image)
                pose_end = time.time()
                timings["pose_detection"].append(pose_end - pose_start)

                landmarks = pose_results.pose_landmarks

                # Display tracking data if enabled
                if display:
                    display_start = time.time()
                    self.display_frame(side, color_image, landmarks)
                    display_end = time.time()
                    timings["display"].append(display_end - display_start)

                if not pose_results.pose_landmarks:
                    continue

                # 3. process each arm
                for current_arm in arms_to_process:
                    # 3a. get coordinates of reachy's hands in reachy's frame
                    coord_start = time.time()
                    shoulder, elbow, hand = current_arm.get_coordinates(
                        landmarks.landmark, depth_frame, w, h, self.intrinsics
                    )
                    coord_end = time.time()
                    timings["get_coordinates"].append(coord_end - coord_start)

                    if shoulder is None or hand is None:
                        continue

                    conv_start = time.time()
                    target_ee_coord = get_reachy_coordinates(
                        hand, shoulder, self.hand_sf, current_arm.side
                    )

                    target_elbow_coord = get_reachy_coordinates(
                        elbow, shoulder, self.elbow_sf, current_arm.side
                    )
                    conv_end = time.time()
                    timings["coordinate_conversion"].append(conv_end - conv_start)

                    # 3b. Process the new ee_position and calculate IK if needed
                    pos_proc_start = time.time()
                    should_update, target_ee_coord_smoothed = (
                        current_arm.process_new_position(
                            target_ee_coord,
                            target_pos_tolerance,
                            movement_min_tolerance,
                        )
                    )
                    pos_proc_end = time.time()
                    timings["position_processing"].append(pos_proc_end - pos_proc_start)

                    if should_update:
                        # ! We're not smoothing the elbow position here
                        # Calculate IK and update joint positions
                        ik_start = time.time()
                        successful_update = (
                            current_arm.calculate_joint_positions_custom_ik(
                                target_ee_coord_smoothed,
                                target_elbow_coord,
                                elbow_weight,
                            )
                        )
                        ik_end = time.time()
                        timings["inverse_kinematics"].append(ik_end - ik_start)
                    else:
                        print("Inverse kinematics skipped!")
                        successful_update = False

                # Apply goal positions directly at controlled rate (maximum 30Hz)
                current_time = time.time()
                if (
                    current_time - last_movement_time >= movement_interval
                    and successful_update
                ):
                    apply_start = time.time()
                    last_movement_time = current_time
                    update_count += 1

                    # Apply joint positions for all arms that have updates
                    for current_arm in arms_to_process:
                        # Apply arm joint positions if there are any to apply
                        if current_arm.joint_dict:
                            for (
                                joint_name,
                                joint_value,
                            ) in current_arm.joint_dict.items():
                                try:
                                    # Apply position directly to the joint
                                    setattr(
                                        getattr(current_arm.arm, joint_name),
                                        "goal_position",
                                        joint_value,
                                    )
                                except Exception as e:
                                    print(
                                        f"Error setting position for {joint_name}: {e}"
                                    )
                    apply_end = time.time()
                    timings["apply_positions"].append(apply_end - apply_start)

                # Calculate loop latency
                loop_end_time = time.time()
                loop_latency = loop_end_time - loop_start_time
                timings["total_loop"].append(loop_latency)

        except Exception as e:
            print(f"Failed to run the shadow pipeline: {e}")
        finally:
            # Print final detailed performance statistics
            print("\n===== PERFORMANCE ANALYSIS =====")
            print(f"Total updates: {update_count}")
            print(f"Number of arms: {arm_count}")

            # Calculate and print average times for each section
            print("\nAverage time per operation (ms):")

            # Sort timings by average time (descending)
            sorted_timings = sorted(
                [(k, np.mean(v) * 1000) for k, v in timings.items() if v],
                key=lambda x: x[1],
                reverse=True,
            )

            total_time = (
                np.mean(timings["total_loop"]) * 1000 if timings["total_loop"] else 0
            )

            for operation, avg_time in sorted_timings:
                if operation != "total_loop":
                    percentage = (avg_time / total_time * 100) if total_time > 0 else 0
                    print(
                        f"{operation:<25}: {avg_time:6.2f} ms ({percentage:5.1f}% of loop)"
                    )

            # Print loop summary
            if timings["total_loop"]:
                avg_loop = np.mean(timings["total_loop"]) * 1000
                min_loop = min(timings["total_loop"]) * 1000
                max_loop = max(timings["total_loop"]) * 1000
                theoretical_max_hz = 1000 / avg_loop if avg_loop > 0 else 0

                print(f"\nLoop timing:")
                print(
                    f"  Average: {avg_loop:.2f} ms (theoretical max: {theoretical_max_hz:.1f} Hz)"
                )
                print(f"  Min: {min_loop:.2f} ms, Max: {max_loop:.2f} ms")

                # Calculate effective update rate
                if update_count > 0 and timings["total_loop"]:
                    total_run_time = sum(timings["total_loop"])
                    effective_hz = (
                        update_count / total_run_time if total_run_time > 0 else 0
                    )
                    print(f"\nEffective update rate: {effective_hz:.2f} Hz")
                    print(
                        f"Target update rate: {1/movement_interval:.1f} Hz (movement_interval={movement_interval}s)"
                    )

            # Print minimize function timing statistics
            minimize_stats = minimize_timer.get_stats()
            if minimize_stats["calls"] > 0:
                print("\n===== MINIMIZE FUNCTION TIMING STATISTICS =====")
                print(f"Total calls: {minimize_stats['calls']}")
                print(f"Average time: {minimize_stats['avg_time']*1000:.2f} ms")
                print(f"Min time: {minimize_stats['min_time']*1000:.2f} ms")
                print(f"Max time: {minimize_stats['max_time']*1000:.2f} ms")
                print(f"Total time: {minimize_stats['total_time']*1000:.2f} ms")
                print(
                    f"Percentage of total loop time: {(minimize_stats['total_time'] / sum(timings['total_loop']) * 100):.1f}%"
                )
                print(f"\nIteration Statistics:")
                print(f"Average iterations: {minimize_stats['avg_iterations']:.1f}")
                print(f"Min iterations: {minimize_stats['min_iterations']}")
                print(f"Max iterations: {minimize_stats['max_iterations']}")
                print(f"Total iterations: {minimize_stats['total_iterations']}")

            self.cleanup()
