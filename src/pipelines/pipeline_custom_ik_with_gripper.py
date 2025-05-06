import time
from typing import List, Literal
import numpy as np
import cv2
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
from collections import defaultdict

from src.utils.hands import flip_hand_labels
from src.utils.three_d import get_reachy_coordinates
from src.pipelines.Pipeline import Pipeline
from src.mapping.get_arm_lengths import get_arm_lengths
from src.mapping.map_to_robot_coordinates import get_scale_factors
from src.sensing.extract_3D_points import get_head_coordinates
from config.CONSTANTS import get_zero_pos
from src.reachy.utils import setup_torque_limits
from src.models.shadow_arms import ShadowArm
from src.models.custom_ik import minimize_timer  # Import the timer


class Pipeline_custom_ik_with_gripper(Pipeline):
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
    """

    def display_frame(self, arm, color_image, pose_landmarks=None, hand_results=None):
        # Display landmarks on the image
        if pose_landmarks:
            self.mp_draw.draw_landmarks(
                color_image,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                self.mp_draw.draw_landmarks(
                    color_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
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
        handedness_certainty_threshold = 0.8
        ########################################

        ############### FLAGS ##################
        cleanup_requested = False
        successful_update = False
        #successful_gripper_update = False
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
            last_hand_movement_time = time.time()

            print("Starting shadowing. Press 'control c' to exit safely")

            while not cleanup_requested:

                loop_start_time = time.time()

                # ! The absence of this might be causing a lag
                #  # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cleanup_requested = True

                # 1. get data from RealSense camera
                camera_start = time.time()
                camera_data = self._get_camera_data()
                camera_end = time.time()
                timings["camera_acquisition"].append(camera_end - camera_start)

                if camera_data is None:
                    continue
                color_frame, depth_frame, color_image, rgb_image, h, w = camera_data

                # 2. get pose and hand landmarks from the image using mediapipe
                pose_start = time.time()
                pose_results = self.pose.process(rgb_image)
                pose_end = time.time()
                timings["pose_detection"].append(pose_end - pose_start)

                landmarks = pose_results.pose_landmarks

                hand_results = self.hands.process(rgb_image)

                # Display tracking data if enabled
                if display:
                    display_start = time.time()
                    self.display_frame(side, color_image, landmarks, hand_results)
                    display_end = time.time()
                    timings["display"].append(display_end - display_start)

                if not pose_results.pose_landmarks:
                    continue

                for current_arm in arms_to_process:
                    # 3. get coordinates of reachy's hands in reachy's frame
                    coord_start = time.time()
                    shoulder, elbow, hand = current_arm.get_coordinates(
                        landmarks.landmark, depth_frame, w, h, self.intrinsics
                    )
                    coord_end = time.time()
                    timings["get_coordinates"].append(coord_end - coord_start)

                    if shoulder is None or hand is None:
                        continue
                    
                    #shoulder, elbow, hand = current_arm.apply_kalman_filter_to_coordinates(shoulder, elbow, hand)

                    conv_start = time.time()
                    target_ee_coord = get_reachy_coordinates(
                        hand, shoulder, self.hand_sf, current_arm.side
                    )

                    target_elbow_coord = get_reachy_coordinates(
                        elbow, shoulder, self.elbow_sf, current_arm.side
                    )
                    conv_end = time.time()
                    timings["coordinate_conversion"].append(conv_end - conv_start)

                    # 4. Process the new ee_position and calculate IK if needed
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
                        # print("Inverse kinematics skipped!")
                        successful_update = False

                hand_data_start = time.time()
                # 5. Set the gripper value in current_arm.joint_dict
                successful_gripper_update = False
                if hand_results.multi_hand_landmarks:
                    # both_hand_landmarks = [None, None]  # left, right
                    already_detected_handedness = None
                    processed_hands = 0
                    if hand_results.multi_hand_landmarks:
                        for index, hand_landmarks in enumerate(
                            hand_results.multi_hand_landmarks
                        ):
                            if (
                                processed_hands >= 2
                            ):  # only execute on the first two hands detected.
                                break
                            handedness_certainty = (
                                hand_results.multi_handedness[index]
                                .classification[0]
                                .score
                            )
                            # handedness certainty can also be verified with a check if the previous hand in 
                            # the multi_hand_landmarks array was detected to be the same, and having a lower_handedness_certainty
                            # This is done in controleMainWithDepth.py in ../../scripts/pince 
                            if handedness_certainty < handedness_certainty_threshold:
                                continue

                            # Could also add some visibility checks here

                            hand_type = flip_hand_labels(
                                hand_results.multi_handedness[index]
                                .classification[0]
                                .label
                            )

                            arm_to_process_hand = None
                            for arm in arms_to_process:
                                if arm.side == hand_type:
                                    arm_to_process_hand = arm
                                    break

                            if (
                                arm_to_process_hand is None
                                or already_detected_handedness == hand_type
                            ):
                                continue
                            processed_hands += 1
                            already_detected_handedness = hand_type
                            # both_hand_landmarks[index] = hand_landmarks
                            hand_landmarks = hand_results.multi_hand_landmarks[index]
                            hand_closed = arm_to_process_hand.get_hand_closedness(
                                hand_landmarks, self.mp_hands
                            )
                            if arm_to_process_hand.hand_closed != hand_closed:
                                # update the hand state - using force
                                successful_gripper_update = True
                                arm_to_process_hand.hand_closed = hand_closed
                                if hand_closed:
                                    arm_to_process_hand.close_hand()
                                else:
                                    arm_to_process_hand.open_hand()


                hand_data_end = time.time()
                timings["hand_data_processing"].append(
                    hand_data_end - hand_data_start
                )

                # 6. Apply goal positions directly at controlled rate (maximum 30Hz)
                current_time = time.time()
                
                if (
                    current_time - last_movement_time >= movement_interval
                    and successful_update
                ):
                    last_movement_time = current_time
                    apply_start = time.time()
                    # Apply joint positions for all arms that have updates
                    for current_arm in arms_to_process:
                        # Apply arm joint positions if there are any to apply
                        if current_arm.joint_dict:
                            for (
                                key,
                                value,
                            ) in current_arm.joint_dict.items():
                                if key == f"{current_arm.prefix}gripper":
                                    continue
                                try:
                                    setattr(
                                        getattr(current_arm.arm, key),
                                        "goal_position",
                                        value,
                                    )
                                except Exception as e:
                                    print(
                                        f"Error setting position for {key}: {e}"
                                    )
                    apply_end = time.time()
                    timings["apply_positions"].append(apply_end - apply_start)
                
                
                # # Gripper Set Position
                if (
                    current_time - last_hand_movement_time >= movement_interval
                    and successful_gripper_update
                ):
                    last_hand_movement_time = current_time
                    gripper_start = time.time()
                    # Apply joint positions for all arms that have updates
                    for current_arm in arms_to_process:
                        
                        # Apply arm joint positions if there are any to apply
                        if current_arm.joint_dict:
                            
                            joint = f"{current_arm.prefix}gripper"
                            if joint in current_arm.joint_dict:
                                try:
                                    print(f"Setting gripper position for {joint}: {current_arm.joint_dict[joint]}")
                                    setattr(
                                        getattr(current_arm.arm, joint),
                                        "goal_position",
                                        current_arm.joint_dict[joint],
                                    )
                                except Exception as e:
                                    print(
                                        f"Error setting position for {joint}: {e}"
                                    )
                            else:
                                print(f"Gripper joint {joint} not found in joint_dict")
                    gripper_end = time.time()
                    timings["gripper_apply"].append(gripper_end - gripper_start)

                if successful_update or successful_gripper_update:
                    update_count += 1

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
