import time
from typing import Literal
import numpy as np
import cv2
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from src.utils.three_d import get_reachy_coordinates
from src.pipelines.Pipeline import Pipeline
from src.reachy.utils import setup_torque_limits
from src.models.shadow_arms import ShadowArm


class Pipeline_custom_ik_threaded(Pipeline):
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
        smoothing_buffer_size = 5
        position_alpha = 0.4  # For EMA position smoothing
        movement_interval = 0.0333  # Send commands at ~30Hz
        max_change = 5.0  # maximum change in degrees per joint per update
        elbow_weight = 0.1
        target_pos_tolerance = (
            0.03  # update current position if it is more than this far from the target
        )
        movement_min_tolerance = (
            0.005  # update current position if it has moved more than this
        )
        torque_limit = 80.0  # as a percentage of maximum
        ########################################

        ############### FLAGS ##################
        cleanup_requested = False
        successful_update = False
        ########################################

        # Performance metrics
        timings = defaultdict(list)
        update_count = 0

        # Threading function for IK calculation
        # NOTE: For threading!
        def calculate_ik_threaded(
            shadow_arm, target_ee_coord_smoothed, target_elbow_coord, elbow_weight
        ):
            try:
                success = shadow_arm.calculate_joint_positions_custom_ik(
                    target_ee_coord_smoothed, target_elbow_coord, elbow_weight
                )
                return shadow_arm.side, success
            except Exception as e:
                print(f"Error in IK thread for {shadow_arm.side} arm: {e}")
                return shadow_arm.side, False

        try:
            # Set torque limits for all motor joints for safety
            setup_torque_limits(self.reachy, torque_limit, side)

            # Create a dictionary mapping arm sides to ShadowArm instances
            shadow_arms = {}
            if side in ["right", "both"]:
                shadow_arms["right"] = ShadowArm(
                    self.reachy.r_arm,
                    "right",
                    smoothing_buffer_size,
                    position_alpha,
                    max_change,
                    self.mp_pose,
                )
            if side in ["left", "both"]:
                shadow_arms["left"] = ShadowArm(
                    self.reachy.l_arm,
                    "left",
                    smoothing_buffer_size,
                    position_alpha,
                    max_change,
                    self.mp_pose,
                )

            arm_count = len(shadow_arms)

            # Create thread pool - only needs max 2 threads (one per arm)
            # This avoids creating/destroying threads in the loop
            executor = ThreadPoolExecutor(max_workers=arm_count)

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

                # 3. process each arm and prepare for IK calculations
                futures = []

                for arm_side, shadow_arm in shadow_arms.items():
                    # 3a. get coordinates of reachy's hands in reachy's frame
                    coord_start = time.time()
                    shoulder, elbow, hand = shadow_arm.get_coordinates(
                        landmarks.landmark, depth_frame, w, h, self.intrinsics
                    )
                    coord_end = time.time()
                    timings["get_coordinates"].append(coord_end - coord_start)

                    if shoulder is None or hand is None:
                        continue

                    conv_start = time.time()
                    target_ee_coord = get_reachy_coordinates(
                        hand, shoulder, self.hand_sf, arm_side
                    )

                    target_elbow_coord = get_reachy_coordinates(
                        elbow, shoulder, self.elbow_sf, arm_side
                    )
                    conv_end = time.time()
                    timings["coordinate_conversion"].append(conv_end - conv_start)

                    # 3b. Process the new ee_position and determine if IK is needed
                    pos_proc_start = time.time()
                    should_update, target_ee_coord_smoothed = (
                        shadow_arm.process_new_position(
                            target_ee_coord,
                            target_pos_tolerance,
                            movement_min_tolerance,
                        )
                    )
                    pos_proc_end = time.time()
                    timings["position_processing"].append(pos_proc_end - pos_proc_start)

                    # If we should update, submit task to thread pool
                    if should_update:
                        future = executor.submit(
                            calculate_ik_threaded,
                            shadow_arm,
                            target_ee_coord_smoothed,
                            target_elbow_coord,
                            elbow_weight,
                        )
                        futures.append(future)

                # Process IK results from thread pool
                ik_start = time.time()
                successful_update = False

                if futures:
                    # Wait for all submitted tasks to complete
                    successful_results = 0
                    for future in futures:
                        try:
                            arm_side, success = future.result()
                            if success:
                                successful_results += 1
                        except Exception as e:
                            print(f"Error getting thread result: {e}")

                    successful_update = successful_results > 0

                ik_end = time.time()
                timings["inverse_kinematics"].append(ik_end - ik_start)

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
                    for shadow_arm in shadow_arms.values():
                        # Apply arm joint positions if there are any to apply
                        if shadow_arm.joint_dict:
                            for (
                                joint_name,
                                joint_value,
                            ) in shadow_arm.joint_dict.items():
                                try:
                                    # Apply position directly to the joint
                                    setattr(
                                        getattr(shadow_arm.arm, joint_name),
                                        "goal_position",
                                        joint_value,
                                    )
                                except Exception as e:
                                    print(
                                        f"Error setting position for {joint_name}: {e}"
                                    )
                    apply_end = time.time()
                    timings["apply_positions"].append(apply_end - apply_start)

        except Exception as e:
            print(f"Failed to run the shadow pipeline: {e}")
        finally:
            # Shutdown thread pool gracefully
            if "executor" in locals():
                executor.shutdown(wait=False)

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

            self.cleanup()
