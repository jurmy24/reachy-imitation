import time
from typing import List, Literal
import numpy as np
import cv2
import asyncio

from src.reachy.utils import setup_torque_limits
from src.utils.three_d import get_reachy_coordinates
from src.pipelines.Pipeline import Pipeline

from src.models.shadow_arms import ShadowArm


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

    async def shadow(
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
        movement_interval = 0.03  # Send commands at ~30Hz
        max_change = 5.0  # maximum change in degrees per joint per update
        ########################################

        ############### FLAGS ##################
        cleanup_requested = False
        successful_update = False
        ########################################

        try:
            # Set torque limits for all motor joints for safety
            setup_torque_limits(self.reachy, 70.0, side)

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

            # For frame rate and movement control
            last_movement_time = time.time()

            print("Starting shadowing. Press 'q' to exit safely.")

            while not cleanup_requested:
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cleanup_requested = True

                loop_start_time = time.time()

                # 1. get data from RealSense camera
                camera_data = self._get_camera_data()
                if camera_data is None:
                    await asyncio.sleep(0.01)
                    continue
                color_frame, depth_frame, color_image, rgb_image, h, w = camera_data

                # 2. get pose landmarks from the image using mediapipe
                pose_results = self.pose.process(rgb_image)

                landmarks = pose_results.pose_landmarks

                # Display tracking data if enabled
                if display:
                    self.display_frame(side, color_image, landmarks)

                if not pose_results.pose_landmarks:
                    await asyncio.sleep(0.01)
                    continue

                # 3. process each arm
                for current_arm in arms_to_process:
                    # Update the arm's joint array with current joint positions
                    # current_arm.joint_array = current_arm.get_joint_array()

                    # TODO: use the elbow too (for the pipeline_one)
                    # 3a. get coordinates of reachy's hands in reachy's frame
                    shoulder, elbow, hand = current_arm.get_coordinates(
                        landmarks.landmark, depth_frame, w, h, self.intrinsics
                    )
                    if shoulder is None or hand is None:
                        await asyncio.sleep(0.01)
                        continue

                    target_ee_coord = get_reachy_coordinates(
                        hand, shoulder, self.hand_sf, current_arm.side
                    )

                    # TODO: Check if the target end effector coordinates are within reachy's reach
                    # 3b. Process the new ee_position and calculate IK if needed
                    should_update, target_ee_coord_smoothed = (
                        current_arm.process_new_position(target_ee_coord)
                    )

                    if should_update:
                        # Calculate IK and update joint positions
                        successful_update = current_arm.calculate_joint_positions(
                            target_ee_coord_smoothed
                        )
                    else:
                        successful_update = False

                # Apply goal positions directly at controlled rate
                current_time = time.time()
                if (
                    current_time - last_movement_time >= movement_interval
                    and successful_update
                ):
                    last_movement_time = current_time

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

                # Ensure we don't hog the CPU
                elapsed = time.time() - loop_start_time
                if elapsed < 0.01:  # Try to maintain reasonable loop time
                    await asyncio.sleep(0.01 - elapsed)
                else:
                    await asyncio.sleep(0.001)  # Minimal yield to event loop
        except Exception as e:
            print(f"Failed to run the shadow pipeline: {e}")
        finally:
            self.cleanup()
