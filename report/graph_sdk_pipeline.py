import time
from typing import List, Literal
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import csv


from src.utils.hands import flip_hand_labels
from src.utils.three_d import get_reachy_coordinates
from src.pipelines.Pipeline import Pipeline
from src.reachy.utils import setup_torque_limits
from src.models.shadow_arms_force import ShadowArm, HAND_STATUS
from src.models.custom_ik import minimize_timer  # Import the timer
import asyncio

import traceback

from reachy_sdk import ReachySDK
from src.mapping.map_to_robot_coordinates import get_scale_factors
from config.CONSTANTS import HUMAN_ELBOW_TO_HAND_DEFAULT, HUMAN_UPPERARM_DEFAULT

from report.helper_kinematics import elbow_hand_forward_kinematics

# Create the overarching Reachy instance for this application
reachy = ReachySDK(host="192.168.100.2")


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
        # For plotting after execution
        hand_target_coords = []
        hand_actual_coords = []
        elbow_target_coords = []
        elbow_actual_coords = []

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

                    target_elbow_coord = get_reachy_coordinates(
                        elbow, shoulder, self.elbow_sf, current_arm.side
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

                hand_target_coords.append(target_ee_coord)
                elbow_target_coords.append(target_elbow_coord)

                # Get actual robot positions via FK
                actual_elbow, actual_hand = elbow_hand_forward_kinematics(np.deg2rad(arms_to_process[0].get_joint_array()), side="left")
                hand_actual_coords.append(actual_hand)
                elbow_actual_coords.append(actual_elbow)


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
            return hand_target_coords, hand_actual_coords, elbow_target_coords, elbow_actual_coords


def plot_comparison(target, actual, label):
    target = np.array(target)
    actual = np.array(actual)
    fig, ax = plt.subplots()
    ax.plot(target[:, 0], label=f"{label} Target X", linestyle='--')
    ax.plot(actual[:, 0], label=f"{label} Actual X")
    ax.plot(target[:, 1], label=f"{label} Target Y", linestyle='--')
    ax.plot(actual[:, 1], label=f"{label} Actual Y")
    ax.plot(target[:, 2], label=f"{label} Target Z", linestyle='--')
    ax.plot(actual[:, 2], label=f"{label} Actual Z")
    ax.set_title(f"{label} Target vs Actual Coordinates")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position (m)")
    ax.legend()
    plt.grid(True)
    plt.show()



def plot_separate_axes(target_coords, actual_coords, joint_name):
    """
    Plot x, y, z components of target vs actual coordinates on separate subplots.

    Args:
        target_coords: list of (x, y, z) tuples.
        actual_coords: list of (x, y, z) tuples.
        joint_name: str, used in plot titles.
    """
    time_steps = list(range(len(target_coords)))
    target_x = [t[0] for t in target_coords]
    target_y = [t[1] for t in target_coords]
    target_z = [t[2] for t in target_coords]

    actual_x = [a[0] for a in actual_coords]
    actual_y = [a[1] for a in actual_coords]
    actual_z = [a[2] for a in actual_coords]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'{joint_name.capitalize()} Tracking: Target vs Actual')

    axs[0].plot(time_steps, target_x, label='Target X', linestyle='--')
    axs[0].plot(time_steps, actual_x, label='Actual X')
    axs[0].set_ylabel('X')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time_steps, target_y, label='Target Y', linestyle='--')
    axs[1].plot(time_steps, actual_y, label='Actual Y')
    axs[1].set_ylabel('Y')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(time_steps, target_z, label='Target Z', linestyle='--')
    axs[2].plot(time_steps, actual_z, label='Actual Z')
    axs[2].set_ylabel('Z')
    axs[2].set_xlabel('Time Step')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


def save_tracking_data_to_csv(target_coords, actual_coords, joint_name, filename):
    """
    Save target and actual 3D coordinate data for a joint to a CSV file.

    Args:
        target_coords: list of (x, y, z) tuples for target positions.
        actual_coords: list of (x, y, z) tuples for actual positions.
        joint_name: str, used in column labels (e.g., 'hand' or 'elbow').
        filename: str, path to save the CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Header row
        writer.writerow([
            'Time Step',
            f'{joint_name}_target_x', f'{joint_name}_actual_x',
            f'{joint_name}_target_y', f'{joint_name}_actual_y',
            f'{joint_name}_target_z', f'{joint_name}_actual_z'
        ])

        # Write rows
        for i, (t, a) in enumerate(zip(target_coords, actual_coords)):
            writer.writerow([
                i,
                t[0], a[0],  # x
                t[1], a[1],  # y
                t[2], a[2]   # z
            ])

if __name__ == "__main__":
    # Example usage
    arm = "left"

    pipeline = Pipeline_one_mini(reachy)

    hand_sf, elbow_sf = get_scale_factors(
        HUMAN_ELBOW_TO_HAND_DEFAULT, HUMAN_UPPERARM_DEFAULT
    )
    pipeline.hand_sf = hand_sf
    pipeline.elbow_sf = elbow_sf
    
    hand_target_coords, hand_actual_coords, elbow_target_coords, elbow_actual_coords = asyncio.run(pipeline.shadow(side=arm, display=True)
    )

    if hand_target_coords and hand_actual_coords:
        #plot_comparison(hand_target_coords, hand_actual_coords, "Hand")
        plot_separate_axes(hand_target_coords, hand_actual_coords, "hand")

    if elbow_target_coords and elbow_actual_coords:
        #plot_comparison(elbow_target_coords, elbow_actual_coords, "Elbow")
        plot_separate_axes(elbow_target_coords, elbow_actual_coords, "elbow")

    save_tracking_data_to_csv(hand_target_coords, hand_actual_coords, "hand", "sdk_hand_tracking_data.csv")
    save_tracking_data_to_csv(elbow_target_coords, elbow_actual_coords, "elbow", "sdk_elbow_tracking_data.csv")
