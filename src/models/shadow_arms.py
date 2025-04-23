import numpy as np
from typing import Tuple, Literal

from src.utils.three_d import get_3D_coordinates, get_3D_coordinates_of_hand


class ShadowArm:
    """Base class for robot arm operations and tracking"""

    def __init__(
        self,
        reachy_arm,
        side: Literal["right", "left"],
        smoothing_buffer_size,
        position_alpha,
        max_change,
        mp_pose,
    ):
        self.arm = reachy_arm
        self.side: Literal["right", "left"] = side
        self.prefix = f"{side[0]}_"  # "r_" or "l_"
        self.mp_pose = mp_pose
        self.landmark_indices = self.get_landmark_indices()
        

        # Movement tracking
        self.joint_array = self.get_joint_array()
        self.joint_dict = {}
        self.position_history = []
        self.smoothing_buffer_size = smoothing_buffer_size
        self.position_alpha = position_alpha  # For EMA position smoothing
        self.max_change = max_change  # maximum change in degrees per joint per update
        self.prev_hand_pos = self.arm.forward_kinematics()[:3, 3]

    def get_landmark_indices(self):
        """Return MediaPipe landmark indices for this arm

        Returns:
            Dictionary mapping landmark names to MediaPipe indices
        """
        if self.side == "right":
            return {
                "shoulder": self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                "elbow": self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                "index": self.mp_pose.PoseLandmark.RIGHT_INDEX,
                "pinky": self.mp_pose.PoseLandmark.RIGHT_PINKY,
            }
        else:
            return {
                "shoulder": self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                "elbow": self.mp_pose.PoseLandmark.LEFT_ELBOW,
                "index": self.mp_pose.PoseLandmark.LEFT_INDEX,
                "pinky": self.mp_pose.PoseLandmark.LEFT_PINKY,
            }

    def get_coordinates(self, landmarks_data, depth_frame, w, h, intrinsics):
        """Get 3D coordinates for this arm"""
        shoulder = get_3D_coordinates(
            landmarks_data[self.landmark_indices["shoulder"]],
            depth_frame,
            w,
            h,
            intrinsics,
        )
        elbow = get_3D_coordinates(
            landmarks_data[self.landmark_indices["elbow"]],
            depth_frame,
            w,
            h,
            intrinsics,
        )
        hand = get_3D_coordinates_of_hand(
            landmarks_data[self.landmark_indices["index"]],
            landmarks_data[self.landmark_indices["pinky"]],
            depth_frame,
            w,
            h,
            intrinsics,
        )
        return shoulder, elbow, hand

    def get_joint_array(self) -> np.ndarray:
        """Get current joint positions as a numpy array

        Returns:
            np.ndarray: Array of current joint positions (excluding gripper)
        """
        # Define joint names based on arm side prefix
        joint_names = [
            f"{self.prefix}shoulder_pitch",
            f"{self.prefix}shoulder_roll",
            f"{self.prefix}arm_yaw",
            f"{self.prefix}elbow_pitch",
            f"{self.prefix}forearm_yaw",
            f"{self.prefix}wrist_pitch",
            f"{self.prefix}wrist_roll",
        ]

        # Get current joint angles directly from the arm
        return np.array(
            [
                getattr(getattr(self.arm, name), "present_position")
                for name in joint_names
            ]
        )

    def process_new_position(
        self, target_ee_coord: np.ndarray
    ) -> Tuple[bool, np.ndarray]:
        """Process a new target end effector position and determine if movement is needed

        Args:
            target_ee_coord: The target hand position in Reachy's coordinate system

        Returns:
            Tuple containing:
                bool: True if the position has changed enough to require an update
                np.ndarray: The smoothed target position
        """
        # TODO: this is currently replacing a Kalman filter
        # Apply position smoothing
        self.position_history.append(target_ee_coord)
        if len(self.position_history) > self.smoothing_buffer_size:
            self.position_history.pop(0)

        # Compute EMA for smoother position
        smoothed_position = (
            self.position_alpha * target_ee_coord
            + (1 - self.position_alpha) * self.prev_hand_pos
        )

        # Check if the desired position is different from current position
        current_ee_pose_matrix = self.arm.forward_kinematics()
        current_pos = current_ee_pose_matrix[:3, 3]
        already_at_position = np.allclose(current_pos, smoothed_position, atol=0.03)

        # Check if the position has changed significantly from the previous position
        should_update_position = (
            not np.allclose(self.prev_hand_pos, smoothed_position, atol=0.02)
            and not already_at_position
        )

        # Update previous position if we're going to move
        if should_update_position:
            self.prev_hand_pos = smoothed_position

        return should_update_position, smoothed_position

    def calculate_joint_positions(self, target_position: np.ndarray) -> bool:
        """Calculate new joint positions using inverse kinematics

        Args:
            target_position: The target hand position

        Returns:
            bool: True if calculation was successful
        """
        try:
            # Get the current transformation matrix
            transform_matrix = self.arm.forward_kinematics()

            # Set the target position in the transformation matrix
            transform_matrix[:3, 3] = target_position

            # Compute IK with current joint positions as starting point
            joint_pos = self.arm.inverse_kinematics(
                transform_matrix, q0=self.joint_array
            )

            # Get joint names for this arm
            joint_names = [
                f"{self.prefix}shoulder_pitch",
                f"{self.prefix}shoulder_roll",
                f"{self.prefix}arm_yaw",
                f"{self.prefix}elbow_pitch",
                f"{self.prefix}forearm_yaw",
                f"{self.prefix}wrist_pitch",
                f"{self.prefix}wrist_roll",
            ]

            # Apply rate limiting and update joint dictionary
            for i, (name, value) in enumerate(zip(joint_names, joint_pos)):
                # Apply rate limiting
                limited_change = np.clip(
                    value - self.joint_array[i], -self.max_change, self.max_change
                )
                self.joint_array[i] += limited_change
                self.joint_dict[name] = self.joint_array[i]

            # Handle gripper separately - maintain closed
            self.joint_dict[f"{self.prefix}gripper"] = 0

            return True
        except Exception as e:
            print(f"{self.side.capitalize()} arm IK calculation error: {e}")
            return False
