import numpy as np
from typing import Tuple, Literal

from src.utils.hands import calculate_2D_distance
from src.utils.three_d import get_3D_coordinates, get_3D_coordinates_of_hand
from src.models.custom_ik import inverse_kinematics_fixed_wrist
from config.CONSTANTS import get_finger_tips
from statistics import mode
from src.models.kalmann import KalmanFilter3D
from enum import Enum


class HAND_STATUS(Enum):
    """Enum representing the state of the robot's hand/gripper"""

    CLOSED = 1
    CLOSING = 2
    OPEN = 3


class ShadowArm:
    """Class for controlling and tracking a Reachy robot arm.

    Handles arm movement, hand state tracking, and force-based gripper control.
    Uses Kalman filtering for smooth motion and MediaPipe for pose estimation.
    """

    def __init__(
        self,
        reachy,
        reachy_arm,
        side: Literal["right", "left"],
        smoothing_buffer_size,
        position_alpha,
        max_change,
        mp_pose,
    ):
        # Core robot components
        self.reachy = reachy
        self.arm = reachy_arm
        self.side: Literal["right", "left"] = side
        self.prefix = f"{side[0]}_"  # "r_" or "l_"
        self.mp_pose = mp_pose
        self.landmark_indices = self.get_landmark_indices()

        # Movement tracking and smoothing
        self.joint_array = self.get_joint_array()
        self.joint_dict = {}
        self.position_history = []
        self.smoothing_buffer_size = smoothing_buffer_size
        self.position_alpha = position_alpha  # For EMA position smoothing
        self.max_change = max_change  # maximum change in degrees per joint per update
        self.prev_target_hand_position = self.arm.forward_kinematics()[:3, 3]

        # Kalman filters for position smoothing
        self.kf_shoulder = KalmanFilter3D()
        self.kf_elbow = KalmanFilter3D()
        self.kf_wrist = KalmanFilter3D()

        # Hand state tracking
        self.hand_closed = HAND_STATUS.OPEN
        self.hand_closed_history = []
        self._hand_history_size = 3

        # Force gripper configuration
        self.force_max_tolerance = 0
        self.force_min_tolerance = 2000000
        self.gripper_movement_threshold = 5 if side == "right" else -5
        self.open_gripper_value = -60 if self.side == "right" else 60
        self.closed_gripper_value = 12 if self.side == "right" else -12

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

    def apply_kalman_filter_to_coordinates(self, shoulder, elbow, hand):
        """Apply Kalman filter to the coordinates of the arm"""
        shoulder = self.kf_shoulder.update(shoulder)
        elbow = self.kf_elbow.update(elbow)
        hand = self.kf_wrist.update(hand)

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
        self,
        target_ee_coord: np.ndarray,
        target_pos_tolerance: float = 0.03,
        movement_min_tolerance: float = 0.02,
    ) -> Tuple[bool, np.ndarray]:
        """Process a new target end effector position and determine if movement is needed.

        Args:
            target_ee_coord: Target hand position in Reachy's coordinate system
            target_pos_tolerance: Maximum allowed distance from target position
            movement_min_tolerance: Minimum movement required to trigger an update

        Returns:
            Tuple containing:
                bool: True if position has changed enough to require an update
                np.ndarray: The smoothed target position
        """
        smoothed_position = target_ee_coord

        # Get current end effector position
        current_ee_pose_matrix = self.arm.forward_kinematics()
        actual_current_position = current_ee_pose_matrix[:3, 3]

        # Check if we're already at the target position
        already_at_position = np.allclose(
            actual_current_position, smoothed_position, atol=target_pos_tolerance
        )

        # Check if new position is significantly different from previous target
        should_update_position = (
            not np.allclose(
                self.prev_target_hand_position,
                smoothed_position,
                atol=movement_min_tolerance,
            )
            and not already_at_position
        )

        if should_update_position:
            self.prev_target_hand_position = smoothed_position

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

            # ! The fact that we're leaving the orientation unchanged might give strange results
            # Set the target position in the transformation matrix
            transform_matrix[:3, 3] = target_position

            # Compute IK with current joint positions as starting point
            joint_pos = self.arm.inverse_kinematics(
                transform_matrix, q0=self.get_joint_array()
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
            # self.joint_dict[f"{self.prefix}gripper"] = 0

            return True
        except Exception as e:
            print(f"{self.side.capitalize()} arm IK calculation error: {e}")
            return False

    def calculate_joint_positions_custom_ik(
        self,
        target_ee_position: np.ndarray,
        target_elbow_position: np.ndarray,
        elbow_weight: float,
    ) -> bool:
        """Calculate new joint positions using inverse kinematics

        Args:
            target_ee_position: The target hand position
            target_elbow_position: The target elbow position
            elbow_weight: The weight of the elbow in the IK calculation

        Returns:
            bool: True if calculation was successful
        """
        try:
            # Compute IK
            joint_pos = inverse_kinematics_fixed_wrist(
                ee_coords=target_ee_position,
                elbow_coords=target_elbow_position,
                initial_guess=self.joint_array,
                elbow_weight=elbow_weight,
                who="reachy",
                length=[0.28, 0.25, 0.075],
                side=self.side,
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

            return True
        except Exception as e:
            print(f"{self.side.capitalize()} arm IK calculation error: {e}")
            return False

    def get_hand_closedness(self, hand_landmarks, mp_hands, conservative=True):
        if conservative:
            is_new_landmark_closed = self._is_hand_half_closed(hand_landmarks, mp_hands)
        else:
            is_new_landmark_closed = self._is_hand_closed(hand_landmarks, mp_hands)

        self.hand_closed_history.append(is_new_landmark_closed)
        if len(self.hand_closed_history) > self._hand_history_size:
            self.hand_closed_history.pop(0)

        if (
            mode(self.hand_closed_history) is True
        ):  # this should never return HAND_STATUS.CLOSING
            return HAND_STATUS.CLOSED
        else:
            return HAND_STATUS.OPEN

    def _is_hand_closed(self, hand_landmarks, mp_hands):
        palm_base = hand_landmarks.landmark[0]
        open_fingers = 0
        for tip in get_finger_tips(mp_hands):
            if calculate_2D_distance(
                hand_landmarks.landmark[tip], palm_base
            ) > calculate_2D_distance(hand_landmarks.landmark[tip - 3], palm_base):
                open_fingers += 1
        return open_fingers <= 2

    def _is_hand_half_closed(self, hand_landmarks, mp_hands):
        """
        Can be used as a less conservative version of is_hand_closed
        """
        palm_base = hand_landmarks.landmark[0]
        half_closed_fingers = 0
        for tip in get_finger_tips(mp_hands):
            tip_to_palm = calculate_2D_distance(hand_landmarks.landmark[tip], palm_base)
            if tip_to_palm < (
                calculate_2D_distance(hand_landmarks.landmark[tip - 2], palm_base)
            ):
                half_closed_fingers += 1
        return (
            half_closed_fingers >= 3
        )  # Considérer la main entreouverte si au moins 3 doigts sont entreouverts

    def get_force_reading(self) -> float:
        """Get the current force reading from the gripper force sensor

        Returns:
            float: Current force reading in grams
        """
        if self.side == "right":
            return self.reachy.force_sensors.r_force_gripper.force
        else:
            return self.reachy.force_sensors.l_force_gripper.force

    def close_hand(self):
        """Close the gripper using force feedback control.

        Uses force sensor readings to determine when to stop closing
        to prevent damage to the gripper or grasped object.
        """
        current_force = self.get_force_reading()
        print("FORCE READING: ", current_force)

        # Initialize gripper position if not set
        if self.joint_dict.get(f"{self.prefix}gripper") is None:
            self.joint_dict[f"{self.prefix}gripper"] = getattr(
                getattr(self.arm, f"{self.prefix}gripper"), "present_position"
            )

        # Handle force-based gripper control
        if (self.side == "right" and current_force > 0) or (
            self.side == "left" and current_force < 0
        ):  # Too much force - stop closing
            value = (
                self.joint_dict[f"{self.prefix}gripper"]
                - self.gripper_movement_threshold
            )
            self.hand_closed = HAND_STATUS.CLOSING

        elif abs(current_force) <= self.force_min_tolerance and abs(
            current_force
        ) >= abs(
            self.force_max_tolerance
        ):  # Force in acceptable range - maintain position
            value = self.joint_dict[f"{self.prefix}gripper"]
            self.hand_closed = HAND_STATUS.CLOSED

        else:  # Too little force - continue closing
            value = (
                self.joint_dict[f"{self.prefix}gripper"]
                + self.gripper_movement_threshold
            )
            self.hand_closed = HAND_STATUS.CLOSING
            # Apply limits to prevent over-closing
            if self.side == "right":
                value = min(value, self.closed_gripper_value)
            else:
                value = max(value, self.closed_gripper_value)

        self.joint_dict[f"{self.prefix}gripper"] = value

    def open_hand(self):
        """Open the gripper to its maximum open position."""
        self.hand_closed = HAND_STATUS.OPEN
        self.joint_dict[f"{self.prefix}gripper"] = self.open_gripper_value
