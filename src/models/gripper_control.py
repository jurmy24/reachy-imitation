import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from src.utils.hands import flip_hand_labels

# Initialize MediaPipe Hands for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, 2)  # Max 2 hands, no static image mode
mp_draw = mp.solutions.drawing_utils

# Constants for UI text positioning
TOP_LEFT_X_LABEL = 10
TOP_LEFT_Y_LABEL = 30

# Confidence thresholds for hand detection
HANDEDNESS_CERTAINTY_UPPER = 0.9  # High confidence threshold
HANDEDNESS_CERTAINTY_LOWER = 0.7  # Minimum confidence threshold

# MediaPipe landmark indices for finger tips
FINGER_TIPS = [
    mp_hands.HandLandmark.THUMB_TIP.value,
    mp_hands.HandLandmark.INDEX_FINGER_TIP.value,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value,
    mp_hands.HandLandmark.RING_FINGER_TIP.value,
    mp_hands.HandLandmark.PINKY_TIP.value,
]


def calculate_distance_with_depth(point1, point2):
    # this should account for depth
    return np.sqrt(
        (point1[0] - point2[0]) ** 2
        + (point1[1] - point2[1]) ** 2
        + (point1[2] - point2[2]) ** 2
    )


def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def is_hand_open(hand_landmarks):
    """Check if hand is open by comparing finger tip distances to palm base"""
    palm_base = hand_landmarks.landmark[0]
    open_fingers = 0
    for tip in FINGER_TIPS:
        if calculate_distance(
            hand_landmarks.landmark[tip], palm_base
        ) > calculate_distance(hand_landmarks.landmark[tip - 3], palm_base):
            open_fingers += 1
    return open_fingers >= 3


def is_hand_half_open(hand_landmarks):
    # Indices des points de repère pour les pointes des doigts et le centre de la paume
    # Point de repère pour le centre de la paume
    palm_base = hand_landmarks.landmark[0]

    open_fingers = 0
    for tip in FINGER_TIPS:
        if calculate_distance(
            hand_landmarks.landmark[tip], palm_base
        ) > calculate_distance(hand_landmarks.landmark[tip - 2], palm_base):
            open_fingers += 1
    return (
        open_fingers >= 3
    )  # Considérer la main ouverte si au moins 3 doigts sont ouverts


def is_hand_closed(hand_landmarks):
    # Indices des points de repère pour les pointes des doigts et le centre de la paume
    # Point de repère pour le centre de la paume
    palm_base = hand_landmarks.landmark[0]

    open_fingers = 0
    for tip in FINGER_TIPS:
        if calculate_distance(
            hand_landmarks.landmark[tip], palm_base
        ) > calculate_distance(hand_landmarks.landmark[tip - 3], palm_base):
            open_fingers += 1
    return (
        open_fingers <= 2
    )  # Considérer la main ferme si au moins 3 doigts sont ouverts


# or half close, depending on your vision, is your glass half empty or half full
def is_hand_half_closed(hand_landmarks):
    # Indices des points de repère pour les pointes des doigts et le centre de la paume

    palm_base = hand_landmarks.landmark[0]
    half_closed_fingers = 0
    for tip in FINGER_TIPS:
        tip_to_palm = calculate_distance(hand_landmarks.landmark[tip], palm_base)
        if tip_to_palm < (
            calculate_distance(hand_landmarks.landmark[tip - 2], palm_base)
        ):
            half_closed_fingers += 1
    return (
        half_closed_fingers >= 3
    )  # Considérer la main entreouverte si au moins 3 doigts sont entreouverts


def get_3d_coordinates(landmark):
    """Convert 2D landmark coordinates to 3D using depth camera data"""
    cx, cy = int(landmark.x * w), int(landmark.y * h)
    if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
        depth = depth_frame.get_distance(cx, cy)
        if depth > 0:
            intrinsics = (
                pipeline.get_active_profile()
                .get_stream(rs.stream.depth)
                .as_video_stream_profile()
                .get_intrinsics()
            )
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy

            # Convert to 3D coordinates (meters)
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy
            return [x, y, depth]
    return [0, 0, 0]


def flip_hand_labels(hand_label: str):
    """Mediapipe doesn't automatically flip to account for camera mirroring"""
    if hand_label == "Right":
        return "Gauche"
    else:
        return "Droit"
