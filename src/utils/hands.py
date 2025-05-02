import numpy as np


def flip_hand_labels(hand_label: str):
    """Mediapipe doesn't automatically flip to account for camera mirroring"""
    if hand_label.lower() == "right":
        return "left"
    else:
        return "right"


def calculate_2D_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
