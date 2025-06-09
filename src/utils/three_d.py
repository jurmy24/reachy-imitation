import numpy as np
from config.CONSTANTS import CAMERA_TO_REACHY_X, CAMERA_TO_REACHY_Z
from config.CONSTANTS import (
    REACHY_R_SHOULDER_COORDINATES,
    REACHY_L_SHOULDER_COORDINATES,
)


def get_3D_coordinates(landmark, depth_frame, w, h, intrinsics):
    """Convert a landmark to 3D coordinates using depth information.

    This function transforms camera coordinates to human frame coordinates while keeping the same origin.
    The transformation follows these rules:
        Human x = -Camera depth
        Human y = -Camera x
        Human z = -Camera y

    Args:
        landmark: Either a landmark object with x, y, z attributes or
                 a numpy array/list with [x, y, z] coordinates (normalized)
        depth_frame: The depth frame from the camera
        w: Image width
        h: Image height
        intrinsics: Camera intrinsic parameters (fx, fy, ppx, ppy)

    Returns:
        np.array: 3D coordinates in human frame [x, y, z]
    """
    # Handle both landmark objects and numpy arrays/lists
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        cx, cy = int(landmark.x * w), int(landmark.y * h)
    else:
        cx, cy = int(landmark[0] * w), int(landmark[1] * h)

    # Check if pixel coordinates are within image bounds
    if 0 <= cx < w and 0 <= cy < h:
        depth = depth_frame.get_distance(cx, cy)
        if depth > 0:
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy

            # Convert to camera 3D coordinates
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy

            return np.array([-depth, x, -y])

    # Default return if coordinates are invalid
    return np.array([0, 0, 0])


def get_3D_coordinates_reachy_perspective(landmark, depth_frame, w, h, intrinsics):
    """Convert a landmark to 3D coordinates in Reachy's frame of reference.

    This function transforms camera coordinates to Reachy's frame, with Reachy at the origin.
    The transformation follows these rules:
        Reachy x = Camera depth + CAMERA_TO_REACHY_X
        Reachy y = -Camera x
        Reachy z = -Camera y + CAMERA_TO_REACHY_Z

    Args:
        landmark: A landmark object with x, y, z attributes
        depth_frame: The depth frame from the camera
        w: Image width
        h: Image height
        intrinsics: Camera intrinsic parameters (fx, fy, ppx, ppy)

    Returns:
        np.array: 3D coordinates in Reachy's frame [x, y, z]
    """
    cx, cy = int(landmark.x * w), int(landmark.y * h)

    # Check if pixel coordinates are within image bounds
    if 0 <= cx < w and 0 <= cy < h:
        depth = depth_frame.get_distance(cx, cy)
        if depth > 0:
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy

            # Convert to camera 3D coordinates
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy

            return np.array([depth + CAMERA_TO_REACHY_X, -x, -y + CAMERA_TO_REACHY_Z])

    # Default return if coordinates are invalid
    return np.array([0, 0, 0])


def get_3D_coordinates_of_hand(
    index_landmark, pinky_landmark, depth_frame, w, h, intrinsics
):
    """Calculate the 3D coordinates of the center of the hand.

    This function averages the positions of the index and pinky fingers to estimate
    the center of the hand, then converts this to 3D coordinates.

    Args:
        index_landmark: Landmark object for the index finger
        pinky_landmark: Landmark object for the pinky finger
        depth_frame: The depth frame from the camera
        w: Image width
        h: Image height
        intrinsics: Camera intrinsic parameters

    Returns:
        np.array: 3D coordinates of the hand center in human frame
    """
    # Calculate average coordinates
    x_pixels = (index_landmark.x + pinky_landmark.x) / 2
    y_pixels = (index_landmark.y + pinky_landmark.y) / 2
    z_zoom_factor = (index_landmark.z + pinky_landmark.z) / 2

    # Create a numpy array with the averaged values
    avg_coords = np.array([x_pixels, y_pixels, z_zoom_factor])

    # Convert to 3D coordinates
    return get_3D_coordinates(avg_coords, depth_frame, w, h, intrinsics)


def get_reachy_coordinates(point, shoulder, sf, side="right"):
    """Transform coordinates from camera-relative to Reachy-relative frame.

    This function:
    1. Subtracts the shoulder position to get relative coordinates
    2. Applies the scaling factor
    3. Adds the appropriate Reachy shoulder coordinates

    Args:
        point: The point in camera-relative coordinates and human frame
        shoulder: The shoulder (on same side as the point) in camera-relative coordinates and human frame
        sf: The scaling factor to match human and robot proportions
        side: Either "right" or "left" to specify which shoulder to use

    Returns:
        np.array: The point transformed to Reachy's coordinate frame
    """
    if side.lower() == "left":
        return (point - shoulder) * sf + REACHY_L_SHOULDER_COORDINATES
    else:
        return (point - shoulder) * sf + REACHY_R_SHOULDER_COORDINATES
