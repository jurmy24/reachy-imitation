import numpy as np
from config.CONSTANTS import CAMERA_TO_REACHY_X, CAMERA_TO_REACHY_Z
from config.CONSTANTS import (
    REACHY_R_SHOULDER_COORDINATES,
    REACHY_L_SHOULDER_COORDINATES,
    LEN_REACHY_ARM,
)


# Helper function to convert a landmark to 3D coordinates from the camera's perspective using the human coordinate system
def get_3D_coordinates(landmark, depth_frame, w, h, intrinsics):
    """Convert a landmark to 3D coordinates using depth information.

    Note that this function converts the camera frame to the human frame whereas the origin remains the same.

    Transforms from camera coordinates to robot coordinates:
    Human x = -Camera depth
    Human y = -Camera x
    Human z = -Camera y

    Args:
        landmark: Either a landmark object with x, y, z attributes or
                 a numpy array/list with [x, y, z] coordinates (normalized)
        depth_frame: The depth frame from the camera
        w: Image width
        h: Image height
        intrinsics: Camera intrinsic parameters
    """
    # Handle both landmark objects and numpy arrays/lists
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        cx, cy = int(landmark.x * w), int(landmark.y * h)
    else:
        # Assume it's a numpy array or list with [x, y, z]
        cx, cy = int(landmark[0] * w), int(landmark[1] * h)

    # Check if pixel coordinates are within image bounds
    if 0 <= cx < w and 0 <= cy < h:
        depth = depth_frame.get_distance(cx, cy)
        if depth > 0:  # Ensure depth is valid
            # Get camera intrinsic parameters
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy

            # Convert to camera 3D coordinates
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy

            # Transform to robot coordinate system
            # TODO: based on the camera's system, it should really be -z, -x, -y
            return np.array([-depth, x, -y])

    # Default return if coordinates are invalid
    return np.array([0, 0, 0])


# Helper function to convert a landmark to 3D coordinates with Reachy's frame and Reachy at the origin instead of the camera
def get_3D_coordinates_reachy_perspective(landmark, depth_frame, w, h, intrinsics):
    """Convert a landmark to 3D coordinates using depth information.

    Transforms from camera coordinates to robot coordinates:
    - Reachy x = Camera depth
    - Reachy y = -Camera x
    - Reachy z = -Camera y
    """
    cx, cy = int(landmark.x * w), int(landmark.y * h)

    # Check if pixel coordinates are within image bounds
    if 0 <= cx < w and 0 <= cy < h:
        depth = depth_frame.get_distance(cx, cy)
        if depth > 0:  # Ensure depth is valid
            # Get camera intrinsic parameters
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy

            # Convert to camera 3D coordinates
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy

            # Transform to robot coordinate system
            return np.array([depth + CAMERA_TO_REACHY_X, -x, -y + CAMERA_TO_REACHY_Z])

    # Default return if coordinates are invalid
    return np.array([0, 0, 0])


def get_3D_coordinates_of_hand(
    index_landmark, pinky_landmark, depth_frame, w, h, intrinsics
):
    # Calculate average coordinates
    x_pixels = (index_landmark.x + pinky_landmark.x) / 2
    y_pixels = (index_landmark.y + pinky_landmark.y) / 2
    z_zoom_factor = (index_landmark.z + pinky_landmark.z) / 2

    # Create a numpy array with the averaged values
    avg_coords = np.array([x_pixels, y_pixels, z_zoom_factor])

    # Call get_3D_coordinates with all required parameters
    return get_3D_coordinates(avg_coords, depth_frame, w, h, intrinsics)


def get_reachy_coordinates(point, shoulder, sf, side="right"):
    """

    Args:
        point: The point in camera-relative coordinates and human frame
        shoulder: The shoulder (on same side as the point) in camera-relative coordinates and human frame
        sf: The scaling factor
        arm: Either "right" or "left" to specify which shoulder

    Returns:
        The point, now relative to Reachy's origin and scaled
    """
    if side.lower() == "left":
        return (point - shoulder) * sf + REACHY_L_SHOULDER_COORDINATES
    else:
        return (point - shoulder) * sf + REACHY_R_SHOULDER_COORDINATES
