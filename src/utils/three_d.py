import numpy as np


# Helper function to convert a landmark to 3D coordinates
def get_3D_coordinates(landmark, depth_frame, w, h, intrinsics):
    """Convert a landmark to 3D coordinates using depth information.

    Transforms from camera coordinates to robot coordinates:
    - Human x = -Camera depth
    - Human y = -Camera x
    - Human z = -Camera y
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
            return np.array([-depth, -x, -y])

    # Default return if coordinates are invalid
    return np.array([0, 0, 0])

# Helper function to convert a landmark to 3D coordinates


def get_3D_coordinates_reachy_perspective(landmark, depth_frame, w, h, intrinsics):
    """Convert a landmark to 3D coordinates using depth information.

    Transforms from camera coordinates to robot coordinates:
    - Robot x = Camera depth
    - Robot y = -Camera x
    - Robot z = -Camera y
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
            return np.array([depth-0.18,-x, 0.35-y])

    # Default return if coordinates are invalid
    return np.array([0, 0, 0])
