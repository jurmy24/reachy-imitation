# This script extracts 3D points from a human using a depth camera (Intel RealSense D435i)
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
from src.utils.three_d import get_3D_coordinates_reachy_perspective, get_3D_coordinates

# # Initialize MediaPipe for hand and body point map detection
# mp_hands = mp.solutions.hands
# mp_pose = mp.solutions.pose

# # Create model instances with optimized parameters
# hands = mp_hands.Hands(
#     static_image_mode=False,  # Faster for video streams
#     max_num_hands=2,  # We only need two hands max
# )

# pose = mp_pose.Pose(
#     static_image_mode=False,  # Faster for video streams
#     smooth_landmarks=True,
# )

# # Configure intel RealSense camera (color and depth streams)
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# # Align depth frame to color frame
# align = rs.align(rs.stream.color)
# profile = pipeline.start(config)

# # Cache intrinsics for repeated use
# intrinsics = (
#     profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
# )


def calculate_arm_coordinates(
    pose, hands, mp_pose, mp_hands, intrinsics, rgb_image, depth_frame, w, h, arm
):
    # Initialize arm coordinates for this frame
    right_arm_coordinates = {}
    left_arm_coordinates = {}

    # Run computer vision pose and hand detection
    pose_results = pose.process(rgb_image)
    hand_results = hands.process(rgb_image)

    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark

        # Process right arm if requested
        if arm == "right" or arm == "both":
            # Get right arm joint positions
            right_shoulder = get_3D_coordinates(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                depth_frame,
                w,
                h,
                intrinsics,
            )
            right_elbow = get_3D_coordinates(
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                depth_frame,
                w,
                h,
                intrinsics,
            )
            right_wrist = get_3D_coordinates(
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
                depth_frame,
                w,
                h,
                intrinsics,
            )

            # Store points relative to shoulder
            right_arm_coordinates["shoulder_right"] = np.array(
                [0, 0, 0])  # Origin
            right_arm_coordinates["elbow_right"] = right_elbow - right_shoulder
            right_arm_coordinates["wrist_right"] = right_wrist - right_shoulder

        # Process left arm if requested
        if arm == "left" or arm == "both":
            # Get left arm joint positions
            left_shoulder = get_3D_coordinates(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                depth_frame,
                w,
                h,
                intrinsics,
            )
            left_elbow = get_3D_coordinates(
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                depth_frame,
                w,
                h,
                intrinsics,
            )
            left_wrist = get_3D_coordinates(
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
                depth_frame,
                w,
                h,
                intrinsics,
            )

            # Store points relative to shoulder
            left_arm_coordinates["shoulder_left"] = np.array(
                [0, 0, 0])  # Origin
            left_arm_coordinates["elbow_left"] = left_elbow - left_shoulder
            left_arm_coordinates["wrist_left"] = left_wrist - left_shoulder

        # Process hands if detected
        if hand_results.multi_hand_landmarks:
            # Process hands when we have both hand and body landmarks
            right_hand_idx = -1
            left_hand_idx = -1

            # Find indices of right and left hands if handedness is available
            if hand_results.multi_handedness:
                for i, handedness in enumerate(hand_results.multi_handedness):
                    hand_type = handedness.classification[0].label
                    # FIXED: Swapping handedness because MediaPipe labels are flipped
                    # When MediaPipe says "Right", it's actually the user's left hand
                    # When MediaPipe says "Left", it's actually the user's right hand
                    if hand_type == "Right":
                        left_hand_idx = i  # This is actually the LEFT hand
                    elif hand_type == "Left":
                        right_hand_idx = i  # This is actually the RIGHT hand

            # If no handedness info, make assumptions based on available hands
            # This is a fallback but not as reliable as the handedness detection
            if right_hand_idx == -1 and left_hand_idx == -1:
                if len(hand_results.multi_hand_landmarks) >= 1:
                    right_hand_idx = 0
                if len(hand_results.multi_hand_landmarks) >= 2:
                    left_hand_idx = 1

            # Process right hand if available and requested
            if (arm == "right" or arm == "both") and right_hand_idx >= 0:
                right_hand = hand_results.multi_hand_landmarks[right_hand_idx]

                # Get finger MCP points
                index_mcp = get_3D_coordinates(
                    right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    depth_frame,
                    w,
                    h,
                    intrinsics,
                )
                pinky_mcp = get_3D_coordinates(
                    right_hand.landmark[mp_hands.HandLandmark.PINKY_MCP],
                    depth_frame,
                    w,
                    h,
                    intrinsics,
                )

                # Store relative to right shoulder
                right_arm_coordinates["index_mcp"] = index_mcp - right_shoulder
                right_arm_coordinates["pinky_mcp"] = pinky_mcp - right_shoulder

            # Process left hand if available and requested
            if (arm == "left" or arm == "both") and left_hand_idx >= 0:
                left_hand = hand_results.multi_hand_landmarks[left_hand_idx]

                # Get finger MCP points
                index_mcp = get_3D_coordinates(
                    left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    depth_frame,
                    w,
                    h,
                    intrinsics,
                )
                pinky_mcp = get_3D_coordinates(
                    left_hand.landmark[mp_hands.HandLandmark.PINKY_MCP],
                    depth_frame,
                    w,
                    h,
                    intrinsics,
                )

                # Store relative to left shoulder
                left_arm_coordinates["index_mcp"] = index_mcp - left_shoulder
                left_arm_coordinates["pinky_mcp"] = pinky_mcp - left_shoulder

    return right_arm_coordinates, left_arm_coordinates


def get_head_coordinates(pose, mp_pose, intrinsics, rgb_image, depth_frame, w, h):
    """
    Extract the 3D coordinates of the human head from a frame.
    Returns a tuple (x, y, z) representing the head position in meters.
    Returns None if no head is detected.
    """
    # Run pose detection
    pose_results = pose.process(rgb_image)

    if not pose_results.pose_landmarks:
        return None

    landmarks = pose_results.pose_landmarks.landmark

    # Use nose as the primary head point
    head_position = get_3D_coordinates_reachy_perspective(
        landmarks[mp_pose.PoseLandmark.NOSE], depth_frame, w, h, intrinsics
    )

    """    # If nose point is invalid (e.g., depth couldn't be measured)
    # try using the center point between the eyes
    if head_position is None or np.isnan(head_position).any():
        left_eye = get_3D_coordinates(
            landmarks[mp_pose.PoseLandmark.LEFT_EYE], depth_frame, w, h, intrinsics
        )

        right_eye = get_3D_coordinates(
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE], depth_frame, w, h, intrinsics
        )

        # If both eyes are detected, use their midpoint
        if (
            left_eye is not None
            and right_eye is not None
            and not np.isnan(left_eye).any()
            and not np.isnan(right_eye).any()
        ):
            head_position = (left_eye + right_eye) / 2 """

    return head_position


# def run(arm: Literal["right", "left", "both"] = "right"):
#     try:
#         while True:
#             # Get frames from RealSense camera
#             frames = pipeline.wait_for_frames()
#             aligned_frames = align.process(frames)
#             color_frame = aligned_frames.get_color_frame()
#             depth_frame = aligned_frames.get_depth_frame()

#             # Check if frames are valid, if not, skip
#             if not color_frame or not depth_frame:
#                 continue

#             # OpenCV uses BGR format, MediaPipe uses RGB format, so we convert the color image
#             color_image = np.asanyarray(color_frame.get_data())
#             rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

#             # Get height and width of the color image
#             h, w, _ = color_image.shape

#             right_arm_coordinates, left_arm_coordinates = calculate_3D_points(
#                 rgb_image, depth_frame, w, h, arm
#             )

#             # Initialize arm coordinates for this frame
#             right_arm_coordinates = {}
#             left_arm_coordinates = {}

#             # Run computer vision pose and hand detection
#             pose_results = pose.process(rgb_image)
#             hand_results = hands.process(rgb_image)

#             if pose_results.pose_landmarks:
#                 landmarks = pose_results.pose_landmarks.landmark

#                 # Process right arm if requested
#                 if arm == "right" or arm == "both":
#                     # Get right arm joint positions
#                     right_shoulder = get_3D_coordinates(
#                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
#                         depth_frame,
#                         w,
#                         h,
#                         intrinsics,
#                     )
#                     right_elbow = get_3D_coordinates(
#                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
#                         depth_frame,
#                         w,
#                         h,
#                         intrinsics,
#                     )
#                     right_wrist = get_3D_coordinates(
#                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
#                         depth_frame,
#                         w,
#                         h,
#                         intrinsics,
#                     )

#                     # Store points relative to shoulder
#                     right_arm_coordinates["shoulder_right"] = np.array(
#                         [0, 0, 0]
#                     )  # Origin
#                     right_arm_coordinates["elbow_right"] = right_elbow - right_shoulder
#                     right_arm_coordinates["wrist_right"] = right_wrist - right_shoulder

#                 # Process left arm if requested
#                 if arm == "left" or arm == "both":
#                     # Get left arm joint positions
#                     left_shoulder = get_3D_coordinates(
#                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
#                         depth_frame,
#                         w,
#                         h,
#                         intrinsics,
#                     )
#                     left_elbow = get_3D_coordinates(
#                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
#                         depth_frame,
#                         w,
#                         h,
#                         intrinsics,
#                     )
#                     left_wrist = get_3D_coordinates(
#                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
#                         depth_frame,
#                         w,
#                         h,
#                         intrinsics,
#                     )

#                     # Store points relative to shoulder
#                     left_arm_coordinates["shoulder_left"] = np.array(
#                         [0, 0, 0]
#                     )  # Origin
#                     left_arm_coordinates["elbow_left"] = left_elbow - left_shoulder
#                     left_arm_coordinates["wrist_left"] = left_wrist - left_shoulder

#                 # Process hands if detected
#                 if hand_results.multi_hand_landmarks:
#                     # Process hands when we have both hand and body landmarks
#                     right_hand_idx = -1
#                     left_hand_idx = -1

#                     # Find indices of right and left hands if handedness is available
#                     if hand_results.multi_handedness:
#                         for i, handedness in enumerate(hand_results.multi_handedness):
#                             hand_type = handedness.classification[0].label
#                             # FIXED: Swapping handedness because MediaPipe labels are flipped
#                             # When MediaPipe says "Right", it's actually the user's left hand
#                             # When MediaPipe says "Left", it's actually the user's right hand
#                             if hand_type == "Right":
#                                 left_hand_idx = i  # This is actually the LEFT hand
#                             elif hand_type == "Left":
#                                 right_hand_idx = i  # This is actually the RIGHT hand

#                     # If no handedness info, make assumptions based on available hands
#                     # This is a fallback but not as reliable as the handedness detection
#                     if right_hand_idx == -1 and left_hand_idx == -1:
#                         if len(hand_results.multi_hand_landmarks) >= 1:
#                             right_hand_idx = 0
#                         if len(hand_results.multi_hand_landmarks) >= 2:
#                             left_hand_idx = 1

#                     # Process right hand if available and requested
#                     if (arm == "right" or arm == "both") and right_hand_idx >= 0:
#                         right_hand = hand_results.multi_hand_landmarks[right_hand_idx]

#                         # Get finger MCP points
#                         index_mcp = get_3D_coordinates(
#                             right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
#                             depth_frame,
#                             w,
#                             h,
#                             intrinsics,
#                         )
#                         pinky_mcp = get_3D_coordinates(
#                             right_hand.landmark[mp_hands.HandLandmark.PINKY_MCP],
#                             depth_frame,
#                             w,
#                             h,
#                             intrinsics,
#                         )

#                         # Store relative to right shoulder
#                         right_arm_coordinates["index_mcp"] = index_mcp - right_shoulder
#                         right_arm_coordinates["pinky_mcp"] = pinky_mcp - right_shoulder

#                     # Process left hand if available and requested
#                     if (arm == "left" or arm == "both") and left_hand_idx >= 0:
#                         left_hand = hand_results.multi_hand_landmarks[left_hand_idx]

#                         # Get finger MCP points
#                         index_mcp = get_3D_coordinates(
#                             left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
#                             depth_frame,
#                             w,
#                             h,
#                             intrinsics,
#                         )
#                         pinky_mcp = get_3D_coordinates(
#                             left_hand.landmark[mp_hands.HandLandmark.PINKY_MCP],
#                             depth_frame,
#                             w,
#                             h,
#                             intrinsics,
#                         )

#                         # Store relative to left shoulder
#                         left_arm_coordinates["index_mcp"] = index_mcp - left_shoulder
#                         left_arm_coordinates["pinky_mcp"] = pinky_mcp - left_shoulder

#             # Display 3D coordinates on the image
#             y_offset = 30

#             # Display right arm coordinates
#             for name, coord in right_arm_coordinates.items():
#                 x, y, z = coord
#                 cv2.putText(
#                     color_image,
#                     f"R_{name}: ({x:.2f}, {y:.2f}, {z:.2f})m",
#                     (50, y_offset),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (0, 255, 0),  # Green for right arm
#                     2,
#                 )
#                 y_offset += 20

#             # Display left arm coordinates
#             for name, coord in left_arm_coordinates.items():
#                 x, y, z = coord
#                 cv2.putText(
#                     color_image,
#                     f"L_{name}: ({x:.2f}, {y:.2f}, {z:.2f})m",
#                     (50, y_offset),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (255, 0, 0),  # Blue for left arm
#                     2,
#                 )
#                 y_offset += 20

#             # Set window title based on which arm(s) is being tracked
#             window_title = "RealSense "
#             if arm == "right":
#                 window_title += "Right Arm"
#             elif arm == "left":
#                 window_title += "Left Arm"
#             elif arm == "both":
#                 window_title += "Both Arms"
#             window_title += " 3D Coordinates"

#             # Display the image
#             cv2.imshow(window_title, color_image)

#             # Quit if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#     finally:
#         pipeline.stop()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     run(arm="right")
