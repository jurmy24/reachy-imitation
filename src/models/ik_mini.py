import numpy as np
from config.CONSTANTS import (
    REACHY_R_SHOULDER_COORDINATES,
    REACHY_L_SHOULDER_COORDINATES,
    LEN_REACHY_ARM,
)



def transform_to_shoulder_origin(point, right_shoulder):
    return point - right_shoulder


def scale_point(sf, point):
    return sf * point


def translate_to_reachy_origin(point, arm="right"):
    """
    Translate a point from shoulder-relative coordinates to Reachy's origin

    Args:
        point: The point in shoulder-relative coordinates
        arm: Either "right" or "left" to specify which shoulder

    Returns:
        The point translated to Reachy's origin
    """
    if arm.lower() == "left":
        return point + REACHY_L_SHOULDER_COORDINATES
    else:  # default to right arm
        return point + REACHY_R_SHOULDER_COORDINATES


def within_reachys_reach(point):
    return np.linalg.norm(point) <= LEN_REACHY_ARM


# async def test_reachy_ik(reachy, hand_sf):

#     prev_reachy_hand_right = reachy.r_arm.forward_kinematics()[0:3, 3]

#     goto_new_position = False
#     its = 0
#     pause_update = True

#     try:
#         while True:
#             its += 1
#             if its == 10:
#                 pause_update = False
#                 its = 0
#             else:
#                 pause_update = True

#             frames = pipeline.wait_for_frames()
#             aligned_frames = align.process(frames)

#             color_frame = aligned_frames.get_color_frame()
#             depth_frame = aligned_frames.get_depth_frame()

#             if not color_frame or not depth_frame:
#                 continue

#             color_image = np.asanyarray(color_frame.get_data())
#             depth_image = np.asanyarray(depth_frame.get_data())

#             rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

#             # Récupérer les paramètres intrinsèques de la caméra
#             intrinsics = (
#                 pipeline.get_active_profile()
#                 .get_stream(rs.stream.depth)
#                 .as_video_stream_profile()
#                 .get_intrinsics()
#             )
#             h, w, _ = color_image.shape

#             # Détection des mains et du corps
#             pose_results = pose.process(rgb_image)
#             # Récupérer les points du bras droit
#             if pose_results.pose_landmarks:
#                 landmarks = pose_results.pose_landmarks.landmark
#                 mp_draw.draw_landmarks(
#                     color_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
#                 )

#                 if (
#                     (
#                         0.3 > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility
#                         or not landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].HasField(
#                             "visibility"
#                         )
#                     )
#                     or (
#                         0.3 > landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].visibility
#                         or not landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].HasField(
#                             "visibility"
#                         )
#                     )
#                     or (
#                         0.3 > landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].visibility
#                         or not landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].HasField(
#                             "visibility"
#                         )
#                     )
#                 ):
#                     cv2.putText(
#                         color_image,
#                         f"je ne vois pas ton bras",
#                         (10, 70),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         (255, 255, 255),
#                         2,
#                     )
#                     cv2.imshow("RealSense Right Arm IK", color_image)

#                     # Quitter la boucle si la touche 'q' est pressée
#                     if cv2.waitKey(1) & 0xFF == ord("q"):
#                         break
#                     continue

#                 # ! NOTE: the second output in the np array used to be positive in get_3d_coordinates, check the code there
#                 human_right_shoulder = get_3D_coordinates(
#                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
#                     depth_frame,
#                     w,
#                     h,
#                     intrinsics,
#                 )
#                 mid_hand_landmark = average_landmarks(
#                     landmarks[mp_pose.PoseLandmark.RIGHT_INDEX],
#                     landmarks[mp_pose.PoseLandmark.RIGHT_PINKY],
#                 )
#                 human_hand_right = get_3D_coordinates_of_hand(
#                     mid_hand_landmark, depth_frame, w, h, intrinsics
#                 )
#                 human_hand_right = transform_to_shoulder_origin(
#                     human_hand_right, human_right_shoulder
#                 )
#                 hand_right = scale_point(hand_sf, human_hand_right)

#                 if not within_reachys_reach(hand_right):
#                     cv2.putText(
#                         color_image,
#                         f"reachy can't reachy there",
#                         (40, 70),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (0, 0, 255),
#                     )

#                     cv2.imshow("RealSense Right Arm IK", color_image)

#                     if cv2.waitKey(1) & 0xFF == ord("q"):
#                         break
#                     continue

#                 reachy_hand_right = translate_to_reachy_origin(hand_right)

#                 # Afficher les coordonnées 3D sur l'image
#                 x, y, z = human_hand_right
#                 cv2.putText(
#                     color_image,
#                     f"Human Hand: ({x:.2f}, {y:.2f}, {z:.2f})m",
#                     (10, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (255, 255, 255),
#                     2,
#                 )
#                 x, y, z = reachy_hand_right
#                 cv2.putText(
#                     color_image,
#                     f"Robot Hand: ({x:.2f}, {y:.2f}, {z:.2f})m",
#                     (10, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (255, 255, 255),
#                     2,
#                 )

#                 a = reachy.r_arm.forward_kinematics()
#                 if not (
#                     np.allclose(prev_reachy_hand_right, reachy_hand_right, atol=0.05)
#                 ):
#                     prev_reachy_hand_right = reachy_hand_right
#                     a[0, 3] = reachy_hand_right[0]
#                     a[1, 3] = reachy_hand_right[1]
#                     a[2, 3] = reachy_hand_right[2]
#                     goto_new_position = True
#                 else:
#                     goto_new_position = False

#                 joint_pos = reachy.r_arm.inverse_kinematics(a)

#                 i = 0
#                 for joint, pos in list(zip(get_ordered_joint_names(reachy), joint_pos)):
#                     cv2.putText(
#                         color_image,
#                         f"{joint}: {pos:.2f} deg",
#                         (400, 30 + i * 20),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         (0, 255, 255),
#                         2,
#                     )
#                     i += 1

#                 if goto_new_position and not pause_update:
#                     await goto_async(
#                         {
#                             joint: pos
#                             for joint, pos in list(
#                                 zip(get_ordered_joint_names(reachy), joint_pos)
#                             )
#                         },
#                         duration=1,
#                         interpolation_mode=InterpolationMode.MINIMUM_JERK,
#                     )
#             else:
#                 cv2.putText(
#                     color_image,
#                     f"je ne vois pas ton corps",
#                     (10, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (255, 255, 255),
#                     2,
#                 )

#             # Afficher l'image
#             cv2.imshow("RealSense Right Arm IK", color_image)

#             # Quitter la boucle si la touche 'q' est pressée
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#     finally:
#         pipeline.stop()
#         cv2.destroyAllWindows()

#     # Define RMSE/best
#     rmse = np.sqrt(np.mean(np.square(target_pos - target_ik)))

#     # Create a visualization output
#     if plots:
#         plt.figure(figsize=(10, 6))

#         # Plot original vs IK predictions
#         plt.subplot(2, 2, 1)
#         x_coord = range(len(get_zero_right_pos(reachy)))

#         # ... existing code ...


# if __name__ == "__main__":
#     hand_sf, elbow_sf = get_scale_factors(0.7, 0.3)
#     import asyncio

#     asyncio.run(test_reachy_ik(hand_sf))
#     if reachy:
#         goto(
#             ZERO_RIGHT_POS,
#             duration=1.5,
#             interpolation_mode=InterpolationMode.MINIMUM_JERK,
#         )
#         reachy.turn_off_smoothly("r_arm")
