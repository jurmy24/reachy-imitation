import numpy as np
import math
import mediapipe as mp
import pyrealsense2 as rs
import cv2
from points_to_shoulder import get_3d_coordinates
from scaling_joint_params import test_scale_factors, L_REACHY_FOREARM, L_REACHY_UPPERARM

L_REACHY_ARM = L_REACHY_FOREARM + L_REACHY_UPPERARM
REACHY_R_SHOULDER_TO_ORIGIN = np.array([0, -0.19, 0]) # Ask victor where these constants should live 

def transform_to_shoulder_origin(point, right_shoulder):
    return point - right_shoulder

def scale_point(sf, point):
    return sf * point

def translate_to_reachy_orogin(point):
    return point + REACHY_R_SHOULDER_TO_ORIGIN


def human_to_robot_coordinates(point, sf):
    """
    Convert human coordinates to robot coordinates
    Args:  
        point (np.array): 3D point in human coordinates (m)
            -> assumes scaled so shoulder is origin, coordinate frame is same as reachy's (i.e. x is depth, z height)
        sf (float): scale factor
    Returns:    
        np.array: 3D point in robot coordinates
    """
    point = scale_point(hand_sf, point)
    point = translate_to_reachy_orogin(point)
    return point


def test_reachy_ik(hand_sf):

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils

    # Configuration caméra RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    align = rs.align(rs.stream.color)

    try:
        while True:
            # Capturer les trames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Récupérer les paramètres intrinsèques de la caméra
            intrinsics = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            h, w, _ = color_image.shape

            # Détection des mains et du corps
            pose_results = pose.process(rgb_image)
            # Récupérer les points du bras droit
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark

                #if not (0.5 < pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility <= 1):


                right_shoulder = get_3d_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], depth_frame, w, h, intrinsics)
                wrist_right = get_3d_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST], depth_frame, w, h, intrinsics)
                wrist_right = transform_to_shoulder_origin(wrist_right, right_shoulder)
                reachy_wrist_right = human_to_robot_coordinates(wrist_right, hand_sf)
                # Afficher les coordonnées 3D sur l'image
                x, y, z = wrist_right
                cv2.putText(
                    color_image,
                    f"Human Wrist: ({x:.2f}, {y:.2f}, {z:.2f})m",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                x, y, z = reachy_wrist_right
                cv2.putText(
                    color_image,
                    f"Robot Wrist: ({x:.2f}, {y:.2f}, {z:.2f})m",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            # Afficher l'image
            cv2.imshow("RealSense Right Arm IK", color_image)

            # Quitter la boucle si la touche 'q' est pressée
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    hand_sf, elbow_sf = test_scale_factors()
    test_reachy_ik(hand_sf)


