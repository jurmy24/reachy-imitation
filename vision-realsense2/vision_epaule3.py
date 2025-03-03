'''récupération pitch et roll épaule quand le buste est dans l'axe de la caméra'''

import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs

# Initialisation MediaPipe Pose
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

# Fonction pour obtenir la profondeur du landmark


def get_depth(landmark, depth_frame, w, h):
    cx, cy = int(landmark.x * w), int(landmark.y * h)
    if 0 <= cx < w and 0 <= cy < h:
        depth = depth_frame.get_distance(cx, cy)
        return depth if depth > 0 else None
    return None

# Fonction pour obtenir les coordonnées 3D


def get_3d_coordinates(landmark, depth_frame, w, h, intrinsics):
    cx, cy = int(landmark.x * w), int(landmark.y * h)
    if 0 <= cx < w and 0 <= cy < h:
        depth = depth_frame.get_distance(cx, cy)
        if depth > 0:
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy
            return np.array([x, y, depth])
    return np.array([0, 0, 0])

# Fonction pour calculer l'angle entre trois points 3D


def calculate_angle(v1, v2):

    produit_scalaire = np.dot(v1, v2)

    # Normes des vecteurs
    norme_v1 = np.linalg.norm(v1)
    norme_v2 = np.linalg.norm(v2)

    # Calcul du cosinus de l'angle
    cos_theta = produit_scalaire / (norme_v1 * norme_v2)

    # Sécurisation pour éviter des erreurs numériques
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calcul de l'angle en radians
    theta_radians = np.arccos(cos_theta)

    # Conversion en degrés
    theta_degres = np.degrees(theta_radians)

    return theta_degres


try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        h, w, _ = color_image.shape
        intrinsics = pipeline.get_active_profile().get_stream(
            rs.stream.depth).as_video_stream_profile().get_intrinsics()

        pose_results = pose.process(rgb_image)

        if pose_results.pose_landmarks:
            mp_draw.draw_landmarks(
                color_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            r_elbow = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            r_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            r_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

            r_elbow_3d = get_3d_coordinates(
                r_elbow, depth_frame, w, h, intrinsics)
            r_shoulder_3d = get_3d_coordinates(
                r_shoulder, depth_frame, w, h, intrinsics)
            r_hip_3d = get_3d_coordinates(r_hip, depth_frame, w, h, intrinsics)

            v1 = [r_elbow_3d[1]-r_shoulder_3d[1],
                  r_elbow_3d[2]-r_shoulder_3d[2]]
            v2 = [r_hip_3d[1]-r_shoulder_3d[1], r_hip_3d[2]-r_shoulder_3d[2]]
            pitch = calculate_angle(v1, v2)
            v3 = [r_elbow_3d[0]-r_shoulder_3d[0],
                  r_elbow_3d[1]-r_shoulder_3d[1]]
            v4 = [r_hip_3d[0]-r_shoulder_3d[0], r_hip_3d[1]-r_shoulder_3d[1]]
            roll = calculate_angle(v3, v4)

            if r_elbow_3d[2] > 0:
                cv2.putText(color_image, f'Pitch: {pitch:.2f} deg', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(color_image, f'Roll: {roll:.2f} deg', (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Right Elbow Depth and Distance', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
