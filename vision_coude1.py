"""Calcul des angles articulaires en 3D"""

import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import math

# Initialiser MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()  # Détection du corps entier
mp_draw = mp.solutions.drawing_utils  # Outils de dessin pour annoter les images

# Configuration de la caméra Intel RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Aligner les flux couleur et profondeur
align = rs.align(rs.stream.color)


def calculate_angle_3d(point1, point2, point3):
    """
    Calculer l'angle entre trois points dans l'espace 3D.
    :param point1: Coordonnées du premier point [x, y, z]
    :param point2: Coordonnées du second point [x, y, z] (sommet de l'angle)
    :param point3: Coordonnées du troisième point [x, y, z]
    :return: Angle en degrés
    """
    # Convertir les points en vecteurs
    vector1 = np.array(
        [point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]]
    )
    vector2 = np.array(
        [point3[0] - point2[0], point3[1] - point2[1], point3[2] - point2[2]]
    )

    # Calculer le produit scalaire et les normes des vecteurs
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # Éviter les divisions par zéro
    if norm1 == 0 or norm2 == 0:
        return 0

    # Calcul de l'angle (cosinus) et conversion en degrés
    cosine_angle = dot_product / (norm1 * norm2)
    cosine_angle = max(
        -1.0, min(1.0, cosine_angle)
    )  # Limiter entre -1 et 1 pour éviter les erreurs numériques
    angle_rad = math.acos(cosine_angle)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def get_3d_coordinates(landmark):
    """
    Obtenir les coordonnées 3D (x, y, z) d'un landmark en utilisant la caméra de profondeur.
    :param landmark: Landmark de MediaPipe (avec des coordonnées normalisées)
    :return: Coordonnées 3D en mètres [x, y, z]
    """
    cx, cy = int(landmark.x * w), int(landmark.y * h)  # Convertir en coordonnées pixels
    if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
        depth = depth_frame.get_distance(cx, cy)  # Obtenir la profondeur (z)
        if depth > 0:  # Vérifier que la profondeur est valide
            # Récupérer les paramètres intrinsèques de la caméra
            intrinsics = (
                pipeline.get_active_profile()
                .get_stream(rs.stream.depth)
                .as_video_stream_profile()
                .get_intrinsics()
            )
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy

            # Convertir les coordonnées en mètres
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy
            return [x, y, depth]
    return [0, 0, 0]  # Si les coordonnées ne sont pas valides


try:
    while True:
        # Capturer les trames de la caméra RealSense
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # Aligner les trames

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())  # Image couleur
        depth_image = np.asanyarray(depth_frame.get_data())  # Image de profondeur

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        pose_results = pose.process(rgb_image)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark  # Récupérer les landmarks
            h, w, c = color_image.shape  # Dimensions de l'image couleur

            # Points pour le bras droit
            r_shoulder = get_3d_coordinates(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            )
            r_elbow = get_3d_coordinates(
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            )
            r_wrist = get_3d_coordinates(
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            )

            # Calculer l'angle du coude droit
            r_elbow_angle = calculate_angle_3d(r_shoulder, r_elbow, r_wrist)

            # Afficher l'angle sur l'image
            cv2.putText(
                color_image,
                f"Elbow Angle: {int(r_elbow_angle)} deg",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        mp_draw.draw_landmarks(
            color_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        cv2.imshow("RealSense Hands and Arms with 3D Angles", color_image)

        # Quitter la boucle si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Libérer les ressources
    pipeline.stop()
    cv2.destroyAllWindows()
