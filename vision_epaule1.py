"""Premier test d analyse des angles de l'épaule par caméra 3D"""

import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs

# Initialiser MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Configuration de la caméra Intel RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Aligner les flux couleur et profondeur
align = rs.align(rs.stream.color)

# Fonction pour calculer l'angle de pitch et roll de l'épaule


def calcul_angles(r_shoulder, r_hip, r_elbow):
    # Définition des points (en 3D)
    r_shoulder = np.array(r_shoulder)  # Remplacer par les coordonnées réelles
    r_hip = np.array(r_hip)  # Remplacer par les coordonnées réelles
    r_elbow = np.array(r_elbow)  # Remplacer par les coordonnées réelles

    # Calcul des vecteurs relatifs
    vector_shoulder_to_hip = r_hip - r_shoulder
    vector_shoulder_to_elbow = r_elbow - r_shoulder

    # Normes des vecteurs
    d_shoulder_to_hip = np.linalg.norm(vector_shoulder_to_hip)
    d_shoulder_to_elbow = np.linalg.norm(vector_shoulder_to_elbow)

    # Calcul de l'angle entre les vecteurs (utilisation du produit scalaire)
    cos_angle = np.dot(vector_shoulder_to_hip, vector_shoulder_to_elbow) / (
        d_shoulder_to_hip * d_shoulder_to_elbow
    )
    # L'angle entre le bras et l'avant-bras
    angle_between = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # Calcul du pitch et roll en utilisant la décomposition du vecteur 3D
    # Calcul du pitch (rotation autour de l'axe Y)
    pitch = np.arctan2(
        vector_shoulder_to_hip[2],
        np.sqrt(vector_shoulder_to_hip[0] ** 2 + vector_shoulder_to_hip[1] ** 2),
    )

    # Calcul du roll (rotation autour de l'axe Z)
    roll = np.arctan2(vector_shoulder_to_elbow[1], vector_shoulder_to_elbow[0])

    # Angles en degrés pour la commande des actionneurs
    shoulder_pitch_deg = np.degrees(pitch)
    shoulder_roll_deg = np.degrees(roll)

    # Retourner les angles
    return shoulder_pitch_deg, shoulder_roll_deg


def get_3d_coordinates(landmark):
    """Obtenir les coordonnées x, y, z pour un landmark en utilisant la caméra de profondeur."""
    cx, cy = int(landmark.x * w), int(landmark.y * h)
    if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
        depth = depth_frame.get_distance(cx, cy)
        if depth > 0:  # Assurez-vous que la profondeur est valide
            intrinsics = (
                pipeline.get_active_profile()
                .get_stream(rs.stream.depth)
                .as_video_stream_profile()
                .get_intrinsics()
            )
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy

            # Convertir les coordonnées en 3D (mètres)
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy
            return [x, y, depth]
    return [0, 0, 0]


try:
    while True:
        # Capturer les trames de la caméra RealSense
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convertir en image NumPy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convertir l'image couleur en RGB pour MediaPipe
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Détection du corps (pose)
        pose_results = pose.process(rgb_image)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            h, w, c = color_image.shape

            # Points pour le bras droit
            r_shoulder = get_3d_coordinates(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            )
            r_elbow = get_3d_coordinates(
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            )
            r_hip = get_3d_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])

            # Calcul des angles pour l'épaule droite
            r_shoulder_pitch, r_shoulder_roll = calcul_angles(
                r_shoulder, r_hip, r_elbow
            )

            # Affichage des angles sur l'image
            cv2.putText(
                color_image,
                f"Pitch: {int(r_shoulder_pitch)} deg",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                color_image,
                f"Roll: {int(r_shoulder_roll)} deg",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        # Dessiner les annotations pour le corps
        mp_draw.draw_landmarks(
            color_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Afficher l'image
        cv2.imshow("RealSense Body Pose and Shoulder Angles", color_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Libérer les ressources
    pipeline.stop()
    cv2.destroyAllWindows()
