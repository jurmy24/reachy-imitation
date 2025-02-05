"""Ce code est un test de controle du coude du bras droit du robot à partir d'une caméra 3D"""

import concurrent.futures
import time
import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import math

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

# Initialiser MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Aligner les flux couleur et profondeur
align = rs.align(rs.stream.color)

# Code d'initialisation (RealSense, MediaPipe, Reachy)


def calculate_angle_3d(point1, point2, point3):
    """
    Calculer l'angle entre trois points dans l'espace 3D.
    """
    # Convertir en vecteurs
    vector1 = np.array([point1[0] - point2[0], point1[1] -
                       point2[1], point1[2] - point2[2]])
    vector2 = np.array([point3[0] - point2[0], point3[1] -
                       point2[1], point3[2] - point2[2]])

    # Produit scalaire et normes
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # Éviter les divisions par zéro
    if norm1 == 0 or norm2 == 0:
        return 0

    # Calcul de l'angle en radians et conversion en degrés
    cosine_angle = dot_product / (norm1 * norm2)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))  # Limiter entre -1 et 1
    angle_rad = math.acos(cosine_angle)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

import numpy as np

def calculate_pitch_roll_yaw(point1,point2,point3):
    """
    Calcule les angles de pitch, roll et yaw de l'épaule à partir des coordonnées des points clés.
    
    Arguments :
        points (dict) : Un dictionnaire contenant les coordonnées des points clés.
                        {
                            "acromion": [x1, y1, z1],
                            "humérus": [x2, y2, z2],
                            "coracoïde": [x3, y3, z3]
                        }
    
    Retourne :
        dict : Angles de pitch, roll et yaw en radians.
    """
    # Récupérer les points clés
    acromion = np.array(points["acromion"])
    humerus = np.array(points["humérus"])
    coracoide = np.array(points["coracoïde"])
    
    # Vecteurs pour définir le repère local
    vec_x = humerus - acromion  # Axe principal entre acromion et humérus
    vec_z = np.cross(vec_x, coracoide - acromion)  # Produit vectoriel pour obtenir l'axe Z
    vec_y = np.cross(vec_z, vec_x)  # Produit vectoriel pour obtenir l'axe Y
    
    # Normaliser les vecteurs pour former un repère orthonormé
    vec_x = vec_x / np.linalg.norm(vec_x)
    vec_y = vec_y / np.linalg.norm(vec_y)
    vec_z = vec_z / np.linalg.norm(vec_z)
    
    # Construire la matrice de rotation (repère local par rapport au repère global)
    R = np.column_stack((vec_x, vec_y, vec_z))
    
    # Calcul des angles d'Euler (yaw, pitch, roll)
    pitch = np.arcsin(-R[2, 0])  # R31
    roll = np.arctan2(R[2, 1], R[2, 2])  # R32, R33
    yaw = np.arctan2(R[1, 0], R[0, 0])  # R21, R11
    
    return {
        "pitch": pitch,
        "roll": roll,
        "yaw": yaw
    }

def process_video(frames, pose, align, depth_frame, depth_image, h, w):
    """
    Fonction pour traiter les trames vidéo, calculer les angles et afficher les résultats.
    """
    # Obtenir la trame couleur
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise ValueError(
            "Color frame is empty. Check if the camera is working properly.")

    # Convertir la trame couleur en tableau NumPy
    color_image = np.asanyarray(color_frame.get_data())

    # Vérifier que la conversion a réussi
    if color_image is None or color_image.size == 0:
        raise ValueError(
            "Color image is invalid. Check the RealSense data stream.")

    # Convertir en RGB pour MediaPipe
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Détection des landmarks avec MediaPipe
    pose_results = pose.process(rgb_image)

    # Calcul des angles si landmarks détectés
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark

        def get_3d_coordinates(landmark):
            """Obtenir les coordonnées x, y, z pour un landmark en utilisant la caméra de profondeur."""
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
                depth = depth_frame.get_distance(cx, cy)
                if depth > 0:  # Assurez-vous que la profondeur est valide
                    intrinsics = pipeline.get_active_profile().get_stream(
                        rs.stream.depth).as_video_stream_profile().get_intrinsics()
                    fx, fy = intrinsics.fx, intrinsics.fy
                    ppx, ppy = intrinsics.ppx, intrinsics.ppy

                    # Convertir les coordonnées en 3D (mètres)
                    x = (cx - ppx) * depth / fx
                    y = (cy - ppy) * depth / fy
                    return [x, y, depth]
            return [0, 0, 0]

        # Exemple pour un bras
        right_shoulder = get_3d_coordinates(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        right_elbow = get_3d_coordinates(
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        right_wrist = get_3d_coordinates(
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        right_hip = get_3d_coordinates(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])

        # Calcul des angles
        right_elbow_angle = calculate_angle_3d(
            right_shoulder, right_elbow, right_wrist)
        right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw == calculate_pitch_roll_yaw(right_shoulder, right_elbow,)

        # Contrôle du bras droit

        return right_elbow_angle, right_shoulder_pitch, right_shoulder_roll,right_shoulder_yaw
    return None


def control_reachy(right_elbow_angle, right_shoulder_pitch, right_shoulder_roll):
    """
    Fonction pour contrôler le bras Reachy en fonction de l'angle calculé.
    """
    if right_shoulder_pitch and right_shoulder_roll and right_elbow_angle:
        right_angled_position = {
            reachy.r_arm.r_shoulder_pitch: 0, #right_shoulder_pitch,
            reachy.r_arm.r_shoulder_roll: 0,#right_shoulder_roll-180,
            reachy.r_arm.r_elbow_pitch: right_elbow_angle - 180,
            reachy.r_arm.r_forearm_yaw: 0,
            reachy.r_arm.r_wrist_pitch: 0,
            reachy.r_arm.r_wrist_roll: 0,
        }

        goto(right_angled_position, duration=1.0,
             interpolation_mode=InterpolationMode.MINIMUM_JERK)


def main_loop():
    """
    Boucle principale exécutant les tâches en parallèle.
    """
    while True:
        # Capturer les trames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convertir en image NumPy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        h, w, _ = color_image.shape

        # Parallélisation des tâches
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            video_future = executor.submit(
                process_video, frames, pose, align, depth_frame, depth_image, h, w)
            angles = video_future.result()
            if angles:
                control_future = executor.submit(control_reachy, *angles)

        # Afficher la vidéo
        cv2.imshow('RealSense Hands and Arms with 3D Angles', color_image)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


reachy = ReachySDK(host='138.195.196.90')
reachy.turn_on('r_arm')
reachy.turn_on('l_arm')

try:
    main_loop()
finally:
    # Libération des ressources
    reachy.turn_off_smoothly('r_arm')
    reachy.turn_off_smoothly('l_arm')

    pipeline.stop()
    cv2.destroyAllWindows()
