import numpy as np
import math
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

def calcul_angles(r_s, r_h, l_s, r_e):
    # Étape 1 : Calcul des vecteurs du plan
    r_s,r_e,r_h,l_s=np.array(r_s),np.array(r_e),np.array(r_h),np.array(l_s)
    v1 = r_h - r_s
    v2 = l_s - r_s
    
    # Vecteur normal au plan
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # Normalisation
    
    # Étape 2 : Définir une base locale
    u = v1 / np.linalg.norm(v1)  # Premier vecteur de la base locale
    w = normal  # Normal au plan
    v = np.cross(w, u)  # Second vecteur de la base locale
    
    # Étape 3 : Exprimer r_e dans cette base
    r_e_local = r_e - r_s  # Vecteur de l'origine au point r_e
    x_prime = np.dot(r_e_local, u)
    y_prime = np.dot(r_e_local, v)
    z_prime = np.dot(r_e_local, w)
    
    # Étape 4 : Calcul des angles sphériques
    theta = np.arctan2(y_prime, x_prime)  # Azimut
    phi = np.arctan2(z_prime, np.sqrt(x_prime**2 + y_prime**2))  # Élévation
    
    # Conversion en degrés
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)
    
    return theta_deg, phi_deg


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
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
            r_elbow = get_3d_coordinates(
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
            r_hip = get_3d_coordinates(
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
            l_shoulder = get_3d_coordinates(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

            # Calcul des angles pour l'épaule droite
            r_shoulder_pitch, r_shoulder_roll = calcul_angles(
                r_shoulder, r_hip,l_shoulder,r_elbow)

            # Affichage des angles sur l'image
            cv2.putText(color_image, f'Pitch: {int(r_shoulder_pitch)} deg',
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(color_image, f'Roll: {int(r_shoulder_roll)} deg',
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Dessiner les annotations pour le corps
        mp_draw.draw_landmarks(
            color_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Afficher l'image
        cv2.imshow('RealSense Body Pose and Shoulder Angles', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Libérer les ressources
    pipeline.stop()
    cv2.destroyAllWindows()
