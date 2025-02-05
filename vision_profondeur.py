'''Affichage des mains et des bras avec la profondeur associée à chaque point'''
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np

# Initialiser MediaPipe pour la détection des mains et du corps
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()  # Détection des mains
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()  # Détection du corps entier
mp_draw = mp.solutions.drawing_utils  

# Configuration de la caméra Intel RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Aligner les flux couleur et profondeur pour synchroniser les données
align = rs.align(rs.stream.color)

try:
    while True:
        # Capturer les trames depuis la caméra RealSense
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # Aligner les flux couleur et profondeur

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())  # Image couleur
        depth_image = np.asanyarray(depth_frame.get_data())  # Image de profondeur

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Effectuer la détection des mains et du corps avec MediaPipe
        hand_results = hands.process(rgb_image)
        pose_results = pose.process(rgb_image)

        # Annoter les points du corps (bras) sur l'image couleur
        if pose_results.pose_landmarks:
            mp_draw.draw_landmarks(color_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = pose_results.pose_landmarks.landmark
            arm_points = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                          mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
                          mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]
            
            for point in arm_points:
                lm = landmarks[point.value]  
                h, w, c = color_image.shape  # Dimensions de l'image
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convertir en coordonnées pixels

                # Vérifier si les coordonnées sont valides pour récupérer la profondeur
                if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
                    depth = depth_frame.get_distance(cx, cy)  # Distance en mètres
                    cv2.putText(color_image, f'{depth:.2f}m', (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Annoter les points des mains sur l'image couleur
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Récupérer et afficher la profondeur pour chaque articulation de la main
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = color_image.shape  # Dimensions de l'image
                    cx, cy = int(lm.x * w), int(lm.y * h)  # Convertir en coordonnées pixels

                    if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
                        depth = depth_frame.get_distance(cx, cy)  # Distance en mètres
                        cv2.putText(color_image, f'{depth:.2f}m', (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Afficher l'image
        cv2.imshow('RealSense Hands and Arms with Depth', color_image)

        # Quitter la boucle si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Libérer les ressources : arrêter la caméra et fermer les fenêtres
    pipeline.stop()
    cv2.destroyAllWindows()
