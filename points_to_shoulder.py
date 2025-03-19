import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np


# Initialiser MediaPipe pour la détection des mains et du corps
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
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

# Fonction pour récupérer les coordonnées 3D d'un point
def get_3d_coordinates(landmark, depth_frame, w, h, intrinsics):
    cx, cy = int(landmark.x * w), int(landmark.y * h)
    if 0 <= cx < w and 0 <= cy < h:
        depth = depth_frame.get_distance(cx, cy)
        if depth > 0:
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy
            #robot coordinate in comparison to camera coordinate: 
            # xCr= -zc ,yr = - xc, zr = - yc
            return np.array([-depth, -x, -y])
    return np.array([0, 0, 0])  # Valeur par défaut si aucune profondeur valide

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
        hand_results = hands.process(rgb_image)

        # Dictionnaire pour stocker les coordonnées 3D
        right_arm_coordinates = {}

        # Récupérer les points du bras droit
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            right_shoulder = get_3d_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], depth_frame, w, h, intrinsics)

            # Définir l'origine du repère à l'épaule droite
            def transform_to_shoulder_origin(point):
                return point - right_shoulder

            right_arm_points = {
                "shoulder_right" :get_3d_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], depth_frame, w, h, intrinsics),
                "elbow_right": get_3d_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW], depth_frame, w, h, intrinsics),
                "wrist_right": get_3d_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST], depth_frame, w, h, intrinsics),
            }

            for name, coord in right_arm_points.items():
                right_arm_coordinates[name] = transform_to_shoulder_origin(coord)

        # Récupérer les points de la main droite (index MCP et pinky MCP)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                index_mcp = get_3d_coordinates(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], depth_frame, w, h, intrinsics)
                pinky_mcp = get_3d_coordinates(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP], depth_frame, w, h, intrinsics)

                right_arm_coordinates["index_mcp"] = transform_to_shoulder_origin(index_mcp)
                right_arm_coordinates["pinky_mcp"] = transform_to_shoulder_origin(pinky_mcp)
        
        # Afficher les coordonnées 3D sur l'image
        for name, coord in right_arm_coordinates.items():
            x, y, z = coord
            cv2.putText(
                color_image,
                f"{name}: ({x:.2f}, {y:.2f}, {z:.2f})m",
                (50, 30 + list(right_arm_coordinates.keys()).index(name) * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        # Afficher l'image
        cv2.imshow("RealSense Right Arm 3D Coordinates (Shoulder as Origin)", color_image)

        # Quitter la boucle si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
