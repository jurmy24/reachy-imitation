import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

# Initialisation de MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils


# Initialisation de RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Lancement du flux
pipeline.start(config)

# Fonction de normalisation
def normalize(v):
    return v / np.linalg.norm(v)

# Calcul de la matrice de rotation
def compute_rotation_matrix(P_right, P_left, P_head, P_buste):
    X = normalize(P_right - P_left)  # Axe X'
    Y = normalize(P_head - P_buste)  # Axe Y'
    Z = np.cross(X, Y)              # Axe Z' (orthogonal)
    Z = normalize(Z)
    Y = np.cross(Z, X)              # Recalcule Y pour assurer l'orthogonalité
    
    R_matrix = np.vstack([X, Y, Z]).T
    return R.from_matrix(R_matrix)

# Fonction pour obtenir les coordonnées 3D d'un landmark avec la profondeur
def get_3d_coordinates(landmark, depth_frame, w, h, intrinsics):
    cx, cy = int(landmark.x * w), int(landmark.y * h)
    if 0 <= cx < w and 0 <= cy < h:
        # Obtenir la profondeur (distance) à partir du frame de profondeur
        depth = depth_frame.get_distance(cx, cy)
        if depth > 0:
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy
            return np.array([x, y, depth])
    return np.array([0, 0, 0])

# Inverse Kinematics pour l'épaule
def inverse_kinematics_shoulder(p_epaule_global, p_coude_global, R_global_to_local):
    # Transformation dans le repère local
    d_global = p_coude_global - p_epaule_global
    R_local_matrix = R_global_to_local.as_matrix()  # Convertit l'objet Rotation en matrice NumPy
    d_local = np.linalg.inv(R_local_matrix) @ d_global

    d_norm = np.linalg.norm(d_local)
    
    # Yaw
    yaw = np.arctan2(d_local[1], d_local[0])

    # Pitch
    d_proj = np.sqrt(d_local[0]**2 + d_local[1]**2)
    pitch = np.arctan2(d_local[2], d_proj)

    # Roll (non utilisé ici)
    roll = 0

    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

# Ouverture de la vidéo ou de la caméra
cap = cv2.VideoCapture(2)

# Main loop
while True:
    # Attendre un frame de la caméra RealSense
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    # Convertir l'image couleur en format RGB
    color_image = np.asanyarray(color_frame.get_data())
    rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Obtenir les paramètres intrinsèques de la caméra
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    img_h, img_w, _ = color_image.shape

    # Traitement MediaPipe pour la pose
    result = pose.process(rgb_frame)
    
    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        mp_draw.draw_landmarks(color_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Récupération des coordonnées 3D
        P_right = get_3d_coordinates(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], depth_frame, img_w, img_h, intrinsics)
        P_left = get_3d_coordinates(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], depth_frame, img_w, img_h, intrinsics)
        P_head = get_3d_coordinates(lm[mp_pose.PoseLandmark.NOSE], depth_frame, img_w, img_h, intrinsics)
        P_buste = (get_3d_coordinates(lm[mp_pose.PoseLandmark.LEFT_HIP], depth_frame, img_w, img_h, intrinsics) + 
                   get_3d_coordinates(lm[mp_pose.PoseLandmark.RIGHT_HIP], depth_frame, img_w, img_h, intrinsics)) / 2
        P_elbow = get_3d_coordinates(lm[mp_pose.PoseLandmark.RIGHT_ELBOW], depth_frame, img_w, img_h, intrinsics)

        # Calcul de la matrice de rotation
        R_global_to_local = compute_rotation_matrix(P_right, P_left, P_head, P_buste)



        # Calcul des angles de l'épaule
        yaw, pitch, roll = inverse_kinematics_shoulder(P_right, P_elbow, R_global_to_local)

        # Affichage des angles à l'écran
        cv2.putText(color_image, f'Yaw: {yaw:.2f} deg', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(color_image, f'Pitch: {pitch:.2f} deg', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Affichage de l'image avec les points et les connexions
    cv2.imshow("Mediapipe Pose", color_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
