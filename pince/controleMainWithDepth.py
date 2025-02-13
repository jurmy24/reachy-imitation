import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
#import utils.angles as utils

# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, 2) # You can also specify confidence levels here
mp_draw = mp.solutions.drawing_utils

TOP_LEFT_X_LABEL = 10
TOP_LEFT_Y_LABEL = 30

HANDEDNESS_CERTAINTY = 0.9  # There's a tradeoff here between whether it recognises 
                            # a hand and whether it confuses between a left and right hand 

""""""
def calculate_distance_with_depth(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2])**2) # this should account for depth 

def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def is_hand_open(hand_landmarks):
    # Indices des points de repère pour les pointes des doigts et le centre de la paume
    finger_tips = [mp_hands.HandLandmark.THUMB_TIP.value, 
                   mp_hands.HandLandmark.INDEX_FINGER_TIP.value, 
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value, 
                   mp_hands.HandLandmark.RING_FINGER_TIP.value, 
                   mp_hands.HandLandmark.PINKY_TIP.value, 
                   ]  # [4, 8, 12, 16, 20]
    
    palm_base = hand_landmarks.landmark[0]  # Point de repère pour le centre de la paume    

    open_fingers = 0
    for tip in finger_tips:
        if calculate_distance(
            hand_landmarks.landmark[tip], palm_base
        ) > calculate_distance(hand_landmarks.landmark[tip - 2], palm_base):
            open_fingers += 1
    return (
        open_fingers >= 3
    )  # Considérer la main ouverte si au moins 3 doigts sont ouverts


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


def flip_hand_labels(hand_label : str):
    """ Mediapipe doesn't automatically flip to account for camera mirroring """
    if (hand_label == "Right"): 
        return "Gauche" # also flipping languages :)
    else: 
        return "Droit"


if __name__ == "__main__":

    #cap = cv2.VideoCapture(0) 

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    #Start streaming
    pipeline.start(config)
    #Aligner les flux couleur et profondeur pour synchroniser les données
    align = rs.align(rs.stream.color)

    while True:
        #ret, frame = cap.read()
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # Aligner les flux couleur et profondeur

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())  # Image couleur
        depth_image = np.asanyarray(depth_frame.get_data())  # Image de profondeur

        h, w, _ = color_image.shape


        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Effectuer la détection des mains et du corps avec MediaPipe
        hand_results = hands.process(rgb_image)


        # Convertir en RGB
        #rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Détection des mains
        results = hands.process(rgb_image)

        # Dessiner les annotations et vérifier si la main est ouverte
        if results.multi_hand_landmarks:
            for hand_num in range(len(results.multi_hand_landmarks)):
                hand_landmarks = results.multi_hand_landmarks[hand_num]
                hand_type = flip_hand_labels(results.multi_handedness[hand_num].classification[0].label)

                certainty = results.multi_handedness[hand_num].classification[0].score

                if certainty < HANDEDNESS_CERTAINTY:
                    hand_type = "Unknown"

                """
                if (hand_type == "Gauche"):
                    print(f"certainty {certainty} of {hand_type} hand")
                """

                mp_draw.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if is_hand_open(hand_landmarks):
                    text = f"Main {hand_type} ouverte"
                    colour = (0, 255, 0)
                else: 
                    text = f"Main {hand_type} ferme"
                    colour = (0, 0, 255)

                if hand_type == "Gauche":
                    label_coordinate = (TOP_LEFT_X_LABEL, TOP_LEFT_Y_LABEL) # top left
                elif hand_type == "Droit": 
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    label_coordinate = (w - text_width - TOP_LEFT_X_LABEL, TOP_LEFT_Y_LABEL) # top right
                else: 
                    continue # don't label if i can't figure out what hand it is ?
                
                # Récupérer et afficher la profondeur pour chaque articulation de la main
                for id, lm in enumerate(hand_landmarks.landmark):
                    #print(lm)
                    #h, w, _ = color_image.shape  # Dimensions de l'image
                    cx, cy = int(lm.x * w), int(
                        lm.y * h
                    )  # Convertir en coordonnées pixels

                    if (
                        0 <= cx < depth_image.shape[1]
                        and 0 <= cy < depth_image.shape[0]
                    ):
                        depth = depth_frame.get_distance(cx, cy)  # Distance en mètres

                cv2.putText(
                        color_image,
                        text,
                        label_coordinate,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        colour,
                        2,
                        cv2.LINE_AA,
                )

        # Afficher l'image
        cv2.imshow("MediaPipe Hands", color_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    #cap.release()
    pipeline.stop()
    cv2.destroyAllWindows()

    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    #    Start streaming
    pipeline.start(config)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir en RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détection des mains
        results = hands.process(rgb_frame)

        # Dessiner les annotations et vérifier si la main est ouverte
        if results.multi_hand_landmarks:
            list_result = process_main(rgb_frame, results)
            for nom_main in list_result:
                if list_result[nom_main] is not None:
                    cv2.putText(frame, f'{nom_main} {list_result[nom_main]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Afficher l'image
        cv2.imshow('MediaPipe Hands', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    """
