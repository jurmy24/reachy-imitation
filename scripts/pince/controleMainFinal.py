import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
# import utils.angles as utils



# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, 2)  # You can also specify confidence levels here
mp_draw = mp.solutions.drawing_utils

TOP_LEFT_X_LABEL = 10
TOP_LEFT_Y_LABEL = 30

HANDEDNESS_CERTAINTY_UPPER = 0.85  # There's a tradeoff here between whether it recognises
# a hand and whether it confuses between a left and right hand
HANDEDNESS_CERTAINTY_LOWER = 0.7
FINGER_TIPS = [mp_hands.HandLandmark.THUMB_TIP.value,
                mp_hands.HandLandmark.INDEX_FINGER_TIP.value,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value,
                mp_hands.HandLandmark.RING_FINGER_TIP.value,
                mp_hands.HandLandmark.PINKY_TIP.value,
                ]  # [4, 8, 12, 16, 20]

def calculate_distance_with_depth(point1, point2):
    # this should account for depth
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2])**2)


def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def is_hand_open(hand_landmarks):
    # Indices des points de repère pour les pointes des doigts et le centre de la paume
    # Point de repère pour le centre de la paume
    palm_base = hand_landmarks.landmark[0]

    open_fingers = 0
    for tip in FINGER_TIPS:
        if calculate_distance(
            hand_landmarks.landmark[tip], palm_base
        ) > calculate_distance(hand_landmarks.landmark[tip - 3], palm_base):
            open_fingers += 1
    return (
        open_fingers >= 3
    )  # Considérer la main ouverte si au moins 3 doigts sont ouverts

def is_hand_half_open(hand_landmarks):
    # Indices des points de repère pour les pointes des doigts et le centre de la paume
    # Point de repère pour le centre de la paume
    palm_base = hand_landmarks.landmark[0]

    open_fingers = 0
    for tip in FINGER_TIPS:
        if calculate_distance(
            hand_landmarks.landmark[tip], palm_base
        ) > calculate_distance(hand_landmarks.landmark[tip - 2], palm_base):
            open_fingers += 1
    return (
        open_fingers >= 3
    )  # Considérer la main ouverte si au moins 3 doigts sont ouverts

def is_hand_closed(hand_landmarks):
    # Indices des points de repère pour les pointes des doigts et le centre de la paume
    # Point de repère pour le centre de la paume
    palm_base = hand_landmarks.landmark[0]

    open_fingers = 0
    for tip in FINGER_TIPS:
        if calculate_distance(
            hand_landmarks.landmark[tip], palm_base
        ) > calculate_distance(hand_landmarks.landmark[tip - 3], palm_base):
            open_fingers += 1
    return (
        open_fingers <= 2
    )  # Considérer la main ferme si au moins 3 doigts sont ouverts


# or half close, depending on your vision, is your glass half empty or half full
def is_hand_half_closed(hand_landmarks):
    # Indices des points de repère pour les pointes des doigts et le centre de la paume


    # Point de repère pour le centre de la paume
    palm_base = hand_landmarks.landmark[0]
    """
    half_open_fingers = 0
    for tip in FINGER_TIPS:
        if calculate_distance(
            hand_landmarks.landmark[tip], palm_base
        ) > (calculate_distance(hand_landmarks.landmark[tip - 2], palm_base) + calculate_distance(hand_landmarks.landmark[tip - 1], palm_base))*0.5:
            half_open_fingers += 1
    return (
        half_open_fingers >= 3
    )  # Considérer la main entreouverte si au moins 3 doigts sont entreouverts
    """
    half_closed_fingers = 0
    for tip in FINGER_TIPS:
        tip_to_palm = calculate_distance(hand_landmarks.landmark[tip], palm_base)
        if tip_to_palm < (calculate_distance(hand_landmarks.landmark[tip - 2], palm_base)):
            half_closed_fingers += 1
    return (
        half_closed_fingers >= 3
    )  # Considérer la main entreouverte si au moins 3 doigts sont entreouverts

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


def flip_hand_labels(hand_label: str):
    """ Mediapipe doesn't automatically flip to account for camera mirroring """
    if (hand_label == "Right"):
        return "Gauche"  # also flipping languages :)
    else:
        return "Droit"

def get_label(index, hand):
    output = "Unknown", 0
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            label = classification.classification[0].index 
            score = classification.classification[0].score
            output = label, score
    return output

if __name__ == "__main__":

    # cap = cv2.VideoCapture(0)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)
    # Aligner les flux couleur et profondeur pour synchroniser les données
    align = rs.align(rs.stream.color)

    previous_hand_type = ["Unknown", "Unknown"]
    while True:
        # ret, frame = cap.read()
        frames = pipeline.wait_for_frames()
        # Aligner les flux couleur et profondeur
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())  # Image couleur
        depth_image = np.asanyarray(
            depth_frame.get_data())  # Image de profondeur

        h, w, _ = color_image.shape

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Effectuer la détection des mains et du corps avec MediaPipe
        hand_results = hands.process(rgb_image)

        # Convertir en RGB
        # rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Détection des mains
        results = hands.process(rgb_image)

        # Dessiner les annotations et vérifier si la main est ouverte

        if results.multi_hand_landmarks:
            for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if index >= len(previous_hand_type):
                    break  # Just in case we have more hands than expected
                #hand_landmarks = results.multi_hand_landmarks[hand_num]

                #hand_type, certainty = get_label(index, results)
                hand_type = flip_hand_labels(
                    results.multi_handedness[index].classification[0].label)
                
                certainty = results.multi_handedness[index].classification[0].score
                
                
                if certainty < HANDEDNESS_CERTAINTY_UPPER:
                    hand_type = "Unknown"

                # else:
                #     if certainty < HANDEDNESS_CERTAINTY_UPPER and previous_hand_type[index] != hand_type:
                #         hand_type = "Unknown"

                # previous_hand_type[index] = hand_type
                

                
                #print(f"certainty {certainty} of {hand_type} hand")
                
                
                mp_draw.draw_landmarks(
                    color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                """
                if is_hand_open(hand_landmarks):
                    text = f"Main {hand_type} ouverte"
                    colour = (0, 255, 0)
                elif is_hand_half_open(hand_landmarks):
                    text = f"Main {hand_type} entreouverte"
                    colour = (255, 0, 0)
                else:
                    text = f"Main {hand_type} ferme"
                    colour = (0, 0, 255)
                """
                #print(hand_landmarks.landmark[0])

                if is_hand_half_closed(hand_landmarks):
                    text = f"Main {hand_type} ferme"
                    colour = (0, 0, 255)
                # elif is_hand_half_closed(hand_landmarks):
                #     text = f"Main {hand_type} entreouverte"
                #     colour = (255, 0, 0)
                else:
                    text = f"Main {hand_type} ouverte"
                    colour = (0, 255, 0)

                if hand_type == "Gauche":
                    label_coordinate = (
                        TOP_LEFT_X_LABEL, TOP_LEFT_Y_LABEL)  # top left
                elif hand_type == "Droit":
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    label_coordinate = (
                        w - text_width - TOP_LEFT_X_LABEL, TOP_LEFT_Y_LABEL)  # top right
                else:
                    continue  # don't label if i can't figure out what hand it is ?

                # Récupérer et afficher la profondeur pour chaque articulation de la main
                for id, lm in enumerate(hand_landmarks.landmark):
                    # print(lm)
                    # h, w, _ = color_image.shape  # Dimensions de l'image
                    cx, cy = int(lm.x * w), int(
                        lm.y * h
                    )  # Convertir en coordonnées pixels

                    if (
                        0 <= cx < depth_image.shape[1]
                        and 0 <= cy < depth_image.shape[0]
                    ):
                        depth = depth_frame.get_distance(
                            cx, cy)  # Distance en mètres

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

    # cap.release()
    pipeline.stop()
    cv2.destroyAllWindows()



    # THIS IS CODE USED FOR THE REGULAR 2D CAMERA
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
