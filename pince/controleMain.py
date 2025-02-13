import cv2
import mediapipe as mp
import numpy as np

# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def is_hand_open(hand_landmarks):
    # Indices des points de repère pour les pointes des doigts et le centre de la paume
    finger_tips = [4, 8, 12, 16, 20]
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


def is_left_hand(hand_landmarks):
    # Vérifier si le point de repère 0 (poignet) est à droite des autres points de repère
    return hand_landmarks.landmark[0].x > hand_landmarks.landmark[9].x


def process_main(frame, results):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    liste_result = {"Main gauche": None, "Main droite": None}
    # Dessiner les annotations et vérifier si la main est ouverte
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_type = "Main gauche" if is_left_hand(hand_landmarks) else "Main droite"
            if is_hand_open(hand_landmarks):
                liste_result[hand_type] = "ouverte"
            else:
                liste_result[hand_type] = "fermee"
    return liste_result


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
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
            for hand_landmarks in results.multi_hand_landmarks:
                print(hand_landmarks.landmark[0])
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_type = (
                    "Main gauche" if is_left_hand(hand_landmarks) else "Main droite"
                )
                if is_hand_open(hand_landmarks):
                    cv2.putText(
                        frame,
                        f"{hand_type} ouverte",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        frame,
                        f"{hand_type} fermee",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

        # Afficher l'image
        cv2.imshow("MediaPipe Hands", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
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
