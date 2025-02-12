"""controle du robot avec vision 3D"""

import cv2
import mediapipe as mp
import numpy as np
from math import sqrt, acos, pi
import time
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def angle_coude_gauche(landmarks):
    shoulder = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
    ]
    elbow = [
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
    ]
    wrist = [
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
    ]

    angle = calculate_angle(shoulder, elbow, wrist)
    return elbow, angle


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def init_longueur():
    longueurs = {}
    longueurs["RightUpperArm"] = 0
    longueurs["LeftUpperArm"] = 0
    longueurs["RightLowerArm"] = 0
    longueurs["LeftLowerArm"] = 0
    tInit = time.time()
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened() and time.time() - tInit < 8:
            ret, frame = cap.read()
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            # Length left upper arm
            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                ]
                elbow = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                ]
                length = distance(shoulder, elbow)
                longueurs["LeftUpperArm"] = max(longueurs["LeftUpperArm"], length)
            except:
                pass

            # Length right upper arm
            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                ]
                elbow = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                ]
                length = distance(shoulder, elbow)
                longueurs["RightUpperArm"] = max(longueurs["RightUpperArm"], length)
            except:
                pass

            # Length right lower arm
            try:
                landmarks = results.pose_landmarks.landmark
                elbow = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                ]
                wrist = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                ]
                length = distance(wrist, elbow)
                longueurs["RightLowerArm"] = max(longueurs["RightLowerArm"], length)
            except:
                pass

            # Length left lower arm
            try:
                landmarks = results.pose_landmarks.landmark
                elbow = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                ]
                wrist = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                ]
                length = distance(wrist, elbow)
                longueurs["LeftLowerArm"] = max(longueurs["LeftLowerArm"], length)
            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(245, 117, 66), thickness=2, circle_radius=2
                ),
                mp_drawing.DrawingSpec(
                    color=(245, 66, 230), thickness=2, circle_radius=2
                ),
            )

            cv2.imshow("Mediapipe Feed", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
    return longueurs


def angleFrontal(vrai_longueur, longueur_percue):
    return acos(longueur_percue / vrai_longueur) * 180 / pi


def affiche_texte1(image, angle, body):
    cv2.putText(
        image,
        str(angle),
        tuple(np.multiply(body, [640, 480]).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def affiche_texte2(image, angle, body, typ):
    cv2.putText(
        image,
        typ + str(angle),
        tuple(np.multiply(body, [640, 480]).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def angle_left_elbow(landmarks, vLongueur):
    shoulder = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
    ]
    elbow = [
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
    ]
    wrist = [
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
    ]
    pLongueur = distance(elbow, wrist)
    # Calculate angle
    sideAngle = round(calculate_angle(shoulder, elbow, wrist), 2)
    frontalAngle = round(angleFrontal(vLongueur, pLongueur), 2)
    return sideAngle, frontalAngle, elbow


def angle_right_elbow(landmarks, vLongueur):
    shoulder = [
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
    ]
    elbow = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
    ]
    wrist = [
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
    ]
    pLongueur = distance(elbow, wrist)
    # Calculate angle
    sideAngle = round(calculate_angle(shoulder, elbow, wrist), 2)
    frontalAngle = round(angleFrontal(vLongueur, pLongueur), 2)
    return sideAngle, frontalAngle, elbow


def angle_left_shoulder(landmarks, vLongueur):
    hip = [
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
    ]
    shoulder = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
    ]
    elbow = [
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
    ]
    pLongueur = distance(shoulder, elbow)
    # Calculate angle
    sideAngle = round(calculate_angle(hip, shoulder, elbow), 2)
    frontalAngle = round(angleFrontal(vLongueur, pLongueur), 2)
    return sideAngle, frontalAngle, shoulder


def angle_right_shoulder(landmarks, vLongueur):
    hip = [
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
    ]
    shoulder = [
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
    ]
    elbow = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
    ]
    pLongueur = distance(shoulder, elbow)
    # Calculate angle
    sideAngle = round(calculate_angle(hip, shoulder, elbow), 2)
    frontalAngle = round(angleFrontal(vLongueur, pLongueur), 2)
    return sideAngle, frontalAngle, shoulder


def convex3point(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (y3 - y2) / (x3 - x2) - (y2 - y1) / (x2 - x1) > 0


def convexite_bras_droit(landmarks):
    shoulder = [
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
    ]
    elbow = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
    ]
    wrist = [
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
    ]
    return convex3point(shoulder, elbow, wrist)


longueurs = init_longueur()
print(longueurs)
qlongueurs = {
    "RightUpperArm": 0.28180931597450054,
    "LeftUpperArm": 0.2412316313128073,
    "RightLowerArm": 0.3757721842678538,
    "LeftLowerArm": 0.3126503633940726,
}
reachy = ReachySDK(host="138.195.196.90")
reachy.turn_on("r_arm")


cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        # Left Elbow
        try:
            landmarks = results.pose_landmarks.landmark
            sideAngle, frontalAngle, left_elbow = angle_left_elbow(
                landmarks, longueurs["LeftLowerArm"]
            )
            x, y = left_elbow

            affiche_texte2(image, sideAngle, [x, y - 0.05], "sa  ")
        except:
            pass

        # Right Elbow
        try:
            landmarks = results.pose_landmarks.landmark
            ElbSideAngle, frontalAngle, right_elbow = angle_right_elbow(
                landmarks, longueurs["RightLowerArm"]
            )
            x, y = right_elbow
            affiche_texte2(image, ElbSideAngle, [x, y - 0.05], "sa: ")
        except:
            pass

        # Left Shoulder
        try:
            landmarks = results.pose_landmarks.landmark
            sideAngle, frontalAngle, left_shoulder = angle_left_shoulder(
                landmarks, longueurs["LeftUpperArm"]
            )
            x, y = left_shoulder
            affiche_texte2(image, frontalAngle, left_shoulder, "fa: ")
            affiche_texte2(image, sideAngle, [x, y - 0.05], "sa: ")
        except:
            pass

        # Right Shoulder
        try:
            landmarks = results.pose_landmarks.landmark
            ShSideAngle, frontalAngle, right_shoulder = angle_right_shoulder(
                landmarks, longueurs["RightUpperArm"]
            )
            x, y = right_shoulder
            affiche_texte2(image, frontalAngle, right_shoulder, "fa: ")
            affiche_texte2(image, ShSideAngle, [x, y - 0.05], "sa: ")
            if convexite_bras_droit(landmarks):
                yam = 90
            else:
                yam = -90
            # Traiter le cas de la position du shoulder_roll grâce à la convexité du bras d'action

            right_angled_position = {
                reachy.r_arm.r_shoulder_pitch: 0,
                reachy.r_arm.r_shoulder_roll: -ShSideAngle,
                reachy.r_arm.r_arm_yaw: -yam,
                reachy.r_arm.r_elbow_pitch: ElbSideAngle - 180,
                reachy.r_arm.r_forearm_yaw: 0,
                reachy.r_arm.r_wrist_pitch: 0,
                reachy.r_arm.r_wrist_roll: 0,
            }
            goto(
                goal_positions=right_angled_position,
                duration=1.0,
                interpolation_mode=InterpolationMode.MINIMUM_JERK,
            )
        except:
            pass

        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        cv2.imshow("Mediapipe Feed", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Remplacez '192.168.X.X' par l'adresse IP de Reachy

# Vérifiez si la connexion fonctionne
# for name, joint in reachy.joints.items():
#    print(f'Joint "{name}" is at pos {joint.present_position} degree.')


reachy.turn_off("r_arm")
