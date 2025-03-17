"""Construction d'un dataset pour la main"""

import numpy as np
import os
import csv
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from math import sqrt, acos, pi
import time 
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

xRW, yRW = 100, 100
# vibrant green
image_folder = "D:/ProjetReachy/DataMain/PhotoMain"
output_csv = "D:/ProjetReachy/DataMain/label.csv"
catégorie = 'mo'
classification = {'mf': 1, 'mo': 0} # mf - closed . mo -> open

def image_main_droite(landmarks, image):
    h, w, _ = image.shape
    xRW = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w)
    yRW = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)

    # Définir la taille de la région de zoom
    zoom_size = 100
    x1 = max(0, xRW - zoom_size)
    y1 = max(0, yRW - zoom_size)
    x2 = min(w, xRW + zoom_size)
    y2 = min(h, yRW + zoom_size)

    # Extraire la région de zoom
    return image[y1:y2, x1:x2]


def findRightHand2():
    id_photo = len(os.listdir(image_folder))
    nb_photo = 1
    with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            def écrire(file, path, marquage):       
                writer = csv.writer(file)
                writer.writerow([path, marquage])

            cap = cv2.VideoCapture(0)
            deb = time.time()
            zoomed_image = None  # Initialiser zoomed_image
            while cap.isOpened():    
                
                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                try:
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        
                        if (time.time() - deb) > nb_photo:
                            marquage = classification[catégorie]
                            imgname = os.path.join(image_folder, 'photo{}.jpg'.format(id_photo))
                            nb_photo += 1
                            id_photo += 1
                            zoomed_image = image_main_droite(landmarks, image)
                            cv2.imwrite(imgname, zoomed_image)
                            écrire(file, imgname, marquage)
                        # Afficher la région de zoom si elle est définie
                        if zoomed_image is not None:
                            cv2.imshow('Zoomed Right Hand', zoomed_image)

                    cv2.imshow('Mediapipe Feed', image)
                except:
                    pass

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    findRightHand2()
