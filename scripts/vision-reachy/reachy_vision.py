"Cette fiche permet de regarder via le caméra de Reachy"

from reachy_sdk import ReachySDK
import cv2 as cv

# Replace with the actual IP you've found.
reachy = ReachySDK(host="138.195.196.90")

while True:
    # This let you access the last frame grabbed by Reachy left eye
    # It's always the most up-to-date image
    left_image = reachy.left_camera.last_frame

    cv.imshow("left image", left_image)
    cv.waitKey(30)
