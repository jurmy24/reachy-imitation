import numpy as np
import math
import mediapipe as mp
import pyrealsense2 as rs
import cv2
from scaling_joint_params import test_scale_factors, L_REACHY_FOREARM, L_REACHY_UPPERARM, get_3d_coordinates, get_scale_factors
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
reachy = ReachySDK(host="138.195.196.90")


L_REACHY_ARM = L_REACHY_FOREARM + L_REACHY_UPPERARM
REACHY_R_SHOULDER_TO_ORIGIN = np.array([0, -0.19, 0]) # Ask victor where these constants should live 
JOINT_ARGS = ["shoulder pitch", "shoulder roll", "arm yaw", "elbow pitch", "forearm yaw", "wrist pitch", "wrist roll"]

def transform_to_shoulder_origin(point, right_shoulder):
    return point - right_shoulder

def scale_point(sf, point):
    return sf * point

def translate_to_reachy_origin(point):
    return point + REACHY_R_SHOULDER_TO_ORIGIN


# def human_to_robot_coordinates(point, sf):
#     """
#     Convert human coordinates to robot coordinates
#     Args:  
#         point (np.array): 3D point in human coordinates (m)
#             -> assumes scaled so shoulder is origin, coordinate frame is same as reachy's (i.e. x is depth, z height)
#         sf (float): scale factor
#     Returns:    
#         np.array: 3D point in robot coordinates
#     """
#     point = scale_point(hand_sf, point)
#     point = translate_to_reachy_origin(point)
#     return point

def within_reachys_reach(point):
    point = point - REACHY_R_SHOULDER_TO_ORIGIN
    return np.linalg.norm(point) < L_REACHY_ARM

def test_reachy_ik(hand_sf):

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils

    # Configuration caméra RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    align = rs.align(rs.stream.color)
    prev_reachy_wrist_right = np.array([0, 0, 0])
    prev_reachy_wrist_right = reachy.r_arm.forward_kinematics()[0:3, 3]
    goto_new_position = False
    #print(prev_reachy_wrist_right)
    reachy.turn_on('r_arm')
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
            # Récupérer les points du bras droit
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                
                if ((0.2 > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility or not landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].HasField('visibility'))
                        or (0.2 > landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].visibility or not landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].HasField('visibility'))):
                    cv2.putText(
                        color_image,
                        f"je ne vois pas ton bras",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("RealSense Right Arm IK", color_image)

                    # Quitter la boucle si la touche 'q' est pressée
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                
                right_shoulder = get_3d_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], depth_frame, w, h, intrinsics)
                wrist_right = get_3d_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST], depth_frame, w, h, intrinsics)
                wrist_right = transform_to_shoulder_origin(wrist_right, right_shoulder)
                wrist_right = scale_point(hand_sf, wrist_right)

                if not within_reachys_reach(wrist_right):
                    cv2.imshow("RealSense Right Arm IK", color_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
                reachy_wrist_right = translate_to_reachy_origin(wrist_right)
                
                #reachy_wrist_right = human_to_robot_coordinates(wrist_right, hand_sf)

                # Afficher les coordonnées 3D sur l'image
                x, y, z = wrist_right
                cv2.putText(
                    color_image,
                    f"Human Wrist: ({x:.2f}, {y:.2f}, {z:.2f})m",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                x, y, z = reachy_wrist_right
                cv2.putText(
                    color_image,
                    f"Robot Wrist: ({x:.2f}, {y:.2f}, {z:.2f})m",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

                #print(prev_reachy_wrist_right)
                #print(reachy_wrist_right)
                a = reachy.r_arm.forward_kinematics()
                if not (np.allclose(prev_reachy_wrist_right, reachy_wrist_right, atol = 1e-2)):
                    prev_reachy_wrist_right = reachy_wrist_right
                    a[0,3] = reachy_wrist_right[0]
                    a[1,3] = reachy_wrist_right[1]
                    a[2,3] = reachy_wrist_right[2]
                    goto_new_position = True
                else: 
                    goto_new_position = False

                joint_pos = reachy.r_arm.inverse_kinematics(a)

                i = 0
                for joint, label in zip(joint_pos, JOINT_ARGS):
                    cv2.putText(
                        color_image,
                        f"{label}: {joint:.2f} deg",
                        (400, 30 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )
                    i+=1

                if goto_new_position:
                    # implement the go to code here
                    goto(
                        {joint: pos for joint,pos in zip(reachy.r_arm.joints.values(), joint_pos)},
                        duration=2.0,
                        interpolation_mode=InterpolationMode.MINIMUM_JERK,
                    )
            # Afficher l'image
            cv2.imshow("RealSense Right Arm IK", color_image)

            # Quitter la boucle si la touche 'q' est pressée
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    #hand_sf, elbow_sf = test_scale_factors()
    #hugo sf 
    hand_sf, elbow_sf = get_scale_factors(0.3, 0.3)
    print(hand_sf)
    test_reachy_ik(hand_sf)
    reachy.turn_off_smoothly("r_arm")


