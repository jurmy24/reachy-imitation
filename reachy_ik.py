import numpy as np
import math
import mediapipe as mp
import pyrealsense2 as rs
import cv2
from scaling_joint_params import test_scale_factors, L_REACHY_FOREARM, L_REACHY_UPPERARM, get_3d_coordinates, get_scale_factors, average_landmarks, get_3d_coordinates_of_hand
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
reachy = ReachySDK(host="138.195.196.90")


L_REACHY_ARM = L_REACHY_FOREARM + L_REACHY_UPPERARM 
REACHY_R_SHOULDER_TO_ORIGIN = np.array([0, -0.19, 0]) # Ask victor where these constants should live 
JOINT_ARGS = ["shoulder pitch", "shoulder roll", "arm yaw", "elbow pitch", "forearm yaw", "wrist pitch", "wrist roll"]

move_reachy = True # Set to False if you don't want Reachy to move

ZERO_RIGHT_POS = {
    reachy.r_arm.r_shoulder_pitch: 0,
    reachy.r_arm.r_shoulder_roll: 0,
    reachy.r_arm.r_arm_yaw: 0,
    reachy.r_arm.r_elbow_pitch: 0,
    reachy.r_arm.r_forearm_yaw: 0,
    reachy.r_arm.r_wrist_pitch: 0,
    reachy.r_arm.r_wrist_roll: 0,
}
ordered_joint_names = [
    reachy.r_arm.r_shoulder_pitch,
    reachy.r_arm.r_shoulder_roll,
    reachy.r_arm.r_arm_yaw,
    reachy.r_arm.r_elbow_pitch,
    reachy.r_arm.r_forearm_yaw,
    reachy.r_arm.r_wrist_pitch,
    reachy.r_arm.r_wrist_roll
]

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
    return np.linalg.norm(point) <= L_REACHY_ARM

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
    prev_reachy_hand_right = reachy.r_arm.forward_kinematics()[0:3, 3]
    goto_new_position = False
    #print(prev_reachy_hand_right)
    its = 0
    pause_update = True

    if move_reachy:
        reachy.turn_on('r_arm')
    try:
        while True:
            its += 1
            if its == 10:
                pause_update = False
                its = 0
            else:
                pause_update = True
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
                mp_draw.draw_landmarks(
                color_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
                
                if ((0.3 > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility or not landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].HasField('visibility'))
                        or (0.3 > landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].visibility or not landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].HasField('visibility'))
                        or (0.3 > landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].visibility or not landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].HasField('visibility'))):
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

                
                human_right_shoulder = get_3d_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], depth_frame, w, h, intrinsics)
                mid_hand_landmark = average_landmarks(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX],landmarks[mp_pose.PoseLandmark.RIGHT_PINKY])
                human_hand_right = get_3d_coordinates_of_hand(mid_hand_landmark, depth_frame, w, h, intrinsics)
                human_hand_right = transform_to_shoulder_origin(human_hand_right, human_right_shoulder)
                hand_right = scale_point(hand_sf, human_hand_right)

                if not within_reachys_reach(hand_right):
                    cv2.putText(
                        color_image,
                        f"reachy can't reachy there",
                        (40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                    )

                    cv2.imshow("RealSense Right Arm IK", color_image)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
                
                reachy_hand_right = translate_to_reachy_origin(hand_right)
                
                #reachy_wrist_right = human_to_robot_coordinates(wrist_right, hand_sf)

                # Afficher les coordonnées 3D sur l'image
                x, y, z = human_hand_right
                cv2.putText(
                    color_image,
                    f"Human Hand: ({x:.2f}, {y:.2f}, {z:.2f})m",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                x, y, z = reachy_hand_right
                cv2.putText(
                    color_image,
                    f"Robot Hand: ({x:.2f}, {y:.2f}, {z:.2f})m",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

                #print(prev_reachy_wrist_right)
                #print(reachy_wrist_right)
                a = reachy.r_arm.forward_kinematics()
                if not (np.allclose(prev_reachy_hand_right, reachy_hand_right, atol = 0.05)):
                    prev_reachy_hand_right = reachy_hand_right
                    a[0,3] = reachy_hand_right[0]
                    a[1,3] = reachy_hand_right[1]
                    a[2,3] = reachy_hand_right[2]
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

                if goto_new_position and move_reachy and not pause_update:
                    # implement the go to code here
                    goto(
                        {joint: pos for joint,pos in list(zip(ordered_joint_names, joint_pos))},
                        duration=2.0,
                        interpolation_mode=InterpolationMode.MINIMUM_JERK,
                    )
            else:
                cv2.putText(
                    color_image,
                    f"je ne vois pas ton corps",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
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
    hand_sf, elbow_sf = get_scale_factors(0.7, 0.3)
    print(hand_sf)
    test_reachy_ik(hand_sf)
    if move_reachy:
        goto(
            ZERO_RIGHT_POS,
            duration=2.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )
        reachy.turn_off_smoothly("r_arm")


