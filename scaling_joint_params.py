
import numpy as np
import math
import mediapipe as mp
import pyrealsense2 as rs
import cv2




L_REACHY_UPPERARM = 0.28 # distance (m) from elbow to elbow
L_REACHY_elbow_to_hand = 0.25 + 0.0325 + 0.075 # distance (m) from elbow to hand

def average_landmarks(landmark1, landmark2):
    """
    Calculate the average of two landmarks.
    
    Args:
        landmark1 (mp.framework.formats.landmark_pb2.NormalizedLandmark): The first landmark.
        landmark2 (mp.framework.formats.landmark_pb2.NormalizedLandmark): The second landmark.
    
    Returns:
        mp.framework.formats.landmark_pb2.NormalizedLandmark: The average landmark.
    """
    #avg_landmark = mp.framework.formats.landmark_pb2.NormalizedLandmark()
    x = (landmark1.x + landmark2.x) / 2
    y = (landmark1.y + landmark2.y) / 2
    z = (landmark1.z + landmark2.z) / 2
    return np.array([x, y, z])

def get_3d_coordinates_of_hand(landmark, depth_frame, w, h, intrinsics):
    cx, cy = int(landmark[0] * w), int(landmark[1] * h)
    if 0 <= cx < w and 0 <= cy < h:
        depth = depth_frame.get_distance(cx, cy)
        if depth > 0:
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy
            #robot coordinate in comparison to camera coordinate: 
            # xCr= -zc ,yr = - xc, zr = - yc
            return np.array([-depth, x, -y]) # x was initially not negative 
    return np.array([0, 0, 0])  # Valeur par défaut si aucune profondeur valide    


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
            return np.array([-depth, x, -y]) # x was initially not negative 
    return np.array([0, 0, 0])  # Valeur par défaut si aucune profondeur valide

def get_arm_lengths(pose_landmarks, mp_pose, depth_frame, w, h, intrinsics):
    required_landmarks = [
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_PINKY,
        mp_pose.PoseLandmark.RIGHT_INDEX,
    ]
    for landmark in required_landmarks:
        if not (0.5 < pose_landmarks.landmark[landmark].visibility <= 1):
            return None, None
        
    r_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    r_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    r_hand = average_landmarks(pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY], pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX])

    r_elbow_3d = get_3d_coordinates(r_elbow, depth_frame, w, h, intrinsics)
    r_shoulder_3d = get_3d_coordinates(r_shoulder, depth_frame, w, h, intrinsics)
    r_hand_3d = get_3d_coordinates_of_hand(r_hand, depth_frame, w, h, intrinsics)

    upper_length = np.linalg.norm(r_elbow_3d - r_shoulder_3d)
    elbow_to_hand_length = np.linalg.norm(r_elbow_3d - r_hand_3d)
    return elbow_to_hand_length, upper_length

def get_scale_factors(elbow_to_hand_length, upper_length):
    """
    Calculate the scale factors for the Reachy arm
    """
    hand_sf = (L_REACHY_elbow_to_hand+L_REACHY_UPPERARM) / (elbow_to_hand_length+upper_length)
    elbow_sf = L_REACHY_UPPERARM / upper_length
    return hand_sf, elbow_sf



def test_scale_factors():
    """
    Test the scaling_joint_params functions
    """
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

    elbow_to_hand_lengths = np.array([])
    upperarm_lengths = np.array([])
    calculate_arm_lengths = True

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            h, w, _ = color_image.shape
            intrinsics = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

            pose_results = pose.process(rgb_image)

            if pose_results.pose_landmarks:

                mp_draw.draw_landmarks(color_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if calculate_arm_lengths:
                    elbow_to_hand_length, upper_length = get_arm_lengths(pose_results.pose_landmarks, mp_pose, depth_frame, w, h, intrinsics)
                    if elbow_to_hand_length is not None and upper_length is not None:
                        elbow_to_hand_lengths = np.append(elbow_to_hand_lengths, elbow_to_hand_length)
                        upperarm_lengths = np.append(upperarm_lengths, upper_length)
                        if len(elbow_to_hand_lengths) > 1000:
                            elbow_to_hand_length = np.median(elbow_to_hand_lengths)
                            upper_length = np.median(upperarm_lengths)
                            calculate_arm_lengths = False
                            hand_sf, elbow_sf = get_scale_factors(elbow_to_hand_length, upper_length)
                        cv2.putText(color_image, f'elbow_to_hand length: {elbow_to_hand_length:.2f} m', (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(color_image, f'Upper arm length: {upper_length:.2f} m', (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else: 
                        cv2.putText(color_image, f'Mettez tout votre bras droit dans le cadre!', (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            elif calculate_arm_lengths: 
                cv2.putText(color_image, f'Mettez tout votre bras droit dans le cadre!', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            if not calculate_arm_lengths:                 
                cv2.putText(color_image, f'elbow_to_hand length: {elbow_to_hand_length:.2f} m w/ Hand SF = {hand_sf} ', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(color_image, f'Upper arm length: {upper_length:.2f} m w/ Elbow SF = {elbow_sf}', (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("RealSense Right Arm Lengths", color_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return hand_sf, elbow_sf

if __name__ == "__main__":
    test_scale_factors()
