
import numpy as np
import math
import mediapipe as mp
import pyrealsense2 as rs
import cv2




L_REACHY_FOREARM = 0.28 # distance (m) from shoulder to elbow
L_REACHY_UPPERARM = 0.25 + 0.0325 # distance (m) from elbow to wrist


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
            return np.array([-depth, -x, -y])
    return np.array([0, 0, 0])  # Valeur par défaut si aucune profondeur valide

def get_arm_lengths(pose_landmarks, mp_pose, depth_frame, w, h, intrinsics):
    required_landmarks = [
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_WRIST
    ]
    for landmark in required_landmarks:
        if not (0.5 < pose_landmarks.landmark[landmark].visibility <= 1):
            return None, None
        
    r_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    r_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    r_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    r_elbow_3d = get_3d_coordinates(r_elbow, depth_frame, w, h, intrinsics)
    r_shoulder_3d = get_3d_coordinates(r_shoulder, depth_frame, w, h, intrinsics)
    r_wrist_3d = get_3d_coordinates(r_wrist, depth_frame, w, h, intrinsics)

    forearm_length = np.linalg.norm(r_elbow_3d - r_shoulder_3d)
    upper_length = np.linalg.norm(r_elbow_3d - r_wrist_3d)
    return forearm_length, upper_length

def get_scale_factors(forearm_length, lower_length):
    """
    Calculate the scale factors for the Reachy arm
    """
    hand_sf = (L_REACHY_FOREARM+L_REACHY_UPPERARM) / (forearm_length+lower_length)
    elbow_sf = L_REACHY_UPPERARM / forearm_length
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

    forearm_lengths = np.array([])
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
                    forearm_length, upper_length = get_arm_lengths(pose_results.pose_landmarks, mp_pose, depth_frame, w, h, intrinsics)
                    if forearm_length is not None and upper_length is not None:
                        forearm_lengths = np.append(forearm_lengths, forearm_length)
                        upperarm_lengths = np.append(upperarm_lengths, upper_length)
                        if len(forearm_lengths) > 100:
                            forearm_length = np.median(forearm_lengths)
                            upper_length = np.median(upperarm_lengths)
                            calculate_arm_lengths = False
                            hand_sf, elbow_sf = get_scale_factors(forearm_length, upper_length)
                        cv2.putText(color_image, f'Forearm length: {forearm_length:.2f} m', (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(color_image, f'Lower arm length: {upper_length:.2f} m', (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else: 
                        cv2.putText(color_image, f'Mettez tout votre bras droit dans le cadre!', (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            elif calculate_arm_lengths: 
                cv2.putText(color_image, f'Mettez tout votre bras droit dans le cadre!', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


                        
            if not calculate_arm_lengths:                 
                cv2.putText(color_image, f'Forearm length: {forearm_length:.2f} m w/ Hand SF = {hand_sf} ', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(color_image, f'Lower arm length: {upper_length:.2f} m w/ Elbow SF = {elbow_sf}', (10, 100),
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
