import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import cv2
from src.utils.config_loader import get_robot_dimensions
from src.mapping.get_arm_lengths import get_arm_lengths

# Load robot dimensions from config
robot_config = get_robot_dimensions()

LEN_REACHY_UPPERARM = robot_config["reachy"]["right_arm"]["upper_arm"]["length"]
LEN_REACHY_ELBOW_TO_END_EFFECTOR = robot_config["reachy"]["right_arm"][
    "elbow_to_end_effector"
]["length"]


def get_scale_factors(forearm_length, lower_length):
    """
    Calculate the scale factors for the Reachy arm
    """
    hand_sf = (LEN_REACHY_ELBOW_TO_END_EFFECTOR + LEN_REACHY_UPPERARM) / (
        forearm_length + lower_length
    )
    elbow_sf = LEN_REACHY_UPPERARM / forearm_length
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
            intrinsics = (
                pipeline.get_active_profile()
                .get_stream(rs.stream.depth)
                .as_video_stream_profile()
                .get_intrinsics()
            )

            pose_results = pose.process(rgb_image)

            if pose_results.pose_landmarks:

                mp_draw.draw_landmarks(
                    color_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                if calculate_arm_lengths:
                    forearm_length, upper_length = get_arm_lengths(
                        pose_results.pose_landmarks,
                        mp_pose,
                        depth_frame,
                        w,
                        h,
                        intrinsics,
                    )
                    if forearm_length is not None and upper_length is not None:
                        forearm_lengths = np.append(forearm_lengths, forearm_length)
                        upperarm_lengths = np.append(upperarm_lengths, upper_length)
                        if len(forearm_lengths) > 100:
                            forearm_length = np.median(forearm_lengths)
                            upper_length = np.median(upperarm_lengths)
                            calculate_arm_lengths = False
                            hand_sf, elbow_sf = get_scale_factors(
                                forearm_length, upper_length
                            )
                        cv2.putText(
                            color_image,
                            f"Forearm length: {forearm_length:.2f} m",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        cv2.putText(
                            color_image,
                            f"Lower arm length: {upper_length:.2f} m",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                    else:
                        cv2.putText(
                            color_image,
                            f"Mettez tout votre bras droit dans le cadre!",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
            elif calculate_arm_lengths:
                cv2.putText(
                    color_image,
                    f"Mettez tout votre bras droit dans le cadre!",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            if not calculate_arm_lengths:
                cv2.putText(
                    color_image,
                    f"Forearm length: {forearm_length:.2f} m w/ Hand SF = {hand_sf} ",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    color_image,
                    f"Lower arm length: {upper_length:.2f} m w/ Elbow SF = {elbow_sf}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("RealSense Right Arm Lengths", color_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return hand_sf, elbow_sf


if __name__ == "__main__":
    test_scale_factors()
