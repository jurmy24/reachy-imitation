from src import ImitationPipeline
import mediapipe as mp
import pyrealsense2 as rs
import cv2
import numpy as np
from src.mapping.get_arm_lengths import get_arm_lengths
from src.mapping.map_to_robot_coordinates import get_scale_factors


class RobotModelPipeline(ImitationPipeline):
    """Approach 1: Uses robot arm model with IK"""

    def __init__(self):
        # Load configs, initialize components (lightweight)
        self.mp_draw = None
        super().__init__()

    def recognize_human(self):
        self.mp_draw = mp.solutions.drawing_utils

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                h, w, _ = color_image.shape
                intrinsics = (
                    self.pipeline.get_active_profile()
                    .get_stream(rs.stream.depth)
                    .as_video_stream_profile()
                    .get_intrinsics()
                )

                pose_results = self.pose.process(rgb_image)

                if pose_results.pose_landmarks:

                    self.mp_draw.draw_landmarks(
                        color_image,
                        pose_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                    )

                    if calculate_arm_lengths:
                        forearm_length, upper_length = get_arm_lengths(
                            pose_results.pose_landmarks,
                            self.mp_pose,
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
        except Exception as e:
            print(f"Failed to run the recognize human pipeline: {e}")

        return hand_sf, elbow_sf

    def process_frame(self):
        # Implementation specific to approach 1
        pass

    def run(self):
        # Implementation specific to approach 1
        pass

    def cleanup(self):
        pass
