# Implement something like this when running imitations in the different approaches
from abc import ABC, abstractmethod
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import cv2


class ImitationPipeline(ABC):
    """Base class for all imitation approaches"""

    def __init__(self):
        # Load configs, initialize components (lightweight)
        self.mp_hands = None
        self.mp_pose = None
        self.hands = None
        self.pose = None
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.initialize()

    def initialize(self):
        """Setup components required for this imitation approach"""
        # Initialize MediaPipe for hand and body point map detection
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

        # Create model instances with optimized parameters
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            smooth_landmarks=True,
        )

        # Configure intel RealSense camera (color and depth streams)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Align depth frame to color frame
        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(config)

        # Cache intrinsics for repeated use
        self.intrinsics = (
            profile.get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )

    @abstractmethod
    def recognize_human(self):
        """Recognize the human in the frame and calculate the scale factors"""
        pass

    @abstractmethod
    def process_frame(self):
        """Process a single frame of input data"""
        pass

    def run(self):
        """Main processing loop - may be overridden by subclasses if needed"""
        self.initialize()
        try:
            while True:
                # Get frames from RealSense camera
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                # Check if frames are valid, if not, skip
                if not color_frame or not depth_frame:
                    continue

                # OpenCV uses BGR format, MediaPipe uses RGB format, so we convert the color image
                color_image = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                # Get height and width of the color image
                h, w, _ = color_image.shape
                self.process_frame()
        finally:
            self.cleanup()

    @abstractmethod
    def cleanup(self):
        """Clean up resources - subclasses can override if needed"""
        pass
