from abc import ABC, abstractmethod
from typing import Literal
import mediapipe as mp
import pyrealsense2 as rs
import cv2
from reachy_sdk import ReachySDK


class Pipeline(ABC):
    """Base class for all imitation approaches"""

    def __init__(self, reachy: ReachySDK = None):
        # Load configs, initialize components (lightweight)
        self.reachy = reachy
        self.mp_hands = None
        self.mp_pose = None
        self.hands = None
        self.pose = None
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.mp_draw = None
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

        # NOTE: This turns on the entire Reachy robot (i.e. head and both arms on stiff mode)
        if self.reachy is not None:
            self.reachy.turn_on("reachy")

    @abstractmethod
    def initiation_protocol(self):
        """Recognize the human in the frame and calculate the scale factors"""
        pass

    @abstractmethod
    def process_frame(self, **kwargs):
        """Process a single frame of input data"""
        pass

    @abstractmethod
    def display_frame(self, **kwargs):
        """Display the processed frame with visualization options"""
        pass

    @abstractmethod
    def run(
        self, arm: Literal["right", "left", "both"] = "right", display: bool = True
    ):
        pass

    def cleanup(self):
        """Clean up resources - subclasses can override if needed"""
        self.pipeline.stop()
        cv2.destroyAllWindows()
        if self.reachy is not None:
            self.reachy.turn_off_smoothly("reachy")
