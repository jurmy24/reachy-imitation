from reachy_sdk import ReachySDK
from src.mapping.map_to_robot_coordinates import get_scale_factors
from src.pipelines.pipeline_custom_ik_with_gripper import (
    Pipeline_custom_ik_with_gripper,
)
from src.pipelines.pipeline_custom_ik_force_gripper import (
    Pipeline_custom_ik_force_gripper,
)
from src.pipelines.pipeline_custom_ik import Pipeline_custom_ik
from config.CONSTANTS import HUMAN_ELBOW_TO_HAND_DEFAULT, HUMAN_UPPERARM_DEFAULT

# Initialize connection to the Reachy robot
reachy = ReachySDK(host="192.168.100.2")


def main():
    """
    Main function to control the Reachy robot's arm movement.
    Supports three modes:
    1. Calibration mode - runs initiation protocol
    2. Human tracking mode - watches human movement
    3. Default mode - uses predefined scale factors
    """
    # Configuration parameters
    calibrate = False
    watch_human = True
    arm = "left"

    # Initialize the pipeline with force gripper control (alternatives are with normal gripper or without gripper)
    pipeline = Pipeline_custom_ik_force_gripper(reachy)

    # Handle different operation modes
    if calibrate:
        print("Running initiation protocol...")
        pipeline.initiation_protocol()
        print(f"Calibrated. Hand SF: {pipeline.hand_sf}, Elbow SF: {pipeline.elbow_sf}")

    elif watch_human:
        print("Running human tracking protocol...")
        pipeline._watch_human()
        print("Using default scale factors")
        hand_sf, elbow_sf = get_scale_factors(
            HUMAN_ELBOW_TO_HAND_DEFAULT, HUMAN_UPPERARM_DEFAULT
        )
        # Convert to numpy float64 to match expected types
        pipeline.hand_sf = float(hand_sf)
        pipeline.elbow_sf = float(elbow_sf)

    else:
        print("Using default scale factors")
        hand_sf, elbow_sf = get_scale_factors(
            HUMAN_ELBOW_TO_HAND_DEFAULT, HUMAN_UPPERARM_DEFAULT
        )
        # Convert to numpy float64 to match expected types
        pipeline.hand_sf = float(hand_sf)
        pipeline.elbow_sf = float(elbow_sf)

    # Execute the main movement tracking pipeline
    print(f"Tracking {arm} arm(s)...")
    pipeline.shadow(side=arm, display=True)


if __name__ == "__main__":
    main()
