from reachy_sdk import ReachySDK
from src.mapping.map_to_robot_coordinates import get_scale_factors
from src.pipelines.pipeline_one_mini import Pipeline_one_mini
from config.CONSTANTS import HUMAN_ELBOW_TO_HAND_DEFAULT, HUMAN_UPPERARM_DEFAULT

# Create the overarching Reachy instance for this application
reachy = ReachySDK(host="138.195.196.90")


async def main():
    calibrate = False
    arm = "left"

    pipeline = Pipeline_one_mini(reachy)
    if calibrate:
        print("Running initiation protocol...")
        pipeline.initiation_protocol()
        print(
            f"Calibrated. Hand SF: {pipeline.hand_sf}, Elbow SF: {pipeline.elbow_sf}")
    else:
        print("Using default scale factors")
        hand_sf, elbow_sf = get_scale_factors(
            HUMAN_ELBOW_TO_HAND_DEFAULT, HUMAN_UPPERARM_DEFAULT
        )
        pipeline.hand_sf = hand_sf
        pipeline.elbow_sf = elbow_sf

    # Run the main pipeline
    print(f"Tracking {arm} arm(s)...")
    await pipeline.shadow(side=arm, display=True)
    # pipeline.cleanup()

    # Example on how to run:
    # python -m src.main


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
