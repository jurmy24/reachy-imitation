import argparse
from reachy_sdk import ReachySDK
from src.pipelines.pipeline_one import Pipeline_one

# Create the overarching Reachy instance for this application
reachy = ReachySDK(host="138.195.196.90")


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run Robot Imitation Pipeline")

    # Add arguments
    parser.add_argument(
        "--arm",
        type=str,
        choices=["right", "left", "both"],
        default="right",
        help="Which arm(s) to track (default: right)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run arm length calibration before tracking",
    )
    parser.add_argument(
        "--only-calibrate",
        action="store_true",
        help="Run only arm length calibration without tracking",
    )

    # Parse arguments
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = Pipeline_one(reachy)

    # Run calibration if requested
    if args.calibrate or args.only_calibrate:
        print("Running initiation protocol...")
        pipeline.initiation_protocol()
        print(
            f"Calibration complete. Hand SF: {pipeline.hand_sf}, Elbow SF: {pipeline.elbow_sf}"
        )

        # Exit early if only calibration was requested
        if args.only_calibrate:
            return

    # Run the main pipeline
    print(f"Tracking {args.arm} arm(s)...")
    pipeline.run(arm=args.arm)

    pipeline.cleanup()

    # Example on how to run:
    # python -m src.main --only-calibrate


if __name__ == "__main__":
    main()
