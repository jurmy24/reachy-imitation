#!/usr/bin/env python3
"""
Test script to demonstrate how RealSense's wait_for_frames() function works.
This script visualizes whether wait_for_frames() returns the most recent frame
or one from the backlog when processing is slower than the camera's frame rate.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time
import argparse
from datetime import datetime


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test RealSense frame acquisition")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Artificial delay in seconds to simulate slow processing",
    )
    parser.add_argument(
        "--frames", type=int, default=100, help="Number of frames to process"
    )
    parser.add_argument(
        "--display", action="store_true", help="Display frames in a window"
    )
    args = parser.parse_args()

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start the pipeline
    profile = pipeline.start(config)

    # Get the device's frame rate
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    frame_rate = depth_sensor.get_supported_modes()[0].fps
    print(f"Camera frame rate: {frame_rate} FPS")
    print(f"Simulated processing delay: {args.delay} seconds")
    print(f"Expected frames dropped per cycle: {frame_rate * args.delay}")

    # Initialize frame counters and timers
    frame_count = 0
    start_time = time.time()
    last_frame_time = start_time
    frame_times = []
    frame_numbers = []

    try:
        while frame_count < args.frames:
            # Get the current time before waiting for frames
            pre_wait_time = time.time()

            # Wait for the next frame
            frames = pipeline.wait_for_frames()

            # Get the current time after receiving the frame
            post_wait_time = time.time()

            # Get the color frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Get the frame number from the metadata
            frame_number = color_frame.get_frame_number()

            # Calculate time since last frame
            current_time = time.time()
            time_since_last = current_time - last_frame_time
            last_frame_time = current_time

            # Store frame timing information
            frame_times.append(time_since_last)
            frame_numbers.append(frame_number)

            # Add text to the image
            cv2.putText(
                color_image,
                f"Frame: {frame_number}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                color_image,
                f"Time since last: {time_since_last:.3f}s",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                color_image,
                f"Wait time: {(post_wait_time - pre_wait_time)*1000:.1f}ms",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Add a timestamp to the image
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            cv2.putText(
                color_image,
                f"Time: {timestamp}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Display the image if requested
            if args.display:
                cv2.imshow("RealSense Frame Test", color_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Simulate processing delay
            time.sleep(args.delay)

            # Increment frame counter
            frame_count += 1

            # Print progress every 10 frames
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{args.frames} frames")

    finally:
        # Stop the pipeline
        pipeline.stop()
        cv2.destroyAllWindows()

        # Calculate and print statistics
        total_time = time.time() - start_time
        avg_frame_time = np.mean(frame_times) if frame_times else 0
        min_frame_time = np.min(frame_times) if frame_times else 0
        max_frame_time = np.max(frame_times) if frame_times else 0

        print("\n===== FRAME ACQUISITION STATISTICS =====")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average frame time: {avg_frame_time*1000:.2f} ms")
        print(f"Min frame time: {min_frame_time*1000:.2f} ms")
        print(f"Max frame time: {max_frame_time*1000:.2f} ms")
        print(f"Effective frame rate: {frame_count/total_time:.2f} FPS")

        # Check for frame number gaps
        if len(frame_numbers) > 1:
            gaps = [
                frame_numbers[i + 1] - frame_numbers[i]
                for i in range(len(frame_numbers) - 1)
            ]
            max_gap = max(gaps)
            avg_gap = sum(gaps) / len(gaps)
            print(f"Average frame number gap: {avg_gap:.2f}")
            print(f"Maximum frame number gap: {max_gap}")

            if max_gap > 1:
                print("\nWARNING: Frame number gaps detected!")
                print(
                    "This indicates that wait_for_frames() is not returning the most recent frame."
                )
                print("Instead, it's returning frames from the backlog.")
            else:
                print("\nNo frame number gaps detected.")
                print(
                    "This suggests that wait_for_frames() is returning the most recent frame."
                )

        # Save frame timing data to a file
        with open("frame_timing_data.csv", "w") as f:
            f.write("Frame Number,Time Since Last Frame (s)\n")
            for i, (frame_num, frame_time) in enumerate(
                zip(frame_numbers, frame_times)
            ):
                f.write(f"{frame_num},{frame_time}\n")
        print("\nFrame timing data saved to 'frame_timing_data.csv'")


if __name__ == "__main__":
    main()
