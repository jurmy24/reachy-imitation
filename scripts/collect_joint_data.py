"""Script to collect joint data during robot movement and store it in a CSV file."""

import csv
import time
from datetime import datetime
from threading import Thread
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

# Initialize Reachy
reachy = ReachySDK(host="138.195.196.90")


def collect_joint_data(duration=3.0, sampling_rate=0.1):
    """Collect joint data and save it to a CSV file.

    Args:
        duration: Total duration of data collection in seconds
        sampling_rate: Time between samples in seconds
    """
    # Create CSV filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"joint_data_{timestamp}.csv"

    # Define the strong man pose
    strong_man_pose = {
        reachy.l_arm.l_shoulder_pitch: 0,
        reachy.l_arm.l_shoulder_roll: 90,
        reachy.l_arm.l_arm_yaw: 90,
        reachy.l_arm.l_elbow_pitch: -90,
        reachy.l_arm.l_forearm_yaw: 0,
        reachy.l_arm.l_wrist_pitch: 0,
        reachy.l_arm.l_wrist_roll: 0,
        reachy.r_arm.r_shoulder_pitch: 0,
        reachy.r_arm.r_shoulder_roll: -90,
        reachy.r_arm.r_arm_yaw: -90,
        reachy.r_arm.r_elbow_pitch: -90,
        reachy.r_arm.r_forearm_yaw: 0,
        reachy.r_arm.r_wrist_pitch: 0,
        reachy.r_arm.r_wrist_roll: 0,
    }

    # Turn on the arms
    reachy.turn_on("l_arm")
    reachy.turn_on("r_arm")

    # Prepare CSV file
    with open(filename, "w", newline="") as csvfile:
        # Create CSV writer
        writer = csv.writer(csvfile)

        # Write header
        header = ["timestamp"]
        for arm in ["l_arm", "r_arm"]:
            for joint in reachy.__getattribute__(arm).joints:
                header.extend(
                    [
                        f"{arm}_{joint.name}_position",
                        f"{arm}_{joint.name}_temperature",
                        f"{arm}_{joint.name}_goal_position",
                        f"{arm}_{joint.name}_pid",
                    ]
                )
        writer.writerow(header)

        # Set up recording thread
        running = True
        data = []

        def record():
            start_time = time.time()
            while running:
                current_time = time.time() - start_time
                row = [current_time]
                for arm in ["l_arm", "r_arm"]:
                    for joint in reachy.__getattribute__(arm).joints:
                        row.extend(
                            [
                                joint.present_position,
                                joint.temperature,
                                joint.goal_position,
                                joint.pid,
                            ]
                        )
                data.append(row)
                time.sleep(sampling_rate)

        # Start recording thread
        t = Thread(target=record)
        t.start()

        # Perform movement
        goto(
            goal_positions=strong_man_pose,
            duration=duration,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )

        # Stop recording
        running = False
        t.join()

        # Write all recorded data to CSV
        for row in data:
            writer.writerow(row)

        # Return to zero position
        zero_pose = {
            reachy.l_arm.l_shoulder_pitch: 0,
            reachy.l_arm.l_shoulder_roll: 0,
            reachy.l_arm.l_arm_yaw: 0,
            reachy.l_arm.l_elbow_pitch: 0,
            reachy.l_arm.l_forearm_yaw: 0,
            reachy.l_arm.l_wrist_pitch: 0,
            reachy.l_arm.l_wrist_roll: 0,
            reachy.r_arm.r_shoulder_pitch: 0,
            reachy.r_arm.r_shoulder_roll: 0,
            reachy.r_arm.r_arm_yaw: 0,
            reachy.r_arm.r_elbow_pitch: 0,
            reachy.r_arm.r_forearm_yaw: 0,
            reachy.r_arm.r_wrist_pitch: 0,
            reachy.r_arm.r_wrist_roll: 0,
        }

        goto(
            goal_positions=zero_pose,
            duration=3.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )

    # Turn off the arms
    reachy.turn_off_smoothly("r_arm")
    reachy.turn_off_smoothly("l_arm")

    print(f"Data collection complete. Data saved to {filename}")


if __name__ == "__main__":
    collect_joint_data(duration=5.0, sampling_rate=0.1)
