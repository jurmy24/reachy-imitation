"""Script to visualize elbow and shoulder joint data from CSV files."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np


def load_all_data():
    """Load all data files from the data directory."""
    # Get the absolute path to the data directory
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "data"

    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)

    # List all CSV files
    data_files = sorted(data_dir.glob("joint_data_*.csv"))

    if not data_files:
        print(f"No data files found in {data_dir}")
        print("Please run collect_joint_data.py first to generate some data files.")
        return []

    print(f"Found {len(data_files)} data files in {data_dir}")
    return data_files


def plot_joint_data(all_dfs, filenames):
    """Plot elbow and shoulder joint positions for all data files on the same graphs."""
    # Define the joints we want to plot with their exact column names
    joints = {
        "shoulder_pitch": {
            "present": "r_arm_r_shoulder_pitch_position",
            "goal": "r_arm_r_shoulder_pitch_goal_position",
        },
        "shoulder_roll": {
            "present": "r_arm_r_shoulder_roll_position",
            "goal": "r_arm_r_shoulder_roll_goal_position",
        },
        "arm_yaw": {
            "present": "r_arm_r_arm_yaw_position",
            "goal": "r_arm_r_arm_yaw_goal_position",
        },
        "elbow_pitch": {
            "present": "r_arm_r_elbow_pitch_position",
            "goal": "r_arm_r_elbow_pitch_goal_position",
        },
    }

    # Create a figure with subplots for each joint
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # fig.suptitle("Right Arm Joint Positions - All Trajectories", fontsize=16)
    axes = axes.flatten()

    # Define colors for different trajectories
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Plot each joint
    for idx, (joint_name, columns) in enumerate(joints.items()):
        ax = axes[idx]
        has_data = False

        # Plot each trajectory
        for df_idx, (df, filename) in enumerate(zip(all_dfs, filenames)):
            # Plot present position
            if columns["present"] in df.columns:
                has_data = True
                ax.plot(
                    df["timestamp"],
                    df[columns["present"]],
                    color=colors[df_idx],
                    label=f"Trial {filename.split('_')[2].strip('.csv')}",
                    linewidth=1.5,
                    alpha=0.9,
                )

            # Plot goal position with thicker red dotted line
            if columns["goal"] in df.columns:
                has_data = True
                ax.plot(
                    df["timestamp"],
                    df[columns["goal"]],
                    color="red",
                    linestyle="--",
                    label=f"Goal position" if df_idx == 0 else None,
                    linewidth=2.5,
                    alpha=0.9,
                )

        if has_data:
            ax.set_title(joint_name.replace("_", " ").title())
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Position (degrees)")
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.set_visible(False)

    plt.tight_layout()

    # Save the figure
    current_dir = Path(__file__).parent
    graphs_dir = current_dir.parent / "data" / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    # Create filename for the graph
    save_path = graphs_dir / "all_trajectories.png"

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved graph to {save_path}")
    plt.close()


def main():
    """Main function to run the visualization."""
    # Load all data files
    data_files = load_all_data()

    if not data_files:
        return

    # Load all dataframes
    all_dfs = []
    filenames = []
    for file_path in data_files:
        print(f"\nProcessing {file_path.name}")
        try:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
            filenames.append(file_path.name)
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")

    if all_dfs:
        plot_joint_data(all_dfs, filenames)


if __name__ == "__main__":
    main()
