"""Script to visualize elbow and shoulder joint data from CSV files."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

def load_all_data():
    """Load all data files from the data directory."""
    # Get the absolute path to the data directory
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / 'data'
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    # List all CSV files
    data_files = sorted(data_dir.glob('joint_data_*.csv'))
    
    if not data_files:
        print(f"No data files found in {data_dir}")
        print("Please run collect_joint_data.py first to generate some data files.")
        return []
    
    print(f"Found {len(data_files)} data files in {data_dir}")
    return data_files

def plot_joint_data(df, filename):
    """Plot elbow and shoulder joint positions for a single data file."""
    # Print data information for debugging
    print(f"\nData shape: {df.shape}")
    
    # Define the joints we want to plot with their exact column names
    joints = {
        'shoulder_pitch': {
            'l_present': 'l_arm_l_shoulder_pitch_position',
            'l_goal': 'l_arm_l_shoulder_pitch_goal_position',
            'r_present': 'r_arm_r_shoulder_pitch_position',
            'r_goal': 'r_arm_r_shoulder_pitch_goal_position'
        },
        'shoulder_roll': {
            'l_present': 'l_arm_l_shoulder_roll_position',
            'l_goal': 'l_arm_l_shoulder_roll_goal_position',
            'r_present': 'r_arm_r_shoulder_roll_position',
            'r_goal': 'r_arm_r_shoulder_roll_goal_position'
        },
        'arm_yaw': {
            'l_present': 'l_arm_l_arm_yaw_position',
            'l_goal': 'l_arm_l_arm_yaw_goal_position',
            'r_present': 'r_arm_r_arm_yaw_position',
            'r_goal': 'r_arm_r_arm_yaw_goal_position'
        },
        'elbow_pitch': {
            'l_present': 'l_arm_l_elbow_pitch_position',
            'l_goal': 'l_arm_l_elbow_pitch_goal_position',
            'r_present': 'r_arm_r_elbow_pitch_position',
            'r_goal': 'r_arm_r_elbow_pitch_goal_position'
        }
    }
    
    # Create a figure with subplots for each joint
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Joint Positions - {filename}', fontsize=16)
    axes = axes.flatten()
    
    # Plot each joint
    for idx, (joint_name, columns) in enumerate(joints.items()):
        ax = axes[idx]
        has_data = False
        
        # Plot left arm
        if columns['l_present'] in df.columns and columns['l_goal'] in df.columns:
            has_data = True
            ax.plot(df['timestamp'], df[columns['l_present']], label='Left Arm (Present)')
            ax.plot(df['timestamp'], df[columns['l_goal']], '--', label='Left Arm (Goal)')
        
        # Plot right arm
        if columns['r_present'] in df.columns and columns['r_goal'] in df.columns:
            has_data = True
            ax.plot(df['timestamp'], df[columns['r_present']], label='Right Arm (Present)')
            ax.plot(df['timestamp'], df[columns['r_goal']], '--', label='Right Arm (Goal)')
        
        if has_data:
            ax.set_title(joint_name.replace('_', ' ').title())
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position (degrees)')
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure
    current_dir = Path(__file__).parent
    graphs_dir = current_dir.parent / 'data' / 'graphs'
    graphs_dir.mkdir(exist_ok=True)
    
    # Create filename for the graph
    graph_filename = filename.replace('.csv', '.png')
    save_path = graphs_dir / graph_filename
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved graph to {save_path}")
    plt.close()

def main():
    """Main function to run the visualization."""
    # Load all data files
    data_files = load_all_data()
    
    if not data_files:
        return
    
    # Plot each file separately
    for file_path in data_files:
        print(f"\nProcessing {file_path.name}")
        try:
            df = pd.read_csv(file_path)
            plot_joint_data(df, file_path.name)
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")

if __name__ == "__main__":
    main() 