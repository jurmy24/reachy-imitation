import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

def load_all_data():
    """Load all data files from the data directory."""
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    data_files = sorted(data_dir.glob('joint_data_*.csv'))

    if not data_files:
        print(f"No data files found in {data_dir}")
        return []

    print(f"Found {len(data_files)} data files.")
    return data_files

def plot_right_shoulder_pitch_all_files(data_files):
    """Plot r_shoulder_pitch position over time for all CSVs."""
    joint_col = 'r_arm_r_shoulder_pitch_position'
    goal_col = 'r_arm_r_shoulder_pitch_goal_position'
    time_col = 'timestamp'

    # Dark mode styling
    mpl.rcParams['axes.edgecolor'] = 'white'
    mpl.rcParams['xtick.color'] = 'white'
    mpl.rcParams['ytick.color'] = 'white'
    mpl.rcParams['text.color'] = 'white'
    mpl.rcParams['axes.labelcolor'] = 'white'

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
    ax.set_facecolor('black')
    ax.grid(True, linestyle=':', color='white', alpha=0.3)

    # Load goal from first file
    df_first = pd.read_csv(data_files[0])
    if goal_col in df_first.columns:
        ax.plot(df_first[time_col], df_first[goal_col], '--', color='white', linewidth=2, label='Goal')

    # Plot each file's right shoulder pitch
    for i, file_path in enumerate(data_files):
        df = pd.read_csv(file_path)
        if joint_col in df.columns:
            label = f'Trial {i + 1}'
            ax.plot(df[time_col], df[joint_col], linewidth=2, label=label)

    #ax.set_title("Right Shoulder Pitch Position Over Time", fontsize=16, weight='bold')
    ax.set_xlabel("Time (s)", fontsize=20)
    ax.set_ylabel("Position (degrees)", fontsize=20)
    ax.legend(loc='lower right', fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig('right_shoulder_pitch_all_files.png', transparent=True, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    data_files = load_all_data()
    if not data_files:
        return

    plot_right_shoulder_pitch_all_files(data_files)

if __name__ == "__main__":
    main()
