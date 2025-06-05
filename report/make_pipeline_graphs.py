import pandas as pd
import matplotlib.pyplot as plt

def make_tracking_plots(sdk_hand, sdk_elbow, custom_hand, custom_elbow, time_steps):
    # Compute hand errors
    # Plotting
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=True)
    # Set an overall title
    fig.suptitle('Live Reachy Hand and Elbow Positions Over Time during left arm Strongman', fontsize=16)

    axes_labels = ['x', 'y', 'z']

    for i, axis in enumerate(axes_labels):
        # Hand errors (left column)
        ax_hand = axes[i][0]
        ax_hand.plot(time_steps, custom_hand[f'hand_target_{axis}'], label='Target', linestyle='--')
        ax_hand.plot(time_steps, custom_hand[f'hand_actual_{axis}'], label='Custom', linestyle='-')
        ax_hand.plot(time_steps, sdk_hand[f'hand_actual_{axis}'], label='SDK', linestyle='-')

        ax_hand.set_ylabel(f'{axis.upper()} (m)')
        #ax_hand.set_title(f'Hand Tracking - {axis.upper()} Axis')
        ax_hand.grid(True)
        if i == 2:
            ax_hand.set_xlabel('Time Step')
        if i == 0:
            ax_hand.legend()

        # Elbow errors (right column)
        ax_elbow = axes[i][1]
        ax_elbow.plot(time_steps, custom_elbow[f'elbow_target_{axis}'], label='Target', linestyle='--')
        ax_elbow.plot(time_steps, custom_elbow[f'elbow_actual_{axis}'], label='Custom', linestyle='-')
        ax_elbow.plot(time_steps, sdk_elbow[f'elbow_actual_{axis}'], label='SDK', linestyle='-')
        ax_elbow.set_ylabel(f'{axis.upper()} (m)')
        #ax_elbow.set_title(f'Elbow Tracking - {axis.upper()} Axis')
        ax_elbow.grid(True)
        if i == 2:
            ax_elbow.set_xlabel('Time Step')
        if i == 0:
            ax_elbow.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
"""
def make_tracking_plots_fancy(sdk_hand, sdk_elbow, custom_hand, custom_elbow, time_steps):
    # Set up plot
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12), sharex=True)
    fig.suptitle('Live Reachy Hand and Elbow Positions Over Time (Left Arm Strongman)', fontsize=18, weight='bold')

    # Axis labels
    axes_labels = ['x', 'y', 'z']
    joint_titles = ['Hand', 'Elbow']
    line_styles = {
        'Target': ('--', 'black'),
        'Custom': ('-', '#1f77b4'),
        'SDK': ('-', '#ff7f0e')
    }

    for i, axis in enumerate(axes_labels):
        # Plot Hand (Left Column)
        ax_hand = axes[i][0]
        ax_hand.plot(time_steps, custom_hand[f'hand_target_{axis}'], label='Target', linestyle=line_styles['Target'][0], color=line_styles['Target'][1], linewidth=2)
        ax_hand.plot(time_steps, custom_hand[f'hand_actual_{axis}'], label='Custom', linestyle=line_styles['Custom'][0], color=line_styles['Custom'][1], linewidth=2)
        ax_hand.plot(time_steps, sdk_hand[f'hand_actual_{axis}'], label='SDK', linestyle=line_styles['SDK'][0], color=line_styles['SDK'][1], linewidth=2)
        ax_hand.set_ylabel(f'{axis.upper()} Position (m)', fontsize=10)
        ax_hand.grid(True, linestyle=':', alpha=0.7)
        if i == 2:
            ax_hand.set_xlabel('Time Step', fontsize=10)
        if i == 0:
            ax_hand.set_title(r'$End \ Effector_{desired}$ and $End \ Effector_{actual}$ vs Time', fontsize=12)

        # Plot Elbow (Right Column)
        ax_elbow = axes[i][1]
        ax_elbow.plot(time_steps, custom_elbow[f'elbow_target_{axis}'], label='Target', linestyle=line_styles['Target'][0], color=line_styles['Target'][1], linewidth=2)
        ax_elbow.plot(time_steps, custom_elbow[f'elbow_actual_{axis}'], label='Custom', linestyle=line_styles['Custom'][0], color=line_styles['Custom'][1], linewidth=2)
        ax_elbow.plot(time_steps, sdk_elbow[f'elbow_actual_{axis}'], label='SDK', linestyle=line_styles['SDK'][0], color=line_styles['SDK'][1], linewidth=2)
        ax_elbow.set_ylabel(f'{axis.upper()} Position (m)', fontsize=10)
        ax_elbow.grid(True, linestyle=':', alpha=0.7)
        if i == 2:
            ax_elbow.set_xlabel('Time Step', fontsize=10)
        if i == 0:
            ax_elbow.set_title(r'$Elbow_{desired}$ and $Elbow_{actual}$ vs Time', fontsize=12)


    # Unified legend (top center)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle and legend
    fig.savefig('tracking_plot.png', dpi=300, transparent=True, bbox_inches='tight')

    plt.show()
"""
def make_tracking_plots_fancy(sdk_hand, sdk_elbow, custom_hand, custom_elbow, time_steps): 
    # Set up plot with wider format and black background
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 8), sharex=True, facecolor='black')
    # fig.suptitle('Live Reachy Hand and Elbow Positions Over Time (Left Arm Strongman)', 
    #              fontsize=20, weight='bold', color='white')
    # fig.text(0.5, -0.04, 'Live Reachy Hand and Elbow Positions Over Time (Left Arm Strongman)', 
    #      ha='center', fontsize=20, weight='bold', color='white')
    # Axis labels and colors
    axes_labels = ['x', 'y', 'z']
    line_styles = {
        'Target': ('--', 'white'),
        'Custom': ('-', '#1f77b4'),
        'SDK': ('-', '#ff7f0e')
    }

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('black')  # Set subplot background
            ax.tick_params(colors='white')  # Tick color
            ax.spines[:].set_color('white')  # Axis spine color
            ax.grid(True, linestyle=':', alpha=0.3, color='white')  # Gridlines

    for i, axis in enumerate(axes_labels):
        # Hand plot (left column)
        ax_hand = axes[i][0]
        ax_hand.plot(time_steps, custom_hand[f'hand_target_{axis}'], label='Target Position',
                     linestyle=line_styles['Target'][0], color=line_styles['Target'][1], linewidth=2)
        ax_hand.plot(time_steps, custom_hand[f'hand_actual_{axis}'], label='Custom IK',
                     linestyle=line_styles['Custom'][0], color=line_styles['Custom'][1], linewidth=2)
        ax_hand.plot(time_steps, sdk_hand[f'hand_actual_{axis}'], label='Simple IK',
                     linestyle=line_styles['SDK'][0], color=line_styles['SDK'][1], linewidth=2)
        ax_hand.set_ylabel(f'{axis.upper()} Position (m)', fontsize=11, color='white')
        if i == 2:
            ax_hand.set_xlabel('Time Step', fontsize=11, color='white')
        if i == 0:
            ax_hand.set_title(r'$End\ Effector_{desired}$ vs $End\ Effector_{actual}$', fontsize=18, color='white')

        # Elbow plot (right column)
        ax_elbow = axes[i][1]
        ax_elbow.plot(time_steps, custom_elbow[f'elbow_target_{axis}'], label='Target',
                      linestyle=line_styles['Target'][0], color=line_styles['Target'][1], linewidth=2)
        ax_elbow.plot(time_steps, custom_elbow[f'elbow_actual_{axis}'], label='Custom',
                      linestyle=line_styles['Custom'][0], color=line_styles['Custom'][1], linewidth=2)
        ax_elbow.plot(time_steps, sdk_elbow[f'elbow_actual_{axis}'], label='SDK',
                      linestyle=line_styles['SDK'][0], color=line_styles['SDK'][1], linewidth=2)
        ax_elbow.set_ylabel(f'{axis.upper()} Position (m)', fontsize=11, color='white')
        if i == 2:
            ax_elbow.set_xlabel('Time Step', fontsize=11, color='white')
        if i == 0:
            ax_elbow.set_title(r'$Elbow_{desired}$ vs $Elbow_{actual}$', fontsize=14, color='white')

    # Legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
               ncol=3, frameon=False, fontsize=20, facecolor='black', labelcolor='white')

    # Layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('tracking_plot_dark.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

    # Time steps
def make_error_plots(sdk_hand, sdk_elbow, custom_hand, custom_elbow, time_steps):
    # Compute hand errors
    sdk_hand_errors = {
        'x': sdk_hand['hand_target_x'] - sdk_hand['hand_actual_x'],
        'y': sdk_hand['hand_target_y'] - sdk_hand['hand_actual_y'],
        'z': sdk_hand['hand_target_z'] - sdk_hand['hand_actual_z']
    }
    custom_hand_errors = {
        'x': custom_hand['hand_target_x'] - custom_hand['hand_actual_x'],
        'y': custom_hand['hand_target_y'] - custom_hand['hand_actual_y'],
        'z': custom_hand['hand_target_z'] - custom_hand['hand_actual_z']
    }

    # Compute elbow errors
    sdk_elbow_errors = {
        'x': sdk_elbow['elbow_target_x'] - sdk_elbow['elbow_actual_x'],
        'y': sdk_elbow['elbow_target_y'] - sdk_elbow['elbow_actual_y'],
        'z': sdk_elbow['elbow_target_z'] - sdk_elbow['elbow_actual_z']
    }
    custom_elbow_errors = {
        'x': custom_elbow['elbow_target_x'] - custom_elbow['elbow_actual_x'],
        'y': custom_elbow['elbow_target_y'] - custom_elbow['elbow_actual_y'],
        'z': custom_elbow['elbow_target_z'] - custom_elbow['elbow_actual_z']
    }

    # Plotting
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=True)

    axes_labels = ['x', 'y', 'z']

    for i, axis in enumerate(axes_labels):
        # Hand errors (left column)
        ax_hand = axes[i][0]
        ax_hand.plot(time_steps, sdk_hand_errors[axis], label='SDK', linestyle='--')
        ax_hand.plot(time_steps, custom_hand_errors[axis], label='Custom', linestyle='-')
        ax_hand.set_ylabel(f'{axis.upper()} Error')
        ax_hand.grid(True)
        if i == 2:
            ax_hand.set_xlabel('Time Step')
        if i == 0:
            ax_hand.legend()
            ax_hand.set_title(f'Hand Tracking - {axis.upper()} Axis')


        # Elbow errors (right column)
        ax_elbow = axes[i][1]
        ax_elbow.plot(time_steps, sdk_elbow_errors[axis], label='SDK', linestyle='--')
        ax_elbow.plot(time_steps, custom_elbow_errors[axis], label='Custom', linestyle='-')
        ax_elbow.set_ylabel(f'{axis.upper()} Error')
        ax_elbow.grid(True)
        if i == 2:
            ax_elbow.set_xlabel('Time Step')
        if i == 0:
            ax_elbow.legend()
            ax_elbow.set_title(f'Elbow Tracking - {axis.upper()} Axis')


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the first 181 rows from each CSV file
    sdk_hand = pd.read_csv('sdk_hand_tracking_data.csv').iloc[:181]
    custom_hand = pd.read_csv('custom_hand_tracking_data.csv').iloc[:181]
    time_steps = custom_hand['Time Step']

    sdk_elbow = pd.read_csv('sdk_elbow_tracking_data.csv').iloc[:181]
    custom_elbow = pd.read_csv('custom_elbow_tracking_data.csv').iloc[:181]

    #make_error_plots(sdk_hand, sdk_elbow, custom_hand, custom_elbow, time_steps)
    #make_tracking_plots(sdk_hand, sdk_elbow, custom_hand, custom_elbow, time_steps)
    make_tracking_plots_fancy(sdk_hand, sdk_elbow, custom_hand, custom_elbow, time_steps)