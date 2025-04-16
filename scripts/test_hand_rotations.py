"""Ce code permet de tester que le robot est bien connecté"""

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

# Remplacez '192.168.X.X' par l'adresse IP de Reachy
reachy = ReachySDK(host="138.195.196.90")
import numpy as np
import time

ordered_joint_names = [
    reachy.r_arm.r_shoulder_pitch,
    reachy.r_arm.r_shoulder_roll,
    reachy.r_arm.r_arm_yaw,
    reachy.r_arm.r_elbow_pitch,
    reachy.r_arm.r_forearm_yaw,
    reachy.r_arm.r_wrist_pitch,
    reachy.r_arm.r_wrist_roll
]




def test_main_droit() -> None:
    zero_position = {
        reachy.r_arm.r_shoulder_pitch: 0,
        reachy.r_arm.r_shoulder_roll: 0,
        reachy.r_arm.r_arm_yaw: 0,
        reachy.r_arm.r_elbow_pitch: 0,
        reachy.r_arm.r_forearm_yaw: 0,
        reachy.r_arm.r_wrist_pitch: 0,
        reachy.r_arm.r_wrist_roll: 0,
    }

    
    A = np.array([
        [0, 0, -1, 0.3],
        [0, 1, 0, -0.4],  
        [1, 0, 0, -0.3],
        [0, 0, 0, 1],  
        ])
    B = np.array([
        [1, 0, 0, 0.3],
        [0, 1, 0, -0.4],  
        [0, 0, 1, -0.3],
        [0, 0, 0, 1],  
        ])
    C = np.array([
        [1, 0, 0, 0.3],
        [0, 0, -1, -0.4],  
        [0, 1, 0, -0.3],
        [0, 0, 0, 1],  
        ])
    joint_pos_A = reachy.r_arm.inverse_kinematics(A)
    joint_pos_B = reachy.r_arm.inverse_kinematics(B)
    joint_pos_C = reachy.r_arm.inverse_kinematics(C)
    reachy.turn_on("r_arm")

    #goto({joint: pos for joint,pos in zip(ordered_joint_names, joint_pos_A)}, duration=1.0)
    #time.sleep(4)
    goto({joint: pos for joint,pos in zip(ordered_joint_names, joint_pos_B)}, duration=1.0)
    time.sleep(4)
    # for name, joint in reachy.joints.items():
    #     print(f'Joint "{name}" is at pos {joint.present_position} degree.')

    goto({joint: pos for joint,pos in zip(ordered_joint_names, joint_pos_C)}, duration=1.0)
    time.sleep(4)

    goto(
    goal_positions=zero_position,
    duration=1.0,
    interpolation_mode=InterpolationMode.MINIMUM_JERK,
    )

if __name__ == "__main__":
    # Vérifiez si la connexion fonctionne
    for name, joint in reachy.joints.items():
        print(f'Joint "{name}" is at pos {joint.present_position} degree.')

    test_main_droit()
    #test_bras_gauche()
    #test_deux_bras()

    reachy.turn_off_smoothly("r_arm")
    reachy.turn_off_smoothly("l_arm")
    reachy.turn_off_smoothly("head")

