"""Imosé une position spécifique aux 2 bras"""

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import numpy as np

# Remplacez '192.168.X.X' par l'adresse IP de Reachy
reachy = ReachySDK(host='138.195.196.90')

# Vérifiez si la connexion fonctionne
# for name, joint in reachy.joints.items():
#    print(f'Joint "{name}" is at pos {joint.present_position} degree.')


angled_position = {
    reachy.l_arm.l_shoulder_pitch: 0,
    reachy.l_arm.l_shoulder_roll: 90,
    reachy.l_arm.l_arm_yaw: -90,
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


reachy.turn_on('l_arm')
reachy.turn_on('r_arm')

goto(
    goal_positions=angled_position,
    duration=1.0,
    interpolation_mode=InterpolationMode.MINIMUM_JERK
)


reachy.turn_off_smoothly('l_arm')
reachy.turn_off_smoothly('r_arm')
