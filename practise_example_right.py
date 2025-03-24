from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto

import time
from reachy_sdk.trajectory.interpolation import InterpolationMode

reachy = ReachySDK(host="138.195.196.90")
import numpy as np


ZERO_RIGHT_POS = {
    reachy.r_arm.r_shoulder_pitch: 0,
    reachy.r_arm.r_shoulder_roll: 0,
    reachy.r_arm.r_arm_yaw: 0,
    reachy.r_arm.r_elbow_pitch: 0,
    reachy.r_arm.r_forearm_yaw: 0,
    reachy.r_arm.r_wrist_pitch: 0,
    reachy.r_arm.r_wrist_roll: 0,
}

ordered_joint_names = [
    reachy.r_arm.r_shoulder_pitch,
    reachy.r_arm.r_shoulder_roll,
    reachy.r_arm.r_arm_yaw,
    reachy.r_arm.r_elbow_pitch,
    reachy.r_arm.r_forearm_yaw,
    reachy.r_arm.r_wrist_pitch,
    reachy.r_arm.r_wrist_roll
]

A = np.array([
  [0, 0, -1, 0.3],
  [0, 1, 0, -0.4],  
  [1, 0, 0, -0.3],
  [0, 0, 0, 1],  
])

B = np.array([
  [0, 0, -1, 0.3],
  [0, 1, 0, -0.4],  
  [1, 0, 0, 0.0],
  [0, 0, 0, 1],  
])

C = np.array([
  [0, 0, -1, 0.3],
  [0, 1, 0, -0.1],  
  [1, 0, 0, 0.0],
  [0, 0, 0, 1],  
])

D = np.array([
  [0, 0, -1, 0.3],
  [0, 1, 0, -0.1],  
  [1, 0, 0, -0.3],
  [0, 0, 0, 1],  
])

joint_pos_A = reachy.r_arm.inverse_kinematics(A)
joint_pos_B = reachy.r_arm.inverse_kinematics(B)
joint_pos_C = reachy.r_arm.inverse_kinematics(C)
joint_pos_D = reachy.r_arm.inverse_kinematics(D)


# print(joint_pos_A)
# print(joint_pos_B)
# print(joint_pos_C)
# print(joint_pos_D)
# print(reachy.r_arm.joints.values())

# put the joints in stiff mode
reachy.turn_on('r_arm')

# use the goto function

goto({joint: pos for joint,pos in zip(ordered_joint_names, joint_pos_A)}, duration=2.0)
time.sleep(1)
goto({joint: pos for joint,pos in zip(ordered_joint_names, joint_pos_B)}, duration=2.0)
time.sleep(0.5)
goto({joint: pos for joint,pos in zip(ordered_joint_names, joint_pos_C)}, duration=2.0)
time.sleep(0.5)

#print(list(zip(ordered_joint_names, joint_pos_D)))
#print({joint: pos for joint,pos in list(zip(ordered_joint_names, joint_pos_D))})
goto({joint: pos for joint,pos in list(zip(ordered_joint_names, joint_pos_D))}, duration=2.0)
time.sleep(1)
goto(
    ZERO_RIGHT_POS,
    duration=2.0,
    interpolation_mode=InterpolationMode.MINIMUM_JERK,
)

# put the joints back to compliant mode
# use turn_off_smoothly to prevent the arm from falling hard
reachy.turn_off_smoothly('r_arm')
