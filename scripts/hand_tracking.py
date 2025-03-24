from reachy_sdk import ReachySDK
import time

reachy = ReachySDK(host="138.195.196.90")

reachy.turn_on('head')

x, y, z = reachy.r_arm.forward_kinematics()[:3, -1]
reachy.head.look_at(x=x, y=y, z=z, duration=1.0)

time.sleep(0.5)

while True:
    x, y, z = reachy.r_arm.forward_kinematics()[:3, -1]
    reachy.head.look_at(x=x, y=y, z=z, duration=0.1)

reachy.turn_off_smoothly("reachy")
