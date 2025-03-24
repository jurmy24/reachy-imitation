"""Ce code permet de tester que le robot est bien connecté"""

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

# Remplacez '192.168.X.X' par l'adresse IP de Reachy
reachy = ReachySDK(host="138.195.196.90")

# Vérifiez si la connexion fonctionne
for name, joint in reachy.joints.items():
    print(f'Joint "{name}" is at pos {joint.present_position} degree.')


def test_bras_droit() -> None:
    right_angled_position = {
        reachy.r_arm.r_shoulder_pitch: 0,
        reachy.r_arm.r_shoulder_roll: 0,
        reachy.r_arm.r_arm_yaw: 0,
        reachy.r_arm.r_elbow_pitch: 0,
        reachy.r_arm.r_forearm_yaw: 0,
        reachy.r_arm.r_wrist_pitch: 0,
        reachy.r_arm.r_wrist_roll: 0,
    }
    reachy.turn_on("r_arm")
    goto(
        goal_positions=right_angled_position,
        duration=1.0,
        interpolation_mode=InterpolationMode.MINIMUM_JERK,
    )

def test_bras_gauche() -> None:
    right_angled_position = {
        reachy.l_arm.l_shoulder_pitch: 0,
        reachy.l_arm.l_shoulder_roll: 30,
        reachy.l_arm.l_arm_yaw: 0,
        reachy.l_arm.l_elbow_pitch: 0,
        reachy.l_arm.l_forearm_yaw: 0,
        reachy.l_arm.l_wrist_pitch: 0,
        reachy.l_arm.l_wrist_roll: 0,
    }
    reachy.turn_on("l_arm")
    goto(
        goal_positions=right_angled_position,
        duration=1.0,
        interpolation_mode=InterpolationMode.MINIMUM_JERK,
    )

def test_deux_bras() -> None:
    angled_position = {
        reachy.l_arm.l_shoulder_pitch: 0,
        reachy.l_arm.l_shoulder_roll: 30,
        reachy.l_arm.l_arm_yaw: 0,
        reachy.l_arm.l_elbow_pitch: 0,
        reachy.l_arm.l_forearm_yaw: 0,
        reachy.l_arm.l_wrist_pitch: 0,
        reachy.l_arm.l_wrist_roll: 0,
        reachy.r_arm.r_shoulder_pitch: 0,
        reachy.r_arm.r_shoulder_roll: -30,
        reachy.r_arm.r_arm_yaw: 0,
        reachy.r_arm.r_elbow_pitch: 0,
        reachy.r_arm.r_forearm_yaw: 0,
        reachy.r_arm.r_wrist_pitch: 0,
        reachy.r_arm.r_wrist_roll: 0,
    }
    reachy.turn_on("l_arm")
    reachy.turn_on("r_arm")
    goto(
        goal_positions=angled_position,
        duration=1.0,
        interpolation_mode=InterpolationMode.MINIMUM_JERK,
    )


if __name__ == "__main__":
    test_bras_droit()
    #test_bras_gauche()
    #test_deux_bras()

    reachy.turn_off_smoothly("r_arm")
    reachy.turn_off_smoothly("l_arm")
    reachy.turn_off_smoothly("head")

