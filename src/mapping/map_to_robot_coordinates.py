from config.CONSTANTS import LEN_REACHY_UPPERARM, LEN_REACHY_ELBOW_TO_END_EFFECTOR


def get_scale_factors(elbow_to_hand_length, upper_arm_length):
    """
    Calculate the scale factors for the Reachy arm
    """
    hand_sf = (LEN_REACHY_ELBOW_TO_END_EFFECTOR + LEN_REACHY_UPPERARM) / (
        elbow_to_hand_length + upper_arm_length
    )
    elbow_sf = LEN_REACHY_UPPERARM / upper_arm_length
    return hand_sf, elbow_sf
