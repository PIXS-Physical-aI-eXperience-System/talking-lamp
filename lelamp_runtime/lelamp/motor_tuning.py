"""Motor gains chosen for the LeLamp's loaded and unloaded joints."""

DEFAULT_POSITION_P_COEFFICIENT = 16
LOADED_PITCH_POSITION_P_COEFFICIENT = 32
LOADED_PITCH_JOINTS = frozenset({"base_pitch", "elbow_pitch"})


def position_p_coefficient(motor: str) -> int:
    """Return enough position gain for loaded pitch joints to resist gravity."""
    if motor in LOADED_PITCH_JOINTS:
        return LOADED_PITCH_POSITION_P_COEFFICIENT
    return DEFAULT_POSITION_P_COEFFICIENT
