"""Motor gains and the measured calibration used by physical motion."""

from collections.abc import Mapping

DEFAULT_POSITION_P_COEFFICIENT = 16
LOADED_PITCH_POSITION_P_COEFFICIENT = 32
LOADED_PITCH_JOINTS = frozenset({"base_pitch", "elbow_pitch"})

EXPECTED_CALIBRATION_ID = "lelamp"
EXPECTED_FOLLOWER_CALIBRATION = {
    "base_yaw": {
        "id": 1, "drive_mode": 0, "homing_offset": -653,
        "range_min": 1072, "range_max": 3086,
    },
    "base_pitch": {
        "id": 2, "drive_mode": 0, "homing_offset": -1406,
        "range_min": 939, "range_max": 3073,
    },
    "elbow_pitch": {
        "id": 3, "drive_mode": 0, "homing_offset": 1767,
        "range_min": 957, "range_max": 3192,
    },
    "wrist_roll": {
        "id": 4, "drive_mode": 0, "homing_offset": -1491,
        "range_min": 1007, "range_max": 3076,
    },
    "wrist_pitch": {
        "id": 5, "drive_mode": 0, "homing_offset": 2003,
        "range_min": 900, "range_max": 3150,
    },
}


def _calibration_field(calibration: object, field: str):
    if isinstance(calibration, Mapping):
        return calibration.get(field)
    return getattr(calibration, field, None)


def validate_calibration_profile(lamp_id: str, calibration: Mapping[str, object]) -> None:
    """Reject motion when the saved LeRobot calibration is not the measured one."""
    if lamp_id != EXPECTED_CALIBRATION_ID:
        raise ValueError(
            f"Motion profile is calibrated for {EXPECTED_CALIBRATION_ID!r}, not {lamp_id!r}"
        )
    if set(calibration) != set(EXPECTED_FOLLOWER_CALIBRATION):
        raise ValueError("Saved calibration joints do not match the motion profile")
    for joint, expected in EXPECTED_FOLLOWER_CALIBRATION.items():
        actual = calibration[joint]
        mismatches = [
            f"{field}={_calibration_field(actual, field)!r} (expected {value!r})"
            for field, value in expected.items()
            if _calibration_field(actual, field) != value
        ]
        if mismatches:
            raise ValueError(
                f"Saved calibration for {joint} does not match the motion profile: "
                + ", ".join(mismatches)
            )


def require_motor_calibration_match(matches_saved_profile: bool) -> None:
    """Fail closed when servo EEPROM does not match the saved calibration."""
    if not matches_saved_profile:
        raise RuntimeError(
            "The motor-resident calibration does not match the saved lelamp profile; "
            "refusing to configure motors or enable torque"
        )


def position_p_coefficient(motor: str) -> int:
    """Return enough position gain for loaded pitch joints to resist gravity."""
    if motor in LOADED_PITCH_JOINTS:
        return LOADED_PITCH_POSITION_P_COEFFICIENT
    return DEFAULT_POSITION_P_COEFFICIENT
