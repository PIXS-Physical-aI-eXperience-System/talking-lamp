import math

import mujoco
import numpy as np
import pytest

from motion.config import JOINT_NAMES, REST_POSE, WORLD_XML
from motion.hardware_alignment import HardwareAlignment


def test_vertical_reference_maps_to_measured_simulator_pose():
    alignment = HardwareAlignment.load()

    q = alignment.normalized_to_radians(alignment.straight_normalized)

    assert np.allclose(q, alignment.straight_radians, atol=1e-12)


def test_rest_pose_reverses_base_pitch_from_the_vertical_reference():
    alignment = HardwareAlignment.load()

    assert REST_POSE[1] == pytest.approx(
        alignment.straight_radians[1] + math.radians(30.0)
    )


def test_rest_pose_lowers_elbow_seventy_degrees_from_previous_home():
    alignment = HardwareAlignment.load()

    # The previous home was 42 degrees from the measured straight reference.
    assert REST_POSE[2] == pytest.approx(
        alignment.straight_radians[2] + math.radians(42.0 + 70.0)
    )


def test_encoder_quarter_turn_uses_each_measured_direction():
    alignment = HardwareAlignment.load()
    raw = alignment.straight_raw + 1024

    q = alignment.raw_to_radians(raw)

    expected = alignment.straight_radians + alignment.direction_sign * (math.pi / 2)
    assert np.allclose(q, expected, atol=1e-12)


def test_normalized_and_simulator_conversions_round_trip():
    alignment = HardwareAlignment.load()
    normalized = np.array([-37.5, 22.0, -11.25, 48.5, 7.75])

    restored = alignment.radians_to_normalized(
        alignment.normalized_to_radians(normalized)
    )

    assert np.allclose(restored, normalized, atol=1e-10)


def test_mujoco_limits_and_reference_poses_match_hardware_alignment():
    alignment = HardwareAlignment.load()
    model = mujoco.MjModel.from_xml_path(str(WORLD_XML))
    actual_limits = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actual_limits.append(model.jnt_range[jid])

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    straight_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "calibration_straight"
    )

    assert np.allclose(actual_limits, alignment.joint_limits, atol=1e-6)
    assert np.allclose(model.key_qpos[home_id], REST_POSE, atol=1e-6)
    assert np.allclose(
        model.key_qpos[straight_id], alignment.straight_radians, atol=1e-6
    )
