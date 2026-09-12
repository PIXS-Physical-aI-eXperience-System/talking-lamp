"""Shared helpers for the Talking Lamp MuJoCo scene.

The 5 hinges are a serial chain (see sim/build_arm.py). Joint = actuator name;
`SERVO_ID` maps to the STS3215 IDs from `LeLamp/docs/2. Servos Setup.md`.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from motion.hardware_alignment import HardwareAlignment

WORLD_XML = Path(__file__).resolve().parent / "world.xml"

JOINT_NAMES = ["base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"]
SERVO_ID = {"base_yaw": 1, "base_pitch": 2, "elbow_pitch": 3, "wrist_roll": 4, "wrist_pitch": 5}

HEAD_SITE = "head"  # tip of the head (where the work light will mount)
HARDWARE_ALIGNMENT = HardwareAlignment.load()


def load() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(WORLD_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)  # "home"
    data.ctrl[actuator_order(model)] = data.qpos[qpos_order(model)]
    mujoco.mj_forward(model, data)
    return model, data


def normalized_to_qpos(normalized: np.ndarray) -> np.ndarray:
    """Convert the five LeRobot normalized positions to aligned MuJoCo radians."""
    return HARDWARE_ALIGNMENT.normalized_to_radians(normalized)


def qpos_to_normalized(qpos: np.ndarray) -> np.ndarray:
    """Convert aligned MuJoCo radians to the five LeRobot normalized positions."""
    return HARDWARE_ALIGNMENT.radians_to_normalized(qpos)


def actuator_order(model: mujoco.MjModel) -> list[int]:
    """ctrl index for each JOINT_NAMES entry."""
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES]


def qpos_order(model: mujoco.MjModel) -> list[int]:
    """qpos index for each JOINT_NAMES entry."""
    return [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES]


def joint_ranges(model: mujoco.MjModel) -> np.ndarray:
    """(5, 2) array of [lo, hi] radians, ordered like JOINT_NAMES."""
    out = np.zeros((5, 2))
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        out[i] = model.jnt_range[jid]
    return out


def head_pose(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, HEAD_SITE)
    return data.site_xpos[sid].copy()
