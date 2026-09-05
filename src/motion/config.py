"""Shared constants for the Talking Lamp motion stack.

The 5 actuated joints, in canonical order. This order is the contract with
every other part (B routes joint-angle arrays, A's behaviour tags reference it,
D hands 3D targets that E turns into these angles). Units are **radians**, sign
and zero follow the MuJoCo model in ``sim/lelamp_arm.xml``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WORLD_XML = REPO_ROOT / "sim" / "world.xml"
RECORDINGS_DIR = REPO_ROOT / "lelamp_runtime" / "lelamp" / "recordings"

JOINT_NAMES: tuple[str, ...] = (
    "base_yaw",
    "base_pitch",
    "elbow_pitch",
    "wrist_roll",
    "wrist_pitch",
)
NJ = len(JOINT_NAMES)
SERVO_ID = {"base_yaw": 1, "base_pitch": 2, "elbow_pitch": 3, "wrist_roll": 4, "wrist_pitch": 5}

HEAD_SITE = "head"

CONTROL_HZ = 100.0
CONTROL_DT = 1.0 / CONTROL_HZ

# Per-joint kinematic limits for the online trajectory generator.
# Conservative defaults for STS3215 @ 12 V with an unballasted head; retune once
# the real head weight is measured (docs/파트-분배.md 4.5, "관절 속도·가속 제한값").
VEL_LIMIT = np.array([3.0, 2.5, 2.5, 4.0, 4.0])      # rad/s
ACC_LIMIT = np.array([12.0, 10.0, 10.0, 20.0, 20.0])  # rad/s^2
JERK_LIMIT = np.array([120.0, 100.0, 100.0, 250.0, 250.0])  # rad/s^3

# A relaxed, "alive"-looking neutral pose: facing forward, leaning over the desk,
# head ~14 cm ahead of the base and 40 cm up, gaze angled down at the work area.
# Used as the idle base pose and the IK null-space bias. All joints mid-range.
REST_POSE = np.array([0.0, 0.9, -0.5, 0.0, -0.3])
