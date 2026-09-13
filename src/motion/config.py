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
# The assembled arm visibly excited its compliant links at the former limits.
# A physical A/B run found that quarter-rate jerk reduced the shake while
# preserving the original velocity and acceleration ceilings.
JERK_LIMIT = np.array([30.0, 25.0, 25.0, 62.5, 62.5])  # rad/s^3

# Relaxed operating pose derived from the manually measured vertical reference.
# Base pitch is reversed from the former forward extension; elbow is 70 deg
# lower than the previous operating pose.
REST_POSE = np.array([
    0.2617993877991494,
    0.07629817559829882,
    -0.04058412526635097,
    0.0,
    0.9402527974893196,
])
