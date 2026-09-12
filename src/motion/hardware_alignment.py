"""Conversions between LeRobot normalized servo positions and MuJoCo joints.

The physical reference is a manually measured, fully vertical lamp pose.  A
positive or negative encoder change is mapped to the corresponding MuJoCo
joint direction, so a physical rotation and a simulated rotation have the same
magnitude around that reference pose.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .config import JOINT_NAMES, REPO_ROOT


ALIGNMENT_FILE = REPO_ROOT / "sim" / "hardware_alignment.json"


@dataclass(frozen=True)
class HardwareAlignment:
    counts_per_revolution: float
    raw_ranges: np.ndarray
    straight_raw: np.ndarray
    straight_normalized: np.ndarray
    straight_radians: np.ndarray
    direction_sign: np.ndarray

    @classmethod
    def load(cls, path: str | Path = ALIGNMENT_FILE) -> "HardwareAlignment":
        payload = json.loads(Path(path).read_text())
        joints = payload["joints"]
        if set(joints) != set(JOINT_NAMES):
            raise ValueError("hardware alignment joints do not match JOINT_NAMES")

        ordered = [joints[name] for name in JOINT_NAMES]
        signs = np.asarray([joint["direction_sign"] for joint in ordered], dtype=float)
        if not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError("every hardware alignment direction_sign must be -1 or 1")

        raw_ranges = np.asarray(
            [joint["calibration_raw_range"] for joint in ordered], dtype=float
        )
        if np.any(raw_ranges[:, 0] >= raw_ranges[:, 1]):
            raise ValueError("hardware calibration raw ranges must increase")

        return cls(
            counts_per_revolution=float(payload["encoder_counts_per_revolution"]),
            raw_ranges=raw_ranges,
            straight_raw=np.asarray([joint["straight_raw"] for joint in ordered], dtype=float),
            straight_normalized=np.asarray(
                [joint["straight_normalized"] for joint in ordered], dtype=float
            ),
            straight_radians=np.asarray(
                [joint["simulation_straight_radians"] for joint in ordered], dtype=float
            ),
            direction_sign=signs,
        )

    @property
    def radians_per_count(self) -> float:
        return 2.0 * math.pi / self.counts_per_revolution

    @property
    def joint_limits(self) -> np.ndarray:
        endpoints = self.raw_to_radians(self.raw_ranges.T)
        return np.stack((np.min(endpoints, axis=0), np.max(endpoints, axis=0)), axis=1)

    def normalized_to_raw(self, normalized: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized, dtype=float)
        low, high = self.raw_ranges[:, 0], self.raw_ranges[:, 1]
        return low + (values + 100.0) * (high - low) / 200.0

    def raw_to_normalized(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=float)
        low, high = self.raw_ranges[:, 0], self.raw_ranges[:, 1]
        return (values - low) * 200.0 / (high - low) - 100.0

    def raw_to_radians(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=float)
        return self.straight_radians + self.direction_sign * (
            values - self.straight_raw
        ) * self.radians_per_count

    def radians_to_raw(self, radians: np.ndarray) -> np.ndarray:
        values = np.asarray(radians, dtype=float)
        return self.straight_raw + self.direction_sign * (
            values - self.straight_radians
        ) / self.radians_per_count

    def normalized_to_radians(self, normalized: np.ndarray) -> np.ndarray:
        return self.raw_to_radians(self.normalized_to_raw(normalized))

    def radians_to_normalized(self, radians: np.ndarray) -> np.ndarray:
        return self.raw_to_normalized(self.radians_to_raw(radians))
