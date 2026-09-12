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
from typing import Mapping

import numpy as np

from .config import JOINT_NAMES, REPO_ROOT


ALIGNMENT_FILE = REPO_ROOT / "sim" / "hardware_alignment.json"


@dataclass(frozen=True)
class HardwareAlignment:
    calibration_id: str
    counts_per_revolution: float
    motor_ids: np.ndarray
    drive_modes: np.ndarray
    homing_offsets: np.ndarray
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
            calibration_id=str(payload["calibration_id"]),
            counts_per_revolution=float(payload["encoder_counts_per_revolution"]),
            motor_ids=np.asarray([joint["motor_id"] for joint in ordered], dtype=int),
            drive_modes=np.asarray([joint["drive_mode"] for joint in ordered], dtype=int),
            homing_offsets=np.asarray(
                [joint["homing_offset"] for joint in ordered], dtype=int
            ),
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

    def validate_calibration_id(self, lamp_id: str) -> None:
        if lamp_id != self.calibration_id:
            raise ValueError(
                f"Hardware alignment is calibrated for {self.calibration_id!r}, "
                f"not {lamp_id!r}"
            )

    def validate_calibration(self, calibration: Mapping[str, object]) -> None:
        if set(calibration) != set(JOINT_NAMES):
            raise ValueError("Saved calibration joints do not match hardware alignment")
        for index, joint in enumerate(JOINT_NAMES):
            actual = calibration[joint]
            expected = {
                "id": int(self.motor_ids[index]),
                "drive_mode": int(self.drive_modes[index]),
                "homing_offset": int(self.homing_offsets[index]),
                "range_min": int(self.raw_ranges[index, 0]),
                "range_max": int(self.raw_ranges[index, 1]),
            }
            mismatches = [
                f"{field}={getattr(actual, field, None)!r} (expected {value!r})"
                for field, value in expected.items()
                if getattr(actual, field, None) != value
            ]
            if mismatches:
                raise ValueError(
                    f"Saved calibration for {joint} does not match hardware alignment: "
                    + ", ".join(mismatches)
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
