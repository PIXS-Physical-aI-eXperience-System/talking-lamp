"""Calibrated radian adapter for the physical LeLamp Feetech follower.

LeRobot is loaded only when constructing a physical backend. Feedback is cached
and sampled by command count, so ``measured()`` never adds serial traffic.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np

from .config import CONTROL_HZ, JOINT_NAMES, NJ
from .hardware_alignment import HardwareAlignment


def _hardware_dependencies():
    try:
        from lelamp.follower.lelamp_follower import LeLampFollower
        from lelamp.follower.config_lelamp_follower import LeLampFollowerConfig
        from lelamp.playback import park_and_disconnect
    except ImportError as exc:
        raise RuntimeError(
            "Physical motion requires the lelamp runtime and LeRobot dependencies. "
            "Add this repository's lelamp_runtime directory to PYTHONPATH and "
            "use an environment containing its dependencies."
        ) from exc

    def factory(*, port, lamp_id):
        return LeLampFollower(LeLampFollowerConfig(
            port=port, id=lamp_id, use_degrees=False, max_relative_target=None,
            cameras={}, disable_torque_on_disconnect=True,
        ))

    return factory, park_and_disconnect


class FeetechBackend:
    """Own the follower connection; call ``close`` to park before torque release.

    ``feedback_hz`` is the nominal rate at 100 commands/s. Missed command slots
    lower the feedback rate too. ``clamped_joints`` counts per-joint commands
    clipped at the calibrated normalized endpoints, and ``clamped_ticks`` counts
    commands with at least one clip.
    """

    def __init__(
        self, *, port: str, lamp_id: str, feedback_hz: float = 20.0,
        alignment: HardwareAlignment | None = None,
        robot_factory: Callable | None = None, park: Callable | None = None,
    ) -> None:
        if not np.isfinite(feedback_hz) or not 0 < feedback_hz <= CONTROL_HZ:
            raise ValueError(f"feedback_hz must be in (0, {CONTROL_HZ:g}]")
        self.alignment = alignment or HardwareAlignment.load()
        self.alignment.validate_calibration_id(lamp_id)
        if robot_factory is None or park is None:
            default_factory, default_park = _hardware_dependencies()
            robot_factory = robot_factory or default_factory
            park = park or default_park
        self._park = park
        self.feedback_hz = float(feedback_hz)
        self._feedback_phase = 0.0
        self.sent_ticks = 0
        self.clamped_ticks = 0
        self.clamped_joints: Counter[str] = Counter()
        self._closed = False
        self.robot = robot_factory(port=port, lamp_id=lamp_id)
        try:
            # The static alignment contains raw encoder references from one
            # exact LeRobot calibration. Reject a typo or recalibration before
            # opening the bus and enabling torque.
            self.alignment.validate_calibration(self.robot.calibration)
            self.robot.connect(calibrate=False)
            self._measured = self._read_pose()
        except BaseException:
            self.close()
            raise

    def _read_pose(self) -> np.ndarray:
        observation = self.robot.get_observation()
        normalized = np.array([observation[f"{j}.pos"] for j in JOINT_NAMES], float)
        if not np.all(np.isfinite(normalized)):
            raise ValueError("Hardware observation must contain five finite positions")
        return self.alignment.normalized_to_radians(normalized)

    def measured(self) -> np.ndarray:
        return self._measured.copy()

    def send(self, q_cmd: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Hardware backend is closed")
        q = np.asarray(q_cmd, float)
        if q.shape != (NJ,) or not np.all(np.isfinite(q)):
            raise ValueError("Hardware command must contain five finite radians")
        normalized = self.alignment.radians_to_normalized(q)
        bounded = np.clip(normalized, -100.0, 100.0)
        clipped = np.abs(normalized - bounded) > 1e-9
        self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINT_NAMES, bounded)})
        self.sent_ticks += 1
        if np.any(clipped):
            self.clamped_ticks += 1
            self.clamped_joints.update(j for j, c in zip(JOINT_NAMES, clipped) if c)
        self._feedback_phase += self.feedback_hz
        if self._feedback_phase >= CONTROL_HZ - 1e-9:
            self._feedback_phase -= CONTROL_HZ
            self._measured = self._read_pose()

    def close(self) -> None:
        if self._closed:
            return
        # Do not release torque if parking fails: the established helper keeps
        # the arm supported and propagates that failure to the operator.
        if self.robot.is_connected:
            self._park(self.robot)
        self._closed = True
