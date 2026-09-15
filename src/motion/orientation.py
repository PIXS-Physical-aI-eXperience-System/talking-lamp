"""Yaw-only speaker-orientation anchor and settle-aware coordinator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blender import BlendContext, LayerOutput
from .config import NJ, REST_POSE


@dataclass(frozen=True)
class OrientationConfig:
    center_yaw: float = float(REST_POSE[0])
    yaw_margin: float = np.deg2rad(5.0)
    deadband: float = np.deg2rad(5.0)
    settle_error: float = np.deg2rad(3.0)
    settle_velocity: float = 0.08
    settle_duration: float = 0.15
    acquire_timeout: float = 2.0
    disconnect_hold: float = 10.0


@dataclass(frozen=True)
class OrientationSnapshot:
    state: str
    speech_id: str | None
    target_yaw: float | None
    current_yaw: float
    clamped: bool
    code: str


class OrientationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _validate_config(cfg: OrientationConfig) -> None:
    values = (
        cfg.center_yaw,
        cfg.yaw_margin,
        cfg.deadband,
        cfg.settle_error,
        cfg.settle_velocity,
        cfg.settle_duration,
        cfg.acquire_timeout,
        cfg.disconnect_hold,
    )
    if not all(np.isfinite(value) for value in values):
        raise OrientationError("invalid_config", "orientation configuration must be finite")
    if cfg.yaw_margin < 0:
        raise OrientationError("invalid_config", "yaw margin must not be negative")
    if any(value <= 0 for value in values[2:]):
        raise OrientationError("invalid_config", "orientation thresholds must be positive")


class BaseYawOrientationLayer:
    """Absolute base-yaw anchor that leaves every other joint unclaimed."""

    name = "orientation"
    priority = 15

    def __init__(self, cfg: OrientationConfig, joint_limits: np.ndarray):
        _validate_config(cfg)
        limits = np.asarray(joint_limits, dtype=float)
        if limits.shape != (NJ, 2) or not np.all(np.isfinite(limits)):
            raise OrientationError("invalid_joint_limits", "joint limits must be a finite (5, 2) array")
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise OrientationError("invalid_joint_limits", "each joint lower limit must be below its upper limit")

        safe_min = float(limits[0, 0] + cfg.yaw_margin)
        safe_max = float(limits[0, 1] - cfg.yaw_margin)
        if safe_min >= safe_max:
            raise OrientationError("invalid_config", "yaw margin eliminates the safe yaw range")
        if not safe_min <= cfg.center_yaw <= safe_max:
            raise OrientationError("invalid_config", "center yaw must be inside the safe yaw range")

        self.cfg = cfg
        self._safe_yaw_limits = (safe_min, safe_max)
        self._target_yaw: float | None = None
        self._clamped = False

    @property
    def safe_yaw_limits(self) -> tuple[float, float]:
        return self._safe_yaw_limits

    def acquire(self, target_yaw: float) -> bool:
        if not np.isfinite(target_yaw):
            raise OrientationError("invalid_target", "target yaw must be finite")
        low, high = self.safe_yaw_limits
        self._clamped = bool(target_yaw < low or target_yaw > high)
        self._target_yaw = float(np.clip(target_yaw, low, high))
        return self._clamped

    def return_center(self) -> None:
        self.acquire(self.cfg.center_yaw)

    def release(self) -> None:
        self._target_yaw = None
        self._clamped = False

    def update(self, ctx: BlendContext) -> LayerOutput:
        if self._target_yaw is None:
            return LayerOutput.inactive()
        value = np.zeros(NJ)
        weight = np.zeros(NJ)
        value[0] = self._target_yaw
        weight[0] = 1.0
        return LayerOutput(value=value, weight=weight, additive=False)


class OrientationCoordinator:
    """Coordinates orientation acquisition, settling, timeout, and return."""

    def __init__(self, layer: BaseYawOrientationLayer, cfg: OrientationConfig):
        _validate_config(cfg)
        if cfg != layer.cfg:
            raise OrientationError("config_mismatch", "layer and coordinator configurations must match")
        self.layer = layer
        self.cfg = cfg
        self._state = "idle"
        self._speech_id: str | None = None
        self._target_yaw: float | None = None
        self._current_yaw = float(cfg.center_yaw)
        self._clamped = False
        self._code = "idle"
        self._started_at: float | None = None
        self._settle_since: float | None = None
        self._disconnected_at: float | None = None

    def _snapshot(self, *, code: str | None = None) -> OrientationSnapshot:
        return OrientationSnapshot(
            state=self._state,
            speech_id=self._speech_id,
            target_yaw=self._target_yaw,
            current_yaw=self._current_yaw,
            clamped=self._clamped,
            code=self._code if code is None else code,
        )

    @staticmethod
    def _finite(value: float, name: str) -> float:
        if not np.isfinite(value):
            raise OrientationError("invalid_input", f"{name} must be finite")
        return float(value)

    def acquire(self, speech_id: str, target_yaw: float, *, now: float,
                current_yaw: float, task_light_busy: bool) -> OrientationSnapshot:
        if not isinstance(speech_id, str) or not speech_id.strip():
            raise OrientationError("invalid_speech_id", "speech ID must be a non-empty string")
        if speech_id == self._speech_id:
            return self._snapshot()

        now = self._finite(now, "now")
        current_yaw = self._finite(current_yaw, "current yaw")
        target_yaw = self._finite(target_yaw, "target yaw")
        self._current_yaw = current_yaw
        if task_light_busy:
            return self._snapshot(code="blocked_by_task_light")

        self._clamped = self.layer.acquire(target_yaw)
        low, high = self.layer.safe_yaw_limits
        self._target_yaw = float(np.clip(target_yaw, low, high))
        self._speech_id = speech_id
        self._started_at = now
        self._settle_since = None
        self._disconnected_at = None
        if abs(current_yaw - self._target_yaw) <= self.cfg.deadband:
            self._state = "aligned"
            self._code = "aligned"
        else:
            self._state = "orienting"
            self._code = "orienting"
        return self._snapshot()

    def release(self) -> OrientationSnapshot:
        """Release the anchor and reset its speech-session lifecycle."""
        self.layer.release()
        self._state = "idle"
        self._speech_id = None
        self._target_yaw = None
        self._clamped = False
        self._code = "idle"
        self._started_at = None
        self._settle_since = None
        self._disconnected_at = None
        return self._snapshot()

    def return_center(self, *, now: float, current_yaw: float,
                      motion_busy: bool) -> OrientationSnapshot:
        if motion_busy:
            raise OrientationError("busy", "cannot return center while motion is busy")
        now = self._finite(now, "now")
        self._current_yaw = self._finite(current_yaw, "current yaw")
        self.layer.return_center()
        self._target_yaw = self.cfg.center_yaw
        self._clamped = False
        self._state = "returning"
        self._code = "returning"
        self._started_at = now
        self._settle_since = None
        self._disconnected_at = None
        return self._snapshot()

    def observe(self, *, now: float, current_yaw: float,
                velocity: float) -> OrientationSnapshot:
        now = self._finite(now, "now")
        self._current_yaw = self._finite(current_yaw, "current yaw")
        velocity = self._finite(velocity, "velocity")

        if self._state in {"orienting", "aligned", "timeout"} and self._disconnected_at is not None:
            if now - self._disconnected_at >= self.cfg.disconnect_hold:
                self.return_center(now=now, current_yaw=current_yaw, motion_busy=False)
                return self._snapshot()

        if self._state not in {"orienting", "returning"}:
            return self._snapshot()

        assert self._target_yaw is not None
        settled = (
            abs(current_yaw - self._target_yaw) <= self.cfg.settle_error
            and abs(velocity) <= self.cfg.settle_velocity
        )
        if settled:
            if self._settle_since is None:
                self._settle_since = now
            if now - self._settle_since >= self.cfg.settle_duration - 1e-12:
                self._state = "aligned" if self._state == "orienting" else "centered"
                self._code = self._state
        else:
            self._settle_since = None

        if self._state == "orienting" and self._started_at is not None:
            if now - self._started_at >= self.cfg.acquire_timeout:
                self._state = "timeout"
                self._code = "timeout"
        return self._snapshot()

    def disconnected(self, *, now: float) -> None:
        self._disconnected_at = self._finite(now, "now")
