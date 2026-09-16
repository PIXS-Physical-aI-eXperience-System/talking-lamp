"""Pure direction-of-arrival statistics, calibration, and stabilization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import statistics
from typing import Literal
from uuid import UUID, uuid4


class DoaError(ValueError):
    """Invalid DOA input or configuration with a machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DoaError("invalid_value", f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DoaError("invalid_value", f"{name} must be a finite number")
    return result


def _degree(value: object, name: str = "DOA") -> float:
    result = _number(value, name)
    if not 0.0 <= result < 360.0:
        raise DoaError("out_of_range", f"{name} must be from 0 inclusive to 360 exclusive")
    return result


def _validated_degrees(samples: Sequence[float]) -> tuple[float, ...]:
    if isinstance(samples, (str, bytes)):
        raise DoaError("invalid_samples", "samples must be a non-empty sequence of degrees")
    values = tuple(_degree(value, "samples") for value in samples)
    if not values:
        raise DoaError("invalid_samples", "samples must be a non-empty sequence of degrees")
    return values


def circular_distance_deg(a: float, b: float) -> float:
    """Return the shortest unsigned angular distance in degrees."""
    first = _number(a, "angle")
    second = _number(b, "angle")
    return abs(((first - second + 180.0) % 360.0) - 180.0)


def circular_medoid_deg(samples: Sequence[float]) -> float:
    """Return the input angle with minimum total circular distance."""
    values = _validated_degrees(samples)
    return min(
        enumerate(values),
        key=lambda item: (
            sum(circular_distance_deg(item[1], other) for other in values),
            item[0],
        ),
    )[1]


def circular_mad_deg(samples: Sequence[float], center: float) -> float:
    """Return the median circular absolute deviation about ``center``."""
    values = _validated_degrees(samples)
    checked_center = _number(center, "center")
    return float(statistics.median(
        circular_distance_deg(value, checked_center) for value in values
    ))


def _wrap_signed_deg(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


@dataclass(frozen=True)
class DoaCalibration:
    zero_deg: float
    direction_sign: int
    front_half_angle_deg: float = 90.0


@dataclass(frozen=True)
class DoaTarget:
    raw_doa_deg: float
    relative_rad: float
    target_yaw: float
    clamped: bool


def calibrate_target(
    doa_deg: float,
    calibration: DoaCalibration,
    *,
    center_yaw: float,
    safe_limits: tuple[float, float],
) -> DoaTarget:
    """Convert a raw array-frame DOA to a safe absolute base-yaw target."""
    raw = _degree(doa_deg)
    zero = _number(calibration.zero_deg, "zero_deg")
    sign = calibration.direction_sign
    if isinstance(sign, bool) or sign not in {-1, 1}:
        raise DoaError("invalid_config", "direction_sign must be -1 or 1")
    front = _number(calibration.front_half_angle_deg, "front_half_angle_deg")
    if not 0.0 < front <= 180.0:
        raise DoaError("invalid_config", "front_half_angle_deg must be in (0, 180]")
    center = _number(center_yaw, "center_yaw")
    try:
        low, high = safe_limits
    except (TypeError, ValueError) as exc:
        raise DoaError("invalid_config", "safe_limits must contain two finite values") from exc
    low = _number(low, "safe yaw minimum")
    high = _number(high, "safe yaw maximum")
    if low >= high:
        raise DoaError("invalid_config", "safe yaw minimum must be less than maximum")
    if not low <= center <= high:
        raise DoaError("invalid_config", "center_yaw must be inside safe_limits")

    relative_deg = _wrap_signed_deg(sign * (raw - zero))
    if abs(relative_deg) > front:
        raise DoaError(
            "rear_direction",
            "direction is outside the configured front half-plane",
        )
    relative_rad = math.radians(relative_deg)
    unclamped = center + relative_rad
    target = min(max(unclamped, low), high)
    return DoaTarget(
        raw_doa_deg=raw,
        relative_rad=relative_rad,
        target_yaw=target,
        clamped=not math.isclose(target, unclamped, rel_tol=0.0, abs_tol=1e-12),
    )


@dataclass(frozen=True)
class DoaSample:
    timestamp: float
    doa_deg: float
    speech_detected: bool


@dataclass(frozen=True)
class DoaDecision:
    state: Literal["ready", "rejected"]
    speech_id: str
    doa_deg: float | None
    dispersion_deg: float | None
    sample_count: int
    code: str
    timestamp: float


class DoaStabilizer:
    """Select exactly one stable direction for each speech rising edge."""

    def __init__(
        self,
        *,
        min_samples: int = 6,
        min_window_sec: float = 0.4,
        max_window_sec: float = 1.0,
        max_dispersion_deg: float = 12.0,
        id_factory: Callable[[], UUID | str] = uuid4,
    ) -> None:
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
            raise DoaError("invalid_config", "min_samples must be a positive integer")
        self.min_window_sec = _number(min_window_sec, "min_window_sec")
        self.max_window_sec = _number(max_window_sec, "max_window_sec")
        self.max_dispersion_deg = _number(max_dispersion_deg, "max_dispersion_deg")
        if not 0.0 <= self.min_window_sec <= self.max_window_sec:
            raise DoaError("invalid_config", "collection windows are invalid")
        if not 0.0 <= self.max_dispersion_deg <= 180.0:
            raise DoaError("invalid_config", "max_dispersion_deg must be in [0, 180]")
        if not callable(id_factory):
            raise DoaError("invalid_config", "id_factory must be callable")
        self.min_samples = min_samples
        self.id_factory = id_factory
        self._speech_detected = False
        self._speech_id: str | None = None
        self._started_at: float | None = None
        self._samples: list[float] = []
        self._terminal = False
        self._last_timestamp: float | None = None

    @property
    def speech_id(self) -> str | None:
        return self._speech_id

    @staticmethod
    def _canonical_id(value: UUID | str) -> str:
        try:
            parsed = UUID(str(value))
        except (ValueError, TypeError, AttributeError) as exc:
            raise DoaError("invalid_id", "id_factory must return a UUID") from exc
        text = str(parsed)
        if str(value) != text:
            raise DoaError("invalid_id", "id_factory must return canonical UUID text")
        return text

    def observe(self, sample: DoaSample) -> DoaDecision | None:
        timestamp = _number(sample.timestamp, "timestamp")
        doa = _degree(sample.doa_deg)
        if not isinstance(sample.speech_detected, bool):
            raise DoaError("invalid_value", "speech_detected must be a boolean")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise DoaError("out_of_order", "sample timestamps must not move backwards")
        self._last_timestamp = timestamp

        rising = sample.speech_detected and not self._speech_detected
        self._speech_detected = sample.speech_detected
        if rising:
            self._speech_id = self._canonical_id(self.id_factory())
            self._started_at = timestamp
            self._samples = []
            self._terminal = False

        if self._speech_id is None or self._started_at is None or self._terminal:
            return None
        if sample.speech_detected:
            self._samples.append(doa)

        elapsed = timestamp - self._started_at
        center = dispersion = None
        if self._samples:
            center = circular_medoid_deg(self._samples)
            dispersion = circular_mad_deg(self._samples, center)

        if (
            elapsed >= self.min_window_sec
            and len(self._samples) >= self.min_samples
            and dispersion is not None
            and dispersion <= self.max_dispersion_deg
        ):
            return self._finish("ready", center, dispersion, "stable", timestamp)
        if elapsed >= self.max_window_sec:
            code = "insufficient_samples" if len(self._samples) < self.min_samples else "unstable"
            return self._finish("rejected", center, dispersion, code, timestamp)
        return None

    def _finish(
        self,
        state: Literal["ready", "rejected"],
        center: float | None,
        dispersion: float | None,
        code: str,
        timestamp: float,
    ) -> DoaDecision:
        assert self._speech_id is not None
        self._terminal = True
        return DoaDecision(
            state=state,
            speech_id=self._speech_id,
            doa_deg=center,
            dispersion_deg=dispersion,
            sample_count=len(self._samples),
            code=code,
            timestamp=timestamp,
        )

