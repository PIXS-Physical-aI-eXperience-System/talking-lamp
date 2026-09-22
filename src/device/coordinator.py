"""Coordinate stabilized XVF direction decisions with the motion daemon."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Protocol

from .doa import DoaCalibration, DoaDecision, DoaError, calibrate_target


class MotionClient(Protocol):
    async def request(
        self, kind: str, payload: dict[str, object], ttl_ms: int = 1000,
    ) -> list[dict[str, object]]: ...


class DirectionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class OrientationEvent:
    state: str
    speech_id: str | None
    raw_doa_deg: float | None
    relative_rad: float | None
    target_yaw: float | None
    current_yaw: float | None
    clamped: bool
    code: str
    message: str
    timestamp: float


class DirectionCoordinator:
    """Convert one terminal DOA decision into one local orientation command."""

    def __init__(
        self,
        motion: MotionClient,
        calibration: DoaCalibration,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(getattr(motion, "request", None)):
            raise DirectionError("invalid_config", "motion client must provide request()")
        if not callable(clock):
            raise DirectionError("invalid_config", "clock must be callable")
        self.motion = motion
        self.calibration = calibration
        self.clock = clock
        self._last_event: OrientationEvent | None = None

    @staticmethod
    def _terminal(events: list[dict[str, object]]) -> dict[str, object]:
        if not events:
            raise DirectionError("invalid_response", "motion returned no response")
        terminal = events[-1]
        if terminal.get("state") != "completed":
            code = terminal.get("code")
            raise DirectionError(
                code if isinstance(code, str) else "motion_failed",
                terminal.get("message") if isinstance(terminal.get("message"), str)
                else "motion request failed",
            )
        data = terminal.get("data")
        if not isinstance(data, dict):
            raise DirectionError("invalid_response", "motion response data is missing")
        return terminal

    @staticmethod
    def _finite_data(data: dict[str, object], name: str) -> float:
        value = data.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise DirectionError("invalid_response", f"motion {name} is not finite")
        return float(value)

    async def handle_decision(self, decision: DoaDecision) -> OrientationEvent:
        if decision.state == "rejected":
            event = OrientationEvent(
                state="rejected",
                speech_id=decision.speech_id,
                raw_doa_deg=decision.doa_deg,
                relative_rad=None,
                target_yaw=None,
                current_yaw=None,
                clamped=False,
                code=decision.code,
                message="DOA stabilization rejected the utterance",
                timestamp=decision.timestamp,
            )
            self._last_event = event
            return event
        if decision.state != "ready" or decision.doa_deg is None:
            raise DirectionError("invalid_decision", "ready DOA decision must include a direction")

        status_terminal = self._terminal(
            await self.motion.request("orientation.status", {}, ttl_ms=1000))
        status = status_terminal["data"]
        assert isinstance(status, dict)
        center_yaw = self._finite_data(status, "center_yaw")
        safe_limits = (
            self._finite_data(status, "safe_yaw_min"),
            self._finite_data(status, "safe_yaw_max"),
        )
        try:
            target = calibrate_target(
                decision.doa_deg,
                self.calibration,
                center_yaw=center_yaw,
                safe_limits=safe_limits,
            )
        except DoaError as exc:
            event = OrientationEvent(
                state="rejected",
                speech_id=decision.speech_id,
                raw_doa_deg=decision.doa_deg,
                relative_rad=None,
                target_yaw=None,
                current_yaw=self._finite_data(status, "current_yaw"),
                clamped=False,
                code=exc.code,
                message=exc.message,
                timestamp=decision.timestamp,
            )
            self._last_event = event
            return event

        terminal = self._terminal(await self.motion.request(
            "orientation.acquire",
            {"speech_id": decision.speech_id, "target_yaw": target.target_yaw},
            ttl_ms=1000,
        ))
        data = terminal["data"]
        assert isinstance(data, dict)
        state = data.get("state")
        code = data.get("code")
        if not isinstance(state, str) or not isinstance(code, str):
            raise DirectionError("invalid_response", "motion orientation state is invalid")
        event = OrientationEvent(
            state=state,
            speech_id=decision.speech_id,
            raw_doa_deg=target.raw_doa_deg,
            relative_rad=target.relative_rad,
            target_yaw=self._finite_data(data, "target_yaw"),
            current_yaw=self._finite_data(data, "current_yaw"),
            clamped=target.clamped or data.get("clamped") is True,
            code=code,
            message=terminal.get("message") if isinstance(terminal.get("message"), str) else "",
            timestamp=decision.timestamp,
        )
        self._last_event = event
        return event

    async def return_center(self) -> OrientationEvent:
        terminal = self._terminal(await self.motion.request(
            "orientation.return_center", {}, ttl_ms=1000))
        data = terminal["data"]
        assert isinstance(data, dict)
        previous = self._last_event
        event = OrientationEvent(
            state=data.get("state") if isinstance(data.get("state"), str) else "centered",
            speech_id=data.get("speech_id") if isinstance(data.get("speech_id"), str) else (
                previous.speech_id if previous else None),
            raw_doa_deg=previous.raw_doa_deg if previous else None,
            relative_rad=previous.relative_rad if previous else None,
            target_yaw=self._finite_data(data, "target_yaw"),
            current_yaw=self._finite_data(data, "current_yaw"),
            clamped=data.get("clamped") is True,
            code=data.get("code") if isinstance(data.get("code"), str) else "centered",
            message=terminal.get("message") if isinstance(terminal.get("message"), str) else "",
            timestamp=float(self.clock()),
        )
        self._last_event = event
        return event

    async def status(self) -> OrientationEvent:
        terminal = self._terminal(await self.motion.request(
            "orientation.status", {}, ttl_ms=1000))
        data = terminal["data"]
        assert isinstance(data, dict)
        state = data.get("state")
        code = data.get("code")
        if not isinstance(state, str) or not isinstance(code, str):
            raise DirectionError("invalid_response", "motion orientation status is invalid")
        previous = self._last_event
        speech_id = data.get("speech_id")
        if not isinstance(speech_id, str):
            speech_id = previous.speech_id if previous else None
        target = data.get("target_yaw")
        if target is None:
            target_yaw = previous.target_yaw if previous else None
        elif isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target):
            raise DirectionError("invalid_response", "motion target_yaw is invalid")
        else:
            target_yaw = float(target)
        event = OrientationEvent(
            state=state,
            speech_id=speech_id,
            raw_doa_deg=previous.raw_doa_deg if previous else None,
            relative_rad=previous.relative_rad if previous else None,
            target_yaw=target_yaw,
            current_yaw=self._finite_data(data, "current_yaw"),
            clamped=data.get("clamped") is True,
            code=code,
            message=terminal.get("message") if isinstance(terminal.get("message"), str) else "",
            timestamp=float(self.clock()),
        )
        self._last_event = event
        return event
