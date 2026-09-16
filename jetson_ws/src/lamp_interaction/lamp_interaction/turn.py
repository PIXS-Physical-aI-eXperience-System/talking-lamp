"""Enforce alignment, concurrent response, center, then idle ordering."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from typing import Any, Protocol
from uuid import UUID


_MOTION = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class TurnResponse:
    speech_id: str
    motion_name: str
    audio_stream: Any

    def __post_init__(self) -> None:
        try:
            canonical = str(UUID(self.speech_id)) == self.speech_id
        except ValueError:
            canonical = False
        if not canonical:
            raise ValueError("speech_id must be a canonical UUID")
        if not isinstance(self.motion_name, str) or not _MOTION.fullmatch(self.motion_name):
            raise ValueError("motion_name is invalid")


@dataclass(frozen=True)
class TurnResult:
    success: bool
    code: str
    message: str


class TurnPorts(Protocol):
    async def wait_orientation(self, speech_id: str, timeout: float): ...
    async def play_audio(self, response: TurnResponse): ...
    async def play_motion(self, name: str): ...
    async def return_center(self): ...
    async def play_idle(self): ...
    async def cancel_audio(self) -> None: ...
    async def cancel_motion(self) -> None: ...
    async def interrupt_motion(self) -> None: ...


def _failure(result: object, fallback: str) -> TurnResult:
    code = getattr(result, "code", fallback)
    message = getattr(result, "message", "")
    return TurnResult(False, code if isinstance(code, str) else fallback,
                      message if isinstance(message, str) else "")


class TurnOrchestrator:
    def __init__(
        self, ports: TurnPorts, *, orientation_timeout: float = 3.0,
        response_timeout: float = 30.0, center_timeout: float = 5.0,
    ) -> None:
        self.ports = ports
        self.orientation_timeout = orientation_timeout
        self.response_timeout = response_timeout
        self.center_timeout = center_timeout

    async def run(self, response: TurnResponse) -> TurnResult:
        try:
            orientation = await asyncio.wait_for(
                self.ports.wait_orientation(
                    response.speech_id, self.orientation_timeout),
                self.orientation_timeout,
            )
        except TimeoutError:
            return TurnResult(False, "orientation_timeout", "alignment timed out")
        if getattr(orientation, "state", None) != "aligned":
            return _failure(orientation, "orientation_failed")

        audio_task = asyncio.create_task(self.ports.play_audio(response))
        motion_task = asyncio.create_task(self.ports.play_motion(response.motion_name))
        tasks = (audio_task, motion_task)
        try:
            audio, motion = await asyncio.wait_for(
                asyncio.gather(*tasks), self.response_timeout)
        except TimeoutError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return TurnResult(False, "response_timeout", "audio or motion timed out")
        except Exception as exc:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return TurnResult(False, "response_failed", str(exc))

        if getattr(audio, "success", False) is not True:
            return _failure(audio, "audio_failed")
        if getattr(motion, "success", False) is not True:
            return _failure(motion, "motion_failed")

        try:
            centered = await asyncio.wait_for(
                self.ports.return_center(), self.center_timeout)
        except TimeoutError:
            return TurnResult(False, "center_timeout", "return-center timed out")
        if (
            getattr(centered, "success", False) is not True
            or getattr(centered, "state", "centered") != "centered"
        ):
            return _failure(centered, "center_failed")

        idle = await self.ports.play_idle()
        if getattr(idle, "success", False) is not True:
            return _failure(idle, "idle_failed")
        return TurnResult(True, "completed", "")

    async def barge_in(self) -> None:
        await asyncio.gather(
            self.ports.cancel_audio(), self.ports.cancel_motion())
        await self.ports.interrupt_motion()


__all__ = ["TurnOrchestrator", "TurnPorts", "TurnResponse", "TurnResult"]
