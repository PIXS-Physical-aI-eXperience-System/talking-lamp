"""One-shot client for the Pi motion daemon's protected Unix control socket."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import math
from pathlib import Path
from typing import Callable
from uuid import UUID, uuid4

from motion.protocol import (
    MAX_LINE_BYTES,
    ProtocolError,
    decode_local_request,
    encode_message,
)


class MotionClientError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class MotionUnixClient:
    """Send one local orientation request per connection, without replay."""

    def __init__(
        self,
        path: str | Path = "/run/talking-lamp/motion-control.sock",
        *,
        response_timeout: float = 5.0,
        id_factory: Callable[[], UUID | str] = uuid4,
    ) -> None:
        if (
            isinstance(response_timeout, bool)
            or not isinstance(response_timeout, (int, float))
            or not math.isfinite(response_timeout)
            or response_timeout <= 0
        ):
            raise MotionClientError("invalid_config", "response_timeout must be finite and positive")
        if not callable(id_factory):
            raise MotionClientError("invalid_config", "id_factory must be callable")
        self.path = Path(path)
        self.response_timeout = float(response_timeout)
        self.id_factory = id_factory

    async def request(
        self,
        kind: str,
        payload: dict[str, object],
        ttl_ms: int = 1000,
    ) -> list[dict[str, object]]:
        ident = str(self.id_factory())
        envelope = {
            "version": 1,
            "id": ident,
            "type": kind,
            "ttl_ms": ttl_ms,
            "payload": payload,
        }
        try:
            line = encode_message(envelope)
            decode_local_request(line, received_at=0.0)
        except (ProtocolError, TypeError, ValueError) as exc:
            raise MotionClientError("invalid_request", str(exc)) from exc

        writer: asyncio.StreamWriter | None = None
        try:
            async with asyncio.timeout(self.response_timeout):
                reader, writer = await asyncio.open_unix_connection(
                    str(self.path), limit=MAX_LINE_BYTES + 1)
                writer.write(line)
                await writer.drain()
                events: list[dict[str, object]] = []
                accepted = False
                while True:
                    try:
                        response = await reader.readline()
                    except ValueError as exc:
                        raise MotionClientError(
                            "line_too_long", "motion response exceeds 16 KiB") from exc
                    if not response:
                        raise MotionClientError(
                            "connection_lost", "motion socket closed before a terminal response")
                    if len(response) > MAX_LINE_BYTES:
                        raise MotionClientError("line_too_long", "motion response exceeds 16 KiB")
                    if not response.endswith(b"\n"):
                        raise MotionClientError("invalid_response", "motion response is not NDJSON")
                    try:
                        event = json.loads(response)
                    except (UnicodeError, json.JSONDecodeError) as exc:
                        raise MotionClientError("invalid_response", "motion response is not JSON") from exc
                    if not isinstance(event, dict) or event.get("id") != ident:
                        raise MotionClientError("invalid_response", "motion response ID does not match")
                    state = event.get("state")
                    if state not in {"accepted", "completed", "failed", "cancelled"}:
                        raise MotionClientError("invalid_response", "motion response state is invalid")
                    if not isinstance(event.get("code"), str):
                        raise MotionClientError("invalid_response", "motion response code is invalid")
                    if state == "accepted":
                        if accepted:
                            raise MotionClientError("invalid_response", "motion response accepted twice")
                        accepted = True
                        events.append(event)
                        continue
                    events.append(event)
                    return events
        except TimeoutError as exc:
            raise MotionClientError("timeout", "motion response timed out") from exc
        except MotionClientError:
            raise
        except OSError as exc:
            raise MotionClientError("connection_failed", f"motion socket failed: {exc}") from exc
        finally:
            if writer is not None:
                writer.close()
                with suppress(OSError):
                    await writer.wait_closed()

