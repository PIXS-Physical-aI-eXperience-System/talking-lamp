"""Authenticated Pi device command handler and single-owner TCP server."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import asdict
import inspect
import json
import math
import time
from typing import Any, Callable, Protocol
from uuid import uuid4

from .coordinator import DirectionError
from .led import LedError
from .motion_client import MotionClientError
from .protocol import (
    MAX_LINE_BYTES,
    DeviceProtocolError,
    DeviceRequest,
    decode_request,
    encode_message,
)


class DeviceCommandError(RuntimeError):
    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class DeviceService(Protocol):
    async def dispatch(self, request: DeviceRequest) -> dict[str, object]: ...
    async def disconnected(self) -> None: ...


class DeviceCommandHandler:
    """Map validated device requests to direction and LED policy objects."""

    def __init__(self, coordinator: Any, led: Any) -> None:
        self.coordinator = coordinator
        self.led = led

    async def dispatch(self, request: DeviceRequest) -> dict[str, object]:
        try:
            if request.type == "orientation.return_center":
                return asdict(await self.coordinator.return_center())
            if request.type == "orientation.status":
                return asdict(await self.coordinator.status())
            if request.type == "led.frame":
                return asdict(self.led.frame(
                    bytes(request.payload["rgb"]),
                    brightness=request.payload["brightness"],
                ))
            if request.type == "led.solid":
                return asdict(self.led.solid(
                    request.payload["rgb"],
                    brightness=request.payload["brightness"],
                ))
            if request.type == "led.clear":
                return asdict(self.led.clear())
            if request.type == "led.status":
                return asdict(self.led.status)
            if request.type == "device.status":
                return {
                    "orientation": asdict(await self.coordinator.status()),
                    "led": asdict(self.led.status),
                }
            if request.type == "system.heartbeat":
                return {}
        except (LedError, DirectionError, MotionClientError) as exc:
            raise DeviceCommandError(exc.code, exc.message) from exc
        raise DeviceCommandError("unknown_type", "unsupported device command")

    async def disconnected(self) -> None:
        try:
            self.led.clear()
        except LedError:
            # The LED controller has already latched its fault. Disconnect
            # cleanup must not prevent the network owner from being released.
            pass


class DeviceTcpServer:
    """Serve one authenticated Jetson owner and ordered device events."""

    def __init__(
        self,
        service: DeviceService,
        *,
        token: str,
        host: str = "127.0.0.1",
        port: int = 8766,
        allowed_hosts: set[str] | None = None,
        heartbeat_timeout: float = 2.5,
        clock: Callable[[], float] = time.monotonic,
        session_factory: Callable[[], object] = uuid4,
    ) -> None:
        if not isinstance(token, str) or not token:
            raise ValueError("token must not be empty")
        if (
            isinstance(heartbeat_timeout, bool)
            or not isinstance(heartbeat_timeout, (int, float))
            or not math.isfinite(heartbeat_timeout)
            or heartbeat_timeout <= 0
        ):
            raise ValueError("heartbeat_timeout must be finite and positive")
        if not host or not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
            raise ValueError("host and port are invalid")
        self.service = service
        self.token = token
        self.host = host
        self.port = port
        self.allowed_hosts = None if allowed_hosts is None else frozenset(allowed_hosts)
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.clock = clock
        self.session_factory = session_factory
        self._server: asyncio.Server | None = None
        self._owner: asyncio.StreamWriter | None = None
        self._connections: set[asyncio.Task] = set()
        self._event_sender: Callable[[dict[str, object]], Any] | None = None
        self._session_id: str | None = None
        self._event_sequence = 0

    @property
    def sockets(self):
        return self._server.sockets if self._server is not None else ()

    @property
    def owner(self) -> asyncio.StreamWriter | None:
        return self._owner

    async def start(self) -> DeviceTcpServer:
        if self._server is None:
            self._server = await asyncio.start_server(
                self._handle, self.host, self.port, limit=MAX_LINE_BYTES + 1)
        return self

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        await self._server.serve_forever()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        tasks = list(self._connections)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def publish_event(self, event: str, data: dict[str, object]) -> bool:
        sender = self._event_sender
        if sender is None or self._session_id is None:
            return False
        self._event_sequence += 1
        message = {
            "version": 1,
            "event": event,
            "session_id": self._session_id,
            "sequence": self._event_sequence,
            "data": data,
        }
        try:
            await sender(message)
        except (ConnectionError, OSError, TimeoutError, DeviceProtocolError):
            return False
        return True

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connection = asyncio.current_task()
        if connection is not None:
            self._connections.add(connection)
        replies: set[asyncio.Task] = set()
        write_lock = asyncio.Lock()
        authenticated = False
        deadline = self.clock() + self.heartbeat_timeout
        errors = 0

        async def send(message: dict[str, object]) -> None:
            async with write_lock:
                writer.write(encode_message(message))
                await asyncio.wait_for(writer.drain(), self.heartbeat_timeout)

        async def forward(request: DeviceRequest) -> None:
            try:
                if self.clock() > request.expires_at:
                    raise DeviceCommandError("expired", "request expired before dispatch")
                data = await self.service.dispatch(request)
                await send({
                    "id": request.id,
                    "state": "completed",
                    "code": "completed",
                    "data": data,
                })
            except DeviceCommandError as exc:
                with suppress(ConnectionError, OSError, TimeoutError, DeviceProtocolError):
                    await send({
                        "id": request.id,
                        "state": "failed",
                        "code": exc.code,
                        "message": exc.message,
                        "data": {},
                    })
            except (ConnectionError, OSError, TimeoutError, DeviceProtocolError):
                writer.close()
            except Exception:
                with suppress(ConnectionError, OSError, TimeoutError, DeviceProtocolError):
                    await send({
                        "id": request.id,
                        "state": "failed",
                        "code": "internal_error",
                        "message": "device command failed",
                        "data": {},
                    })

        try:
            peer = writer.get_extra_info("peername")
            if self.allowed_hosts is not None and (
                not peer or peer[0] not in self.allowed_hosts
            ):
                await send({"state": "failed", "code": "host_not_allowed", "data": {}})
                return
            while True:
                try:
                    timeout = max(0.0, deadline - self.clock())
                    async with asyncio.timeout(timeout):
                        line = await reader.readline()
                except ValueError:
                    await send({"state": "failed", "code": "line_too_long", "data": {}})
                    return
                if not line:
                    return
                received_at = self.clock()
                try:
                    request = decode_request(line, token=self.token, received_at=received_at)
                except DeviceProtocolError as exc:
                    message: dict[str, object] = {
                        "state": "failed", "code": exc.code,
                        "message": exc.message, "data": {},
                    }
                    with suppress(ValueError, UnicodeError):
                        raw = json.loads(line)
                        if isinstance(raw, dict) and isinstance(raw.get("id"), str):
                            message["id"] = raw["id"][:64]
                    await send(message)
                    errors += 1
                    if exc.code in {"unauthorized", "line_too_long", "missing_newline"} or errors >= 3:
                        return
                    continue
                errors = 0
                if not authenticated:
                    if self._owner is not None:
                        await send({
                            "id": request.id,
                            "state": "failed",
                            "code": "controller_connected",
                            "data": {},
                        })
                        return
                    authenticated = True
                    self._owner = writer
                    self._session_id = str(self.session_factory())
                    self._event_sequence = 0
                    self._event_sender = send
                    deadline = received_at + self.heartbeat_timeout
                if request.type == "system.heartbeat":
                    deadline = received_at + self.heartbeat_timeout
                if len(replies) >= 64:
                    await send({
                        "id": request.id,
                        "state": "failed",
                        "code": "too_many_pending",
                        "data": {},
                    })
                    return
                # Confirm admission before scheduling command work.  This keeps
                # every accepted acknowledgement observable even when the next
                # request trips the pending limit and closes the connection.
                await send({
                    "id": request.id,
                    "state": "accepted",
                    "code": "accepted",
                    "data": {},
                })
                task = asyncio.create_task(forward(request))
                replies.add(task)
                task.add_done_callback(replies.discard)
        except (ConnectionError, OSError, TimeoutError):
            pass
        finally:
            if authenticated:
                self._event_sender = None
                self._session_id = None
                self._event_sequence = 0
                if self._owner is writer:
                    self._owner = None
                result = self.service.disconnected()
                if inspect.isawaitable(result):
                    with suppress(Exception):
                        await result
            for task in replies:
                task.cancel()
            await asyncio.gather(*replies, return_exceptions=True)
            writer.close()
            with suppress(ConnectionError, OSError):
                await writer.wait_closed()
            if connection is not None:
                self._connections.discard(connection)
