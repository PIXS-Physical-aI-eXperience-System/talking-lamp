"""Reconnecting authenticated NDJSON client for the Pi device service."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import math
from typing import Any
from uuid import UUID, uuid4


MAX_LINE_BYTES = 16 * 1024


class TransportError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def _encode(message: dict[str, object]) -> bytes:
    try:
        line = json.dumps(
            message, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise TransportError("invalid_request", "request is not finite JSON") from exc
    if len(line) > MAX_LINE_BYTES:
        raise TransportError("line_too_long", "request exceeds 16 KiB")
    return line


class DeviceTransport:
    def __init__(
        self,
        host: str,
        port: int,
        token: str,
        *,
        heartbeat_interval: float = 1.0,
        reconnect_delay: float = 0.25,
    ) -> None:
        if not isinstance(host, str) or not host or not isinstance(token, str) or not token:
            raise TransportError("invalid_config", "host and token must not be empty")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise TransportError("invalid_config", "port must be in 1..65535")
        for name, value in (
            ("heartbeat_interval", heartbeat_interval), ("reconnect_delay", reconnect_delay)):
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
            ):
                raise TransportError("invalid_config", f"{name} must be finite and positive")
        self.host = host
        self.port = port
        self.token = token
        self.heartbeat_interval = float(heartbeat_interval)
        self.reconnect_delay = float(reconnect_delay)
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._write_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future] = {}
        self._events: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=256)
        self._reader_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._closed = False
        self._connected = asyncio.Event()
        self._session_id: str | None = None
        self._session_sequence = 0
        self._old_sessions: set[str] = set()

    @property
    def connected(self) -> bool:
        return self._connected.is_set() and self._writer is not None

    async def start(self) -> None:
        """Attempt once, then keep reconnecting without failing the owner node."""
        try:
            await self.connect()
        except TransportError:
            self._schedule_reconnect()

    async def connect(self) -> None:
        if self._closed:
            raise TransportError("closed", "transport is closed")
        async with self._connect_lock:
            if self.connected:
                return
            try:
                reader, writer = await asyncio.open_connection(
                    self.host, self.port, limit=MAX_LINE_BYTES + 1)
            except OSError as exc:
                raise TransportError("connection_failed", str(exc)) from exc
            self._reader = reader
            self._writer = writer
            self._connected.set()
            self._reader_task = asyncio.create_task(self._read_loop())
            if self._heartbeat_task is None or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def wait_connected(self, timeout: float = 5.0) -> None:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout)
        except TimeoutError as exc:
            raise TransportError("connection_timeout", "device reconnect timed out") from exc

    async def request(
        self, kind: str, payload: dict[str, object], ttl_ms: int = 1000,
        *, response_timeout: float | None = None,
    ) -> dict[str, object]:
        if not self.connected:
            raise TransportError("not_connected", "device transport is not connected")
        if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or not 1 <= ttl_ms <= 10_000:
            raise TransportError("invalid_request", "ttl_ms must be in 1..10000")
        if response_timeout is None:
            terminal_timeout = ttl_ms / 1000.0 + 2.0
        elif (
            isinstance(response_timeout, bool)
            or not isinstance(response_timeout, (int, float))
            or not math.isfinite(response_timeout)
            or response_timeout <= 0
        ):
            raise TransportError(
                "invalid_request", "response_timeout must be finite and positive")
        else:
            terminal_timeout = float(response_timeout)
        ident = str(uuid4())
        line = _encode({
            "version": 1, "id": ident, "type": kind, "ttl_ms": ttl_ms,
            "token": self.token, "payload": payload,
        })
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[ident] = future
        try:
            async with self._write_lock:
                writer = self._writer
                if writer is None:
                    raise TransportError("connection_lost", "device connection was lost")
                writer.write(line)
                await writer.drain()
            try:
                return await asyncio.wait_for(
                    asyncio.shield(future), timeout=terminal_timeout)
            except TimeoutError as exc:
                raise TransportError("timeout", "device request timed out") from exc
        except (ConnectionError, OSError) as exc:
            raise TransportError("connection_lost", str(exc)) from exc
        finally:
            self._pending.pop(ident, None)

    async def next_event(self) -> dict[str, object]:
        return await self._events.get()

    async def _read_loop(self) -> None:
        reason = "device connection closed"
        try:
            assert self._reader is not None
            while not self._closed:
                try:
                    line = await self._reader.readline()
                except ValueError as exc:
                    reason = "device response exceeds 16 KiB"
                    raise TransportError("line_too_long", reason) from exc
                if not line:
                    break
                if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
                    reason = "invalid device NDJSON frame"
                    break
                try:
                    message = json.loads(line)
                except (UnicodeError, json.JSONDecodeError):
                    reason = "invalid device JSON response"
                    break
                if not isinstance(message, dict):
                    reason = "device response is not an object"
                    break
                if "event" in message:
                    self._accept_event(message)
                    continue
                ident = message.get("id")
                state = message.get("state")
                if not isinstance(ident, str) or state not in {
                    "accepted", "completed", "failed", "cancelled",
                }:
                    reason = "invalid correlated device response"
                    break
                future = self._pending.get(ident)
                if future is not None and state != "accepted" and not future.done():
                    future.set_result(message)
        except (ConnectionError, OSError, TransportError) as exc:
            reason = str(exc)
        finally:
            await self._lost(reason)

    def _accept_event(self, message: dict[str, Any]) -> None:
        if set(message) != {"version", "event", "session_id", "sequence", "data"}:
            return
        session = message["session_id"]
        sequence = message["sequence"]
        if (
            message["version"] != 1 or not isinstance(message["event"], str)
            or not _uuid(session) or isinstance(sequence, bool)
            or not isinstance(sequence, int) or sequence < 1
            or not isinstance(message["data"], dict)
        ):
            return
        assert isinstance(session, str)
        if self._session_id is None:
            if sequence != 1:
                return
            self._session_id = session
            self._session_sequence = 0
        elif session != self._session_id:
            if session in self._old_sessions or sequence != 1:
                return
            self._old_sessions.add(self._session_id)
            self._session_id = session
            self._session_sequence = 0
        if sequence != self._session_sequence + 1:
            return
        self._session_sequence = sequence
        if self._events.full():
            with suppress(asyncio.QueueEmpty):
                self._events.get_nowait()
        self._events.put_nowait(message)

    async def _lost(self, reason: str) -> None:
        self._connected.clear()
        writer, self._writer = self._writer, None
        self._reader = None
        if writer is not None:
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(TransportError("connection_lost", reason))
        self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        if not self._closed and (
            self._reconnect_task is None or self._reconnect_task.done()
        ):
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        delay = self.reconnect_delay
        while not self._closed and not self.connected:
            await asyncio.sleep(delay)
            try:
                await self.connect()
            except TransportError:
                delay = min(delay * 2.0, 5.0)

    async def _heartbeat_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.heartbeat_interval)
            if not self.connected:
                continue
            try:
                await self.request("system.heartbeat", {}, ttl_ms=1000)
            except TransportError:
                pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connected.clear()
        current = asyncio.current_task()
        tasks = [task for task in (
            self._reader_task, self._heartbeat_task, self._reconnect_task,
        ) if task is not None and task is not current]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(TransportError("closed", "transport closed"))
        self._pending.clear()
