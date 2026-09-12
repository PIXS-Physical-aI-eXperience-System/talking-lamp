"""Async NDJSON transport. Only MotionController may mutate the runtime."""
from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import asdict
import json
import math
import time

from .controller import CommandTicket, MotionController
from .protocol import MAX_LINE_BYTES, ProtocolError, decode_request, encode_message


class MotionTcpServer:
    def __init__(self, controller: MotionController, *, token: str,
                 host: str = "127.0.0.1", port: int = 8765,
                 allowed_hosts: set[str] | None = None,
                 heartbeat_timeout: float = 2.5) -> None:
        if not token:
            raise ValueError("token must not be empty")
        if not math.isfinite(heartbeat_timeout) or heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout must be finite and positive")
        self.controller = controller
        self.token = token
        self.host, self.port = host, port
        self.allowed_hosts = None if allowed_hosts is None else frozenset(allowed_hosts)
        self.heartbeat_timeout = heartbeat_timeout
        self._server: asyncio.Server | None = None
        self._owner: asyncio.StreamWriter | None = None
        self._connections: set[asyncio.Task] = set()

    @property
    def sockets(self):
        return self._server.sockets if self._server else ()

    async def start(self) -> MotionTcpServer:
        if self._server is None:
            self._server = await asyncio.start_server(
                self._handle, self.host, self.port, limit=MAX_LINE_BYTES + 1)
        return self

    async def serve_forever(self) -> None:
        await self.start()
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

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connection = asyncio.current_task()
        self._connections.add(connection)
        replies: set[asyncio.Task] = set()
        write_lock = asyncio.Lock()
        authenticated = False
        deadline = time.monotonic() + self.heartbeat_timeout
        errors = 0

        async def send(message):
            async with write_lock:
                writer.write(encode_message(message))
                await asyncio.wait_for(writer.drain(), self.heartbeat_timeout)

        async def forward(ticket: CommandTicket):
            try:
                # Cancelling transport waits must not cancel shared controller futures.
                accepted = await asyncio.shield(asyncio.wrap_future(ticket.accepted))
                message = asdict(accepted)
                message["id"] = message.pop("request_id")
                await send(message)
                if accepted.state == "accepted":
                    completed = await asyncio.shield(asyncio.wrap_future(ticket.completed))
                    message = asdict(completed)
                    message["id"] = message.pop("request_id")
                    await send(message)
            except (ConnectionError, OSError, TimeoutError, ProtocolError):
                writer.close()

        try:
            peer = writer.get_extra_info("peername")
            if self.allowed_hosts is not None and (not peer or peer[0] not in self.allowed_hosts):
                await send(dict(state="failed", code="host_not_allowed"))
                return
            while True:
                try:
                    async with asyncio.timeout_at(asyncio.get_running_loop().time() +
                                                  max(0, deadline - time.monotonic())):
                        line = await reader.readline()
                except ValueError:  # StreamReader's limit exceeded, with or without newline.
                    await send(dict(state="failed", code="line_too_long"))
                    return
                if not line:
                    return
                received_at = time.monotonic()
                try:
                    request = decode_request(line, token=self.token, received_at=received_at)
                except ProtocolError as exc:
                    message = dict(state="failed", code=exc.code, message=exc.message)
                    # Correlate schema errors where possible, never echo the supplied token.
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
                        await send(dict(id=request.id, state="failed", code="controller_connected"))
                        return
                    self._owner = writer
                    authenticated = True
                    deadline = received_at + self.heartbeat_timeout
                if request.type == "system.heartbeat":
                    deadline = received_at + self.heartbeat_timeout
                if len(replies) >= 64:
                    await send(dict(id=request.id, state="failed", code="too_many_pending"))
                    return
                ticket = self.controller.submit(request)
                reply = asyncio.create_task(forward(ticket))
                replies.add(reply)
                reply.add_done_callback(replies.discard)
        except (ConnectionError, OSError, TimeoutError):
            pass
        finally:
            if authenticated:
                self.controller.remote_disconnected()
                self._owner = None
            for task in replies:
                task.cancel()
            await asyncio.gather(*replies, return_exceptions=True)
            writer.close()
            with suppress(ConnectionError, OSError):
                await writer.wait_closed()
            self._connections.discard(connection)
