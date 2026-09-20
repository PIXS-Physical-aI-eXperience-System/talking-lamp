"""Protected local Unix transport for Pi-only orientation commands.

This transport deliberately owns no motion runtime state.  It validates local
NDJSON messages and forwards controller tickets; the controller's one owner
thread remains the only component that can advance hardware.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import asdict
import json
import os
from pathlib import Path
import stat
import time

from .controller import CommandTicket, MotionController
from .protocol import MAX_LINE_BYTES, ProtocolError, decode_local_request, encode_message


DEFAULT_SOCKET_PATH = Path("/run/talking-lamp/motion-control.sock")


class MotionUnixServer:
    """Serve local orientation RPC without remote token or owner semantics."""

    def __init__(self, controller: MotionController,
                 path: str | Path = DEFAULT_SOCKET_PATH) -> None:
        self.controller = controller
        self.path = Path(path)
        self._server: asyncio.AbstractServer | None = None
        self._connections: set[asyncio.Task] = set()
        self._owns_socket = False
        # One ticket may be intentionally shared by duplicate IDs from several
        # local clients. Only the last disconnected holder may invalidate it.
        self._ticket_holders: dict[int, tuple[CommandTicket, int]] = {}

    @property
    def sockets(self):
        return self._server.sockets if self._server else ()

    async def start(self) -> MotionUnixServer:
        if self._server is not None:
            return self
        self._remove_stale_socket()
        try:
            self._server = await asyncio.start_unix_server(
                self._handle, path=str(self.path), limit=MAX_LINE_BYTES + 1
            )
            self._owns_socket = True
            os.chmod(self.path, 0o660)
        except BaseException:
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
                self._server = None
            self._remove_owned_socket()
            raise
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
        self._remove_owned_socket()

    def _remove_stale_socket(self) -> None:
        """Remove only a socket node; a file or symlink is never ours to unlink."""
        try:
            mode = os.lstat(self.path).st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(mode):
            raise FileExistsError(f"refusing to replace non-socket path: {self.path}")
        os.unlink(self.path)

    def _remove_owned_socket(self) -> None:
        if not self._owns_socket:
            return
        self._owns_socket = False
        try:
            mode = os.lstat(self.path).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(mode):
            os.unlink(self.path)

    def _hold_ticket(self, ticket: CommandTicket) -> int:
        key = id(ticket)
        existing = self._ticket_holders.get(key)
        if existing is None:
            self._ticket_holders[key] = (ticket, 1)
        else:
            self._ticket_holders[key] = (ticket, existing[1] + 1)
        return key

    def _release_ticket(self, key: int) -> None:
        existing = self._ticket_holders.get(key)
        if existing is None:
            return
        ticket, holders = existing
        if holders > 1:
            self._ticket_holders[key] = (ticket, holders - 1)
            return
        del self._ticket_holders[key]
        self.controller.invalidate_unstarted(ticket)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connection = asyncio.current_task()
        if connection is not None:
            self._connections.add(connection)
        replies: set[asyncio.Task] = set()
        held_tickets: dict[int, int] = {}
        next_hold = 0
        write_lock = asyncio.Lock()
        errors = 0

        async def send(message: dict[str, object]) -> None:
            async with write_lock:
                writer.write(encode_message(message))
                await writer.drain()

        def release_holder(hold: int) -> None:
            key = held_tickets.pop(hold, None)
            if key is not None:
                self._release_ticket(key)

        async def forward(ticket: CommandTicket, hold: int) -> None:
            try:
                accepted = await asyncio.shield(asyncio.wrap_future(ticket.accepted))
                # A terminal result or accepted result means the controller has
                # either finished this ticket or atomically begun it. EOF can no
                # longer invalidate it, so do not retain the local holder.
                release_holder(hold)
                message = asdict(accepted)
                message["id"] = message.pop("request_id")
                await send(message)
                if accepted.state == "accepted":
                    completed = await asyncio.shield(asyncio.wrap_future(ticket.completed))
                    message = asdict(completed)
                    message["id"] = message.pop("request_id")
                    await send(message)
            except (ConnectionError, OSError, ProtocolError):
                writer.close()
            finally:
                release_holder(hold)

        try:
            while True:
                try:
                    line = await reader.readline()
                except ValueError:
                    await send({"state": "failed", "code": "line_too_long"})
                    return
                if not line:
                    return
                try:
                    request = decode_local_request(line, received_at=time.monotonic())
                except ProtocolError as exc:
                    message: dict[str, object] = {
                        "state": "failed", "code": exc.code, "message": exc.message,
                    }
                    with suppress(ValueError, UnicodeError):
                        raw = json.loads(line)
                        if isinstance(raw, dict) and isinstance(raw.get("id"), str):
                            message["id"] = raw["id"][:64]
                    await send(message)
                    errors += 1
                    if exc.code in {"line_too_long", "missing_newline"} or errors >= 3:
                        return
                    continue
                errors = 0
                if len(replies) >= 64:
                    await send({"id": request.id, "state": "failed", "code": "too_many_pending"})
                    return
                ticket = self.controller.submit(request, origin="local")
                hold = next_hold
                next_hold += 1
                held_tickets[hold] = self._hold_ticket(ticket)
                reply = asyncio.create_task(forward(ticket, hold))
                replies.add(reply)
                reply.add_done_callback(replies.discard)
        except (ConnectionError, OSError):
            pass
        finally:
            for task in replies:
                task.cancel()
            await asyncio.gather(*replies, return_exceptions=True)
            for hold in list(held_tickets):
                release_holder(hold)
            writer.close()
            with suppress(ConnectionError, OSError):
                await writer.wait_closed()
            if connection is not None:
                self._connections.discard(connection)
