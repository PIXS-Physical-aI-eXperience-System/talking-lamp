"""Async NDJSON transport. Only MotionController may mutate the runtime."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Callable

from .catalog import MotionCatalog
from .config import CONTROL_HZ, RECORDINGS_DIR, REST_POSE
from .controller import CommandTicket, MotionController
from .protocol import MAX_LINE_BYTES, ProtocolError, decode_request, encode_message
from .runtime import MotionRuntime, NullBackend
from .trajectory import TrajectoryGenerator


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


async def _serve(controller: MotionController, server: MotionTcpServer,
                 shutdown_requested: Callable[[], bool]) -> None:
    """Join the sole motion owner before the caller parks the backend."""
    owner = threading.Thread(target=controller.run, name="motion-owner")
    serving = None
    try:
        if shutdown_requested():
            controller.stop("shutdown during startup")
        owner.start()
        serving = asyncio.create_task(server.serve_forever())
        while not shutdown_requested() and owner.is_alive() and not serving.done():
            await asyncio.sleep(.05)
        if serving.done():
            await serving  # Surface bind/transport failures to systemd.
    finally:
        controller.stop("daemon shutdown")
        # No timeout: parking must never race a still-running motor writer.
        if owner.ident is not None:
            owner.join()
        try:
            if serving is not None:
                serving.cancel()
                with suppress(asyncio.CancelledError):
                    await serving
        finally:
            await server.close()
    fault = controller.snapshot().fault
    if fault:
        raise RuntimeError(f"Motion controller failed: {fault}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Authenticated Pi motion daemon")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--port", help="Feetech serial device (physical mode)")
    mode.add_argument("--null-backend", action="store_true",
                      help="Commission transport without importing or opening hardware")
    parser.add_argument("--lamp-id", help="Existing LeRobot calibration id; required with --port")
    parser.add_argument("--catalog", type=Path, default=RECORDINGS_DIR / "catalog.toml")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=8765)
    parser.add_argument("--allow-host", action="append", help="Allowed peer IP; repeat for multiple peers")
    parser.add_argument("--heartbeat-timeout", type=float, default=2.5)
    parser.add_argument("--feedback-hz", type=float, default=20.)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.null_backend and args.lamp_id is not None:
        parser.error("--lamp-id cannot be used with --null-backend")
    if not args.null_backend and (not args.port or not args.lamp_id):
        parser.error("physical mode requires --port and --lamp-id")
    if not args.bind or not 1 <= args.tcp_port <= 65535:
        parser.error("--bind must be non-empty and --tcp-port must be in 1..65535")
    if not math.isfinite(args.heartbeat_timeout) or args.heartbeat_timeout <= 0:
        parser.error("--heartbeat-timeout must be finite and positive")
    if not math.isfinite(args.feedback_hz) or not 0 < args.feedback_hz <= CONTROL_HZ:
        parser.error(f"--feedback-hz must be in (0, {CONTROL_HZ:g}]")
    try:
        token = os.environ.get("TALKING_LAMP_TOKEN", "")
        if not token.strip():
            raise ValueError("TALKING_LAMP_TOKEN must be set and non-empty")
        catalog = MotionCatalog.load(args.catalog)
        library = catalog.library()  # Validate all enabled CSVs before torque.
        if not args.null_backend and TrajectoryGenerator(REST_POSE).backend != "ruckig":
            raise RuntimeError("Physical motion requires Ruckig for jerk limits; install ruckig")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Motion daemon preflight failed: {exc}", file=sys.stderr)
        return 2

    stopping = False
    previous_handlers = {}
    backend = controller = None
    def request_shutdown(*_):
        # Python handlers may re-enter on a repeated signal. Event.set() and
        # controller.stop() acquire locks, so only assign this main-thread flag.
        nonlocal stopping
        stopping = True
    try:
        # Install before opening hardware. A signal during construction requests
        # shutdown without interrupting assignment of the backend we must close.
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_shutdown)
        try:
            if args.null_backend:
                backend = NullBackend()
            else:
                from .hardware_backend import FeetechBackend
                backend = FeetechBackend(port=args.port, lamp_id=args.lamp_id,
                                         feedback_hz=args.feedback_hz)
            if not stopping:
                runtime = MotionRuntime(backend=backend, initial_pose=backend.measured(),
                                        primitives=library)
                controller = MotionController(runtime, catalog)
                server = MotionTcpServer(controller, token=token, host=args.bind,
                    port=args.tcp_port, allowed_hosts=None if args.allow_host is None else set(args.allow_host),
                    heartbeat_timeout=args.heartbeat_timeout)
                asyncio.run(_serve(controller, server, lambda: stopping))
        finally:
            if controller is not None:
                controller.stop("daemon shutdown")
            if backend is not None:
                backend.close()
    except Exception as exc:
        print(f"Motion daemon failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
