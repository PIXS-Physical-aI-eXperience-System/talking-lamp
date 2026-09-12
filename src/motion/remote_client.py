"""Dependency-free persistent client and diagnostic CLI for Pi motion commands.

A lost connection fails outstanding requests. Reconnection never replays them;
callers explicitly decide whether another request (with a new UUID) is warranted.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import json
import math
import os
import re
import sys
from uuid import uuid4

from .protocol import MAX_LINE_BYTES, ProtocolError, decode_request, encode_message

RECONNECT_DELAYS = (.25, .5, 1., 2., 5.)


class MotionClient:
    def __init__(self, host: str, port: int = 8765, *, token: str,
                 heartbeat_interval: float = 1.) -> None:
        if not token:
            raise ValueError("token must not be empty")
        if not math.isfinite(heartbeat_interval) or heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be finite and positive")
        self.host, self.port, self.token = host, port, token
        self.heartbeat_interval = heartbeat_interval
        self._writer: asyncio.StreamWriter | None = None
        self._connected = asyncio.Event()
        self._closed = False
        self._runner: asyncio.Task | None = None
        self._accepted: dict[str, asyncio.Future] = {}
        self._terminal: dict[str, asyncio.Future] = {}

    @property
    def connected(self) -> bool:
        return self._connected.is_set() and not self._closed

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def connect(self) -> None:
        if self._closed:
            raise ConnectionError("client is closed")
        if self._runner is None:
            self._runner = asyncio.create_task(self._run())
        await self._connected.wait()
        if self._closed:
            raise ConnectionError("client is closed")

    async def close(self) -> None:
        self._closed = True
        if self._runner is not None:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
            self._runner = None
        self._fail_pending()
        self._connected.set()  # Wake callers waiting for an unavailable server.

    def _fail_pending(self) -> None:
        for future in (*self._accepted.values(), *self._terminal.values()):
            if not future.done():
                future.set_exception(ConnectionError("motion connection lost; request was not replayed"))

    async def _run(self) -> None:
        attempt = 0
        while not self._closed:
            heartbeat = None
            writer = None
            try:
                reader, writer = await asyncio.open_connection(
                    self.host, self.port, limit=MAX_LINE_BYTES + 1)
                self._writer = writer
                self._connected.set()
                heartbeat = asyncio.create_task(self._heartbeats())
                while line := await reader.readline():
                    if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
                        raise ConnectionError("invalid response framing")
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise ConnectionError("invalid response")
                    ident, state = message.get("id"), message.get("state")
                    if not isinstance(ident, str) or not isinstance(state, str) or state not in {"accepted", "completed", "failed", "cancelled"}:
                        raise ConnectionError("invalid response")
                    attempt = 0
                    if ident not in self._accepted:
                        continue
                    accepted = self._accepted[ident]
                    if not accepted.done():
                        accepted.set_result(message)
                    if state != "accepted" and not self._terminal[ident].done():
                        self._terminal[ident].set_result(message)
            except (OSError, ValueError, ConnectionError):
                pass
            finally:
                self._connected.clear()
                self._writer = None
                self._fail_pending()
                if heartbeat:
                    heartbeat.cancel()
                    await asyncio.gather(heartbeat, return_exceptions=True)
                if writer:
                    writer.close()
                    with suppress(OSError):
                        await writer.wait_closed()
            await asyncio.sleep(RECONNECT_DELAYS[min(attempt, len(RECONNECT_DELAYS) - 1)])
            attempt += 1

    async def _heartbeats(self) -> None:
        try:
            while True:
                async with asyncio.timeout(self.heartbeat_interval + 1.):
                    events = await self.request("system.heartbeat", {})
                if events[-1]["state"] != "completed":
                    raise ConnectionError("heartbeat rejected")
                await asyncio.sleep(self.heartbeat_interval)
        except (ConnectionError, TimeoutError, OSError):
            if self._writer:
                self._writer.close()

    async def request(self, type: str, payload: dict | None = None,
                      ttl_ms: int = 1000) -> list[dict]:
        ident = str(uuid4())
        line = encode_message(dict(version=1, id=ident, type=type, ttl_ms=ttl_ms,
                                   token=self.token, payload={} if payload is None else payload))
        decode_request(line, token=self.token, received_at=0.)
        await self.connect()
        writer = self._writer
        if writer is None:
            raise ConnectionError("motion connection lost")
        loop = asyncio.get_running_loop()
        accepted, terminal = loop.create_future(), loop.create_future()
        self._accepted[ident], self._terminal[ident] = accepted, terminal
        try:
            writer.write(line)
            await writer.drain()
            first = await accepted
            last = await terminal
            return [first, last] if first["state"] == "accepted" else [last]
        finally:
            self._accepted.pop(ident, None)
            self._terminal.pop(ident, None)
            for future in (accepted, terminal):
                if not future.done():
                    future.cancel()
                elif not future.cancelled():
                    future.exception()  # Retrieve a sibling failure when the first await failed.


def _number(low, high, integer=False):
    def parse(value):
        try:
            result = int(value) if integer else float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("expected a number") from exc
        if not math.isfinite(result) or not low <= result <= high:
            raise argparse.ArgumentTypeError(f"must be from {low} to {high}")
        return result
    return parse


def _motion_name(value):
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        raise argparse.ArgumentTypeError("invalid motion name")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=_number(1, 65535, True), default=8765)
    parser.add_argument("--ttl-ms", type=_number(1, 10000, True), default=1000)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "status", "interrupt"):
        sub.add_parser(command)
    play = sub.add_parser("play")
    play.add_argument("name", type=_motion_name)
    play.add_argument("--replace-current", action="store_true")
    play.add_argument("--intensity", type=_number(0., 1.), default=1.)
    play.add_argument("--repeat", type=_number(1, 3, True), default=1)
    for command in ("track-point", "task-light"):
        point = sub.add_parser(command)
        point.add_argument("point", nargs=3, type=_number(-2., 2.), metavar="METRES")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    token = os.environ.get("TALKING_LAMP_TOKEN")
    if not token:
        print("TALKING_LAMP_TOKEN must be set", file=sys.stderr)
        return 2
    kind = {"list": "motion.list", "status": "motion.status", "play": "motion.play",
            "interrupt": "motion.interrupt", "track-point": "track.point",
            "task-light": "task_light.place"}[args.command]
    payload = {}
    if args.command == "play":
        payload = dict(name=args.name, replace_current=args.replace_current,
                       intensity=args.intensity, repeat=args.repeat)
    elif args.command in {"track-point", "task-light"}:
        payload = dict(point=args.point)

    async def run():
        async with MotionClient(args.host, args.port, token=token) as client:
            events = await client.request(kind, payload, args.ttl_ms)
            for event in events:
                print(json.dumps(event))
            return 0 if events[-1]["state"] == "completed" else 1
    try:
        return asyncio.run(run())
    except (ConnectionError, OSError, ProtocolError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
