import asyncio
from contextlib import asynccontextmanager
import json
from uuid import UUID

import pytest

from device.motion_client import MotionClientError, MotionUnixClient


@asynccontextmanager
async def unix_server(tmp_path, handler):
    path = tmp_path / "motion.sock"
    server = await asyncio.start_unix_server(handler, path=path)
    try:
        yield path
    finally:
        server.close()
        await server.wait_closed()


def test_unix_client_sends_exact_token_free_envelope_and_correlates_events(tmp_path):
    async def scenario():
        received = []

        async def handler(reader, writer):
            request = json.loads(await reader.readline())
            received.append(request)
            for state in ("accepted", "completed"):
                writer.write((json.dumps({
                    "id": request["id"], "state": state, "code": state, "data": {},
                }) + "\n").encode())
                await writer.drain()
            writer.close()
            await writer.wait_closed()

        async with unix_server(tmp_path, handler) as path:
            client = MotionUnixClient(path, id_factory=lambda: UUID(
                "10000000-0000-0000-0000-000000000001"))
            events = await client.request("orientation.status", {}, ttl_ms=750)

        assert [event["state"] for event in events] == ["accepted", "completed"]
        assert received == [{
            "version": 1,
            "id": "10000000-0000-0000-0000-000000000001",
            "type": "orientation.status",
            "ttl_ms": 750,
            "payload": {},
        }]
    asyncio.run(scenario())


def test_unix_client_does_not_replay_after_accepted_then_eof(tmp_path):
    async def scenario():
        connections = 0

        async def handler(reader, writer):
            nonlocal connections
            connections += 1
            request = json.loads(await reader.readline())
            writer.write((json.dumps({
                "id": request["id"], "state": "accepted", "code": "accepted",
            }) + "\n").encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        async with unix_server(tmp_path, handler) as path:
            client = MotionUnixClient(path)
            with pytest.raises(MotionClientError) as error:
                await client.request("orientation.acquire", {
                    "speech_id": "20000000-0000-0000-0000-000000000001",
                    "target_yaw": 0.2,
                })
            await asyncio.sleep(0)
        assert error.value.code == "connection_lost"
        assert connections == 1
    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kind,payload,ttl",
    [
        ("motion.play", {}, 1000),
        ("orientation.status", {"extra": True}, 1000),
        ("orientation.status", {}, 0),
        ("orientation.acquire", {"speech_id": "bad", "target_yaw": 0.0}, 1000),
    ],
)
def test_unix_client_rejects_invalid_requests_before_connect(tmp_path, kind, payload, ttl):
    async def scenario():
        client = MotionUnixClient(tmp_path / "missing.sock")
        with pytest.raises(MotionClientError) as error:
            await client.request(kind, payload, ttl_ms=ttl)
        assert error.value.code == "invalid_request"
    asyncio.run(scenario())


def test_unix_client_rejects_mismatched_or_oversized_response(tmp_path):
    async def run_case(response):
        async def handler(reader, writer):
            await reader.readline()
            writer.write(response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        async with unix_server(tmp_path, handler) as path:
            with pytest.raises(MotionClientError) as error:
                await MotionUnixClient(path).request("orientation.status", {})
            return error.value.code

    mismatch = (json.dumps({
        "id": "30000000-0000-0000-0000-000000000001",
        "state": "completed",
        "code": "completed",
    }) + "\n").encode()
    assert asyncio.run(run_case(mismatch)) == "invalid_response"

    second_dir = tmp_path / "second"
    second_dir.mkdir()
    tmp_path = second_dir
    assert asyncio.run(run_case(b"x" * (16 * 1024 + 1) + b"\n")) == "line_too_long"


def test_unix_client_times_out_once_without_reconnect(tmp_path):
    async def scenario():
        connections = 0

        async def handler(reader, writer):
            nonlocal connections
            connections += 1
            await reader.readline()
            await asyncio.sleep(1)
            writer.close()

        async with unix_server(tmp_path, handler) as path:
            client = MotionUnixClient(path, response_timeout=0.02)
            with pytest.raises(MotionClientError) as error:
                await client.request("orientation.status", {})
        assert error.value.code == "timeout"
        assert connections == 1
    asyncio.run(scenario())
