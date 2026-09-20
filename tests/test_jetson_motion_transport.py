import asyncio
import json

import pytest

from lamp_motion_bridge.transport import MotionTransport, TransportError


def test_motion_transport_correlates_terminal_response_and_never_replays():
    async def scenario():
        requests = []

        async def handler(reader, writer):
            request = json.loads(await reader.readline())
            requests.append(request)
            writer.write(json.dumps({
                "id": request["id"], "state": "accepted", "code": "accepted", "data": {},
            }).encode() + b"\n")
            writer.write(json.dumps({
                "id": request["id"], "state": "completed", "code": "completed",
                "data": {"motions": ["idle", "nod"]},
            }).encode() + b"\n")
            await writer.drain()
            await reader.read()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        transport = MotionTransport(
            "127.0.0.1", server.sockets[0].getsockname()[1], "motion-token",
            heartbeat_interval=60)
        try:
            await transport.connect()
            result = await transport.request("motion.list", {})
            assert result["data"]["motions"] == ["idle", "nod"]
            assert requests[0]["token"] == "motion-token"
            assert requests[0]["type"] == "motion.list"
        finally:
            await asyncio.wait_for(transport.close(), 1)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_motion_transport_rejects_nonfinite_or_oversized_request_locally():
    async def scenario():
        async def handler(reader, writer):
            await reader.read()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        transport = MotionTransport(
            "127.0.0.1", server.sockets[0].getsockname()[1], "motion-token",
            heartbeat_interval=60)
        try:
            await transport.connect()
            with pytest.raises(TransportError) as nonfinite:
                await transport.request("track.point", {"point": [0.0, float("nan"), 0.0]})
            assert nonfinite.value.code == "invalid_request"
            with pytest.raises(TransportError) as large:
                await transport.request("motion.play", {"name": "x" * 20_000})
            assert large.value.code == "line_too_long"
        finally:
            await asyncio.wait_for(transport.close(), 1)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
