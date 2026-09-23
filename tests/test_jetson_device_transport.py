import asyncio
import json
import socket
from uuid import UUID, uuid4

import pytest

from lamp_device_bridge.transport import DeviceTransport, TransportError


TOKEN = "jetson-device-token"


async def send(writer, message):
    writer.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
    await writer.drain()


def test_device_transport_correlates_requests_and_orders_session_events():
    async def scenario():
        observed = []

        async def handler(reader, writer):
            request = json.loads(await reader.readline())
            observed.append(request)
            assert request["token"] == TOKEN
            assert str(UUID(request["id"])) == request["id"]
            await send(writer, {
                "id": request["id"], "state": "accepted", "code": "accepted", "data": {}})
            await send(writer, {
                "id": request["id"], "state": "completed", "code": "completed",
                "data": {"active": False}})
            session_id = str(uuid4())
            await send(writer, {
                "version": 1, "event": "led.status", "session_id": session_id,
                "sequence": 1, "data": {"active": False}})
            await send(writer, {
                "version": 1, "event": "orientation.status", "session_id": session_id,
                "sequence": 2, "data": {"state": "aligned"}})
            await reader.read()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        transport = DeviceTransport(
            "127.0.0.1", port, TOKEN, heartbeat_interval=60, reconnect_delay=.01)
        try:
            await asyncio.wait_for(transport.connect(), 1)
            result = await transport.request("led.status", {})
            assert result["state"] == "completed"
            assert result["data"] == {"active": False}
            first = await asyncio.wait_for(transport.next_event(), 1)
            second = await asyncio.wait_for(transport.next_event(), 1)
            assert [first["sequence"], second["sequence"]] == [1, 2]
        finally:
            await asyncio.wait_for(transport.close(), 1)
            server.close()
            await server.wait_closed()
        assert observed[0]["type"] == "led.status"
        assert observed[0]["ttl_ms"] == 1000

    asyncio.run(scenario())


def test_disconnect_fails_pending_request_and_reconnect_never_replays_it():
    async def scenario():
        requests = []
        connections = 0

        async def handler(reader, writer):
            nonlocal connections
            connections += 1
            line = await reader.readline()
            if line:
                requests.append(json.loads(line))
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        transport = DeviceTransport(
            "127.0.0.1", port, TOKEN, heartbeat_interval=60, reconnect_delay=.01)
        try:
            await asyncio.wait_for(transport.connect(), 1)
            with pytest.raises(TransportError) as error:
                await transport.request("device.status", {})
            assert error.value.code == "connection_lost"
            for _ in range(100):
                if connections >= 2 and transport.connected:
                    break
                await asyncio.sleep(.01)
            assert transport.connected
            assert len(requests) == 1
        finally:
            await asyncio.wait_for(transport.close(), 1)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_stale_or_out_of_order_events_are_discarded():
    async def scenario():
        async def handler(reader, writer):
            request = json.loads(await reader.readline())
            await send(writer, {
                "id": request["id"], "state": "completed", "code": "completed", "data": {}})
            old = str(uuid4())
            new = str(uuid4())
            for session, sequence in ((old, 1), (old, 1), (new, 1), (old, 2), (new, 3), (new, 2)):
                await send(writer, {
                    "version": 1, "event": "led.status", "session_id": session,
                    "sequence": sequence, "data": {"sequence": sequence}})
            await reader.read()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        transport = DeviceTransport("127.0.0.1", port, TOKEN, heartbeat_interval=60)
        try:
            await asyncio.wait_for(transport.connect(), 1)
            await transport.request("device.status", {})
            events = [await asyncio.wait_for(transport.next_event(), 1) for _ in range(3)]
            assert [event["sequence"] for event in events] == [1, 1, 2]
        finally:
            await asyncio.wait_for(transport.close(), 1)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_initial_connection_failure_retries_until_server_appears():
    async def scenario():
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        transport = DeviceTransport(
            "127.0.0.1", port, TOKEN,
            heartbeat_interval=60, reconnect_delay=.01)
        server = None
        try:
            await transport.start()
            assert not transport.connected

            async def handler(reader, writer):
                await reader.read()
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_server(handler, "127.0.0.1", port)
            await transport.wait_connected(timeout=1)
            assert transport.connected
        finally:
            await asyncio.wait_for(transport.close(), 1)
            if server is not None:
                server.close()
                await server.wait_closed()

    asyncio.run(scenario())


def test_terminal_response_timeout_is_independent_from_receive_ttl():
    async def scenario():
        async def handler(reader, writer):
            request = json.loads(await reader.readline())
            await send(writer, {
                "id": request["id"], "state": "accepted",
                "code": "accepted", "data": {},
            })
            await asyncio.sleep(0.05)
            await send(writer, {
                "id": request["id"], "state": "completed",
                "code": "completed", "data": {"finished": True},
            })
            await reader.read()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        transport = DeviceTransport(
            "127.0.0.1", server.sockets[0].getsockname()[1], TOKEN,
            heartbeat_interval=60)
        try:
            await transport.connect()
            result = await transport.request(
                "motion.play", {}, ttl_ms=1, response_timeout=0.2)
            assert result["state"] == "completed"
            assert result["data"] == {"finished": True}
        finally:
            await asyncio.wait_for(transport.close(), 1)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_transport_uses_caller_request_id_for_correlated_cancellation():
    async def scenario():
        request_id = str(uuid4())

        async def handler(reader, writer):
            request = json.loads(await reader.readline())
            assert request["id"] == request_id
            await send(writer, {
                "id": request_id, "state": "completed",
                "code": "completed", "data": {},
            })
            await reader.read()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        transport = DeviceTransport(
            "127.0.0.1", server.sockets[0].getsockname()[1], TOKEN,
            heartbeat_interval=60)
        try:
            await transport.connect()
            result = await transport.request(
                "motion.play", {}, request_id=request_id)
            assert result["state"] == "completed"
        finally:
            await transport.close()
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
