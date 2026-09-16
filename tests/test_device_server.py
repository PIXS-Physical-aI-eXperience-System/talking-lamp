import asyncio
from dataclasses import dataclass
import json
from uuid import uuid4

import pytest

from device.coordinator import OrientationEvent
from device.audio import AudioStatus
from device.led import LedStatus
from device.protocol import encode_message
from device.server import DeviceCommandHandler, DeviceTcpServer


TOKEN = "device-server-token"


def envelope(kind="device.status", *, ident=None, payload=None, token=TOKEN, ttl_ms=1000):
    return encode_message({
        "version": 1,
        "id": str(uuid4()) if ident is None else ident,
        "type": kind,
        "ttl_ms": ttl_ms,
        "token": token,
        "payload": {} if payload is None else payload,
    })


async def receive(reader):
    line = await asyncio.wait_for(reader.readline(), 2)
    assert line
    return json.loads(line)


class FakeCoordinator:
    def __init__(self):
        self.calls = []

    def event(self, state):
        return OrientationEvent(
            state=state, speech_id="speech", raw_doa_deg=100.0,
            relative_rad=0.1, target_yaw=0.2, current_yaw=0.2,
            clamped=False, code=state, message="", timestamp=1.0,
        )

    async def status(self):
        self.calls.append("status")
        return self.event("aligned")

    async def return_center(self):
        self.calls.append("return_center")
        return self.event("centered")


class FakeLed:
    def __init__(self):
        self.calls = []
        self._status = LedStatus(False, 0.0, 0.0, False, None)

    @property
    def status(self):
        return self._status

    def frame(self, frame, *, brightness):
        self.calls.append(("frame", bytes(frame), brightness))
        self._status = LedStatus(True, brightness, min(brightness, 0.1), brightness > 0.1, None)
        return self._status

    def solid(self, rgb, *, brightness):
        self.calls.append(("solid", tuple(rgb), brightness))
        self._status = LedStatus(True, brightness, min(brightness, 0.1), brightness > 0.1, None)
        return self._status

    def clear(self):
        self.calls.append(("clear",))
        self._status = LedStatus(False, 0.0, 0.0, False, None)
        return self._status


class FakeAudio:
    def __init__(self):
        self.calls = []
        self._status = AudioStatus(True, False, None, "idle", "capture_running", "")

    @property
    def status(self):
        return self._status

    async def play_start(self, payload):
        self.calls.append(("start", dict(payload)))
        self._status = AudioStatus(True, True, payload["stream_id"], "playing", "playing", "")
        return self._status

    async def play_stop(self, stream_id):
        self.calls.append(("stop", stream_id))
        self._status = AudioStatus(True, False, None, "idle", "drained", "")
        return self._status

    async def disconnect(self):
        self.calls.append(("disconnect",))
        self._status = AudioStatus(True, False, None, "idle", "disconnected", "")

def test_command_handler_dispatches_orientation_led_and_device_status():
    async def scenario():
        coordinator, led, audio = FakeCoordinator(), FakeLed(), FakeAudio()
        handler = DeviceCommandHandler(coordinator, led, audio)

        stream_id = "20000000-0000-0000-0000-000000000002"
        playing = await handler.dispatch(type("Request", (), {
            "type": "audio.play.start", "payload": {
                "stream_id": stream_id, "sample_rate": 16000,
                "channels": 1, "encoding": "pcm_s16le"}})())
        drained = await handler.dispatch(type("Request", (), {
            "type": "audio.play.stop", "payload": {"stream_id": stream_id}})())

        solid = await handler.dispatch(type("Request", (), {
            "type": "led.solid", "payload": {"rgb": [1, 2, 3], "brightness": .8}})())
        frame = await handler.dispatch(type("Request", (), {
            "type": "led.frame", "payload": {"rgb": list(range(192)), "brightness": .05}})())
        orientation = await handler.dispatch(type("Request", (), {
            "type": "orientation.return_center", "payload": {}})())
        status = await handler.dispatch(type("Request", (), {
            "type": "device.status", "payload": {}})())

        assert solid["applied_brightness"] == pytest.approx(.1)
        assert frame["applied_brightness"] == pytest.approx(.05)
        assert orientation["state"] == "centered"
        assert status["orientation"]["state"] == "aligned"
        assert status["led"]["active"] is True
        assert status["audio"]["capture_running"] is True
        assert playing["state"] == "playing"
        assert drained["code"] == "drained"
        assert led.calls[:2] == [
            ("solid", (1, 2, 3), .8),
            ("frame", bytes(range(192)), .05),
        ]
        await handler.disconnected()
        assert led.calls[-1] == ("clear",)
        assert audio.calls[-1] == ("disconnect",)
    asyncio.run(scenario())


class RecordingService:
    def __init__(self):
        self.requests = []
        self.disconnects = 0

    async def dispatch(self, request):
        self.requests.append(request)
        return {"kind": request.type}

    async def disconnected(self):
        self.disconnects += 1


async def start_server(service=None, **kwargs):
    server = DeviceTcpServer(
        service or RecordingService(), token=TOKEN, host="127.0.0.1", port=0,
        heartbeat_timeout=kwargs.pop("heartbeat_timeout", 1.0), **kwargs)
    await server.start()
    port = server.sockets[0].getsockname()[1]
    return server, port


def test_server_roundtrip_returns_accepted_then_completed():
    async def scenario():
        service = RecordingService()
        server, port = await start_server(service)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            ident = str(uuid4())
            writer.write(envelope("led.status", ident=ident))
            await writer.drain()
            accepted, completed = await receive(reader), await receive(reader)
            assert accepted == {"id": ident, "state": "accepted", "code": "accepted", "data": {}}
            assert completed["id"] == ident
            assert completed["state"] == "completed"
            assert completed["data"] == {"kind": "led.status"}
            writer.close()
            await writer.wait_closed()
        finally:
            await server.close()
        assert [request.type for request in service.requests] == ["led.status"]
    asyncio.run(scenario())


def test_server_rejects_host_before_authentication():
    async def scenario():
        server, port = await start_server(allowed_hosts={"192.168.100.1"})
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            result = await receive(reader)
            assert result["state"] == "failed"
            assert result["code"] == "host_not_allowed"
            assert await reader.read() == b""
            writer.close()
            await writer.wait_closed()
        finally:
            await server.close()
    asyncio.run(scenario())


def test_second_authenticated_owner_cannot_disrupt_first():
    async def scenario():
        server, port = await start_server(heartbeat_timeout=2)
        try:
            first_reader, first_writer = await asyncio.open_connection("127.0.0.1", port)
            first_writer.write(envelope("system.heartbeat"))
            await first_writer.drain()
            await receive(first_reader)
            await receive(first_reader)

            second_reader, second_writer = await asyncio.open_connection("127.0.0.1", port)
            second_writer.write(envelope("device.status"))
            await second_writer.drain()
            rejected = await receive(second_reader)
            assert rejected["code"] == "controller_connected"
            second_writer.close()
            await second_writer.wait_closed()

            ident = str(uuid4())
            first_writer.write(envelope("system.heartbeat", ident=ident))
            await first_writer.drain()
            assert (await receive(first_reader))["id"] == ident
            assert (await receive(first_reader))["state"] == "completed"
            first_writer.close()
            await first_writer.wait_closed()
        finally:
            await server.close()
    asyncio.run(scenario())


def test_heartbeat_timeout_disconnects_and_runs_safe_cleanup_once():
    async def scenario():
        service = RecordingService()
        server, port = await start_server(service, heartbeat_timeout=.05)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(envelope("system.heartbeat"))
            await writer.drain()
            await receive(reader)
            await receive(reader)
            assert await asyncio.wait_for(reader.read(), 1) == b""
            await asyncio.sleep(0)
            assert service.disconnects == 1
            writer.close()
            await writer.wait_closed()
        finally:
            await server.close()
        assert service.disconnects == 1
    asyncio.run(scenario())


def test_pushed_events_are_ordered_and_new_session_resets_sequence():
    async def connect(port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(envelope("system.heartbeat"))
        await writer.drain()
        await receive(reader)
        await receive(reader)
        return reader, writer

    async def scenario():
        server, port = await start_server(heartbeat_timeout=2)
        try:
            reader, writer = await connect(port)
            assert await server.publish_event("orientation.status", {"state": "aligned"})
            first = await receive(reader)
            assert await server.publish_event("led.status", {"active": True})
            second = await receive(reader)
            assert first["sequence"] == 1 and second["sequence"] == 2
            assert first["session_id"] == second["session_id"]
            old_session = first["session_id"]
            writer.close()
            await writer.wait_closed()
            for _ in range(100):
                if server.owner is None:
                    break
                await asyncio.sleep(.01)

            reader, writer = await connect(port)
            assert await server.publish_event("led.status", {"active": False})
            fresh = await receive(reader)
            assert fresh["sequence"] == 1
            assert fresh["session_id"] != old_session
            writer.close()
            await writer.wait_closed()
        finally:
            await server.close()
    asyncio.run(scenario())


class BlockingService(RecordingService):
    def __init__(self):
        super().__init__()
        self.release = asyncio.Event()

    async def dispatch(self, request):
        self.requests.append(request)
        await self.release.wait()
        return {}


def test_server_bounds_pending_requests_at_64():
    async def scenario():
        service = BlockingService()
        server, port = await start_server(service, heartbeat_timeout=5)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            ids = [str(uuid4()) for _ in range(65)]
            for ident in ids:
                writer.write(envelope("device.status", ident=ident))
            await writer.drain()
            events = [await receive(reader) for _ in range(65)]
            assert sum(event["state"] == "accepted" for event in events) == 64
            rejected = [event for event in events if event["state"] == "failed"]
            assert rejected == [{
                "id": ids[-1], "state": "failed", "code": "too_many_pending", "data": {}}]
            service.release.set()
            writer.close()
            await writer.wait_closed()
        finally:
            await server.close()
    asyncio.run(scenario())


def test_expired_request_never_reaches_service():
    async def scenario():
        service = RecordingService()
        readings = iter((0.0, 0.0, 0.0, 1.0))

        def clock():
            return next(readings, 1.0)

        server, port = await start_server(service, clock=clock)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(envelope("device.status", ttl_ms=1))
            await writer.drain()
            first = await receive(reader)
            terminal = await receive(reader)
            assert first["state"] == "accepted"
            assert terminal["state"] == "failed"
            assert terminal["code"] == "expired"
            assert service.requests == []
        finally:
            await server.close()
    asyncio.run(scenario())
