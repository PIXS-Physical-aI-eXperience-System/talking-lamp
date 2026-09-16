import asyncio
import json

import pytest

from device.daemon import (
    DeviceConfig,
    DeviceDaemon,
    DeviceDaemonError,
    NullPixelSink,
    RuntimeDeviceService,
    load_calibration,
)
from device.audio import AudioStatus
from device.doa import DoaCalibration, DoaStabilizer
from device.xvf3800 import XvfError


class FakeServer:
    def __init__(self, service, order):
        self.service = service
        self.order = order
        self.events = []

    async def start(self):
        self.order.append("server.start")

    async def publish_event(self, name, data):
        self.events.append((name, data))
        return True

    async def close(self):
        self.order.append("server.close")


class FakeXvf:
    def __init__(self, order):
        self.order = order
        self.reads = 0

    def read_version(self):
        self.order.append("xvf.version")
        return (1, 0, 3)

    def read_doa(self):
        self.order.append("xvf.read")
        self.reads += 1
        return type("Doa", (), {"doa_deg": 10, "speech_detected": False})()

    def close(self):
        self.order.append("xvf.close")


class FakeLed:
    def __init__(self, order):
        self.order = order
        self.status = type("Status", (), {
            "active": False, "requested_brightness": 0.0,
            "applied_brightness": 0.0, "clamped": False, "fault": None,
        })()

    def clear(self):
        self.order.append("led.clear")
        return self.status

    def close(self):
        self.order.append("led.close")


class FakeAudio:
    def __init__(self, order):
        self.order = order
        self._status = AudioStatus(False, False, None, "idle", "idle", "")

    @property
    def status(self):
        return self._status

    async def start(self):
        self.order.append("audio.start")
        self._status = AudioStatus(True, False, None, "idle", "capture_running", "")
        return self._status

    async def disconnect(self):
        self.order.append("audio.disconnect")

    @staticmethod
    def capture_rtp_timestamp(now):
        return int(now * 48_000) & 0xFFFFFFFF

    async def close(self):
        self.order.append("audio.close")
        self._status = AudioStatus(False, False, None, "closed", "closed", "")


class FakeCoordinator:
    async def status(self):
        raise AssertionError("not used")

    async def return_center(self):
        raise AssertionError("not used")

    async def handle_decision(self, decision):
        raise AssertionError("no terminal decision expected")


class FailingCoordinator(FakeCoordinator):
    async def handle_decision(self, decision):
        from device.motion_client import MotionClientError
        raise MotionClientError("connection_failed", "motion socket is absent")


def config():
    return DeviceConfig(
        token="secret", calibration=DoaCalibration(0.0, 1),
        bind="192.168.100.2", port=8766,
        allowed_hosts=frozenset({"192.168.100.1"}),
        motion_socket="/run/talking-lamp/motion-control.sock",
        sample_rate_hz=20.0, gpio_pin=12, enable_led_hardware=False,
    )


def test_load_calibration_requires_exact_finite_schema(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "doa_zero_deg": 12.5,
        "doa_direction_sign": -1,
        "front_half_angle_deg": 80.0,
    }))
    assert load_calibration(path) == DoaCalibration(12.5, -1, 80.0)

    path.write_text('{"doa_zero_deg": 0, "doa_direction_sign": 1, "extra": true}')
    with pytest.raises(DeviceDaemonError, match="fields"):
        load_calibration(path)


def test_load_calibration_accepts_commissioned_voice_bench_v2_schema(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "conv": 2,
        "offset_deg": 90.49852890666492,
        "sign": 1,
        "n": 47,
        "std": 1.8557965149873001,
        "side_raw": 158.0,
        "side_delta": 67.50147109333508,
        "side_std": 0.0,
        "side": "r",
    }))
    assert load_calibration(path) == DoaCalibration(
        90.49852890666492, 1, 90.0)

    payload = json.loads(path.read_text())
    payload["conv"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(DeviceDaemonError, match="version"):
        load_calibration(path)


def test_null_pixel_sink_tracks_clear_and_never_opens_gpio():
    sink = NullPixelSink()
    sink.write(((1, 2, 3),) * 64, 0.1)
    assert sink.write_count == 1
    assert sink.last_brightness == pytest.approx(0.1)
    sink.clear()
    assert sink.last_pixels == ((0, 0, 0),) * 64
    sink.close()
    assert sink.closed


def test_daemon_binds_before_usb_or_led_and_shutdown_order_is_safe():
    async def scenario():
        order = []
        gate = RuntimeDeviceService()
        fake_server = FakeServer(gate, order)
        stop = asyncio.Event()

        async def sleep(_delay):
            order.append("poll.sleep")
            stop.set()

        def led_factory():
            order.append("led.open")
            return FakeLed(order)

        def xvf_factory():
            order.append("xvf.open")
            return FakeXvf(order)

        daemon = DeviceDaemon(
            config(), server=fake_server, service=gate,
            coordinator=FakeCoordinator(), stabilizer=DoaStabilizer(),
            led_factory=led_factory, xvf_factory=xvf_factory,
            audio_factory=lambda: FakeAudio(order),
            clock=lambda: 0.0, sleep=sleep,
        )
        assert await daemon.run(stop) == 0
        assert order.index("server.start") < order.index("led.open")
        assert order.index("server.start") < order.index("xvf.open")
        assert order.index("led.open") < order.index("audio.start") < order.index("xvf.open")
        assert order.index("xvf.read") < order.index("poll.sleep")
        assert order[-5:] == [
            "server.close", "audio.close", "led.clear", "led.close", "xvf.close"]

    asyncio.run(scenario())


def test_adapter_start_failure_closes_server_and_returns_nonzero():
    async def scenario():
        order = []
        gate = RuntimeDeviceService()
        server = FakeServer(gate, order)

        def fail_xvf():
            order.append("xvf.open")
            raise RuntimeError("USB absent")

        daemon = DeviceDaemon(
            config(), server=server, service=gate,
            coordinator=FakeCoordinator(), stabilizer=DoaStabilizer(),
            led_factory=lambda: FakeLed(order), xvf_factory=fail_xvf,
            audio_factory=lambda: FakeAudio(order),
        )
        assert await daemon.run(asyncio.Event()) == 1
        assert "server.close" in order
        assert order[-3:] == ["audio.close", "led.clear", "led.close"]

    asyncio.run(scenario())


def test_motion_socket_failure_is_reported_without_stopping_device_service():
    async def scenario():
        order = []
        gate = RuntimeDeviceService()
        server = FakeServer(gate, order)
        stop = asyncio.Event()
        ticks = iter(index * 0.1 for index in range(20))

        class SpeakingXvf(FakeXvf):
            def read_doa(self):
                self.order.append("xvf.read")
                self.reads += 1
                return type("Doa", (), {"doa_deg": 90, "speech_detected": True})()

        async def sleep(_delay):
            if order.count("xvf.read") >= 8:
                stop.set()

        daemon = DeviceDaemon(
            config(), server=server, service=gate,
            coordinator=FailingCoordinator(), stabilizer=DoaStabilizer(),
            led_factory=lambda: FakeLed(order),
            xvf_factory=lambda: SpeakingXvf(order),
            audio_factory=lambda: FakeAudio(order),
            clock=lambda: next(ticks), sleep=sleep,
        )
        assert await daemon.run(stop) == 0
        assert order.count("xvf.read") == 8
        assert server.events[-1][0] == "orientation.status"
        assert server.events[-1][1]["state"] == "fault"
        assert server.events[-1][1]["code"] == "connection_failed"

    asyncio.run(scenario())


def test_runtime_xvf_loss_emits_fault_and_rediscovers_without_stopping_tcp():
    async def scenario():
        order = []
        gate = RuntimeDeviceService()
        server = FakeServer(gate, order)
        stop = asyncio.Event()
        devices = []

        class DisconnectingXvf(FakeXvf):
            def read_doa(self):
                self.order.append("xvf.read.fail")
                raise XvfError("usb_io", "device disconnected")

        class ReconnectedXvf(FakeXvf):
            def read_doa(self):
                self.order.append("xvf.read.reconnected")
                stop.set()
                return type("Doa", (), {"doa_deg": 90, "speech_detected": False})()

        def xvf_factory():
            device = DisconnectingXvf(order) if not devices else ReconnectedXvf(order)
            devices.append(device)
            order.append("xvf.open")
            return device

        daemon = DeviceDaemon(
            config(), server=server, service=gate,
            coordinator=FakeCoordinator(), stabilizer=DoaStabilizer(),
            led_factory=lambda: FakeLed(order),
            audio_factory=lambda: FakeAudio(order), xvf_factory=xvf_factory,
            sleep=lambda _delay: asyncio.sleep(0),
        )
        assert await daemon.run(stop) == 0
        assert len(devices) == 2
        assert "xvf.close" in order
        faults = [data for name, data in server.events if name == "device.status"]
        assert faults[0]["xvf"]["connected"] is False
        assert faults[-1]["xvf"]["connected"] is True

    asyncio.run(scenario())


def test_capture_process_exit_emits_fault_and_restarts_without_stopping_device():
    async def scenario():
        order = []
        gate = RuntimeDeviceService()
        server = FakeServer(gate, order)
        stop = asyncio.Event()

        class RestartingAudio(FakeAudio):
            def __init__(self, order):
                super().__init__(order)
                self.running = False
                self.starts = 0

            @property
            def status(self):
                return AudioStatus(
                    self.running, False, None,
                    "idle" if self.running else "fault",
                    "capture_running" if self.running else "capture_exited", "")

            async def start(self):
                self.starts += 1
                self.running = True
                self.order.append("audio.start")
                return self.status

        audio = RestartingAudio(order)

        async def sleep(_delay):
            if order.count("xvf.read") == 1 and audio.starts == 1:
                audio.running = False
            elif audio.starts == 2:
                stop.set()

        daemon = DeviceDaemon(
            config(), server=server, service=gate,
            coordinator=FakeCoordinator(), stabilizer=DoaStabilizer(),
            led_factory=lambda: FakeLed(order), audio_factory=lambda: audio,
            xvf_factory=lambda: FakeXvf(order), sleep=sleep,
        )
        assert await daemon.run(stop) == 0
        assert audio.starts == 2
        faults = [data for name, data in server.events if name == "audio.status"]
        assert faults[0]["code"] == "capture_exited"
        assert faults[-1]["code"] == "capture_running"

    asyncio.run(scenario())


def test_vad_edges_publish_one_audio_activity_event_with_shared_speech_id():
    async def scenario():
        order = []
        gate = RuntimeDeviceService()
        server = FakeServer(gate, order)
        stop = asyncio.Event()
        states = iter((False, True, True, False))
        ticks = iter((0.0, 0.05, 0.10, 0.15))

        class VadXvf(FakeXvf):
            def read_doa(self):
                state = next(states)
                self.order.append("xvf.read")
                if len([item for item in self.order if item == "xvf.read"]) == 4:
                    stop.set()
                return type("Doa", (), {"doa_deg": 90, "speech_detected": state})()

        daemon = DeviceDaemon(
            config(), server=server, service=gate,
            coordinator=FakeCoordinator(), stabilizer=DoaStabilizer(),
            led_factory=lambda: FakeLed(order), audio_factory=lambda: FakeAudio(order),
            xvf_factory=lambda: VadXvf(order), clock=lambda: next(ticks),
            sleep=lambda _delay: asyncio.sleep(0),
        )
        assert await daemon.run(stop) == 0
        events = [data for name, data in server.events if name == "audio.activity"]
        assert [event["active"] for event in events] == [True, False]
        assert events[0]["speech_id"] == events[1]["speech_id"]
        assert events[0]["rtp_timestamp"] < events[1]["rtp_timestamp"]

    asyncio.run(scenario())


def test_active_playback_suppresses_self_speech_vad_and_orientation():
    async def scenario():
        order = []
        gate = RuntimeDeviceService()
        server = FakeServer(gate, order)
        stop = asyncio.Event()
        ticks = iter(index * 0.05 for index in range(20))

        class PlaybackAudio(FakeAudio):
            @property
            def status(self):
                return AudioStatus(
                    True, True, "20000000-0000-0000-0000-000000000002",
                    "playing", "playing", "")

            async def start(self):
                self.order.append("audio.start")
                return self.status

        class SelfSpeechXvf(FakeXvf):
            def read_doa(self):
                self.order.append("xvf.read")
                self.reads += 1
                if self.reads == 10:
                    stop.set()
                return type("Doa", (), {
                    "doa_deg": 90, "speech_detected": True})()

        daemon = DeviceDaemon(
            config(), server=server, service=gate,
            coordinator=FakeCoordinator(), stabilizer=DoaStabilizer(),
            led_factory=lambda: FakeLed(order),
            audio_factory=lambda: PlaybackAudio(order),
            xvf_factory=lambda: SelfSpeechXvf(order),
            clock=lambda: next(ticks), sleep=lambda _delay: asyncio.sleep(0),
        )

        assert await daemon.run(stop) == 0
        assert not [
            event for event in server.events
            if event[0] in {"audio.activity", "orientation.status"}
        ]

    asyncio.run(scenario())
