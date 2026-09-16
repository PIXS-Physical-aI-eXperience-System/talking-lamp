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
from device.doa import DoaCalibration, DoaStabilizer


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


class FakeCoordinator:
    async def status(self):
        raise AssertionError("not used")

    async def return_center(self):
        raise AssertionError("not used")

    async def handle_decision(self, decision):
        raise AssertionError("no terminal decision expected")


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
            clock=lambda: 0.0, sleep=sleep,
        )
        assert await daemon.run(stop) == 0
        assert order.index("server.start") < order.index("led.open")
        assert order.index("server.start") < order.index("xvf.open")
        assert order.index("xvf.read") < order.index("poll.sleep")
        assert order[-4:] == ["server.close", "led.clear", "led.close", "xvf.close"]

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
        )
        assert await daemon.run(asyncio.Event()) == 1
        assert "server.close" in order
        assert order[-2:] == ["led.clear", "led.close"]

    asyncio.run(scenario())
