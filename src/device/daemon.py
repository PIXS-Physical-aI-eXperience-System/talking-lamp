"""Supervised Raspberry Pi XVF/DOA/LED device service."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from dataclasses import asdict, dataclass
import json
import logging
import math
import os
from pathlib import Path
import signal
import time
from typing import Any, Awaitable, Callable

from .coordinator import DirectionCoordinator
from .doa import DoaCalibration, DoaSample, DoaStabilizer
from .led import LedController, LedMapping, PIXEL_COUNT, Ws281xSink
from .motion_client import MotionUnixClient
from .server import DeviceCommandError, DeviceCommandHandler, DeviceTcpServer
from .xvf3800 import XVF_PRODUCT_ID, XVF_VENDOR_ID, Xvf3800


LOG = logging.getLogger("talking_lamp.device")


class DeviceDaemonError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceConfig:
    token: str
    calibration: DoaCalibration
    bind: str = "192.168.100.2"
    port: int = 8766
    allowed_hosts: frozenset[str] = frozenset({"192.168.100.1"})
    motion_socket: str = "/run/talking-lamp/motion-control.sock"
    sample_rate_hz: float = 20.0
    gpio_pin: int = 12
    enable_led_hardware: bool = False
    max_brightness: float = 0.10
    xvf_vid: int = XVF_VENDOR_ID
    xvf_pid: int = XVF_PRODUCT_ID

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not self.token:
            raise DeviceDaemonError("TALKING_LAMP_TOKEN must not be empty")
        if not isinstance(self.bind, str) or not self.bind:
            raise DeviceDaemonError("bind address must not be empty")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise DeviceDaemonError("port must be in 1..65535")
        if not self.allowed_hosts or any(not isinstance(host, str) or not host for host in self.allowed_hosts):
            raise DeviceDaemonError("at least one non-empty allowed host is required")
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, (int, float))
            or not math.isfinite(self.sample_rate_hz)
            or self.sample_rate_hz <= 0
        ):
            raise DeviceDaemonError("sample rate must be finite and positive")
        if self.xvf_vid != XVF_VENDOR_ID or self.xvf_pid != XVF_PRODUCT_ID:
            raise DeviceDaemonError("only the commissioned XVF USB identity 2886:0022 is allowed")


def load_calibration(path: str | Path) -> DoaCalibration:
    source = Path(path)
    try:
        payload = json.loads(source.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeviceDaemonError(f"cannot read calibration {source}: {exc}") from exc
    expected = {"doa_zero_deg", "doa_direction_sign", "front_half_angle_deg"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise DeviceDaemonError(
            "calibration fields must be exactly doa_zero_deg, doa_direction_sign, "
            "front_half_angle_deg")
    zero = payload["doa_zero_deg"]
    sign = payload["doa_direction_sign"]
    front = payload["front_half_angle_deg"]
    if isinstance(zero, bool) or not isinstance(zero, (int, float)) or not math.isfinite(zero):
        raise DeviceDaemonError("doa_zero_deg must be finite")
    if isinstance(sign, bool) or sign not in {-1, 1}:
        raise DeviceDaemonError("doa_direction_sign must be -1 or 1")
    if (
        isinstance(front, bool)
        or not isinstance(front, (int, float))
        or not math.isfinite(front)
        or not 0 < front <= 180
    ):
        raise DeviceDaemonError("front_half_angle_deg must be finite and in (0, 180]")
    return DoaCalibration(float(zero), sign, float(front))


class NullPixelSink:
    """In-memory commissioning sink used until GPIO output is approved."""

    def __init__(self) -> None:
        self.last_pixels = ((0, 0, 0),) * PIXEL_COUNT
        self.last_brightness = 0.0
        self.write_count = 0
        self.clear_count = 0
        self.closed = False

    def write(self, pixels, brightness: float) -> None:
        if self.closed:
            raise RuntimeError("null LED sink is closed")
        self.last_pixels = tuple(pixels)
        self.last_brightness = float(brightness)
        self.write_count += 1

    def clear(self) -> None:
        if self.closed:
            return
        self.last_pixels = ((0, 0, 0),) * PIXEL_COUNT
        self.last_brightness = 0.0
        self.clear_count += 1

    def close(self) -> None:
        self.closed = True


class RuntimeDeviceService:
    """Late-bound handler so TCP can bind before USB or GPIO is opened."""

    def __init__(self) -> None:
        self._handler: DeviceCommandHandler | None = None

    def attach(self, coordinator: Any, led: Any) -> None:
        if self._handler is not None:
            raise DeviceDaemonError("runtime service is already attached")
        self._handler = DeviceCommandHandler(coordinator, led)

    async def dispatch(self, request):
        if self._handler is None:
            raise DeviceCommandError("starting", "device adapters are not ready")
        return await self._handler.dispatch(request)

    async def disconnected(self) -> None:
        if self._handler is not None:
            await self._handler.disconnected()


class DeviceDaemon:
    def __init__(
        self,
        config: DeviceConfig,
        *,
        server: Any,
        service: RuntimeDeviceService,
        coordinator: DirectionCoordinator,
        stabilizer: DoaStabilizer,
        led_factory: Callable[[], Any],
        xvf_factory: Callable[[], Any],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.config = config
        self.server = server
        self.service = service
        self.coordinator = coordinator
        self.stabilizer = stabilizer
        self.led_factory = led_factory
        self.xvf_factory = xvf_factory
        self.clock = clock
        self.sleep = sleep

    async def run(self, stop: asyncio.Event) -> int:
        led = None
        xvf = None
        failed = False
        try:
            # This ordering is deliberate: bad bind/allowlist configuration
            # must fail before libusb or a GPIO peripheral is touched.
            await self.server.start()
            led = self.led_factory()
            xvf = self.xvf_factory()
            version = xvf.read_version()
            if version != (1, 0, 3):
                raise DeviceDaemonError(
                    f"uncommissioned XVF3800 firmware {version}; expected (1, 0, 3)")
            self.service.attach(self.coordinator, led)
            interval = 1.0 / float(self.config.sample_rate_hz)
            while not stop.is_set():
                reading = xvf.read_doa()
                decision = self.stabilizer.observe(DoaSample(
                    timestamp=float(self.clock()),
                    doa_deg=reading.doa_deg,
                    speech_detected=reading.speech_detected,
                ))
                if decision is not None:
                    event = await self.coordinator.handle_decision(decision)
                    await self.server.publish_event("orientation.status", asdict(event))
                await self.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.error("device daemon failed: %s", exc)
            failed = True
        finally:
            with suppress(Exception):
                await self.server.close()
            if led is not None:
                with suppress(Exception):
                    led.clear()
                with suppress(Exception):
                    led.close()
            if xvf is not None:
                with suppress(Exception):
                    xvf.close()
        return 1 if failed else 0


def build_daemon(config: DeviceConfig) -> DeviceDaemon:
    service = RuntimeDeviceService()
    motion = MotionUnixClient(config.motion_socket)
    coordinator = DirectionCoordinator(motion, config.calibration)
    server = DeviceTcpServer(
        service,
        token=config.token,
        host=config.bind,
        port=config.port,
        allowed_hosts=set(config.allowed_hosts),
    )

    def led_factory():
        sink = (
            Ws281xSink(enable_hardware=True, gpio_pin=config.gpio_pin)
            if config.enable_led_hardware else NullPixelSink()
        )
        return LedController(
            sink, LedMapping(), max_brightness=config.max_brightness)

    return DeviceDaemon(
        config,
        server=server,
        service=service,
        coordinator=coordinator,
        stabilizer=DoaStabilizer(),
        led_factory=led_factory,
        xvf_factory=Xvf3800.discover,
    )


def _integer(text: str) -> int:
    return int(text, 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--allow-host", action="append", default=None)
    parser.add_argument("--motion-socket", default="/run/talking-lamp/motion-control.sock")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--sample-rate", type=float, default=20.0)
    parser.add_argument("--gpio-pin", type=int, default=12)
    parser.add_argument("--xvf-vid", type=_integer, default=XVF_VENDOR_ID)
    parser.add_argument("--xvf-pid", type=_integer, default=XVF_PRODUCT_ID)
    parser.add_argument("--max-brightness", type=float, default=0.10)
    parser.add_argument("--enable-led-hardware", action="store_true")
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = DeviceConfig(
        token=os.environ.get("TALKING_LAMP_TOKEN", ""),
        calibration=load_calibration(args.calibration),
        bind=args.bind,
        port=args.port,
        allowed_hosts=frozenset(args.allow_host or ["192.168.100.1"]),
        motion_socket=args.motion_socket,
        sample_rate_hz=args.sample_rate,
        gpio_pin=args.gpio_pin,
        enable_led_hardware=args.enable_led_hardware,
        max_brightness=args.max_brightness,
        xvf_vid=args.xvf_vid,
        xvf_pid=args.xvf_pid,
    )
    daemon = build_daemon(config)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(signum, stop.set)
    return await daemon.run(stop)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        return asyncio.run(async_main(argv))
    except (DeviceDaemonError, ValueError) as exc:
        LOG.error("invalid device configuration: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
