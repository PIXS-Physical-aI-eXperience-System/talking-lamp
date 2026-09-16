"""Pure 8x8 LED mapping and explicitly gated WS2812B hardware output."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol, Sequence


PIXEL_COUNT = 64
FRAME_BYTES = PIXEL_COUNT * 3


class LedError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class LedMapping:
    layout: str = "row_major"
    origin: str = "top_left"
    rotation_deg: int = 0
    color_order: str = "GRB"

    def __post_init__(self) -> None:
        if self.layout not in {"row_major", "serpentine"}:
            raise LedError("invalid_config", "layout must be row_major or serpentine")
        if self.origin not in {"top_left", "top_right", "bottom_left", "bottom_right"}:
            raise LedError("invalid_config", "origin is invalid")
        if isinstance(self.rotation_deg, bool) or self.rotation_deg not in {0, 90, 180, 270}:
            raise LedError("invalid_config", "rotation_deg must be 0, 90, 180, or 270")
        if (
            not isinstance(self.color_order, str)
            or len(self.color_order) != 3
            or set(self.color_order) != {"R", "G", "B"}
        ):
            raise LedError("invalid_config", "color_order must be a permutation of RGB")


def _channel(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise LedError("invalid_frame", "RGB channels must be integers from 0 to 255")
    return value


def _normalize_frame(frame: object) -> tuple[tuple[int, int, int], ...]:
    if isinstance(frame, (bytes, bytearray, memoryview)):
        raw = bytes(frame)
        if len(raw) != FRAME_BYTES:
            raise LedError("invalid_frame", "rgb8 frame must contain exactly 192 bytes")
        return tuple(tuple(raw[offset:offset + 3]) for offset in range(0, FRAME_BYTES, 3))
    if isinstance(frame, (str, Sequence)):
        try:
            pixels = tuple(frame)
        except TypeError as exc:
            raise LedError("invalid_frame", "frame must contain exactly 64 RGB pixels") from exc
        if len(pixels) != PIXEL_COUNT:
            raise LedError("invalid_frame", "frame must contain exactly 64 RGB pixels")
        normalized = []
        for pixel in pixels:
            if isinstance(pixel, (str, bytes)):
                raise LedError("invalid_frame", "each pixel must contain three RGB channels")
            try:
                channels = tuple(pixel)
            except TypeError as exc:
                raise LedError("invalid_frame", "each pixel must contain three RGB channels") from exc
            if len(channels) != 3:
                raise LedError("invalid_frame", "each pixel must contain three RGB channels")
            normalized.append(tuple(_channel(channel) for channel in channels))
        return tuple(normalized)
    raise LedError("invalid_frame", "frame must contain exactly 64 RGB pixels")


def _physical_coordinate(row: int, column: int, mapping: LedMapping) -> tuple[int, int]:
    if mapping.rotation_deg == 90:
        row, column = column, 7 - row
    elif mapping.rotation_deg == 180:
        row, column = 7 - row, 7 - column
    elif mapping.rotation_deg == 270:
        row, column = 7 - column, row
    if mapping.origin in {"bottom_left", "bottom_right"}:
        row = 7 - row
    if mapping.origin in {"top_right", "bottom_right"}:
        column = 7 - column
    return row, column


def map_frame(
    frame: object,
    mapping: LedMapping,
) -> tuple[tuple[int, int, int], ...]:
    """Map a logical top-left row-major RGB frame to physical ordered pixels."""
    if not isinstance(mapping, LedMapping):
        raise LedError("invalid_config", "mapping must be a LedMapping")
    logical = _normalize_frame(frame)
    physical: list[tuple[int, int, int] | None] = [None] * PIXEL_COUNT
    channel_indices = tuple("RGB".index(channel) for channel in mapping.color_order)
    for row in range(8):
        for column in range(8):
            physical_row, physical_column = _physical_coordinate(row, column, mapping)
            if mapping.layout == "serpentine" and physical_row % 2:
                physical_column = 7 - physical_column
            index = physical_row * 8 + physical_column
            rgb = logical[row * 8 + column]
            physical[index] = tuple(rgb[channel] for channel in channel_indices)
    assert all(pixel is not None for pixel in physical)
    return tuple(pixel for pixel in physical if pixel is not None)


class PixelSink(Protocol):
    def write(self, pixels: tuple[tuple[int, int, int], ...], brightness: float) -> None: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class LedStatus:
    active: bool
    requested_brightness: float
    applied_brightness: float
    clamped: bool
    fault: str | None


def _brightness(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise LedError("invalid_brightness", f"{name} must be finite and in [0, 1]")
    return float(value)


class LedController:
    """Apply mapping and a hard brightness ceiling before touching a sink."""

    def __init__(
        self,
        sink: PixelSink,
        mapping: LedMapping | None = None,
        *,
        max_brightness: float = 0.10,
    ) -> None:
        self.sink = sink
        self.mapping = mapping or LedMapping()
        self.max_brightness = _brightness(max_brightness, "max_brightness")
        self._closed = False
        self._status = LedStatus(False, 0.0, 0.0, False, None)
        try:
            self.sink.clear()
        except Exception as exc:
            self._status = LedStatus(False, 0.0, 0.0, False, str(exc))
            raise LedError("sink_failed", f"LED startup clear failed: {exc}") from exc

    @property
    def status(self) -> LedStatus:
        return self._status

    def frame(self, frame: object, *, brightness: float = 1.0) -> LedStatus:
        if self._closed:
            raise LedError("closed", "LED controller is closed")
        mapped = map_frame(frame, self.mapping)
        requested = _brightness(brightness, "brightness")
        applied = min(requested, self.max_brightness)
        clamped = applied != requested
        try:
            self.sink.write(mapped, applied)
        except Exception as exc:
            try:
                self.sink.clear()
            except Exception:
                pass
            self._status = LedStatus(False, requested, 0.0, clamped, str(exc))
            raise LedError("sink_failed", f"LED write failed: {exc}") from exc
        self._status = LedStatus(True, requested, applied, clamped, None)
        return self._status

    def solid(self, color: Sequence[int], *, brightness: float = 1.0) -> LedStatus:
        try:
            rgb = tuple(color)
        except TypeError as exc:
            raise LedError("invalid_frame", "solid color must contain three RGB channels") from exc
        return self.frame([rgb] * PIXEL_COUNT, brightness=brightness)

    def clear(self) -> LedStatus:
        if self._closed:
            return self._status
        try:
            self.sink.clear()
        except Exception as exc:
            self._status = LedStatus(False, 0.0, 0.0, False, str(exc))
            raise LedError("sink_failed", f"LED clear failed: {exc}") from exc
        self._status = LedStatus(False, 0.0, 0.0, False, None)
        return self._status

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failure: Exception | None = None
        try:
            self.sink.clear()
        except Exception as exc:
            failure = exc
        try:
            self.sink.close()
        except Exception as exc:
            failure = failure or exc
        self._status = LedStatus(False, 0.0, 0.0, False,
                                 str(failure) if failure else None)
        if failure:
            raise LedError("sink_failed", f"LED shutdown failed: {failure}") from failure


class Ws281xSink:
    """Lazy Raspberry Pi GPIO adapter, disabled unless explicitly commissioned."""

    def __init__(
        self,
        *,
        enable_hardware: bool,
        gpio_pin: int = 12,
        pixel_count: int = PIXEL_COUNT,
    ) -> None:
        if enable_hardware is not True:
            raise LedError("hardware_disabled", "LED hardware requires explicit enable_hardware=True")
        if isinstance(gpio_pin, bool) or not isinstance(gpio_pin, int) or gpio_pin < 0:
            raise LedError("invalid_config", "gpio_pin must be a non-negative integer")
        if pixel_count != PIXEL_COUNT:
            raise LedError("invalid_config", "this service requires exactly 64 pixels")
        try:
            import rpi_ws281x as ws
        except ImportError as exc:
            raise LedError("dependency_missing", "rpi-ws281x is required for LED hardware") from exc
        self._ws = ws
        self._strip = ws.PixelStrip(
            pixel_count, gpio_pin, 800_000, 10, False, 255, 0,
            ws.WS2811_STRIP_RGB,
        )
        self._strip.begin()
        self._closed = False

    def write(self, pixels: tuple[tuple[int, int, int], ...], brightness: float) -> None:
        if self._closed:
            raise LedError("closed", "LED sink is closed")
        self._strip.setBrightness(round(brightness * 255))
        for index, channels in enumerate(pixels):
            self._strip.setPixelColor(index, self._ws.Color(*channels))
        self._strip.show()

    def clear(self) -> None:
        if self._closed:
            return
        for index in range(PIXEL_COUNT):
            self._strip.setPixelColor(index, 0)
        self._strip.show()

    def close(self) -> None:
        if self._closed:
            return
        self.clear()
        self._closed = True
