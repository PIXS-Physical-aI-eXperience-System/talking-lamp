import itertools
import math
import sys
from types import SimpleNamespace

import pytest

from device.led import (
    LedController,
    LedError,
    LedMapping,
    Ws281xSink,
    map_frame,
)


def marker_frame(row, column, color=(1, 2, 3)):
    frame = [(0, 0, 0)] * 64
    frame[row * 8 + column] = color
    return frame


@pytest.mark.parametrize(
    "rotation,origin,expected_index",
    [
        (0, "top_left", 1),
        (0, "top_right", 6),
        (0, "bottom_left", 57),
        (0, "bottom_right", 62),
        (90, "top_left", 15),
        (180, "top_left", 62),
        (270, "top_left", 48),
    ],
)
def test_row_major_rotation_and_origin_mapping(rotation, origin, expected_index):
    mapped = map_frame(
        marker_frame(0, 1),
        LedMapping(layout="row_major", origin=origin, rotation_deg=rotation,
                   color_order="RGB"),
    )

    assert mapped[expected_index] == (1, 2, 3)
    assert sum(pixel != (0, 0, 0) for pixel in mapped) == 1


def test_serpentine_reverses_odd_physical_rows():
    mapped = map_frame(
        marker_frame(1, 0),
        LedMapping(layout="serpentine", origin="top_left", rotation_deg=0,
                   color_order="RGB"),
    )

    assert mapped[15] == (1, 2, 3)


def test_every_mapping_combination_is_a_permutation():
    frame = [(index, 0, 0) for index in range(64)]
    for layout, origin, rotation in itertools.product(
        ("row_major", "serpentine"),
        ("top_left", "top_right", "bottom_left", "bottom_right"),
        (0, 90, 180, 270),
    ):
        mapped = map_frame(frame, LedMapping(
            layout=layout, origin=origin, rotation_deg=rotation, color_order="RGB"))
        assert sorted(pixel[0] for pixel in mapped) == list(range(64))


@pytest.mark.parametrize(
    "order,expected",
    [("RGB", (1, 2, 3)), ("GRB", (2, 1, 3)), ("BRG", (3, 1, 2))],
)
def test_color_order_is_configurable(order, expected):
    assert map_frame(
        marker_frame(0, 0),
        LedMapping(color_order=order),
    )[0] == expected


@pytest.mark.parametrize(
    "frame",
    [
        b"\x00" * 191,
        [(0, 0, 0)] * 63,
        [(0, 0, 0)] * 63 + [(0, 0, 256)],
        [(0, 0, 0)] * 63 + [(0, False, 0)],
        [(0, 0, 0)] * 63 + [(0, 1.5, 0)],
    ],
)
def test_frame_validation_happens_before_mapping(frame):
    with pytest.raises(LedError) as error:
        map_frame(frame, LedMapping())
    assert error.value.code == "invalid_frame"


def test_bytes_frame_decodes_as_rgb_and_does_not_mutate_input():
    raw = bytearray(192)
    raw[:3] = bytes([10, 20, 30])
    before = bytes(raw)

    mapped = map_frame(raw, LedMapping(color_order="RGB"))

    assert mapped[0] == (10, 20, 30)
    assert bytes(raw) == before


@pytest.mark.parametrize(
    "values",
    [
        {"layout": "columns"},
        {"origin": "middle"},
        {"rotation_deg": 45},
        {"color_order": "RRG"},
    ],
)
def test_mapping_rejects_unknown_installation_values(values):
    with pytest.raises(LedError) as error:
        LedMapping(**values)
    assert error.value.code == "invalid_config"


class FakeSink:
    def __init__(self, fail_write=False):
        self.events = []
        self.fail_write = fail_write

    def write(self, pixels, brightness):
        self.events.append(("write", pixels, brightness))
        if self.fail_write:
            raise OSError("pixel bus failed")

    def clear(self):
        self.events.append(("clear",))

    def close(self):
        self.events.append(("close",))


def test_controller_clears_at_start_clamps_to_default_ten_percent_and_closes():
    sink = FakeSink()
    controller = LedController(sink, LedMapping())

    status = controller.solid((100, 50, 25), brightness=0.75)

    assert sink.events[0] == ("clear",)
    assert sink.events[1][0] == "write"
    assert sink.events[1][2] == pytest.approx(0.10)
    assert status.requested_brightness == pytest.approx(0.75)
    assert status.applied_brightness == pytest.approx(0.10)
    assert status.clamped is True
    assert status.active is True

    controller.close()
    controller.close()
    assert sink.events[-2:] == [("clear",), ("close",)]


def test_controller_clears_after_sink_failure():
    sink = FakeSink(fail_write=True)
    controller = LedController(sink)

    with pytest.raises(LedError) as error:
        controller.solid((1, 2, 3), brightness=0.05)

    assert error.value.code == "sink_failed"
    assert sink.events[-1] == ("clear",)
    assert controller.status.active is False
    assert controller.status.fault == "pixel bus failed"


@pytest.mark.parametrize("brightness", [-0.1, 1.1, math.nan, True])
def test_controller_rejects_invalid_brightness_before_sink_write(brightness):
    sink = FakeSink()
    controller = LedController(sink)

    with pytest.raises(LedError) as error:
        controller.solid((1, 2, 3), brightness=brightness)

    assert error.value.code == "invalid_brightness"
    assert sink.events == [("clear",)]


def test_hardware_sink_requires_explicit_enable_before_import_or_gpio_access():
    with pytest.raises(LedError) as error:
        Ws281xSink(enable_hardware=False)
    assert error.value.code == "hardware_disabled"


def test_hardware_sink_rejects_non_pi5_hardware():
    with pytest.raises(LedError) as error:
        Ws281xSink(
            enable_hardware=True, hardware_model="Raspberry Pi 4 Model B Rev 1.5")
    assert error.value.code == "unsupported_hardware"


def test_hardware_sink_uses_pi5_pio_and_scales_pixels_before_transmit(monkeypatch):
    events = []

    def write(pin, data):
        events.append(("write", pin.id, bytes(data)))

    monkeypatch.setitem(sys.modules, "adafruit_raspberry_pi5_neopixel_write", SimpleNamespace(
        neopixel_write=write,
        free_pio=lambda: events.append(("free",)),
    ))
    sink = Ws281xSink(
        enable_hardware=True, gpio_pin=12, hardware_model="Raspberry Pi 5 Model B Rev 1.1")
    pixels = ((10, 20, 30),) + ((0, 0, 0),) * 63

    sink.write(pixels, 0.10)
    sink.close()

    assert events[0] == ("write", 12, bytes((1, 2, 3)) + bytes(189))
    assert events[1] == ("write", 12, bytes(192))
    assert events[2] == ("free",)
