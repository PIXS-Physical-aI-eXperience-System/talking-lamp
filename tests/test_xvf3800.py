from dataclasses import dataclass

import pytest

from device.xvf3800 import Xvf3800, XvfError


@dataclass
class FakeDevice:
    responses: list[bytes]
    idVendor: int = 0x2886
    idProduct: int = 0x0022
    bus: int = 3
    address: int = 2

    def __post_init__(self):
        self.calls = []

    def ctrl_transfer(self, *args):
        self.calls.append(args)
        return self.responses.pop(0)


def test_version_read_uses_exact_vendor_control_transfer():
    device = FakeDevice([bytes([0, 1, 0, 3])])
    xvf = Xvf3800(device)

    assert xvf.read_version() == (1, 0, 3)
    assert device.calls == [(0xC0, 0, 0x80, 48, 4, 100_000)]


def test_doa_read_decodes_little_endian_angle_and_speech_flag():
    device = FakeDevice([bytes([0, 126, 0, 1, 0])])
    xvf = Xvf3800(device)

    result = xvf.read_doa()

    assert result.doa_deg == 126
    assert result.speech_detected is True
    assert device.calls == [(0xC0, 0, 0x92, 20, 5, 100_000)]


def test_retry_status_is_bounded_and_then_succeeds_without_writes():
    sleeps = []
    device = FakeDevice([bytes([64, 0, 0, 0, 0]), bytes([0, 5, 0, 0, 0])])
    xvf = Xvf3800(device, sleep=sleeps.append)

    assert xvf.read_doa().doa_deg == 5
    assert sleeps == [0.01]
    assert len(device.calls) == 2
    assert not hasattr(xvf, "write")


def test_retry_status_fails_after_exact_attempt_limit():
    device = FakeDevice([bytes([64, 0, 0, 0, 0])] * 3)
    xvf = Xvf3800(device, max_attempts=3, sleep=lambda _: None)

    with pytest.raises(XvfError) as error:
        xvf.read_doa()

    assert error.value.code == "retry_exhausted"
    assert len(device.calls) == 3


@pytest.mark.parametrize(
    "response, code",
    [
        (bytes([7, 0, 0, 0, 0]), "device_status"),
        (bytes([0, 0, 0]), "invalid_response"),
        (bytes([0, 104, 1, 0, 0]), "invalid_response"),  # 360 degrees
        (bytes([0, 0, 0, 2, 0]), "invalid_response"),
    ],
)
def test_doa_read_rejects_device_and_payload_errors(response, code):
    with pytest.raises(XvfError) as error:
        Xvf3800(FakeDevice([response])).read_doa()

    assert error.value.code == code


def test_discovery_requires_exact_identity_and_unique_selection():
    good = FakeDevice([bytes([0, 1, 0, 3])], bus=3, address=2)
    other = FakeDevice([bytes([0, 1, 0, 3])], bus=4, address=7)
    calls = []

    def finder(**kwargs):
        calls.append(kwargs)
        return [good, other]

    with pytest.raises(XvfError) as error:
        Xvf3800.discover(finder=finder)
    assert error.value.code == "multiple_devices"

    selected = Xvf3800.discover(bus=3, address=2, finder=finder)
    assert selected.device is good
    assert calls == [
        {"find_all": True, "idVendor": 0x2886, "idProduct": 0x0022},
        {"find_all": True, "idVendor": 0x2886, "idProduct": 0x0022},
    ]


def test_discovery_rejects_absent_or_wrong_identity():
    with pytest.raises(XvfError) as missing:
        Xvf3800.discover(finder=lambda **_: [])
    assert missing.value.code == "not_found"

    wrong = FakeDevice([], idProduct=0x9999)
    with pytest.raises(XvfError) as identity:
        Xvf3800(wrong)
    assert identity.value.code == "wrong_device"


def test_close_is_idempotent_and_disposes_only_owned_device():
    device = FakeDevice([])
    disposed = []
    borrowed = Xvf3800(device, disposer=disposed.append)
    borrowed.close()
    assert disposed == []

    owned = Xvf3800(device, owns_device=True, disposer=disposed.append)
    owned.close()
    owned.close()
    assert disposed == [device]


def test_constructor_rejects_invalid_attempt_limit():
    with pytest.raises(XvfError) as error:
        Xvf3800(FakeDevice([]), max_attempts=0)
    assert error.value.code == "invalid_config"
