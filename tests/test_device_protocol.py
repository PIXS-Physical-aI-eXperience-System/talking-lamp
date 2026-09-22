import json
import math
from uuid import UUID

import pytest

from device.protocol import (
    DEVICE_COMMAND_TYPES,
    MAX_LINE_BYTES,
    DeviceProtocolError,
    decode_request,
    encode_message,
)


TOKEN = "device-test-token"
REQUEST_ID = "10000000-0000-0000-0000-000000000001"


def wire(kind="device.status", *, payload=None, **updates):
    message = {
        "version": 1,
        "id": REQUEST_ID,
        "type": kind,
        "ttl_ms": 1000,
        "token": TOKEN,
        "payload": {} if payload is None else payload,
    }
    message.update(updates)
    return encode_message(message)


def test_device_protocol_exposes_audio_orientation_and_led_commands():
    assert DEVICE_COMMAND_TYPES == {
        "audio.play.start",
        "audio.play.stop",
        "audio.status",
        "orientation.return_center",
        "orientation.status",
        "led.frame",
        "led.solid",
        "led.clear",
        "led.status",
        "device.status",
        "system.heartbeat",
    }


def test_decode_authenticated_request_uses_receive_time_ttl():
    request = decode_request(wire(), token=TOKEN, received_at=12.5)

    assert request.id == REQUEST_ID
    assert request.type == "device.status"
    assert request.payload == {}
    assert request.expires_at == pytest.approx(13.5)


@pytest.mark.parametrize(
    "line,code",
    [
        (b"{}", "missing_newline"),
        (b"[]\n", "invalid_message"),
        (b"{bad}\n", "invalid_json"),
        (b"\xff\n", "invalid_encoding"),
        (b"x" * (MAX_LINE_BYTES + 1) + b"\n", "line_too_long"),
    ],
)
def test_decode_rejects_invalid_framing_before_dispatch(line, code):
    with pytest.raises(DeviceProtocolError) as error:
        decode_request(line, token=TOKEN, received_at=0.0)
    assert error.value.code == code


@pytest.mark.parametrize(
    "updates,code",
    [
        ({"version": 2}, "unsupported_version"),
        ({"id": "not-a-uuid"}, "invalid_id"),
        ({"ttl_ms": 0}, "invalid_ttl"),
        ({"ttl_ms": True}, "invalid_ttl"),
        ({"token": "wrong"}, "unauthorized"),
        ({"extra": 1}, "invalid_message"),
        ({"payload": []}, "invalid_payload"),
    ],
)
def test_decode_rejects_invalid_envelope(updates, code):
    with pytest.raises(DeviceProtocolError) as error:
        decode_request(wire(**updates), token=TOKEN, received_at=0.0)
    assert error.value.code == code


@pytest.mark.parametrize(
    "kind",
    [
        "orientation.return_center",
        "orientation.status",
        "audio.status",
        "led.clear",
        "led.status",
        "device.status",
        "system.heartbeat",
    ],
)
def test_empty_payload_commands_reject_extra_fields(kind):
    with pytest.raises(DeviceProtocolError) as error:
        decode_request(wire(kind, payload={"extra": 1}), token=TOKEN, received_at=0.0)
    assert error.value.code == "invalid_payload"


def test_led_frame_accepts_exact_rgb8_and_brightness():
    rgb = list(range(192))
    request = decode_request(
        wire("led.frame", payload={"rgb": rgb, "brightness": 0.1}),
        token=TOKEN,
        received_at=0.0,
    )
    assert request.payload["rgb"] == rgb
    assert request.payload["brightness"] == pytest.approx(0.1)


def test_led_solid_accepts_exact_rgb_and_brightness():
    request = decode_request(
        wire("led.solid", payload={"rgb": [1, 2, 3], "brightness": 1.0}),
        token=TOKEN,
        received_at=0.0,
    )
    assert request.payload == {"rgb": [1, 2, 3], "brightness": 1.0}


def test_audio_commands_accept_exact_stream_contract():
    stream_id = "20000000-0000-0000-0000-000000000002"
    request = decode_request(wire("audio.play.start", payload={
        "stream_id": stream_id,
        "sample_rate": 16000,
        "channels": 1,
        "encoding": "pcm_s16le",
    }), token=TOKEN, received_at=1.0)
    assert request.payload["stream_id"] == stream_id

    stopped = decode_request(wire(
        "audio.play.stop", payload={"stream_id": stream_id}),
        token=TOKEN, received_at=1.0)
    assert stopped.payload == {"stream_id": stream_id}


@pytest.mark.parametrize("kind,payload", [
    ("audio.play.start", {
        "stream_id": "bad", "sample_rate": 16000,
        "channels": 1, "encoding": "pcm_s16le"}),
    ("audio.play.start", {
        "stream_id": "20000000-0000-0000-0000-000000000002", "sample_rate": 48000,
        "channels": 1, "encoding": "pcm_s16le"}),
    ("audio.play.start", {
        "stream_id": "20000000-0000-0000-0000-000000000002", "sample_rate": 16000,
        "channels": 2, "encoding": "pcm_s16le"}),
    ("audio.play.start", {
        "stream_id": "20000000-0000-0000-0000-000000000002", "sample_rate": 16000,
        "channels": 1, "encoding": "opus"}),
    ("audio.play.stop", {"stream_id": "BAD"}),
])
def test_audio_payload_rejects_invalid_stream_before_process_creation(kind, payload):
    with pytest.raises(DeviceProtocolError) as error:
        decode_request(wire(kind, payload=payload), token=TOKEN, received_at=0.0)
    assert error.value.code == "invalid_payload"


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("led.frame", {"rgb": [0] * 191, "brightness": 0.1}),
        ("led.frame", {"rgb": [0] * 191 + [256], "brightness": 0.1}),
        ("led.frame", {"rgb": [0] * 191 + [True], "brightness": 0.1}),
        ("led.frame", {"rgb": [0] * 192, "brightness": math.nan}),
        ("led.frame", {"rgb": [0] * 192, "brightness": 1.1}),
        ("led.solid", {"rgb": [0, 0], "brightness": 0.1}),
        ("led.solid", {"rgb": [0, 0, 0], "brightness": -0.1}),
        ("led.solid", {"rgb": [0, 0, 0], "brightness": 0.1, "extra": 1}),
    ],
)
def test_led_payload_rejects_invalid_data_before_hardware(kind, payload):
    with pytest.raises(DeviceProtocolError) as error:
        decode_request(wire(kind, payload=payload), token=TOKEN, received_at=0.0)
    assert error.value.code in {"invalid_payload", "non_finite", "invalid_message"}


def test_encoder_is_compact_finite_ndjson_and_bounded():
    encoded = encode_message({"value": 1})
    assert encoded == b'{"value":1}\n'

    with pytest.raises(DeviceProtocolError) as non_finite:
        encode_message({"value": math.nan})
    assert non_finite.value.code == "invalid_message"

    with pytest.raises(DeviceProtocolError) as too_long:
        encode_message({"value": "x" * MAX_LINE_BYTES})
    assert too_long.value.code == "line_too_long"


def test_uuid_must_be_canonical_lowercase_text():
    uppercase = "abcdefab-cdef-abcd-efab-cdefabcdefab".upper()
    with pytest.raises(DeviceProtocolError) as error:
        decode_request(wire(id=uppercase), token=TOKEN, received_at=0.0)
    assert error.value.code == "invalid_id"
