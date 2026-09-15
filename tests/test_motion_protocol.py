import json
import math
from pathlib import Path
from uuid import UUID

import pytest

from motion.protocol import (
    LOCAL_COMMAND_TYPES,
    REMOTE_COMMAND_TYPES,
    ProtocolError,
    decode_local_request,
    decode_request,
    encode_message,
)


REQUEST_ID = "0199f3c0-0b5f-7b55-a020-7c8b4012d8ce"


def request_line(**overrides: object) -> bytes:
    message: dict[str, object] = {
        "version": 1,
        "id": REQUEST_ID,
        "type": "track.point",
        "ttl_ms": 1_000,
        "token": "secret",
        "payload": {"point": [0.2, 0.0, 0.3]},
    }
    message.update(overrides)
    return json.dumps(message, allow_nan=True).encode() + b"\n"


def test_decode_uses_pi_receive_time_for_ttl():
    request = decode_request(request_line(ttl_ms=750), token="secret", received_at=12.5)

    assert request.expires_at == pytest.approx(13.25)


@pytest.mark.parametrize(
    ("line", "code"),
    [
        (b"{}\n", "missing_field"),
        (request_line(version=2), "unsupported_version"),
        (request_line(type="motor.raw"), "unknown_type"),
        (request_line(payload={"point": [1.0, float("nan"), 2.0]}), "non_finite"),
    ],
)
def test_invalid_requests_are_rejected(line, code):
    with pytest.raises(ProtocolError, match=code):
        decode_request(line, token="secret", received_at=1.0)


@pytest.mark.parametrize(
    ("line", "code"),
    [
        (request_line()[:-1], "missing_newline"),
        (b"[1]\n", "invalid_message"),
        (request_line(id="not-a-uuid"), "invalid_id"),
        (request_line(token="wrong"), "unauthorized"),
        (request_line(ttl_ms=0), "invalid_ttl"),
        (request_line(ttl_ms=True), "invalid_ttl"),
        (request_line(payload={"point": [2.1, 0.0, 0.0]}), "out_of_range"),
        (request_line(payload={"point": [0.0, 0.0]}), "invalid_payload"),
        (request_line(payload={"point": [0.0, 0.0, 0.0], "raw_motor": 1}), "invalid_payload"),
        (request_line(type="track.bearing", payload={"direction": [0.0, 0.0, 0.0]}), "invalid_direction"),
        (request_line(type="motion.play", payload={"name": "Nod", "replace_current": True, "intensity": 1.0, "repeat": 1}), "invalid_name"),
        (request_line(type="motion.play", payload={"name": "nod", "replace_current": 1, "intensity": 1.0, "repeat": 1}), "invalid_payload"),
    ],
)
def test_decoder_rejects_malformed_or_unsafe_requests(line, code):
    with pytest.raises(ProtocolError, match=code):
        decode_request(line, token="secret", received_at=1.0)


def test_decoder_rejects_a_line_over_16_kib():
    line = b" " * (16 * 1024) + b"\n"

    with pytest.raises(ProtocolError, match="line_too_long"):
        decode_request(line, token="secret", received_at=1.0)


@pytest.mark.parametrize(
    ("supplied_token", "expected_token"),
    [
        ("sëcret", "secret"),
        ("secret", "sëcret"),
        ("sëcret", "sëcret"),
    ],
)
def test_decoder_fails_closed_for_non_ascii_tokens(supplied_token, expected_token):
    with pytest.raises(ProtocolError, match="unauthorized"):
        decode_request(
            request_line(token=supplied_token), token=expected_token, received_at=1.0
        )


@pytest.mark.parametrize(
    ("request_type", "payload", "code"),
    [
        (
            "motion.play",
            {"name": "nod", "replace_current": True, "intensity": 10**400, "repeat": 1},
            "invalid_payload",
        ),
        ("track.point", {"point": [10**400, 0.0, 0.0]}, "out_of_range"),
        ("track.bearing", {"direction": [10**400, 0.0, 0.0]}, "out_of_range"),
    ],
)
def test_decoder_rejects_huge_integer_payload_numbers(request_type, payload, code):
    with pytest.raises(ProtocolError, match=code):
        decode_request(
            request_line(type=request_type, payload=payload), token="secret", received_at=1.0
        )


def test_encoder_is_compact_and_terminates_with_one_newline():
    encoded = encode_message({"type": "event", "payload": {"state": "accepted"}})

    assert encoded == b'{"type":"event","payload":{"state":"accepted"}}\n'


def test_golden_requests_decode_and_round_trip_as_json():
    vectors = json.loads(
        (Path(__file__).parent / "fixtures" / "motion_protocol_vectors.json").read_text()
    )

    assert len(vectors["requests"]) == 12
    for message in vectors["requests"]:
        request = decode_request(
            encode_message(message), token="vector-token", received_at=100.0
        )
        assert request.id == message["id"]
        assert request.type == message["type"]
        assert request.payload == message["payload"]
        assert json.loads(encode_message(message)) == message


def test_golden_server_events_round_trip_as_json():
    vectors = json.loads(
        (Path(__file__).parent / "fixtures" / "motion_protocol_vectors.json").read_text()
    )

    assert {event["state"] for event in vectors["events"]} == {
        "accepted",
        "completed",
        "cancelled",
        "failed",
    }
    for message in vectors["events"]:
        UUID(message["id"])
        assert json.loads(encode_message(message)) == message


SPEECH_ID = "00000000-0000-0000-0000-000000000001"
LOCAL_ID = "00000000-0000-0000-0000-000000000002"


def local_request_line(kind="orientation.acquire", *, payload=None, **overrides) -> bytes:
    message = {
        "version": 1,
        "id": LOCAL_ID,
        "type": kind,
        "ttl_ms": 1_000,
        "payload": payload if payload is not None else {
            "speech_id": SPEECH_ID,
            "target_yaw": .4,
        },
    }
    message.update(overrides)
    return json.dumps(message, allow_nan=True).encode() + b"\n"


def test_remote_decoder_rejects_local_orientation_acquire_with_stable_code():
    with pytest.raises(ProtocolError, match="local_only"):
        decode_request(
            request_line(type="orientation.acquire", payload={
                "speech_id": SPEECH_ID,
                "target_yaw": .4,
            }),
            token="secret",
            received_at=1.,
        )


def test_local_decoder_accepts_exact_orientation_payload_without_token():
    request = decode_local_request(local_request_line(), received_at=1.)

    assert request.payload == {"speech_id": SPEECH_ID, "target_yaw": .4}
    assert request.expires_at == 2.


@pytest.mark.parametrize(
    ("line", "code"),
    [
        (local_request_line(payload={"speech_id": "0199F3C0-0B5F-7B55-A020-7C8B4012D8CE", "target_yaw": .4}), "invalid_id"),
        (local_request_line(payload={"speech_id": SPEECH_ID, "target_yaw": float("nan")}), "non_finite"),
        (local_request_line(payload={"speech_id": SPEECH_ID, "target_yaw": math.pi + .001}), "out_of_range"),
        (local_request_line(token="secret"), "invalid_message"),
        (local_request_line("motion.status", payload={}), "remote_only"),
        (local_request_line("orientation.return_center", payload={"extra": True}), "invalid_payload"),
        (local_request_line("orientation.status", payload={"extra": True}), "invalid_payload"),
    ],
)
def test_local_decoder_rejects_non_local_or_unsafe_requests(line, code):
    with pytest.raises(ProtocolError, match=code):
        decode_local_request(line, received_at=1.)


@pytest.mark.parametrize("kind", ["orientation.return_center", "orientation.status"])
def test_local_decoder_requires_empty_payload_for_return_and_status(kind):
    request = decode_local_request(local_request_line(kind, payload={}), received_at=1.)

    assert request.type == kind
    assert request.payload == {}


def test_command_type_exports_keep_orientation_off_authenticated_remote_surface():
    assert "orientation.acquire" in LOCAL_COMMAND_TYPES
    assert "orientation.acquire" not in REMOTE_COMMAND_TYPES
    assert "system.heartbeat" in LOCAL_COMMAND_TYPES & REMOTE_COMMAND_TYPES
