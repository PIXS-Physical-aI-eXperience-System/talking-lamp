"""Strict authenticated NDJSON protocol for the Pi device service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hmac
import json
import math
from typing import Any
from uuid import UUID


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 16 * 1024
DEVICE_COMMAND_TYPES = {
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
_EMPTY_PAYLOAD_TYPES = {
    "audio.status",
    "orientation.return_center",
    "orientation.status",
    "led.clear",
    "led.status",
    "device.status",
    "system.heartbeat",
}
_FIELDS = {"version", "id", "type", "ttl_ms", "token", "payload"}


class DeviceProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class DeviceRequest:
    id: str
    type: str
    payload: dict[str, object]
    expires_at: float


def encode_message(message: Mapping[str, object]) -> bytes:
    try:
        encoded = json.dumps(
            message, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise DeviceProtocolError(
            "invalid_message", "message cannot be encoded as finite JSON",
        ) from exc
    if len(encoded) > MAX_LINE_BYTES:
        raise DeviceProtocolError("line_too_long", "encoded message exceeds 16 KiB")
    return encoded


def decode_request(line: bytes, *, token: str, received_at: float) -> DeviceRequest:
    if not isinstance(line, bytes):
        raise DeviceProtocolError("invalid_line", "request line must be bytes")
    if len(line) > MAX_LINE_BYTES:
        raise DeviceProtocolError("line_too_long", "request line exceeds 16 KiB")
    if not line.endswith(b"\n"):
        raise DeviceProtocolError("missing_newline", "request line must end with newline")
    if line.endswith(b"\n\n"):
        raise DeviceProtocolError("invalid_line", "request line must contain one message")
    try:
        text = line[:-1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DeviceProtocolError("invalid_encoding", "request must be UTF-8") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeviceProtocolError("invalid_json", "request is not JSON") from exc
    if not isinstance(raw, dict):
        raise DeviceProtocolError("invalid_message", "request must be a JSON object")
    _reject_non_finite(raw)
    _validate_envelope(raw, token)
    kind = raw["type"]
    payload = raw["payload"]
    assert isinstance(kind, str) and isinstance(payload, dict)
    _validate_payload(kind, payload)
    ttl_ms = raw["ttl_ms"]
    ident = raw["id"]
    assert isinstance(ttl_ms, int) and isinstance(ident, str)
    if not isinstance(received_at, (int, float)) or not math.isfinite(received_at):
        raise DeviceProtocolError("invalid_time", "received_at must be finite")
    return DeviceRequest(
        id=ident,
        type=kind,
        payload=payload,
        expires_at=float(received_at) + ttl_ms / 1000.0,
    )


def _validate_envelope(message: dict[str, Any], token: str) -> None:
    missing = _FIELDS - set(message)
    if missing:
        raise DeviceProtocolError("missing_field", f"missing required field: {sorted(missing)[0]}")
    extra = set(message) - _FIELDS
    if extra:
        raise DeviceProtocolError("invalid_message", f"unexpected field: {sorted(extra)[0]}")
    version = message["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != PROTOCOL_VERSION:
        raise DeviceProtocolError("unsupported_version", "version must be 1")
    ident = message["id"]
    if not isinstance(ident, str) or not _canonical_uuid(ident):
        raise DeviceProtocolError("invalid_id", "id must be canonical UUID text")
    kind = message["type"]
    if not isinstance(kind, str) or kind not in DEVICE_COMMAND_TYPES:
        raise DeviceProtocolError("unknown_type", "type is not an allowed device command")
    ttl_ms = message["ttl_ms"]
    if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or not 1 <= ttl_ms <= 10_000:
        raise DeviceProtocolError("invalid_ttl", "ttl_ms must be an integer from 1 to 10000")
    supplied = message["token"]
    if (
        not isinstance(token, str)
        or not token
        or not isinstance(supplied, str)
        or not hmac.compare_digest(supplied.encode(), token.encode())
    ):
        raise DeviceProtocolError("unauthorized", "token does not match")
    if not isinstance(message["payload"], dict):
        raise DeviceProtocolError("invalid_payload", "payload must be an object")


def _validate_payload(kind: str, payload: dict[str, object]) -> None:
    if kind in _EMPTY_PAYLOAD_TYPES:
        _exact(payload, set())
        return
    if kind == "led.frame":
        _exact(payload, {"rgb", "brightness"})
        _rgb(payload["rgb"], 192)
        _brightness(payload["brightness"])
        return
    if kind == "led.solid":
        _exact(payload, {"rgb", "brightness"})
        _rgb(payload["rgb"], 3)
        _brightness(payload["brightness"])
        return
    if kind == "audio.play.start":
        _exact(payload, {"stream_id", "sample_rate", "channels", "encoding"})
        if not isinstance(payload["stream_id"], str) or not _canonical_uuid(payload["stream_id"]):
            raise DeviceProtocolError("invalid_payload", "stream_id must be canonical UUID text")
        if (
            payload["sample_rate"] != 16000
            or isinstance(payload["sample_rate"], bool)
            or payload["channels"] != 1
            or isinstance(payload["channels"], bool)
            or payload["encoding"] != "pcm_s16le"
        ):
            raise DeviceProtocolError(
                "invalid_payload", "audio stream must be 16 kHz mono pcm_s16le")
        return
    if kind == "audio.play.stop":
        _exact(payload, {"stream_id"})
        if not isinstance(payload["stream_id"], str) or not _canonical_uuid(payload["stream_id"]):
            raise DeviceProtocolError("invalid_payload", "stream_id must be canonical UUID text")
        return
    raise AssertionError(kind)


def _exact(payload: dict[str, object], fields: set[str]) -> None:
    if set(payload) != fields:
        raise DeviceProtocolError("invalid_payload", "payload fields do not match command schema")


def _rgb(value: object, length: int) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise DeviceProtocolError("invalid_payload", f"rgb must contain exactly {length} channels")
    if any(isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255
           for channel in value):
        raise DeviceProtocolError("invalid_payload", "rgb channels must be integers from 0 to 255")


def _brightness(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeviceProtocolError("invalid_payload", "brightness must be a number")
    if not math.isfinite(value):
        raise DeviceProtocolError("non_finite", "brightness must be finite")
    if not 0.0 <= value <= 1.0:
        raise DeviceProtocolError("invalid_payload", "brightness must be from 0 to 1")


def _canonical_uuid(value: str) -> bool:
    try:
        return str(UUID(value)) == value
    except (ValueError, AttributeError):
        return False


def _reject_non_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DeviceProtocolError("non_finite", "request contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_non_finite(item)
