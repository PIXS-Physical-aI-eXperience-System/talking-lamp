"""Validation and encoding for the Pi motion command wire protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hmac
import json
import math
import re
from typing import Any
from uuid import UUID


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 16 * 1024
COMMAND_TYPES = frozenset(
    {
        "motion.play",
        "motion.cancel",
        "motion.interrupt",
        "motion.status",
        "motion.list",
        "track.point",
        "track.bearing",
        "track.clear",
        "task_light.place",
        "task_light.cancel",
        "task_light.clear",
        "system.heartbeat",
    }
)

_REQUIRED_FIELDS = frozenset({"version", "id", "type", "ttl_ms", "token", "payload"})
_MOTION_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_EMPTY_PAYLOAD_TYPES = frozenset(
    {"motion.interrupt", "motion.status", "motion.list", "track.clear", "task_light.clear", "system.heartbeat"}
)


class ProtocolError(ValueError):
    """A request that cannot safely reach the motion controller."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class Request:
    id: str
    type: str
    payload: dict[str, object]
    expires_at: float


def decode_request(line: bytes, *, token: str, received_at: float) -> Request:
    """Decode an authenticated command received at Pi monotonic time ``received_at``."""
    if not isinstance(line, bytes):
        raise ProtocolError("invalid_line", "request line must be bytes")
    if len(line) > MAX_LINE_BYTES:
        raise ProtocolError("line_too_long", "request line exceeds 16 KiB")
    if not line.endswith(b"\n"):
        raise ProtocolError("missing_newline", "request line must end with one newline")
    if line.endswith(b"\n\n"):
        raise ProtocolError("invalid_line", "request line must contain exactly one message")

    try:
        raw = line[:-1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("invalid_encoding", "request line must be UTF-8") from exc
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("invalid_json", "request line is not JSON") from exc

    if not isinstance(message, dict):
        raise ProtocolError("invalid_message", "request must be a JSON object")
    _reject_non_finite(message)
    _validate_envelope(message, token)
    request_type = message["type"]
    assert isinstance(request_type, str)
    payload = message["payload"]
    assert isinstance(payload, dict)
    _validate_payload(request_type, payload)

    ttl_ms = message["ttl_ms"]
    assert isinstance(ttl_ms, int)
    request_id = message["id"]
    assert isinstance(request_id, str)
    return Request(
        id=request_id,
        type=request_type,
        payload=payload,
        expires_at=received_at + ttl_ms / 1000.0,
    )


def encode_message(message: Mapping[str, object]) -> bytes:
    """Serialize one compact JSON wire message with its required line terminator."""
    try:
        encoded = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise ProtocolError("invalid_message", "message cannot be encoded as finite JSON") from exc
    if len(encoded) > MAX_LINE_BYTES:
        raise ProtocolError("line_too_long", "encoded message exceeds 16 KiB")
    return encoded


def _validate_envelope(message: dict[str, Any], expected_token: str) -> None:
    fields = frozenset(message)
    missing = _REQUIRED_FIELDS - fields
    if missing:
        raise ProtocolError("missing_field", f"missing required field: {sorted(missing)[0]}")
    extras = fields - _REQUIRED_FIELDS
    if extras:
        raise ProtocolError("invalid_message", f"unexpected field: {sorted(extras)[0]}")

    version = message["version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise ProtocolError("unsupported_version", "version must be an integer")
    if version != PROTOCOL_VERSION:
        raise ProtocolError("unsupported_version", f"unsupported version: {version}")

    request_id = message["id"]
    if not isinstance(request_id, str) or not _is_canonical_uuid(request_id):
        raise ProtocolError("invalid_id", "id must be canonical UUID text")

    request_type = message["type"]
    if not isinstance(request_type, str) or request_type not in COMMAND_TYPES:
        raise ProtocolError("unknown_type", "type is not an allowed command")

    ttl_ms = message["ttl_ms"]
    if not isinstance(ttl_ms, int) or isinstance(ttl_ms, bool) or not 1 <= ttl_ms <= 10_000:
        raise ProtocolError("invalid_ttl", "ttl_ms must be an integer from 1 to 10000")

    supplied_token = message["token"]
    if not _tokens_match(supplied_token, expected_token):
        raise ProtocolError("unauthorized", "token does not match")

    if not isinstance(message["payload"], dict):
        raise ProtocolError("invalid_payload", "payload must be a JSON object")


def _validate_payload(request_type: str, payload: dict[str, object]) -> None:
    if request_type == "motion.play":
        _require_exact_fields(payload, {"name", "replace_current", "intensity", "repeat"})
        name = payload["name"]
        if not isinstance(name, str) or not _MOTION_NAME.fullmatch(name):
            raise ProtocolError("invalid_name", "motion name is invalid")
        if not isinstance(payload["replace_current"], bool):
            raise ProtocolError("invalid_payload", "replace_current must be a JSON boolean")
        intensity = payload["intensity"]
        if not _is_number(intensity) or not 0.0 <= intensity <= 1.0:
            raise ProtocolError("invalid_payload", "intensity must be from 0.0 to 1.0")
        repeat = payload["repeat"]
        if not isinstance(repeat, int) or isinstance(repeat, bool) or not 1 <= repeat <= 3:
            raise ProtocolError("invalid_payload", "repeat must be an integer from 1 to 3")
        return

    if request_type in {"motion.cancel", "task_light.cancel"}:
        _require_exact_fields(payload, {"request_id"})
        request_id = payload["request_id"]
        if not isinstance(request_id, str) or not _is_canonical_uuid(request_id):
            raise ProtocolError("invalid_id", "request_id must be canonical UUID text")
        return

    if request_type in _EMPTY_PAYLOAD_TYPES:
        _require_exact_fields(payload, set())
        return

    if request_type in {"track.point", "task_light.place"}:
        _require_exact_fields(payload, {"point"})
        _validate_vector(payload["point"], name="point")
        return

    if request_type == "track.bearing":
        _require_exact_fields(payload, {"direction"})
        direction = _validate_vector(payload["direction"], name="direction")
        if sum(component * component for component in direction) == 0.0:
            raise ProtocolError("invalid_direction", "bearing direction must have nonzero norm")
        return

    raise AssertionError(f"unhandled command type: {request_type}")


def _require_exact_fields(payload: dict[str, object], expected: set[str]) -> None:
    if set(payload) != expected:
        raise ProtocolError("invalid_payload", "payload fields are not allowed for this command")


def _validate_vector(value: object, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3 or not all(_is_number(component) for component in value):
        raise ProtocolError("invalid_payload", f"{name} must contain exactly three numbers")
    if any(abs(component) > 2.0 for component in value):
        raise ProtocolError("out_of_range", f"{name} coordinates must be within 2 metres")
    vector = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in vector):
        raise ProtocolError("non_finite", f"{name} contains a non-finite number")
    return vector  # type: ignore[return-value]


def _reject_non_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolError("non_finite", "message contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_non_finite(item)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _tokens_match(supplied_token: object, expected_token: object) -> bool:
    if not isinstance(supplied_token, str) or not isinstance(expected_token, str):
        return False
    try:
        return hmac.compare_digest(supplied_token, expected_token)
    except TypeError:
        return False


def _is_canonical_uuid(value: str) -> bool:
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False
