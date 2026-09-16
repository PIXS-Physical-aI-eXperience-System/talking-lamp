"""Small, original 8x8 RGB expression catalog for the lamp face."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable


class ExpressionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class Expression:
    name: str
    rgb: bytes


_PALETTE = {
    ".": (0, 0, 0),
    "W": (255, 180, 96),
    "Y": (255, 170, 0),
    "C": (0, 210, 255),
    "B": (32, 96, 255),
    "R": (255, 24, 8),
    "O": (255, 80, 0),
    "V": (150, 48, 255),
    "P": (255, 24, 110),
}


def _compile(name: str, rows: tuple[str, ...]) -> Expression:
    if len(rows) != 8 or any(len(row) != 8 for row in rows):
        raise ValueError(f"{name} must be exactly 8x8")
    try:
        rgb = bytes(channel for row in rows for token in row for channel in _PALETTE[token])
    except KeyError as exc:
        raise ValueError(f"{name} uses unknown color token {exc.args[0]!r}") from exc
    return Expression(name=name, rgb=rgb)


# Logical orientation is top-left, row-major. Physical rotation, origin and
# serpentine wiring are applied by the Raspberry Pi LED controller.
_EXPRESSIONS = tuple(_compile(name, rows) for name, rows in (
    ("neutral", (
        "........", "........", ".WW..WW.", ".WW..WW.",
        "........", "..WWWW..", "........", "........",
    )),
    ("happy", (
        "........", ".YY..YY.", ".YY..YY.", "........",
        "Y......Y", ".Y....Y.", "..YYYY..", "........",
    )),
    ("excited", (
        "C..CC..C", ".C.CC.C.", "..C..C..", "........",
        "..CCCC..", ".C....C.", "..CCCC..", "........",
    )),
    ("sad", (
        "........", ".BB..BB.", ".BB..BB.", "......B.",
        "......B.", "..BBBB..", ".B....B.", "B......B",
    )),
    ("angry", (
        "R......R", ".R....R.", "..RRRR..", "..R..R..",
        "........", "..RRRR..", ".R....R.", "........",
    )),
    ("surprised", (
        "........", ".OO..OO.", ".OO..OO.", "........",
        "...OO...", "..O..O..", "..O..O..", "...OO...",
    )),
    ("curious", (
        "CC......", "..CC....", ".CC...C.", ".CC..CC.",
        ".....C..", "....C...", "........", "....C...",
    )),
    ("thinking", (
        "........", ".VV..VV.", ".VV..VV.", "........",
        "..VV....", ".....V..", "......V.", ".......V",
    )),
    ("shy", (
        "........", ".WW..WW.", ".WW..WW.", "........",
        "PP....PP", "...PP...", "........", "........",
    )),
    ("love", (
        "RRR..RRR", "R.R..R.R", ".R....R.", "........",
        "P......P", ".P....P.", "..PPPP..", "........",
    )),
))
_BY_NAME = {expression.name: expression for expression in _EXPRESSIONS}


def expression_names() -> tuple[str, ...]:
    return tuple(expression.name for expression in _EXPRESSIONS)


def get_expression(name: str) -> Expression:
    try:
        return _BY_NAME[name]
    except (KeyError, TypeError) as exc:
        available = ", ".join(expression_names())
        raise ExpressionError(
            "unknown_expression", f"unknown expression {name!r}; available: {available}") from exc


def expression_payload(name: str, brightness: float) -> dict[str, object]:
    if (
        isinstance(brightness, bool)
        or not isinstance(brightness, (int, float))
        or not math.isfinite(brightness)
        or not 0.0 <= brightness <= 1.0
    ):
        raise ExpressionError(
            "invalid_brightness", "brightness must be a finite number in 0..1")
    expression = get_expression(name)
    return {"rgb": list(expression.rgb), "brightness": float(brightness)}


def request_expression(
    requester: Callable[[str, dict[str, object]], object],
    name: str,
    brightness: float,
) -> object:
    """Validate and resolve the face before performing any transport call."""
    payload = expression_payload(name, brightness)
    return requester("led.frame", payload)
