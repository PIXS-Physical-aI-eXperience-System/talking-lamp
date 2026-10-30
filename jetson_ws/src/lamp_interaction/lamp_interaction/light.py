"""Turn the lamp light on, off, up or down from what the user said.

The voice agent only talks back; it has no way to drive the LED. It publishes
what it heard on ``/lamp/voice/heard`` and this module reads light requests
out of that text. The LLM is told to answer such requests briefly, so the
spoken reply and the light change happen together.

Kept to plain Korean phrases a user says to a desk lamp. Anything unclear is
ignored: a light that does nothing is better than one that flips on a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Warm white, the same tone as the "W" pixels of the face expressions.
WARM_WHITE = (255, 180, 96)
# Levels are a share of the Pi's LED ceiling (--max-brightness 0.08): any
# request above it is clamped, so 0.8 and 0.55 used to look the same. Below a
# quarter of that ceiling the 8-bit channels round to almost nothing.
DEVICE_MAX_BRIGHTNESS = 0.08
DEFAULT_BRIGHTNESS = 1.0
# Halve and double: equal linear steps looked too alike, and the eye judges
# brightness by ratio.
STEP = 2.0
MIN_BRIGHTNESS = 0.25

_TARGET = re.compile(r"불|조명|전등|라이트|스탠드")
# "불편", "불안", "불가능" are not the light.
_NOT_LIGHT = re.compile(r"불(?:편|안|만|가능|쌍|러|행|량|법|구|과|리)")
_NEGATED = re.compile(r"(?:켜|꺼|끄)지\s*(?:마|말)")
_ACTIONS = (
    ("off", re.compile(r"꺼|끄|끌|꺼줘|끕")),
    ("brighter", re.compile(r"밝게|밝혀|밝아|올려")),
    ("dimmer", re.compile(r"어둡게|흐리게|어두워|낮춰|줄여")),
    ("on", re.compile(r"켜|켤|킬|켜줘")),
)


@dataclass(frozen=True)
class LightCommand:
    kind: str           # "on", "off", "brighter" or "dimmer"


def light_command(text: str) -> LightCommand | None:
    cleaned = _NOT_LIGHT.sub("", text)
    if not _TARGET.search(cleaned) or _NEGATED.search(cleaned):
        return None
    found = [kind for kind, pattern in _ACTIONS if pattern.search(cleaned)]
    # "밝게 켜 줘" is a brighter light; any other mix is unclear.
    if found == ["brighter", "on"]:
        return LightCommand("brighter")
    if len(found) != 1:
        return None
    return LightCommand(found[0])


def device_brightness(level: float) -> float:
    """Map a 0..1 level onto what the Pi will actually show."""
    return round(level * DEVICE_MAX_BRIGHTNESS, 4)


class LightState:
    """Track brightness so "brighter" and "dimmer" step from where it is."""

    def __init__(self) -> None:
        self.brightness = 0.0

    def apply(self, command: LightCommand) -> float:
        """Return the new brightness; 0 means off."""
        if command.kind == "on":
            self.brightness = DEFAULT_BRIGHTNESS
        elif command.kind == "off":
            self.brightness = 0.0
        elif command.kind == "brighter":
            if self.brightness == 0.0:
                self.brightness = DEFAULT_BRIGHTNESS
            else:
                self.brightness = min(1.0, self.brightness * STEP)
        else:
            if self.brightness == 0.0:
                return 0.0
            self.brightness = max(MIN_BRIGHTNESS, self.brightness / STEP)
        return self.brightness
