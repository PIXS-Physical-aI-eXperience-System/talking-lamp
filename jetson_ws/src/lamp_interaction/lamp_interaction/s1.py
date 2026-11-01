"""S1: light the book on the desk when the user asks for it.

The vision node does the looking (``/lamp/vision/aim_object``): it tilts the
head down, sweeps left and right, and centres a book in the image with the same
small steps S2 uses to follow a face. Then it reaches out: the arm takes the
task-light pose 15 cm in front of the base, keeping the base yaw it centred the
book at. This flow only decides when to start, waits for the reach to end,
turns the light on, and lets go.

Placing the light from a 3D desk point was tried first (2026-10-30) and needs
a camera mount the lamp does not have: tilting only the wrist moved the same
book from 0.13 m to 0.33 m on the desk. Centring in the image gives the
direction without a mount; the distance is fixed at the middle of the reach.

Only on request, so the head never goes looking on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

GIVE_UP_S = 40.0        # tilt down (5 s) and two full sweeps at 2 deg / 0.5 s
# Vision aim states that end the search. A failed reach still leaves the head
# centred on the book, so it is lit either way.
_DONE = ("placed", "reach_failed")
# Once the book is in view, keep going past give_up_s, up to this.
_FOUND = ("centering", "centered", "reaching")
FOUND_LIMIT_S = 90.0

# STT spells it several ways: "책 빗쳐줘", "채 빗쳐줘" (2026-10-30).
_SHINE = re.compile(r"[비빗빛](?:춰|추|쳐|출)")
_STOP = re.compile(r"그만|치워|원래대로|제자리|(?:비추|비춰)지\s*(?:마|말)")
# The voice agent sends "픽스야" for a bare call. The lamp keeps lighting the
# book and only listens; whatever the user asks next ends the task light.
_WAKE_CALL = re.compile(r"^\s*(?:픽스|빅스|픽세|빅세|피스)\s*(?:야|아)?[\s,.!?~…]*$")


@dataclass(frozen=True)
class TaskLightRequest:
    kind: str           # "place" or "release"


def task_light_command(text: str) -> TaskLightRequest | None:
    """"책 비춰줘" places the light; "그만 비춰", "조명 치워" release it."""
    if _STOP.search(text) and (_SHINE.search(text) or re.search(r"조명|불|라이트", text)):
        return TaskLightRequest("release")
    if _SHINE.search(text):
        return TaskLightRequest("place")
    return None


def ends_task_light(text: str, light_kind: str | None) -> bool:
    """Does this request, heard while the book is lit, end the task light?
    A bare call does not, and neither does turning the light on or changing
    its brightness: that is still lighting the book. ``light_kind`` is the
    ``LightCommand`` kind parsed from the same text, if any."""
    if not text.strip() or _WAKE_CALL.match(text):
        return False
    return light_kind not in ("on", "brighter", "dimmer")


class TaskLightFlow:
    """idle -> pending (waits for the conversation) -> aiming -> lit."""

    def __init__(self, give_up_s: float = GIVE_UP_S) -> None:
        self.give_up_s = give_up_s
        self.state = "idle"
        self._since = 0.0

    @property
    def holds_head(self) -> bool:
        return self.state in ("aiming", "lit")

    @property
    def aiming(self) -> bool:
        return self.state == "aiming"

    def request(self) -> None:
        self.state = "pending"

    def release(self) -> bool:
        """Drop the request. True if vision was aiming or holding for it."""
        held = self.holds_head
        self.state = "idle"
        return held

    def tick(self, now: float, *, busy: bool, aim: str) -> list[str]:
        """``busy``: a conversation or motion still has the head. ``aim``: the
        vision node's aim state. Returns "start", "lit" or "give_up"."""
        if self.state == "pending" and not busy:
            self.state = "aiming"
            self._since = now
            return ["start"]
        if self.state != "aiming":
            return []
        if aim in _DONE:
            self.state = "lit"
            return ["lit"]
        waited = now - self._since
        if waited >= FOUND_LIMIT_S or (aim not in _FOUND and waited >= self.give_up_s):
            self.state = "idle"
            return ["give_up"]
        return []
