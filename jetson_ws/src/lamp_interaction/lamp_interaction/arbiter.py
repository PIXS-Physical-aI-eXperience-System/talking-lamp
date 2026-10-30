"""Decide which scenario owns the lamp head.

Several nodes used to move the head on their own: S2 face following, S4 away
pose, the vision node re-centring, cognition motions, and the Pi turning
toward every utterance. Each was right alone and they fought together (one
real example: a sound turn swung the base 56 deg while the user sat in front
of the camera, and S4 then decided the user had left).

The priority, highest first, agreed on 2026-10-30:

    safety > S2 conversation > S1 task light > S6 sound turn > S4 away > idle

Barge-in is not an owner. It interrupts whatever plays and S2 keeps the head.
Motor safety itself stays on the Pi; here it only blocks every scenario.

Time is passed in so tests need no clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Owner(str, Enum):
    SAFETY = "safety"
    S2 = "s2"
    S1 = "s1"
    S6 = "s6"
    S4 = "s4"
    IDLE = "idle"


# A turn toward a sound that no wake word follows is undone after this long.
SOUND_TURN_HOLD_S = 6.0
# Sound turns stay held this long after vision loses the face, so a user who
# leans out of view for a moment is not turned away from when they speak.
VIEW_GRACE_S = 5.0
# Orientation states that mean the base is turning or has turned.
TURNING_STATES = frozenset({"orienting", "aligned"})


@dataclass(frozen=True)
class Action:
    kind: str           # "sound_return": undo an unanswered sound turn


class HeadArbiter:
    def __init__(self, sound_turn_hold_s: float = SOUND_TURN_HOLD_S) -> None:
        if sound_turn_hold_s <= 0:
            raise ValueError("sound_turn_hold_s must be positive")
        self.sound_turn_hold_s = sound_turn_hold_s
        self.fault = False
        self.conversation = False
        self.task_light = False
        self.away = False
        self.greeting = False           # S4 follows the face after the return greeting
        self.motion_busy = False        # a cognition motion or greeting is playing
        self.user_in_view = False       # vision presence
        self._view_lost_at: float | None = None
        self._sound_id: str | None = None
        self._sound_at = 0.0
        self._answered: dict[str, None] = {}     # insertion-ordered, bounded

    # -- inputs -------------------------------------------------------------

    def motion_status(self, *, connected: bool, fault: str) -> None:
        self.fault = not connected or bool(fault)

    def orientation(self, speech_id: str, state: str, now: float) -> None:
        """The Pi turned (or failed to turn) toward one utterance."""
        if not speech_id or speech_id in self._answered:
            return
        if state in TURNING_STATES:
            if self._sound_id != speech_id:
                self._sound_id = speech_id
                self._sound_at = now
        elif speech_id == self._sound_id:
            self._sound_id = None

    def wake(self, speech_id: str) -> None:
        """A wake word answered this utterance: the turn belongs to S2 now."""
        self.conversation = True
        if speech_id:
            self._answered[speech_id] = None
            if len(self._answered) > 16:
                self._answered.pop(next(iter(self._answered)))
        if speech_id == self._sound_id or not speech_id:
            self._sound_id = None

    def presence(self, present: bool, now: float) -> None:
        if self.user_in_view and not present:
            self._view_lost_at = now
        self.user_in_view = present

    def conversation_ended(self) -> None:
        self.conversation = False

    # -- decisions ----------------------------------------------------------

    def owner(self) -> Owner:
        if self.fault:
            return Owner.SAFETY
        if self.conversation:
            return Owner.S2
        if self.task_light:
            return Owner.S1
        if self._sound_id is not None:
            return Owner.S6
        if self.away or self.greeting:
            return Owner.S4
        return Owner.IDLE

    def may_leave(self) -> bool:
        """May S4 count the user as gone and take the away pose?"""
        return self.owner() is Owner.IDLE and not self.motion_busy

    def holds_sound_turn(self, now: float) -> bool:
        """Should the Pi skip sound turns? Not while the user is in view:
        turning toward their voice only turns the head away from them."""
        if self.user_in_view:
            return True
        return self._view_lost_at is not None and now - self._view_lost_at < VIEW_GRACE_S

    def may_play(self) -> bool:
        """May a cognition motion start now?"""
        return self.owner() in (Owner.S2, Owner.IDLE)

    def tick(self, now: float) -> list[Action]:
        if self._sound_id is None or now - self._sound_at < self.sound_turn_hold_s:
            return []
        if self.motion_busy:
            # Do not pull the head back in the middle of a motion. Decide once
            # it ends, when the greeting gaze may own the head.
            return []
        self._sound_id = None
        if self.owner() is not Owner.IDLE:
            # Someone else holds the head now and will put it where it wants.
            return []
        return [Action("sound_return")]
