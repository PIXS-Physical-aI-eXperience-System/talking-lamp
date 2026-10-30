"""Decide how one spoken turn controls gaze and the return to rest.

The voice process owns audio. This class owns only the S2 motion decisions.
It accepts events in either arrival order because Pi orientation can complete
before the wake-word detector reports the same speech ID.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from uuid import UUID


STATES = frozenset({"wake", "speaking", "done", "interrupted"})
# The Pi refused to turn because the head is already on the user: following
# the face (S2 or the S4 greeting), the face is in view, or about to leave the
# away pose for them.
# Following needs no turn first in that case. A DOA rejection is different:
# the direction is unknown, so S2 still does not follow.
HELD_CODES = frozenset({"face_follow_active", "away_pose_active", "user_in_view"})


@dataclass(frozen=True)
class VoiceEvent:
    speech_id: str
    state: str
    code: str = ""

    @classmethod
    def from_json(cls, payload: str) -> "VoiceEvent":
        data = json.loads(payload)
        if not isinstance(data, dict) or set(data) != {"version", "speech_id", "state", "code"}:
            raise ValueError("invalid voice event fields")
        speech_id = data["speech_id"]
        if not isinstance(speech_id, str) or str(UUID(speech_id)) != speech_id:
            raise ValueError("invalid speech ID")
        if (type(data["version"]) is not int or data["version"] != 1
                or not isinstance(data["state"], str) or data["state"] not in STATES):
            raise ValueError("invalid voice event state")
        if not isinstance(data["code"], str):
            raise ValueError("invalid voice event code")
        return cls(speech_id, data["state"], data["code"])


@dataclass(frozen=True)
class Action:
    kind: str
    speech_id: str
    enabled: bool | None = None


class S2Coordinator:
    """Return effects for the active turn; ignore results from older turns."""

    def __init__(self) -> None:
        self.active_id: str | None = None
        self.following = False
        self._orientation: dict[str, str] = {}

    def orientation(self, speech_id: str, state: str, code: str = "") -> list[Action]:
        if state == "fault" and code in HELD_CODES:
            state = "aligned"
        if speech_id:
            self._orientation[speech_id] = state
            # The only purpose of this cache is to cover orientation-before-wake.
            if len(self._orientation) > 16:
                self._orientation.pop(next(iter(self._orientation)))
        if speech_id != self.active_id or state != "aligned" or self.following:
            return []
        self.following = True
        return [Action("follow", speech_id, True)]

    def voice(self, event: VoiceEvent) -> list[Action]:
        sid = event.speech_id
        if event.state == "wake":
            self.active_id = sid
            if self._orientation.get(sid) == "aligned" and not self.following:
                self.following = True
                return [Action("follow", sid, True)]
            return []
        if sid != self.active_id:
            return []
        if event.state == "interrupted":
            # The next turn starts immediately. Keep tracking the same face.
            return [Action("interrupt", sid)]
        if event.state == "speaking":
            return []
        if event.state == "done":
            self.active_id = None
            effects = []
            if self.following:
                self.following = False
                effects.append(Action("follow", sid, False))
            if event.code != "cancelled":
                effects.append(Action("center", sid))
            return effects
        return []
