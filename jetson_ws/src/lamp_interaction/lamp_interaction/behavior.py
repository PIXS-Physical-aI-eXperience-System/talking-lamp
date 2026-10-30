"""Validate cognition output before it can command a physical motion."""

import json
import math


MOTION_TAGS = frozenset({
    "nod", "headshake", "curious", "excited", "happy_wiggle",
    "sad", "shy", "shock", "scanning", "wake_up", "idle",
})


def motion_commands(raw: str) -> tuple[tuple[str, float], ...]:
    response = json.loads(raw)
    if not isinstance(response, dict) or set(response) != {"say", "actions", "end_turn"}:
        raise ValueError("expected say, actions, end_turn")
    if not isinstance(response["say"], str) or not isinstance(response["end_turn"], bool):
        raise ValueError("invalid cognition response")
    actions = response["actions"]
    if not isinstance(actions, list) or len(actions) > 2:
        raise ValueError("expected at most two actions")
    commands = []
    for action in actions:
        if not isinstance(action, dict) or set(action) != {"tag", "intensity"}:
            raise ValueError("invalid action")
        tag, intensity = action["tag"], action["intensity"]
        if tag not in MOTION_TAGS or isinstance(intensity, bool) or not isinstance(intensity, (int, float)):
            raise ValueError("unsupported action")
        if not math.isfinite(intensity) or not 0 <= intensity <= 1:
            raise ValueError("invalid intensity")
        if tag != "idle":
            commands.append((tag, float(intensity)))
    return tuple(commands)
