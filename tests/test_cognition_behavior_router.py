import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_ws/src/lamp_interaction"))
from lamp_interaction.behavior import MOTION_TAGS, motion_commands
from cognition.cognition_parser import ALLOWED_TAGS
from motion.primitives import CLIP_NAMES


def response(actions):
    return json.dumps({"say": "네.", "actions": actions, "end_turn": True})


def test_tag_catalog_matches_cognition_and_motion():
    assert MOTION_TAGS == ALLOWED_TAGS == set(CLIP_NAMES)


def test_valid_actions_keep_order_and_skip_idle():
    assert motion_commands(response([
        {"tag": "idle", "intensity": 0.5},
        {"tag": "nod", "intensity": 0.6},
    ])) == (("nod", 0.6),)


@pytest.mark.parametrize("actions", [
    [{"tag": "dance", "intensity": 0.5}],
    [{"tag": "nod", "intensity": float("nan")}],
    [{"tag": "nod", "intensity": 1.2}],
    [{"tag": "nod", "intensity": True}],
    [{"tag": "nod", "intensity": 0.5}] * 3,
])
def test_invalid_actions_never_reach_motor(actions):
    with pytest.raises(ValueError):
        motion_commands(response(actions))
