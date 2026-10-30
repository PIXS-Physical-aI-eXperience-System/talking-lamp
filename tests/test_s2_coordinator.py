import json

import pytest

from lamp_interaction.s2 import Action, S2Coordinator, VoiceEvent


A = "20000000-0000-0000-0000-000000000001"
B = "20000000-0000-0000-0000-000000000002"


def test_orientation_can_arrive_before_wake_and_done_returns_to_center():
    s2 = S2Coordinator()
    assert s2.orientation(A, "aligned") == []
    assert s2.voice(VoiceEvent(A, "wake")) == [Action("follow", A, True)]
    assert s2.voice(VoiceEvent(A, "speaking")) == []
    assert s2.voice(VoiceEvent(A, "done", "drained")) == [
        Action("follow", A, False), Action("center", A),
    ]
    assert s2.voice(VoiceEvent(A, "done", "drained")) == []


def test_barge_in_keeps_gaze_and_stale_completion_cannot_center_new_turn():
    s2 = S2Coordinator()
    s2.voice(VoiceEvent(A, "wake"))
    assert s2.orientation(A, "aligned") == [Action("follow", A, True)]
    assert s2.voice(VoiceEvent(A, "interrupted")) == [Action("interrupt", A)]
    assert s2.voice(VoiceEvent(B, "wake")) == []
    assert s2.voice(VoiceEvent(A, "done", "cancelled")) == []
    assert s2.voice(VoiceEvent(B, "done", "drained")) == [
        Action("follow", B, False), Action("center", B),
    ]


def test_rejected_orientation_never_enables_face_following():
    s2 = S2Coordinator()
    assert s2.orientation(A, "rejected") == []
    assert s2.voice(VoiceEvent(A, "wake")) == []
    assert s2.voice(VoiceEvent(A, "done", "drained")) == [Action("center", A)]


def test_voice_event_requires_explicit_version_and_canonical_id():
    raw = {"version": 1, "speech_id": A, "state": "wake", "code": ""}
    assert VoiceEvent.from_json(json.dumps(raw)) == VoiceEvent(A, "wake")
    for changed in ({**raw, "speech_id": "not-an-id"},
                    {**raw, "state": "move_anywhere"},
                    {**raw, "version": 2}):
        with pytest.raises(ValueError):
            VoiceEvent.from_json(json.dumps(changed))


def test_turn_refused_because_the_head_is_on_the_user_still_follows():
    s2 = S2Coordinator()
    assert s2.orientation(A, "fault", "face_follow_active") == []
    assert s2.voice(VoiceEvent(A, "wake")) == [Action("follow", A, True)]


def test_turn_refused_because_the_user_is_in_view_still_follows():
    s2 = S2Coordinator()
    assert s2.orientation(A, "fault", "user_in_view") == []
    assert s2.voice(VoiceEvent(A, "wake")) == [Action("follow", A, True)]


def test_other_orientation_faults_do_not_follow():
    s2 = S2Coordinator()
    s2.voice(VoiceEvent(A, "wake"))
    assert s2.orientation(A, "fault", "shoulder_over_temperature") == []
    assert not s2.following
