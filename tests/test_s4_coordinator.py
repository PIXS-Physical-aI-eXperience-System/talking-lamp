import pytest

from lamp_interaction.s4 import Action, S4Coordinator


def test_short_absence_does_nothing():
    s4 = S4Coordinator(away_after_s=30)
    assert s4.presence(False, 0.0) == []
    assert s4.tick(29.9) == []
    assert s4.presence(True, 29.9) == []
    assert s4.tick(100.0) == []


def test_long_absence_centers_once_then_return_greets_once():
    s4 = S4Coordinator(away_after_s=30)
    s4.presence(False, 0.0)
    assert s4.tick(30.0) == [Action("center")]
    assert s4.tick(60.0) == []
    assert s4.presence(False, 61.0) == []
    assert s4.tick(200.0) == []
    assert s4.presence(True, 201.0) == []
    assert s4.tick(202.0) == []
    assert s4.tick(203.0) == [Action("greet")]
    assert s4.tick(204.0) == []


def test_conversation_restarts_the_away_wait():
    s4 = S4Coordinator(away_after_s=30)
    s4.presence(False, 0.0)
    assert s4.tick(25.0, busy=True) == []
    assert s4.tick(40.0) == []
    assert s4.tick(55.0) == [Action("center")]


def test_brief_face_while_away_does_not_end_the_away_state():
    s4 = S4Coordinator(away_after_s=30, return_after_s=2)
    s4.presence(False, 0.0)
    assert s4.tick(30.0) == [Action("center")]
    assert s4.presence(True, 30.8) == []
    assert s4.presence(False, 31.5) == []
    assert s4.tick(40.0) == []
    assert s4.away
    assert s4.presence(True, 50.0) == []
    assert s4.tick(52.0) == [Action("greet")]


def test_away_threshold_must_be_positive():
    with pytest.raises(ValueError):
        S4Coordinator(away_after_s=0)


def _away_then_back(s4):
    s4.presence(False, 0.0)
    assert s4.tick(30.0) == [Action("center")]
    s4.presence(True, 100.0)
    assert s4.tick(102.0) == [Action("greet")]


def test_greeting_follows_the_face_for_a_while():
    s4 = S4Coordinator(away_after_s=30, greet_follow_s=10)
    _away_then_back(s4)
    assert not s4.greeting
    assert s4.greeted(106.0) == [Action("follow_on")]
    assert s4.greeting
    assert s4.tick(115.9) == []
    assert s4.tick(116.0) == [Action("follow_off")]
    assert not s4.greeting
    assert s4.tick(117.0) == []


def test_greeting_gaze_ends_when_a_conversation_starts():
    s4 = S4Coordinator(away_after_s=30, greet_follow_s=10)
    _away_then_back(s4)
    s4.greeted(106.0)
    assert s4.conversation_started(0.0) == [Action("follow_off")]
    assert s4.conversation_started(0.0) == []
    assert s4.tick(120.0) == []


def test_leaving_again_during_greeting_gaze_drops_the_follow_off():
    s4 = S4Coordinator(away_after_s=5, greet_follow_s=60)
    s4.presence(False, 0.0)
    s4.tick(5.0)
    s4.presence(True, 10.0)
    assert s4.tick(12.0) == [Action("greet")]
    s4.greeted(16.0)
    s4.presence(False, 17.0)
    assert s4.tick(22.0) == [Action("center")]
    assert s4.tick(100.0) == []


def test_greeting_gaze_can_be_disabled():
    s4 = S4Coordinator(away_after_s=30, greet_follow_s=0)
    _away_then_back(s4)
    assert s4.greeted(106.0) == []


def test_follow_off_and_away_in_the_same_tick():
    s4 = S4Coordinator(away_after_s=5, greet_follow_s=4)
    s4.presence(False, 0.0)
    s4.tick(5.0)
    s4.presence(True, 10.0)
    s4.tick(12.0)
    s4.greeted(16.0)
    s4.presence(False, 15.0 + 1.5)
    assert s4.tick(20.0) == [Action("follow_off")]
    assert s4.tick(21.5) == [Action("center")]


def test_wake_word_while_away_resumes_without_a_greeting():
    s4 = S4Coordinator(away_after_s=30)
    s4.presence(False, 0.0)
    assert s4.tick(30.0) == [Action("center")]
    assert s4.conversation_started(31.0) == [Action("resume")]
    assert not s4.away
    s4.presence(True, 40.0)
    assert s4.tick(80.0) == []


def test_resume_with_no_face_in_view_restarts_the_away_wait():
    s4 = S4Coordinator(away_after_s=30)
    s4.presence(False, 0.0)
    assert s4.tick(30.0) == [Action("center")]
    assert s4.conversation_started(40.0) == [Action("resume")]
    assert s4.tick(69.0, busy=False) == []
    assert s4.tick(70.0) == [Action("center")]


def test_nobody_in_view_from_the_start_still_leads_to_the_away_pose():
    s4 = S4Coordinator(away_after_s=20)
    assert s4.tick(5.0) == []
    assert s4.tick(24.9) == []
    assert s4.tick(25.0) == [Action("center")]
