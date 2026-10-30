import pytest

from lamp_interaction.arbiter import Action, HeadArbiter, Owner


def test_idle_by_default_and_s4_may_start():
    arbiter = HeadArbiter()
    assert arbiter.owner() is Owner.IDLE
    assert arbiter.may_leave()
    assert arbiter.may_play()


def test_priority_order():
    arbiter = HeadArbiter()
    arbiter.away = True
    assert arbiter.owner() is Owner.S4
    arbiter.orientation("a", "orienting", 0.0)
    assert arbiter.owner() is Owner.S6
    arbiter.task_light = True
    assert arbiter.owner() is Owner.S1
    arbiter.wake("a")
    assert arbiter.owner() is Owner.S2
    arbiter.motion_status(connected=True, fault="shoulder_over_temperature")
    assert arbiter.owner() is Owner.SAFETY
    arbiter.motion_status(connected=False, fault="")
    assert arbiter.owner() is Owner.SAFETY
    arbiter.motion_status(connected=True, fault="")
    assert arbiter.owner() is Owner.S2


def test_s4_waits_for_everything_else():
    arbiter = HeadArbiter()
    arbiter.motion_busy = True
    assert not arbiter.may_leave()
    arbiter.motion_busy = False
    arbiter.orientation("a", "aligned", 0.0)
    assert not arbiter.may_leave()


def test_cognition_motion_only_in_conversation_or_idle():
    arbiter = HeadArbiter()
    arbiter.away = True
    assert not arbiter.may_play()
    arbiter.wake("a")
    assert arbiter.may_play()
    arbiter.conversation_ended()
    arbiter.away = False
    arbiter.task_light = True
    assert not arbiter.may_play()


def test_unanswered_sound_turn_is_undone_once():
    arbiter = HeadArbiter(sound_turn_hold_s=6.0)
    arbiter.orientation("a", "orienting", 10.0)
    arbiter.orientation("a", "aligned", 11.0)       # same turn, clock not restarted
    assert arbiter.tick(15.9) == []
    assert arbiter.tick(16.0) == [Action("sound_return")]
    assert arbiter.tick(30.0) == []
    assert arbiter.owner() is Owner.IDLE


def test_wake_word_keeps_the_turn():
    arbiter = HeadArbiter()
    arbiter.orientation("a", "aligned", 0.0)
    arbiter.wake("a")
    assert arbiter.tick(100.0) == []
    # A late status for the answered utterance does not start a new turn.
    arbiter.conversation_ended()
    arbiter.orientation("a", "aligned", 101.0)
    assert arbiter.owner() is Owner.IDLE


def test_failed_turn_needs_no_return():
    arbiter = HeadArbiter()
    arbiter.orientation("a", "orienting", 0.0)
    arbiter.orientation("a", "fault", 0.5)
    assert arbiter.owner() is Owner.IDLE
    assert arbiter.tick(10.0) == []
    arbiter.orientation("b", "rejected", 11.0)
    assert arbiter.owner() is Owner.IDLE


def test_newer_utterance_restarts_the_hold():
    arbiter = HeadArbiter(sound_turn_hold_s=6.0)
    arbiter.orientation("a", "aligned", 0.0)
    arbiter.orientation("b", "aligned", 5.0)
    assert arbiter.tick(6.0) == []
    assert arbiter.tick(11.0) == [Action("sound_return")]


def test_no_return_when_a_higher_owner_took_the_head():
    arbiter = HeadArbiter(sound_turn_hold_s=6.0)
    arbiter.orientation("a", "aligned", 0.0)
    arbiter.task_light = True
    assert arbiter.tick(6.0) == []
    arbiter.task_light = False
    assert arbiter.owner() is Owner.IDLE


def test_greeting_gaze_owns_the_head_and_keeps_a_sound_turn():
    arbiter = HeadArbiter(sound_turn_hold_s=6.0)
    arbiter.greeting = True
    assert arbiter.owner() is Owner.S4
    assert not arbiter.may_leave()
    arbiter.orientation("a", "aligned", 0.0)
    assert arbiter.tick(6.0) == []
    arbiter.greeting = False
    assert arbiter.owner() is Owner.IDLE


def test_sound_return_waits_for_a_motion_to_finish():
    arbiter = HeadArbiter(sound_turn_hold_s=6.0)
    arbiter.orientation("a", "aligned", 0.0)
    arbiter.motion_busy = True
    assert arbiter.tick(6.0) == []
    assert arbiter.tick(9.0) == []
    arbiter.motion_busy = False
    assert arbiter.tick(9.5) == [Action("sound_return")]


def test_motion_then_greeting_gaze_drops_the_sound_return():
    arbiter = HeadArbiter(sound_turn_hold_s=6.0)
    arbiter.orientation("a", "aligned", 0.0)
    arbiter.motion_busy = True
    assert arbiter.tick(6.0) == []
    arbiter.motion_busy = False
    arbiter.greeting = True
    assert arbiter.tick(7.0) == []
    arbiter.greeting = False
    assert arbiter.tick(20.0) == []


def test_sound_turns_are_held_while_the_user_is_in_view_and_briefly_after():
    arbiter = HeadArbiter()
    assert not arbiter.holds_sound_turn(0.0)
    arbiter.presence(True, 1.0)
    assert arbiter.holds_sound_turn(100.0)
    arbiter.presence(False, 100.0)
    assert arbiter.holds_sound_turn(104.9)
    assert not arbiter.holds_sound_turn(105.0)
    # A repeated "absent" does not restart the grace.
    arbiter.presence(False, 106.0)
    assert not arbiter.holds_sound_turn(106.0)


def test_invalid_hold():
    with pytest.raises(ValueError):
        HeadArbiter(sound_turn_hold_s=0)
