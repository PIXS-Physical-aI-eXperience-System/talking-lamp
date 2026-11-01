"""The ROS wiring around the head arbiter, checked without rclpy."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source(relative):
    return (ROOT / relative).read_text()


def test_interaction_publishes_the_owner_latched_and_feeds_the_arbiter():
    text = source("jetson_ws/src/lamp_interaction/lamp_interaction/node.py")
    assert '"/lamp/head_owner"' in text
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in text
    for topic in ('"/lamp/motion_status"', '"/lamp/vision/status"',
                  '"/lamp/orientation_status"'):
        assert topic in text
    assert "busy=not self.arbiter.may_leave()" in text
    assert "self.arbiter.may_play()" in text


def test_vision_reads_the_owner_latched_and_reports_placing():
    text = source("jetson_ws/src/lamp_vision/lamp_vision/node.py")
    assert '"/lamp/head_owner"' in text
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in text
    assert 'self._head_owner != "idle"' in text
    assert '"placing": self._placing' in text


def test_motion_bridge_publishes_the_filtered_shoulder_temperature():
    text = source("jetson_ws/src/lamp_motion_bridge/lamp_motion_bridge/node.py")
    assert '"/lamp/motion/shoulder_temperature_c"' in text
    assert "self._publish_temperature(data)" in text


def test_interaction_ends_a_turn_the_voice_agent_never_closes():
    text = source("jetson_ws/src/lamp_interaction/lamp_interaction/node.py")
    assert "VOICE_SILENCE_S = 60.0" in text
    assert 'VoiceEvent(self.s2.active_id, "done", "timeout")' in text


def test_interaction_drives_the_light_from_what_was_heard():
    text = source("jetson_ws/src/lamp_interaction/lamp_interaction/node.py")
    assert '"/lamp/voice/heard"' in text
    assert '"/lamp/led/set_solid"' in text
    assert '"/lamp/led/clear"' in text


def test_interaction_runs_the_task_light_on_request():
    text = source("jetson_ws/src/lamp_interaction/lamp_interaction/node.py")
    assert '"/lamp/vision/aim_object"' in text
    assert '"/lamp/track_point"' not in text
    assert "self.arbiter.task_light = self._vision_placing or self.s1.holds_head" in text
    vision = source("jetson_ws/src/lamp_vision/lamp_vision/node.py")
    assert '"/lamp/vision/aim_object"' in vision


def test_interaction_reacts_when_no_book_is_found_or_a_call_is_not_understood():
    text = source("jetson_ws/src/lamp_interaction/lamp_interaction/node.py")
    assert '"/lamp/voice/say"' in text
    assert 'NO_BOOK_MOTION = ("headshake", 1.0)' in text
    assert 'CONFUSED_MOTION = ("scanning", 0.7)' in text
    give_up = text[text.index('elif action == "give_up":'):]
    assert "self.say_pub.publish(String(data=NO_BOOK_SAY))" in give_up[:600]
    assert "self._react(NO_BOOK_MOTION, wait_s=REACT_WAIT_S)" in give_up[:600]
    assert 'event.state == "confused"' in text
    voice = source("voice-bench/ros/lamp_voice_node.py")
    assert '"/lamp/voice/say"' in voice
    assert '"confused"' in voice


def test_the_confused_voice_event_is_accepted():
    import sys
    sys.path.insert(0, str(ROOT / "jetson_ws/src/lamp_interaction"))
    from lamp_interaction.s2 import S2Coordinator, VoiceEvent

    sid = "00000000-0000-4000-8000-000000000001"
    event = VoiceEvent.from_json(
        '{"version":1,"speech_id":"%s","state":"confused","code":""}' % sid)
    s2 = S2Coordinator()
    s2.voice(VoiceEvent(sid, "wake"))
    assert s2.voice(event) == []
    assert s2.active_id == sid
