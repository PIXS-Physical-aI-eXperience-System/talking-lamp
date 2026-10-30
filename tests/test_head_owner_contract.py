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
