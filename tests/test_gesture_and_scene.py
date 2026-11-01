"""Gestures picked from the user's words, and camera labels in the prompt."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jetson_ws/src/lamp_interaction"))
sys.path.insert(0, str(ROOT / "voice-bench"))

from lamp_interaction.gesture import SPEAK_NOD, gesture_for  # noqa: E402
from voice import link  # noqa: E402
from voice.agent import VoiceAgent  # noqa: E402


def test_praise_wiggles_and_agreement_nods():
    assert gesture_for("잘했어 픽스")[0] == "happy_wiggle"
    assert gesture_for("우와 너 진짜 귀엽다")[0] == "happy_wiggle"
    assert gesture_for("잘했어 고마워")[0] == "happy_wiggle"
    assert gesture_for("응 맞아")[0] == "nod"
    assert gesture_for("좋아 그렇게 하자")[0] == "nod"
    assert gesture_for("고마워 덕분이야")[0] == "nod"


def test_other_feelings_and_plain_questions():
    assert gesture_for("대박 나 합격했어")[0] == "excited"
    assert gesture_for("나 오늘 너무 피곤해")[0] == "sad"
    assert gesture_for("아니 그거 말고")[0] == "headshake"
    assert gesture_for("싫어")[0] == "headshake"
    assert gesture_for("내 책상에 뭐 있어?") is None
    assert gesture_for("어떻게 지냈어?") is None       # "어" alone is agreement only
    assert SPEAK_NOD[1] < gesture_for("응")[1]


def test_interaction_wires_gestures_light_off_and_speaking_nod():
    text = (ROOT / "jetson_ws/src/lamp_interaction/lamp_interaction/node.py").read_text()
    assert "gesture = gesture_for(msg.data)" in text
    assert "self._react(SPEAK_NOD)" in text
    away = text[text.index('"사용자 자리 비움: 대기 자세"'):]
    assert 'self._set_light(LightCommand("off"))' in away[:500]


def test_the_reply_prompt_carries_what_the_camera_sees():
    asked = []

    class Stt:
        def transcribe(self, _audio, _rate):
            return "내 책상에 뭐 있어?"

    class Tts:
        samplerate = 16000

        def synth(self, _text):
            return np.zeros(320, dtype=np.float32)

    agent = None

    def send(kind, _payload):
        if kind == link.SPEAK_END:
            agent.on_play_done(agent._speaking_id, "drained")

    def llm(text):
        asked.append(text)
        return "컵이 보여요."

    agent = VoiceAgent(Stt(), Tts(), None, llm, send)
    agent._think_and_speak(np.zeros(320, np.float32), "s1")
    agent.on_labels(["사람", "컵"])
    agent._think_and_speak(np.zeros(320, np.float32), "s2")
    agent.on_labels([])
    agent._think_and_speak(np.zeros(320, np.float32), "s3")

    assert asked[0] == "내 책상에 뭐 있어?"
    assert asked[1] == "[상황] 카메라 분류결과: 사람, 컵\n[사용자] 내 책상에 뭐 있어?"
    assert asked[2].startswith("[상황] 카메라 분류결과: 없음\n")


def test_no_nod_on_the_bare_call_answer():
    from lamp_interaction.s1 import is_wake_call

    assert is_wake_call("픽스야") and is_wake_call("피스야?")
    assert not is_wake_call("픽스야 불 켜줘")
    text = (ROOT / "jetson_ws/src/lamp_interaction/lamp_interaction/node.py").read_text()
    heard = text[text.index("def _heard"):]
    assert heard.index("is_wake_call(msg.data)") < heard.index("gesture_for(msg.data)")
