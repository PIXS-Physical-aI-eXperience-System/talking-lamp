"""A wake call without a request must not be sent to the language model."""

import sys
import time
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "voice-bench"))

from voice import link  # noqa: E402
from voice.agent import IDLE, LISTENING, VoiceAgent, is_wake_only  # noqa: E402


def test_wake_only_spellings_do_not_match_a_real_request():
    assert is_wake_only("픽스야?")
    assert is_wake_only("빅세야, 픽스야!")
    assert not is_wake_only("픽스야, 안녕?")
    assert not is_wake_only("몇 시야?")


def test_wake_only_replies_without_calling_language_model():
    class Stt:
        def transcribe(self, _audio, _rate):
            return "픽스야?"

    class Tts:
        samplerate = 16000

        def synth(self, text):
            spoken.append(text)
            return np.zeros(320, dtype=np.float32)

    spoken = []
    sent = []
    agent = None

    def send(kind, payload):
        sent.append(kind)
        if kind == link.SPEAK_END:
            agent.on_play_done(agent._speaking_id, "drained")

    def llm(_text):
        raise AssertionError("the language model should not answer a wake-only call")

    agent = VoiceAgent(Stt(), Tts(), None, llm, send)
    agent._think_and_speak(np.zeros(320, dtype=np.float32), "speech-1")

    assert spoken == ["네?"]
    assert link.SPEAK_BEGIN in sent
    assert link.SPEAK_END in sent


def test_wake_ack_does_not_transcribe_call_and_accepts_followup_without_wake():
    class Stt:
        def __init__(self):
            self.calls = 0

        def transcribe(self, _audio, _rate):
            self.calls += 1
            return "안녕?"

    class Tts:
        samplerate = 16000

        def synth(self, text):
            spoken.append(text)
            return np.zeros(320, dtype=np.float32)

    class Wake:
        def __init__(self):
            self.calls = 0

        def detect(self, _pcm):
            self.calls += 1
            return True

        def reset(self):
            pass

    stt = Stt()
    wake = Wake()
    spoken = []
    llm_inputs = []
    agent = None

    def send(kind, _payload):
        if kind == link.SPEAK_END:
            agent.on_play_done(agent._speaking_id, "drained")

    def llm(text):
        llm_inputs.append(text)
        return "안녕하세요."

    agent = VoiceAgent(stt, Tts(), wake, llm, send, ack_on_wake=True)
    voiced = np.ones(320, dtype=np.float32) * 0.1
    silent = np.zeros(320, dtype=np.float32)
    for _ in range(25):
        agent.on_capture("first", voiced)
    for _ in range(31):
        agent.on_capture("", silent)

    deadline = time.monotonic() + 2
    while agent.state != IDLE and time.monotonic() < deadline:
        time.sleep(0.01)
    assert agent.state == IDLE
    assert spoken == ["네?"]
    assert stt.calls == 0
    assert llm_inputs == []

    wake_calls = wake.calls
    agent.on_capture("second", voiced)
    assert agent.state == LISTENING
    assert wake.calls == wake_calls
    for _ in range(24):
        agent.on_capture("second", voiced)
    for _ in range(31):
        agent.on_capture("", silent)

    deadline = time.monotonic() + 2
    while agent.state != IDLE and time.monotonic() < deadline:
        time.sleep(0.01)
    assert agent.state == IDLE
    assert stt.calls == 1
    assert llm_inputs == ["안녕?"]
    assert spoken == ["네?", "안녕하세요."]
