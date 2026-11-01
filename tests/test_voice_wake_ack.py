"""A wake call without a request must not be sent to the language model."""

import sys
import time
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "voice-bench"))

from voice import link  # noqa: E402
from voice.agent import IDLE, LISTENING, VoiceAgent, heard_wake_word, is_wake_only  # noqa: E402


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


def test_wake_word_spellings_are_heard_in_a_transcript():
    assert heard_wake_word("픽스야")
    assert heard_wake_word("빅 스야, 불 켜줘")
    assert heard_wake_word("피스야")
    assert not heard_wake_word("")
    assert not heard_wake_word("롤러스의 롤러스는")


def test_wake_ack_checks_the_call_and_accepts_followup_without_wake():
    class Stt:
        def __init__(self):
            self.calls = 0

        def transcribe(self, _audio, _rate):
            self.calls += 1
            return "픽스야" if self.calls == 1 else "안녕?"

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
    assert stt.calls == 1
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
    assert stt.calls == 2
    assert llm_inputs == ["안녕?"]
    assert spoken == ["네?", "안녕하세요."]


def test_a_wake_on_music_is_dropped_without_a_word():
    class Stt:
        def transcribe(self, _audio, _rate):
            return ""                   # what the music around a false wake gave

    class Tts:
        samplerate = 16000

        def synth(self, text):
            spoken.append(text)
            return np.zeros(320, dtype=np.float32)

    spoken, sent = [], []
    agent = VoiceAgent(Stt(), Tts(), None, lambda _t: "", lambda k, _p: sent.append(k))
    agent._think_and_speak(np.zeros(320, dtype=np.float32), "speech-1", wake_ack=True,
                           wake_sent=False)
    assert spoken == []
    assert link.WAKE not in sent            # S2 never hears of it
    assert link.HEARD not in sent and link.SPEAK_BEGIN not in sent
    assert agent.state == IDLE


def test_a_confirmed_call_announces_the_wake_after_the_check():
    class Stt:
        def transcribe(self, _audio, _rate):
            return "픽스야"

    class Tts:
        samplerate = 16000

        def synth(self, _text):
            return np.zeros(320, dtype=np.float32)

    sent = []
    agent = None

    def send(kind, _payload):
        sent.append(kind)
        if kind == link.SPEAK_END:
            agent.on_play_done(agent._speaking_id, "drained")

    class Wake:
        def detect(self, _pcm):
            return True

        def reset(self):
            pass

    agent = VoiceAgent(Stt(), Tts(), Wake(), lambda _t: "", send, ack_on_wake=True)
    voiced = np.ones(320, dtype=np.float32) * 0.1
    for _ in range(25):
        agent.on_capture("call", voiced)
    assert link.WAKE not in sent            # not before the transcript says so
    for _ in range(31):
        agent.on_capture("", np.zeros(320, dtype=np.float32))
    deadline = time.monotonic() + 2
    while agent.state != IDLE and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sent.count(link.WAKE) == 1
    assert sent.index(link.WAKE) < sent.index(link.SPEAK_BEGIN)


def test_the_wake_check_uses_the_wake_transcription():
    class Stt:
        def transcribe(self, _audio, _rate):
            raise AssertionError("the command hints turned real calls into '불 켜줘'")

        def transcribe_wake(self, _audio, _rate):
            return "피스야"

    class Tts:
        samplerate = 16000

        def synth(self, _text):
            return np.zeros(320, dtype=np.float32)

    sent = []
    agent = None

    def send(kind, _payload):
        sent.append(kind)
        if kind == link.SPEAK_END:
            agent.on_play_done(agent._speaking_id, "drained")

    agent = VoiceAgent(Stt(), Tts(), None, lambda _t: "", send)
    agent._think_and_speak(np.zeros(320, dtype=np.float32), "speech-1", wake_ack=True,
                           wake_sent=False)
    assert link.WAKE in sent and link.SPEAK_BEGIN in sent
