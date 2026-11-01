"""Barge-in: a loud rise or the wake word stops the lamp mid-reply."""

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "voice-bench"))

from voice import link  # noqa: E402
from voice.agent import LISTENING, SPEAKING, BargeInDetector, VoiceAgent  # noqa: E402


def frame(db):
    return np.full(320, 10 ** (db / 20), dtype=np.float32)


class Wake:
    hit = False

    def detect(self, _pcm):
        return self.hit

    def reset(self):
        pass


def speaking_agent(wake):
    class Tts:
        samplerate = 16000

    sent = []
    agent = VoiceAgent(None, Tts(), wake, lambda text: "", lambda k, p: sent.append(k),
                       ack_on_wake=True)
    agent.barge = BargeInDetector(hold_s=0.0)
    with agent._lock:
        agent._set(SPEAKING)
    return agent, sent


def test_a_loud_rise_still_interrupts():
    agent, sent = speaking_agent(Wake())
    for _ in range(20):
        agent.on_capture("", frame(-60))
    for _ in range(3):
        agent.on_capture("", frame(-30))
    assert agent.state == LISTENING
    assert link.BARGE_IN in sent
    assert not agent._wake_ack_pending


def test_the_wake_word_interrupts_a_quiet_voice():
    wake = Wake()
    agent, sent = speaking_agent(wake)
    for _ in range(20):
        agent.on_capture("", frame(-50))
    for _ in range(5):
        agent.on_capture("", frame(-37))     # too quiet for the level rule
    assert agent.state == SPEAKING

    wake.hit = True
    agent.on_capture("", frame(-37))
    assert agent.state == LISTENING
    assert link.BARGE_IN in sent
    assert agent._wake_ack_pending


def test_wake_score_from_the_lamp_voice_is_dropped_when_speech_ends():
    class CountingWake(Wake):
        resets = 0

        def reset(self):
            self.resets += 1

    wake = CountingWake()
    agent, _ = speaking_agent(wake)
    before = wake.resets
    with agent._lock:
        agent._set("대기")
    assert wake.resets == before + 1
