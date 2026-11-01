"""A mumble the STT dresses up as a hint phrase must not become a command."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "voice-bench"))

from voice.stt import Stt  # noqa: E402


def _stt(text, logprob, no_speech):
    stt = Stt.__new__(Stt)
    seg = SimpleNamespace(text=text, avg_logprob=logprob, no_speech_prob=no_speech,
                          compression_ratio=0.5)
    stt.m = SimpleNamespace(transcribe=lambda *a, **k: (iter([seg]), None))
    return stt


def test_a_mumble_read_as_turn_on_the_light_is_dropped():
    # Measured on the lamp 2026-11-01.
    assert _stt("불 켜줘", -0.78, 0.81).transcribe(np.zeros(320, np.float32)) == ""


def test_a_clear_command_is_kept():
    assert _stt("불 켜줘", -0.23, 0.11).transcribe(np.zeros(320, np.float32)) == "불 켜줘"
    # A real "불 꺼줘" scored this on the lamp.
    assert _stt("불 꺼줘", -0.40, 0.47).transcribe(np.zeros(320, np.float32)) == "불 꺼줘"
    # One bad score alone is not enough.
    assert _stt("안녕", -0.91, 0.2).transcribe(np.zeros(320, np.float32)) == "안녕"


def test_the_wake_check_never_drops_for_clarity():
    assert _stt("픽스야", -0.9, 0.9).transcribe_wake(np.zeros(320, np.float32)) == "픽스야"
