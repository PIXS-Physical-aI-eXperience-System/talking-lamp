from pathlib import Path
import sys
import threading


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "voice-bench"))
from voice.agent import VoiceAgent  # noqa: E402


def test_speech_waits_for_the_matching_orientation_result():
    agent = VoiceAgent(None, None, None, lambda _: None, lambda *_: None,
                       orientation_wait_s=0.5)
    finished = threading.Event()
    thread = threading.Thread(target=lambda: (agent._wait_orientation("turn-a"),
                                              finished.set()))
    thread.start()
    agent.on_orientation("turn-b", "aligned")
    assert not finished.wait(0.03)
    agent.on_orientation("turn-a", "aligned")
    assert finished.wait(0.2)
    thread.join()
