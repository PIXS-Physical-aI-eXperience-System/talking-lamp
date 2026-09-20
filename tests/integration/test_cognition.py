import json
from queue import Queue

import pytest

from cognition.contract import ALLOWED_MOTIONS, CognitionResult
from cognition.voice_adapter import VlmReply
from motion.primitives import CLIP_NAMES
from motion.runtime import MotionRuntime, NullBackend


def answer(motion="nod", **changes):
    return json.dumps(dict(observation="A lamp.", speech_ko="램프가 보이네요.",
                           motion=motion, **changes), ensure_ascii=False)


def test_voice_to_real_motion_runtime():
    queue = Queue()
    image = object()
    seen = []

    def infer(frame, question):
        seen.append((frame, question))
        return answer()

    adapter = VlmReply(infer, lambda: image, queue.put)
    assert list(adapter.reply("뭐가 보여?")) == ["램프가 보이네요."]
    assert seen == [(image, "뭐가 보여?")]
    runtime = MotionRuntime(backend=NullBackend())
    result = queue.get_nowait()
    runtime.play_primitive(result.motion)
    for _ in range(100):
        runtime.step()
    assert runtime.backend.measured().shape == (5,)
    assert result.handoff()["motion"] == {"type": "motion.play", "payload": {"name": "nod"}}


def test_motion_contract_matches_main():
    assert ALLOWED_MOTIONS == set(CLIP_NAMES)


@pytest.mark.parametrize("raw", [answer("dance"), "```json\n" + answer() + "\n```",
                                    '{"speech_ko": "안녕"}', 'null',
                                    answer().replace('"nod"', '["nod"]')])
def test_invalid_response_never_reaches_motion(raw):
    queue = Queue()
    adapter = VlmReply(lambda *_: raw, lambda: None, queue.put)
    with pytest.raises((ValueError, TypeError)):
        adapter.reply("안녕")
    assert queue.empty()


def test_idle_and_empty_input():
    queue = Queue()
    adapter = VlmReply(lambda *_: answer("idle"), lambda: None, queue.put)
    assert list(adapter.reply("")) == []
    assert list(adapter.reply("안녕")) == ["램프가 보이네요."]
    assert queue.empty()
    assert CognitionResult.from_text(answer("idle")).handoff()["motion"] is None
