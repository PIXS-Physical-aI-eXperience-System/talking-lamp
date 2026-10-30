"""Adapter for voice.Agent(on_utterance=adapter.reply).

infer(image, question) returns raw model JSON. submit_motion(result) must enqueue
onto the orchestrator/motion owner thread, never mutate a running runtime from
the voice worker. Camera snapshots and inference stay outside the 100 Hz loop.
"""
from .contract import CognitionResult


class VlmReply:
    name = "Talking Lamp visual cognition"

    def __init__(self, infer, snapshot, submit_motion=None):
        self.infer = infer
        self.snapshot = snapshot
        self.submit_motion = submit_motion

    def reply(self, text):
        if not isinstance(text, str) or not text.strip():
            return iter(())
        # Eager validation before returning an iterator: voice.Agent catches
        # exceptions from on_utterance(), before it starts speaking.
        result = CognitionResult.from_text(self.infer(self.snapshot(), text))
        if self.submit_motion is not None and result.motion != "idle":
            self.submit_motion(result)
        return iter((result.speech_ko,))
