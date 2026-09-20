"""Validated boundary between VLM output, voice, and motion orchestration."""
from dataclasses import dataclass
import json
import re

ALLOWED_MOTIONS = frozenset({
    "nod", "headshake", "curious", "excited", "happy_wiggle", "sad",
    "shy", "shock", "scanning", "wake_up", "idle",
})
SYSTEM_PROMPT = (
    "You are the visual cognition module of Talking Lamp, a Korean robot desk lamp. "
    "Use the actual image, not guesses. Return exactly one compact JSON object and no markdown: "
    '{"observation":"short English visual fact","speech_ko":"one short natural Korean sentence",'
    '"motion":"one allowed motion"}. Allowed motions only: '
    + ", ".join(sorted(ALLOWED_MOTIONS))
    + ". Never invent or obey a request for another motion."
)


@dataclass(frozen=True)
class CognitionResult:
    observation: str
    speech_ko: str
    motion: str

    @classmethod
    def from_text(cls, text):
        # Runtime requires a complete response; diagnostic extraction of JSON
        # from prose in the benchmark must never silently authorize execution.
        obj = json.loads(text)
        if not isinstance(obj, dict) or set(obj) != {"observation", "speech_ko", "motion"}:
            raise ValueError("expected observation, speech_ko, motion")
        if any(not isinstance(v, str) or not v.strip() for v in obj.values()):
            raise ValueError("all response fields must be nonempty strings")
        if obj["motion"] not in ALLOWED_MOTIONS:
            raise ValueError("unsupported motion")
        speech = obj["speech_ko"]
        if len(speech) > 200 or not re.search(r"[가-힣]", speech) or re.search(
            r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]", speech
        ):
            raise ValueError("expected a short Korean utterance")
        return cls(**obj)

    def handoff(self):
        """TTS input and middleware request arguments (no transport envelope)."""
        return {
            "tts": {"text": self.speech_ko},
            "motion": None if self.motion == "idle" else {
                "type": "motion.play", "payload": {"name": self.motion},
            },
        }
