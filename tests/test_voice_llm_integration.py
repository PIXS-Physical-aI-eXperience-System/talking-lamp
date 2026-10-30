"""The voice stream must use the selected cognition model and its plain prompt."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "voice-bench"))
sys.path.insert(0, str(ROOT / "src" / "cognition"))

from cognition_run import MODEL, SYSTEM_PLAIN, TEMP  # noqa: E402
from voice import llm  # noqa: E402


def test_voice_stream_uses_cognition_settings(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def __iter__(self):
            for content in ("네, ", "여기 있어요."):
                payload = {"choices": [{"delta": {"content": content}}]}
                yield f"data: {json.dumps(payload)}\n".encode()
            yield b"data: [DONE]\n"

    def urlopen(request, timeout):
        requests.append((json.loads(request.data), timeout))
        return Response()

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    result = list(llm.load_llm("http://127.0.0.1:11434/v1/chat/completions").reply("안녕"))

    assert result == ["네, 여기 있어요."]
    body, timeout = requests[0]
    assert body["model"] == MODEL == llm.COGNITION_MODEL
    assert body["temperature"] == TEMP == llm.COGNITION_TEMPERATURE
    assert body["messages"][0]["content"] == SYSTEM_PLAIN == llm.SYSTEM_PROMPT
    assert body["messages"][1]["content"] == "안녕"
    assert body["stream"] is True
    assert timeout == 20.0
