"""답할 말을 만드는 자리. 모델은 젯슨에 올릴 예정이고 아직 없다.

핵심은 스트리밍이다. LLM 이 답을 다 만들 때까지 기다렸다 TTS 를 돌리면 그
시간이 그대로 지연에 얹힌다. 토큰이 나오는 대로 받아 **문장이 완성될 때마다**
내보내면, TTS 는 첫 문장부터 합성해 바로 말하기 시작한다.

실측으로 확인된 것: 문장 단위로 쪼개 합성하면 첫 소리까지 4.56초 → 0.59초,
최고 메모리 1750 → 1245 MB (results 참조). LLM 쪽도 같은 방식으로 잇는다.

붙이는 법:

    voice_agent.py --llm http://127.0.0.1:8080/v1/chat/completions

OpenAI 호환 엔드포인트를 가정한다. llama.cpp 서버, vLLM, Ollama(호환 모드)가
모두 이 형식을 낸다. 젯슨에 무엇을 올리든 이 규격만 맞으면 된다.
"""
import json
import re
import urllib.error
import urllib.request

# 램프가 할 말의 성격. 길게 답하면 지연도 메모리도 같이 커진다.
SYSTEM_PROMPT = (
    "너는 책상 위 스탠드 램프다. 사람의 말을 듣고 짧게 대답한다. "
    "한 번에 두 문장을 넘기지 않는다. 목록이나 기호를 쓰지 않고 말하듯 답한다. "
    "모르면 모른다고 짧게 말한다."
)

MAX_REPLY_CHARS = 200          # 넘으면 자른다. 말이 길면 대화가 아니라 낭독이 된다
SENTENCE_END = re.compile(r"[.!?。？！]|다\s|요\s")


class EchoLlm:
    """모델이 없을 때 쓰는 대역. 들은 말을 되받는다."""

    name = "되받아 말하기 (모델 없음)"

    def reply(self, text):
        yield f"{text}, 라고 하셨네요."


class HttpLlm:
    """OpenAI 호환 엔드포인트. 토큰을 받아 문장 단위로 내보낸다."""

    def __init__(self, url, model="local", timeout=20.0,
                 system=SYSTEM_PROMPT, max_chars=MAX_REPLY_CHARS):
        self.url = url
        self.model = model
        self.timeout = timeout
        self.system = system
        self.max_chars = max_chars
        self.name = f"HTTP {url} ({model})"

    def reply(self, text):
        """문장이 완성될 때마다 내보낸다. 실패하면 아무것도 안 내보낸다."""
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": self.system},
                         {"role": "user", "content": text}],
            "stream": True,
            "max_tokens": 200,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"})
        buf, total = "", 0
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                for raw in res:
                    piece = _sse_delta(raw)
                    if piece is None:
                        continue
                    buf += piece
                    # 문장이 끝났으면 바로 내보낸다. 기다리면 그만큼 늦게 말한다.
                    while True:
                        m = SENTENCE_END.search(buf)
                        if not m:
                            break
                        cut = m.end()
                        part, buf = buf[:cut].strip(), buf[cut:]
                        if part:
                            total += len(part)
                            yield part
                        if total >= self.max_chars:
                            return
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"  ! LLM 호출 실패: {type(e).__name__}: {e}")
            return
        tail = buf.strip()
        if tail and total < self.max_chars:
            yield tail


def _sse_delta(raw):
    """SSE 한 줄에서 늘어난 글자를 꺼낸다. 아니면 None."""
    line = raw.decode("utf-8", "replace").strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
        return obj["choices"][0]["delta"].get("content")
    except (ValueError, KeyError, IndexError):
        return None


def load_llm(url=None, model="local", timeout=20.0):
    return HttpLlm(url, model, timeout) if url else EchoLlm()
