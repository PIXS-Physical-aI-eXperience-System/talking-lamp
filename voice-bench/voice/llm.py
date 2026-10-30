"""답할 말을 만드는 자리. Jetson의 온보드 LLM을 스트리밍으로 호출한다.

핵심은 스트리밍이다. LLM 이 답을 다 만들 때까지 기다렸다 TTS 를 돌리면 그
시간이 그대로 지연에 얹힌다. 토큰이 나오는 대로 받아 **문장이 완성될 때마다**
내보내면, TTS 는 첫 문장부터 합성해 바로 말하기 시작한다.

실측으로 확인된 것: 문장 단위로 쪼개 합성하면 첫 소리까지 4.56초 → 0.59초,
최고 메모리 1750 → 1245 MB (results 참조). LLM 쪽도 같은 방식으로 잇는다.

붙이는 법:

    voice_agent.py --llm http://127.0.0.1:11434/v1/chat/completions

OpenAI 호환 엔드포인트를 가정한다. llama.cpp 서버, vLLM, Ollama(호환 모드)가
모두 이 형식을 낸다. 젯슨에 무엇을 올리든 이 규격만 맞으면 된다.
"""
import json
import re
import urllib.error
import urllib.request

# A파트의 src/cognition/cognition_run.py SYSTEM_PLAIN과 동일한 평문 규칙.
# JSON 프롬프트를 그대로 쓰면 TTS가 중괄호까지 읽으므로 평문 모드를 쓴다.
COGNITION_MODEL = "hf.co/kexplo/HyperCLOVAX-SEED-Text-Instruct-1.5B-Q4_K_M-GGUF"
COGNITION_TEMPERATURE = 0.2
SYSTEM_PROMPT = (
    "너는 '픽스'라는 이름의 책상 위 스탠드 램프다. 사용자 곁에서 빛을 비추고 말동무가 되어준다.\n"
    "다정하지만 담백하다. 항상 한국어로만, 두 문장 이내로 짧게 말한다. 영어로 답하지 않는다.\n"
    "목록·기호·이모지를 쓰지 않고 말하듯 답한다.\n"
    "시간·날짜·위치처럼 알 수 없는 사실은 절대 지어내지 말고 모른다고 짧게 말한다.\n"
    "'카메라 분류결과'에 있는 것만 근거로 삼고, 목록에 없는 물건은 언급하지 않는다. "
    "분류결과를 그대로 나열하지 않고 사용자의 말에 자연스럽게 반응한다.\n"
    "불을 켜거나 끄거나 밝기를 바꿔 달라는 말에는 램프가 직접 조명을 바꾸니 "
    "'네, 켤게요'처럼 한 문장으로 짧게 답한다. 못 한다고 말하지 않는다."
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

    def __init__(self, url, model=COGNITION_MODEL, timeout=20.0,
                 system=SYSTEM_PROMPT, max_chars=MAX_REPLY_CHARS,
                 temperature=COGNITION_TEMPERATURE):
        self.url = url
        self.model = model
        self.timeout = timeout
        self.system = system
        self.max_chars = max_chars
        self.temperature = temperature
        self.name = f"HTTP {url} ({model})"

    def reply(self, text):
        """문장이 완성될 때마다 내보낸다. 실패하면 아무것도 안 내보낸다."""
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": self.system},
                         {"role": "user", "content": text}],
            "stream": True,
            "max_tokens": 200,
            "temperature": self.temperature,
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


def load_llm(url=None, model=COGNITION_MODEL, timeout=20.0):
    return HttpLlm(url, model, timeout) if url else EchoLlm()
