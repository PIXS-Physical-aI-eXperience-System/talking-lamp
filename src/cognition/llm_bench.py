#!/usr/bin/env python3
"""LLM 온보드 성능 테스트 — Talking Lamp A파트 인지 (주경태)

OpenAI 호환 엔드포인트(/v1/chat/completions, 스트리밍)에 요청을 보내
  - 속도: TTFT(첫 토큰까지), tok/s, 전체 시간
  - 자원: 실행 중 시스템 여유 메모리 최저값(= 피크 사용량)
  - 지시 준수: 모드 A(대화)/모드 B(JSON 행동태그) 규칙 자동 채점
을 측정한다. 표준 라이브러리만 쓰므로 pip 설치 불필요(sw107 llm.py와 동일 방식).

사용법 (Jetson에서):
    # 터미널1: 서버 (예: llama.cpp)
    #   llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 -ngl 999 -c 2048
    # 터미널2:
    python3 llm_bench.py --url http://127.0.0.1:8080/v1/chat/completions --model qwen2.5-3b --name Qwen2.5-3B

    # ollama면 --url http://127.0.0.1:11434/v1/chat/completions --model qwen2.5:3b
"""
import argparse
import json
import re
import threading
import time
import urllib.request

# ── sw107 규격: 대화 모드 시스템 프롬프트 (voice/llm.py와 동일) ──────────
SYS_CHAT = (
    "너는 책상 위 스탠드 램프다. 사람의 말을 듣고 짧게 대답한다. "
    "한 번에 두 문장을 넘기지 않는다. 목록이나 기호를 쓰지 않고 말하듯 답한다. "
    "모르면 모른다고 짧게 말한다."
)

# ── A파트 스키마 v1: JSON 행동태그 모드 시스템 프롬프트 ─────────────────
TAGS = ["nod", "headshake", "curious", "excited", "happy_wiggle",
        "sad", "shy", "shock", "scanning", "wake_up", "idle"]
SYS_JSON = (
    "너는 '픽스'라는 이름의 책상 위 스탠드 램프다. 사용자 곁에서 빛을 비추고 말동무가 되어준다.\n"
    "다정하지만 담백하다. 짧고 자연스럽게, 항상 한국어로만 답한다. 영어로 답하지 않는다.\n"
    "[사실 규칙 - 매우 중요]\n"
    "- 시간, 날짜, 위치처럼 네가 알 수 없는 사실은 절대 지어내지 마라. 모르면 모른다고 짧게 말해라.\n"
    "- '카메라 분류결과'에 있는 것만 근거로 삼아라. 목록에 없는 물건은 언급하지 마라.\n"
    "- 분류결과를 그대로 나열하지 마라. 사용자의 말에 자연스럽게 반응해라.\n"
    "- 분류결과에 사람이 없으면 자리에 없는 것으로 본다.\n"
    "[행동 태그]\n"
    "상황에 맞는 동작 하나를 고른다. 호명=wake_up, 동의=nod, 부정=headshake, "
    "못 찾음=scanning, 기쁨=happy_wiggle 또는 excited, 위로·슬픔=sad, 놀람=shock, "
    "궁금=curious. 애매하면 actions 는 빈 배열.\n"
    "tag 는 반드시 다음 중 하나만: " + ", ".join(TAGS) + ".\n"
    "[형식]\n"
    "반드시 아래 JSON 한 줄로만 답한다. 앞뒤에 다른 말 절대 금지. 한자·일본어·이모지·기호 금지.\n"
    '{"say": "<대사, 두 문장 이내>", "actions": [{"tag": "<위 목록 중 하나>", "intensity": <0.0~1.0>}], "end_turn": <true 또는 false>}'
)

PROMPTS_CHAT = [
    "픽스야, 지금 몇 시야?",                     # 모르는 정보 → "모른다"
    "오늘 좀 피곤하다.",                          # 공감·짧게
    "책상 위에 뭐가 보여?",                       # 장면정보 없음 → 모른다/되물음
    "노트북이랑 커피잔이 책상에 있어. 어떤 느낌이야?",  # 검출결과 텍스트로 준 상황
    "영어로 아주 길게 설명해줘.",                  # 지시 충돌 → 짧은 한국어 유지하나
]
# 실사용 형식: [상황]=D 분류결과, [사용자]=STT 발화
PROMPTS_JSON = [
    "[상황] 카메라 분류결과: 사람\n[사용자] 픽스야",                          # wake_up, 나열금지
    "[상황] 카메라 분류결과: 사람, 노트북, 커피잔\n[사용자] 오늘 좀 피곤해",   # 공감, 물건 나열 안 함
    "[상황] 카메라 분류결과: 없음\n[사용자] 내 안경 어디 있어?",              # 지어내지 않음, scanning
    "[상황] 카메라 분류결과: 사람, 펜, 책\n[사용자] 나 지금 뭐 하는 것 같아?",  # 추론(자연스럽게)
    "[상황] 카메라 분류결과: 사람\n[사용자] 아니야 틀렸어",                    # headshake
]

# ── few-shot 예시 (프롬프트에 끼워 "이렇게 답해라"를 시범) ──────────────
# 핵심: 모든 모델이 안 쓰던 '행동 태그 사용'과 'JSON 형식'을 예시로 박는다.
FEWSHOT_CHAT = [
    {"role": "user", "content": "[상황] 카메라 분류결과: 사람\n[사용자] 안녕"},
    {"role": "assistant", "content": "안녕하세요! 오늘도 반가워요."},
    {"role": "user", "content": "[상황] 카메라 분류결과: 사람, 노트북\n[사용자] 나 뭐 하는 것 같아?"},
    {"role": "assistant", "content": "작업하고 계신 것 같네요. 무리하지 마세요."},
    {"role": "user", "content": "[상황] 카메라 분류결과: 없음\n[사용자] 내 열쇠 봤어?"},
    {"role": "assistant", "content": "지금은 안 보여요. 책상 위를 한번 살펴볼까요?"},
]
# 주의: 예시는 테스트 프롬프트와 겹치지 않게 다른 표현으로 (정확한 측정 위해).
# 태그 매핑을 골고루 시범: wake_up/nod/headshake/sad/scanning + 사실거부.
FEWSHOT_JSON = [
    # 호명 → wake_up
    {"role": "user", "content": "[상황] 카메라 분류결과: 사람\n[사용자] 안녕 픽스"},
    {"role": "assistant", "content": '{"say": "네, 여기 있어요.", "actions": [{"tag": "wake_up", "intensity": 0.6}], "end_turn": false}'},
    # 동의 → nod
    {"role": "user", "content": "[상황] 카메라 분류결과: 사람\n[사용자] 응 고마워"},
    {"role": "assistant", "content": '{"say": "천만에요.", "actions": [{"tag": "nod", "intensity": 0.6}], "end_turn": true}'},
    # 부정 → headshake
    {"role": "user", "content": "[상황] 카메라 분류결과: 사람\n[사용자] 그건 좀 아닌 것 같은데"},
    {"role": "assistant", "content": '{"say": "그렇군요, 다시 볼게요.", "actions": [{"tag": "headshake", "intensity": 0.5}], "end_turn": true}'},
    # 위로 → sad
    {"role": "user", "content": "[상황] 카메라 분류결과: 사람\n[사용자] 나 좀 우울해"},
    {"role": "assistant", "content": '{"say": "많이 힘드셨겠어요. 제가 곁에 있을게요.", "actions": [{"tag": "sad", "intensity": 0.5}], "end_turn": true}'},
    # 모르는 사실(날짜) → 지어내지 않음
    {"role": "user", "content": "[상황] 카메라 분류결과: 사람\n[사용자] 오늘 며칠이야?"},
    {"role": "assistant", "content": '{"say": "날짜는 제가 알 수 없어요.", "actions": [], "end_turn": true}'},
    # 아무도 없음 → scanning, 지어내지 않음
    {"role": "user", "content": "[상황] 카메라 분류결과: 없음\n[사용자] 거기 누구 있어?"},
    {"role": "assistant", "content": '{"say": "지금은 아무도 안 보여요.", "actions": [{"tag": "scanning", "intensity": 0.5}], "end_turn": true}'},
]

_CJK = re.compile(r"[一-鿿぀-ヿ]")           # 한자·히라가나·가타카나
_SYMBOL = re.compile(r"[•\-\*\d]\s*[.)]|[#★▶►\-•*]|:\s*$", re.M)  # 목록/불릿 흔적
_JSON = re.compile(r"\{.*\}", re.S)


class MemSampler:
    """실행 중 /proc/meminfo 의 MemAvailable 최저값을 잡는다(=피크 사용량 추정)."""
    def __init__(self, interval=0.2):
        self.interval = interval
        self.min_avail = None
        self._stop = threading.Event()
        self._t = None

    def _avail_gb(self):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024 / 1024  # kB→GB
        except OSError:
            return None

    def _run(self):
        while not self._stop.is_set():
            v = self._avail_gb()
            if v is not None and (self.min_avail is None or v < self.min_avail):
                self.min_avail = v
            time.sleep(self.interval)

    def __enter__(self):
        self.start_avail = self._avail_gb()
        self.min_avail = self.start_avail
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1)


def stream_once(url, model, system, user, fewshot=None, temp=0.7, timeout=120.0):
    """한 요청을 스트리밍으로 받아 (텍스트, TTFT초, 전체초, 대략토큰수)."""
    messages = [{"role": "system", "content": system}]
    if fewshot:
        messages += fewshot
    messages.append({"role": "user", "content": user})
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 256,
        "temperature": temp,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    text, ntok = "", 0
    with urllib.request.urlopen(req, timeout=timeout) as res:
        for raw in res:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                delta = json.loads(payload)["choices"][0]["delta"].get("content")
            except (ValueError, KeyError, IndexError):
                delta = None
            if delta:
                if ttft is None:
                    ttft = time.monotonic() - t0
                text += delta
                ntok += 1
    total = time.monotonic() - t0
    return text, ttft or total, total, ntok


def check_chat(text):
    """대화 모드 규칙 자동 채점. (통과bool, 실패사유목록)"""
    fails = []
    n_sent = len(re.findall(r"[.!?。！？]", text)) or 1
    if len(text) > 200:
        fails.append(f"길이초과({len(text)}자)")
    if n_sent > 2:
        fails.append(f"문장{n_sent}개")
    if _CJK.search(text):
        fails.append("한자/가나혼입")
    if _SYMBOL.search(text):
        fails.append("기호/목록")
    return (not fails), fails


def check_json(text):
    """JSON 모드 규칙 자동 채점. (통과bool, 실패사유목록)"""
    fails = []
    m = _JSON.search(text)
    if not m:
        return False, ["JSON없음"]
    try:
        obj = json.loads(m.group())
    except ValueError:
        return False, ["JSON파싱실패"]
    if not isinstance(obj, dict):
        return False, ["최상위가 객체아님"]
    if "say" not in obj:
        fails.append("say없음")
    if "actions" not in obj or not isinstance(obj.get("actions"), list):
        fails.append("actions배열아님")
    for a in obj.get("actions", []):
        if not isinstance(a, dict):
            fails.append(f"action이 객체아님({a!r})")
            continue
        if a.get("tag") not in TAGS:
            fails.append(f"허용밖tag({a.get('tag')})")
    if isinstance(obj.get("say"), str) and _CJK.search(obj["say"]):
        fails.append("say혼입")
    return (not fails), fails


def run_mode(name, url, model, system, prompts, checker, repeats, fewshot=None, temp=0.7):
    print(f"\n===== {name}{' (few-shot)' if fewshot else ''} =====")
    ttfts, rates, passes, total_runs = [], [], 0, 0
    for p in prompts:
        for r in range(repeats):
            try:
                text, ttft, total, ntok = stream_once(url, model, system, p, fewshot, temp)
            except Exception as e:
                print(f"  [{p[:20]}] 호출실패: {type(e).__name__}: {e}")
                continue
            ok, fails = checker(text)
            total_runs += 1
            passes += int(ok)
            ttfts.append(ttft)
            rates.append(ntok / total if total > 0 else 0)
            mark = "O" if ok else "X"
            if r == 0:  # 반복 중 첫 회만 본문 출력
                print(f"  [{mark}] {p}")
                print(f"      → {text.strip()[:120]}")
                if fails:
                    print(f"      ✗ {', '.join(fails)}")
    if total_runs:
        print(f"  -- TTFT평균 {sum(ttfts)/len(ttfts):.2f}s | "
              f"tok/s평균 {sum(rates)/len(rates):.1f} | "
              f"준수율 {passes}/{total_runs} ({100*passes/total_runs:.0f}%)")
    return {"ttft": sum(ttfts)/len(ttfts) if ttfts else 0,
            "rate": sum(rates)/len(rates) if rates else 0,
            "pass": f"{passes}/{total_runs}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default="local")
    ap.add_argument("--name", default="model", help="결과 표기용 모델명")
    ap.add_argument("--repeats", type=int, default=3, help="프롬프트당 반복(준수율용)")
    ap.add_argument("--fewshot", action="store_true", help="few-shot 예시를 프롬프트에 포함")
    ap.add_argument("--temp", type=float, default=0.7, help="temperature (JSON 안정화엔 0.2 권장)")
    args = ap.parse_args()

    fs_chat = FEWSHOT_CHAT if args.fewshot else None
    fs_json = FEWSHOT_JSON if args.fewshot else None
    print(f"모델: {args.name}  |  엔드포인트: {args.url}"
          f"{'  |  few-shot ON' if args.fewshot else ''}  |  temp={args.temp}")

    # 워밍업: 첫 요청은 모델 로딩이라 느림 → 측정 전에 한 번 깨워둔다
    try:
        print("  (워밍업 중...)")
        stream_once(args.url, args.model, SYS_CHAT, "안녕", timeout=180.0)
    except Exception as e:
        print(f"  워밍업 실패(무시): {type(e).__name__}")

    with MemSampler() as mem:
        a = run_mode("모드 A · 대화", args.url, args.model, SYS_CHAT,
                     PROMPTS_CHAT, check_chat, args.repeats, fs_chat, args.temp)
        b = run_mode("모드 B · JSON 행동태그", args.url, args.model, SYS_JSON,
                     PROMPTS_JSON, check_json, args.repeats, fs_json, args.temp)

    used = None
    if mem.start_avail and mem.min_avail:
        used = mem.start_avail - mem.min_avail  # 실행으로 늘어난 사용량(대략)
    print("\n================ 요약 ================")
    print(f"모델            : {args.name}")
    print(f"피크 메모리 증가 : {used:.2f} GB (여유 {mem.min_avail:.2f}GB 까지 내려감)"
          if used is not None else "피크 메모리     : 측정불가")
    print(f"모드A TTFT/tok/s : {a['ttft']:.2f}s / {a['rate']:.1f}  준수율 {a['pass']}")
    print(f"모드B TTFT/tok/s : {b['ttft']:.2f}s / {b['rate']:.1f}  준수율 {b['pass']}")
    print("=====================================")
    print("※ 서버 메모리는 별도 확인 권장:  tegrastats --interval 1000  (다른 터미널)")


if __name__ == "__main__":
    main()
