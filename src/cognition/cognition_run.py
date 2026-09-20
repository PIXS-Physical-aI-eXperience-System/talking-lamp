#!/usr/bin/env python3
"""인지 모듈 (A파트 · 주경태) — LLM + 스키마(파서) 통합.

분류라벨 + 사용자말  →  프롬프트 조립  →  LLM(HyperCLOVA)  →  파서  →  {say, actions}

이게 A파트의 완성형 모듈이다. B(통합)는 이 함수를 부르고, 나온 say 는 C(음성),
actions 는 E(모터)로 라우팅한다. D(비전) 분류기의 라벨은 `labels` 인자로 들어온다.

    from cognition_run import respond
    r = respond(["사람", "노트북"], "오늘 좀 피곤해")
    r.say       # → C(TTS)
    r.actions   # → E(모터)

검증된 설정(2026-09-21 벤치마크): HyperCLOVA 1.5B, temp 0.2, few-shot → JSON 100%.

CLI:
    python3 cognition_run.py --labels "사람,노트북" --text "오늘 좀 피곤해"
    python3 cognition_run.py            # 대화형 (라벨/말 직접 입력)
"""
import argparse
import json
import urllib.request

from cognition_parser import ALLOWED_TAGS, parse

# ── 확정 설정 ──────────────────────────────────────────────────────
URL = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "hf.co/kexplo/HyperCLOVAX-SEED-Text-Instruct-1.5B-Q4_K_M-GGUF"
TEMP = 0.2
TAG_ORDER = ["nod", "headshake", "curious", "excited", "happy_wiggle",
             "sad", "shy", "shock", "scanning", "wake_up", "idle"]

# ── 검증된 시스템 프롬프트 (llm_bench v3) ──────────────────────────
SYSTEM = (
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
    "tag 는 반드시 다음 중 하나만: " + ", ".join(TAG_ORDER) + ".\n"
    "[형식]\n"
    "반드시 아래 JSON 한 줄로만 답한다. 앞뒤에 다른 말 절대 금지. 한자·일본어·이모지·기호 금지.\n"
    '{"say": "<대사, 두 문장 이내>", "actions": [{"tag": "<위 목록 중 하나>", "intensity": <0.0~1.0>}], "end_turn": <true 또는 false>}'
)

# ── 평문용 시스템 프롬프트 (파서 없이 쓰는 버전) ─────────────────────
# 음성(C/최승원)이 파서 없이 바로 스트리밍하려면 이걸 llm.py 의 SYSTEM_PROMPT
# 에 넣는다. JSON(SYSTEM)과 달리 평문으로 답해 TTS 로 바로 흘릴 수 있다.
# 같은 접지 규칙(한국어만·환각 억제·나열 금지)을 유지하되 행동태그는 없다.
SYSTEM_PLAIN = (
    "너는 '픽스'라는 이름의 책상 위 스탠드 램프다. 사용자 곁에서 빛을 비추고 말동무가 되어준다.\n"
    "다정하지만 담백하다. 항상 한국어로만, 두 문장 이내로 짧게 말한다. 영어로 답하지 않는다.\n"
    "목록·기호·이모지를 쓰지 않고 말하듯 답한다.\n"
    "시간·날짜·위치처럼 알 수 없는 사실은 절대 지어내지 말고 모른다고 짧게 말한다.\n"
    "'카메라 분류결과'에 있는 것만 근거로 삼고, 목록에 없는 물건은 언급하지 않는다. "
    "분류결과를 그대로 나열하지 않고 사용자의 말에 자연스럽게 반응한다."
)

FEWSHOT = [
    {"role": "user", "content": "[상황] 카메라 분류결과: 사람\n[사용자] 안녕 픽스"},
    {"role": "assistant", "content": '{"say": "네, 여기 있어요.", "actions": [{"tag": "wake_up", "intensity": 0.6}], "end_turn": false}'},
    {"role": "user", "content": "[상황] 카메라 분류결과: 사람\n[사용자] 응 고마워"},
    {"role": "assistant", "content": '{"say": "천만에요.", "actions": [{"tag": "nod", "intensity": 0.6}], "end_turn": true}'},
    {"role": "user", "content": "[상황] 카메라 분류결과: 사람\n[사용자] 그건 좀 아닌 것 같은데"},
    {"role": "assistant", "content": '{"say": "그렇군요, 다시 볼게요.", "actions": [{"tag": "headshake", "intensity": 0.5}], "end_turn": true}'},
    {"role": "user", "content": "[상황] 카메라 분류결과: 사람\n[사용자] 나 좀 우울해"},
    {"role": "assistant", "content": '{"say": "많이 힘드셨겠어요. 제가 곁에 있을게요.", "actions": [{"tag": "sad", "intensity": 0.5}], "end_turn": true}'},
    {"role": "user", "content": "[상황] 카메라 분류결과: 사람\n[사용자] 오늘 며칠이야?"},
    {"role": "assistant", "content": '{"say": "날짜는 제가 알 수 없어요.", "actions": [], "end_turn": true}'},
    {"role": "user", "content": "[상황] 카메라 분류결과: 없음\n[사용자] 거기 누구 있어?"},
    {"role": "assistant", "content": '{"say": "지금은 아무도 안 보여요.", "actions": [{"tag": "scanning", "intensity": 0.5}], "end_turn": true}'},
]


def build_user(labels, text):
    """D의 분류라벨 + 사용자말 → LLM 입력 한 덩어리."""
    lab = ", ".join(labels) if labels else "없음"
    return f"[상황] 카메라 분류결과: {lab}\n[사용자] {text}"


def respond(labels, text, url=URL, model=MODEL, temp=TEMP, timeout=120.0):
    """분류라벨 + 사용자말 → 파싱된 응답(ParsedResponse). 절대 예외 안 던짐."""
    messages = [{"role": "system", "content": SYSTEM}] + FEWSHOT + \
               [{"role": "user", "content": build_user(labels, text)}]
    body = json.dumps({
        "model": model, "messages": messages,
        "stream": False, "max_tokens": 256, "temperature": temp,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            obj = json.loads(res.read())
        raw = obj["choices"][0]["message"]["content"]
    except Exception as e:
        # LLM 실패해도 램프가 죽지 않게 안전 응답
        from cognition_parser import ParsedResponse
        return ParsedResponse(say="", actions=[], end_turn=True, ok=False,
                              notes=[f"LLM 호출 실패: {type(e).__name__}"]), ""
    return parse(raw), raw


def _show(labels, text):
    r, raw = respond(labels, text)
    print(f"\n[입력] 분류={labels or '없음'} | 말=\"{text}\"")
    print(f"  LLM 원문 : {raw.strip()[:150]}")
    print(f"  → say    : {r.say!r}")
    print(f"  → actions: {r.actions}   (E 모터로)")
    print(f"  → end_turn: {r.end_turn}")
    if r.notes:
        print(f"  (파서 방어: {r.notes})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="", help='쉼표구분, 예: "사람,노트북"')
    ap.add_argument("--text", default=None, help="사용자 발화")
    args = ap.parse_args()

    if args.text is not None:
        labels = [x.strip() for x in args.labels.split(",") if x.strip()]
        _show(labels, args.text)
    else:
        # 데모: 대표 케이스 몇 개
        print("=== 인지 모듈 데모 (LLM + 파서 통합) ===")
        _show(["사람"], "픽스야")
        _show(["사람", "노트북", "커피잔"], "오늘 좀 피곤해")
        _show([], "내 안경 어디 있어?")
        _show(["사람", "펜", "책"], "이거 맞아?")
        print("\n대화형으로 쓰려면: python3 cognition_run.py --labels \"사람,노트북\" --text \"안녕\"")
