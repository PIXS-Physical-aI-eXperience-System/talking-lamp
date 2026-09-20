#!/usr/bin/env python3
"""행동 태그 파서 — Talking Lamp A파트 인지 (주경태)

LLM 이 뱉은 원문(문자열)을 [행동태그-스키마-v1] 로 안전하게 뜯어낸다.
LLM 은 완벽한 JSON 을 주지 않는다(코드펜스, 앞뒤 잡소리, 잘림, 없는 태그,
한자 혼입 등) — 파서가 전부 방어해서 절대 죽지 않고 항상 재생 가능한 결과를 낸다.

    from cognition_parser import parse
    r = parse(llm_raw_text)
    r.say         # C(TTS)로 보낼 대사 (한국어만, 정화됨)
    r.actions     # E(모터)로 보낼 [{"tag","intensity"}...]  (검증됨)
    r.end_turn    # B 상태머신용

방어 규칙(스키마 4절):
  1) enum 밖 tag → idle 로 대체
  2) JSON 파싱 실패 → say 만 최대한 살리고 actions=[]
  3) say 정화 → 한자·가나·이모지·괄호설명·목록기호 제거
  4) actions 최대 2개
  5) intensity 0.0~1.0 클램프
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# E 의 모션 CSV 와 1:1 (src/motion/primitives.py CLIP_NAMES)
ALLOWED_TAGS = {
    "nod", "headshake", "curious", "excited", "happy_wiggle",
    "sad", "shy", "shock", "scanning", "wake_up", "idle",
}
MAX_ACTIONS = 2
MAX_SAY_CHARS = 200
DEFAULT_INTENSITY = 0.5

# 제거 대상: 한자, 히라가나·가타카나, 이모지/기호픽토그램
_CJK = re.compile(
    r"[一-鿿㐀-䶿぀-ヿｦ-ﾟ]"
)
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF\U00002B00-\U00002BFF️]"
)
# 괄호 안 지문: (고개를 끄덕이며), [밝게], （…）
_STAGE = re.compile(r"[\(\[（][^\)\]）]*[\)\]）]")
# 줄머리 목록/불릿 기호
_BULLET = re.compile(r"^[\s]*[-*•·▶►#]+[\s]*", re.M)
# 코드펜스
_FENCE = re.compile(r"```[a-zA-Z]*|```")
# 깨진 JSON 에서라도 say 값만 뽑기
_SAY_FALLBACK = re.compile(r'"say"\s*:\s*"((?:[^"\\]|\\.)*)"')


@dataclass
class ParsedResponse:
    say: str = ""
    actions: list = field(default_factory=list)   # [{"tag","intensity"}]
    end_turn: bool = True
    ok: bool = True          # JSON 을 정상 파싱했나 (False = fallback 경로)
    notes: list = field(default_factory=list)     # 어떤 방어가 발동했나(디버그)

    def to_dict(self):
        return {"say": self.say, "actions": self.actions,
                "end_turn": self.end_turn}


def sanitize_say(text: str) -> str:
    """대사 정화 — TTS 로 넘어가면 안 되는 것 제거."""
    if not text:
        return ""
    text = _STAGE.sub("", text)      # 괄호 지문
    text = _CJK.sub("", text)        # 한자·가나
    text = _EMOJI.sub("", text)      # 이모지
    text = _BULLET.sub("", text)     # 목록 기호
    text = text.replace("\\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_SAY_CHARS:
        text = text[:MAX_SAY_CHARS].rstrip()
    return text


def _clamp_intensity(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return DEFAULT_INTENSITY
    return max(0.0, min(1.0, v))


def _normalize_actions(raw_actions, notes) -> list:
    """actions 배열 검증: tag enum 강제, intensity 클램프, 개수 제한."""
    out = []
    if not isinstance(raw_actions, list):
        notes.append("actions가 배열이 아님 → []")
        return out
    for a in raw_actions:
        if not isinstance(a, dict):
            continue
        tag = a.get("tag")
        if tag not in ALLOWED_TAGS:
            notes.append(f"허용밖 tag '{tag}' → idle")
            tag = "idle"
        out.append({"tag": tag,
                    "intensity": _clamp_intensity(a.get("intensity",
                                                         DEFAULT_INTENSITY))})
        if len(out) >= MAX_ACTIONS:
            if len(raw_actions) > MAX_ACTIONS:
                notes.append(f"actions {len(raw_actions)}개 → {MAX_ACTIONS}개로 제한")
            break
    return out


def _extract_json(raw: str):
    """원문에서 JSON 객체를 최대한 뽑아 dict 로. 실패하면 None."""
    if not raw:
        return None
    s = _FENCE.sub("", raw).strip()
    # 1) 통째로
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass
    # 2) 첫 { ~ 짝 맞는 } 까지 잘라서
    start = s.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        break
    return None


def parse(raw: str) -> ParsedResponse:
    """LLM 원문 → ParsedResponse. 절대 예외를 던지지 않는다."""
    notes = []
    obj = _extract_json(raw)

    if obj is not None:
        say = sanitize_say(str(obj.get("say", "")))
        actions = _normalize_actions(obj.get("actions", []), notes)
        end_turn = obj.get("end_turn", True)
        end_turn = bool(end_turn) if isinstance(end_turn, (bool, int)) else True
        if not say:
            notes.append("say 비어있음")
        return ParsedResponse(say=say, actions=actions, end_turn=end_turn,
                              ok=True, notes=notes)

    # --- fallback: JSON 실패 ---
    notes.append("JSON 파싱 실패 → fallback")
    m = _SAY_FALLBACK.search(raw or "")
    if m:                                   # 깨진 JSON 속 say 만 건짐
        say = sanitize_say(m.group(1))
    else:                                   # JSON 흔적 통째로 지우고 본문만
        stripped = _FENCE.sub("", raw or "")
        stripped = re.sub(r"[{}\[\]\"]", "", stripped)
        stripped = re.sub(r"\b(say|actions|tag|intensity|end_turn)\b\s*:?", "",
                          stripped)
        say = sanitize_say(stripped)
    return ParsedResponse(say=say, actions=[], end_turn=True,
                          ok=False, notes=notes)


# ─────────────────────────── 자체 테스트 ───────────────────────────
if __name__ == "__main__":
    cases = [
        ('정상',
         '{"say": "네, 맞아요.", "actions": [{"tag":"nod","intensity":0.7}], "end_turn": true}'),
        ('코드펜스',
         '```json\n{"say":"안녕하세요!","actions":[{"tag":"wake_up"}]}\n```'),
        ('앞뒤 잡소리',
         '물론이죠! {"say":"반가워요","actions":[]} 도움이 됐길 바라요'),
        ('없는 tag',
         '{"say":"춤춰볼게요","actions":[{"tag":"dance","intensity":2}]}'),
        ('한자·이모지·괄호지문',
         '{"say":"안녕하세요 你好 😊 (고개를 끄덕이며)","actions":[{"tag":"nod"}]}'),
        ('actions 3개 초과',
         '{"say":"와","actions":[{"tag":"nod"},{"tag":"excited"},{"tag":"happy_wiggle"}]}'),
        ('JSON 깨짐 - say만 건지기',
         '{"say":"좀 피곤하네요", "actions":[{"tag":"sad", inten'),
        ('완전 평문 (형식 무시)',
         '그냥 안녕하세요 반갑습니다'),
    ]
    for name, raw in cases:
        r = parse(raw)
        print(f"[{name}]  ok={r.ok}")
        print(f"   say     : {r.say!r}")
        print(f"   actions : {r.actions}")
        print(f"   end_turn: {r.end_turn}")
        if r.notes:
            print(f"   notes   : {r.notes}")
        print()
