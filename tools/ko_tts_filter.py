#!/usr/bin/env python3
"""언어 혼입 후처리 필터 — A(주경태) → C(최승원) 인계물.

목적:
    InternVL이 한국어 답변에 중국어(한자)·일본어(가나)를 섞어 낼 때가 있다.
    C(최승원)의 TTS는 한국어·영어만 처리하므로, 이물질 문자가 그대로 넘어가면
    음성 합성이 깨진다. 이 필터가 **TTS 직전 마지막 안전망**으로, TTS가 절대
    한자/가나를 보지 못하도록 보장한다.

방어선 2단 (전체 그림):
    1차 예방 — 디코딩 단계에서 한자/가나 토큰 자체를 억제 (vlm_lite.py --ko-only).
               대부분 여기서 안 나온다.
    2차 안전망 — 이 후처리 필터. 1차를 뚫고 새어나온 잔여물을 제거 + 재생성 플래그.

C(최승원)와의 계약(인계 규칙):
    - TTS에 넘기기 직전 sanitize_for_tts(text) 를 통과시킨다.
    - 반환된 clean 문자열에는 한자/가나가 없음을 보장한다.
    - report.needs_regen 이 True면 A(주경태)의 런타임이 VLM 재생성을 고려한다
      (문장이 많이 깨졌다는 신호). TTS는 그와 무관하게 clean 을 써도 안전하다.

API:
    clean, report = sanitize_for_tts("안녕 参 하세요")
    # clean == "안녕 하세요" (또는 정리된 형태), report.was_contaminated == True
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass, field


# --- 코드포인트 판정 ---------------------------------------------------------
def _in(cp: int, lo: int, hi: int) -> bool:
    return lo <= cp <= hi


def is_hanja(cp: int) -> bool:
    return (_in(cp, 0x4E00, 0x9FFF) or _in(cp, 0x3400, 0x4DBF)
            or _in(cp, 0x20000, 0x2A6DF) or _in(cp, 0xF900, 0xFAFF))


def is_kana(cp: int) -> bool:
    return (_in(cp, 0x3040, 0x309F) or _in(cp, 0x30A0, 0x30FF)
            or _in(cp, 0x31F0, 0x31FF) or _in(cp, 0xFF66, 0xFF9D))


def _is_foreign(ch: str) -> bool:
    cp = ord(ch)
    return is_hanja(cp) or is_kana(cp)


@dataclass
class FilterReport:
    was_contaminated: bool = False
    removed: list[str] = field(default_factory=list)  # 제거된 이물질 문자들
    n_removed: int = 0
    orig_len: int = 0
    needs_regen: bool = False  # 많이 깨져서 재생성이 나을 때

    def as_dict(self) -> dict:
        return {
            "was_contaminated": self.was_contaminated,
            "n_removed": self.n_removed,
            "removed": "".join(self.removed),
            "needs_regen": self.needs_regen,
        }


# 재생성 권고 임계: 이물질이 전체의 이 비율을 넘으면 문장이 많이 깨진 것으로 본다.
REGEN_RATIO_THRESHOLD = 0.15


def sanitize_for_tts(text: str, regen_threshold: float = REGEN_RATIO_THRESHOLD) -> tuple[str, FilterReport]:
    """한자/가나를 제거해 TTS 안전 문자열을 만든다.

    반환: (clean_text, report)
      - clean_text: 한자/가나가 제거된 문자열. 제거로 생긴 이중 공백·
                    구두점 앞 공백을 정리한다.
      - report: 오염 여부/제거 수/재생성 권고.
    """
    text = unicodedata.normalize("NFC", text)
    report = FilterReport(orig_len=len([c for c in text if not c.isspace()]))

    kept_chars = []
    for ch in text:
        if _is_foreign(ch):
            report.removed.append(ch)
        else:
            kept_chars.append(ch)
    clean = "".join(kept_chars)

    report.n_removed = len(report.removed)
    report.was_contaminated = report.n_removed > 0

    if report.was_contaminated:
        # 제거 흔적 정리: 여러 공백 → 하나, 구두점 앞 공백 제거, 양끝 트림
        clean = re.sub(r"[ \t]{2,}", " ", clean)
        clean = re.sub(r"\s+([,.!?;:)\]}」』】”’])", r"\1", clean)
        clean = re.sub(r"([(\[{「『【“‘])\s+", r"\1", clean)
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        clean = clean.strip()

        denom = report.orig_len or 1
        report.needs_regen = (report.n_removed / denom) > regen_threshold

    return clean, report


def contains_foreign(text: str) -> bool:
    """빠른 판정 — 재생성/로깅 결정용."""
    return any(_is_foreign(ch) for ch in text)


def _self_test() -> int:
    cases = [
        # (입력, 기대 clean, 기대 오염여부)
        ("안녕하세요, 반갑습니다.", "안녕하세요, 반갑습니다.", False),
        ("책상 위 参 키보드를 정리할까요?", "책상 위 키보드를 정리할까요?", True),
        ("오늘 날씨가 良い 좋네요", "오늘 날씨가 좋네요", True),
        ("キーボード를 정리했어요.", "를 정리했어요.", True),
        ("English is fine 123", "English is fine 123", False),
    ]
    ok = True
    for text, expected_clean, expected_contam in cases:
        clean, rep = sanitize_for_tts(text)
        pass_clean = clean == expected_clean
        pass_contam = rep.was_contaminated == expected_contam
        pass_no_foreign = not contains_foreign(clean)  # 계약: clean엔 이물질 없음
        mark = "OK" if (pass_clean and pass_contam and pass_no_foreign) else "FAIL"
        if mark == "FAIL":
            ok = False
        print(f"  [{mark}] {text!r}")
        print(f"        → clean={clean!r} (기대 {expected_clean!r})")
        print(f"        오염={rep.was_contaminated} 제거={rep.n_removed} 재생성={rep.needs_regen} 이물질잔존={contains_foreign(clean)}")
    print("\n결과:", "전부 통과" if ok else "실패 있음")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="언어 혼입 후처리 필터 (TTS 보호)")
    ap.add_argument("--text", help="정제할 텍스트")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if args.text is not None:
        clean, rep = sanitize_for_tts(args.text)
        print("clean :", clean)
        print("report:", rep.as_dict())
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
