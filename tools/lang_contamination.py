#!/usr/bin/env python3
"""한국어 출력 언어 혼입(language contamination) 측정기.

VLM이 한국어로 답해야 하는데 중간에 한자(중국어)나 가나(일본어)를
섞어 내는 빈도를 정량화한다. A파트 VLM 후보 선정에서 '한국어 품질'
지표를 숫자로 만들기 위한 도구.

사용:
    # 텍스트 한 줄 검사
    python lang_contamination.py --text "안녕하세요 今日は良い天気"

    # JSONL 파일 검사 (각 줄에 {"response": "..."} 형태)
    python lang_contamination.py --jsonl responses.jsonl --field response

    # 여러 모델 응답 비교 (파일당 한 모델, 줄당 응답 하나)
    python lang_contamination.py --files internvl.txt smolvlm.txt gemma.txt

측정 지표:
    - contaminated_rate : 이물질 문자를 하나라도 포함한 응답 비율(%)  ← 핵심 지표
    - char_contam_rate  : 전체 문자 중 이물질(한자/가나) 비율(%)
    - by_script         : 스크립트별(한자/히라가나/가타카나) 등장 응답 수
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass, field


# 유니코드 범위 정의 --------------------------------------------------------
# 한국어 발화체 대화에서는 한자·가나가 거의 나오지 않으므로 모두 '이물질'로 본다.
# (한자 병기가 정당한 문맥이 있다면 --allow-hanja 로 한자를 제외할 수 있다.)
def _in(cp: int, lo: int, hi: int) -> bool:
    return lo <= cp <= hi


def is_hangul(cp: int) -> bool:
    return (
        _in(cp, 0xAC00, 0xD7A3)   # 완성형 음절
        or _in(cp, 0x1100, 0x11FF)  # 자모
        or _in(cp, 0x3130, 0x318F)  # 호환 자모
        or _in(cp, 0xA960, 0xA97F)  # 확장 자모 A
        or _in(cp, 0xD7B0, 0xD7FF)  # 확장 자모 B
    )


def is_hanja(cp: int) -> bool:
    return (
        _in(cp, 0x4E00, 0x9FFF)   # CJK 통합 한자
        or _in(cp, 0x3400, 0x4DBF)  # 확장 A
        or _in(cp, 0x20000, 0x2A6DF)  # 확장 B
        or _in(cp, 0xF900, 0xFAFF)  # 호환 한자
    )


def is_hiragana(cp: int) -> bool:
    return _in(cp, 0x3040, 0x309F)


def is_katakana(cp: int) -> bool:
    return _in(cp, 0x30A0, 0x30FF) or _in(cp, 0x31F0, 0x31FF) or _in(cp, 0xFF66, 0xFF9D)


SCRIPTS = {
    "hanja": is_hanja,
    "hiragana": is_hiragana,
    "katakana": is_katakana,
}


@dataclass
class Report:
    total: int = 0
    contaminated: int = 0
    total_chars: int = 0
    contam_chars: int = 0
    by_script: dict[str, int] = field(default_factory=lambda: {k: 0 for k in SCRIPTS})
    samples: list[tuple[str, str]] = field(default_factory=list)  # (스크립트태그, 원문 일부)

    def add(self, text: str, allow_hanja: bool = False, keep_samples: int = 5) -> None:
        self.total += 1
        scripts_here = set()
        contam_chars = 0
        # 공백을 뺀 실질 문자만 분모로
        real_chars = [c for c in text if not c.isspace()]
        self.total_chars += len(real_chars)
        for c in real_chars:
            cp = ord(c)
            for name, fn in SCRIPTS.items():
                if name == "hanja" and allow_hanja:
                    continue
                if fn(cp):
                    scripts_here.add(name)
                    contam_chars += 1
                    break
        if scripts_here:
            self.contaminated += 1
            self.contam_chars += contam_chars
            for s in scripts_here:
                self.by_script[s] += 1
            if len(self.samples) < keep_samples:
                snippet = text.strip().replace("\n", " ")
                if len(snippet) > 80:
                    snippet = snippet[:77] + "..."
                self.samples.append(("+".join(sorted(scripts_here)), snippet))

    def summary(self) -> dict:
        return {
            "responses": self.total,
            "contaminated_responses": self.contaminated,
            "contaminated_rate_pct": round(100 * self.contaminated / self.total, 2) if self.total else 0.0,
            "char_contam_rate_pct": round(100 * self.contam_chars / self.total_chars, 3) if self.total_chars else 0.0,
            "by_script": self.by_script,
        }


def print_report(name: str, rep: Report) -> None:
    s = rep.summary()
    print(f"\n=== {name} ===")
    print(f"  응답 수                : {s['responses']}")
    print(f"  혼입 응답 수           : {s['contaminated_responses']}")
    print(f"  ▶ 혼입률(응답)         : {s['contaminated_rate_pct']} %   (핵심 지표)")
    print(f"  ▶ 혼입률(문자)         : {s['char_contam_rate_pct']} %")
    print(f"  스크립트별(등장 응답수): "
          f"한자 {s['by_script']['hanja']}, "
          f"히라가나 {s['by_script']['hiragana']}, "
          f"가타카나 {s['by_script']['katakana']}")
    if rep.samples:
        print("  예시:")
        for tag, snip in rep.samples:
            print(f"    [{tag}] {snip}")


def read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def read_jsonl(path: str, field_name: str) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            val = obj.get(field_name)
            if isinstance(val, str):
                out.append(val)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="한국어 출력 언어 혼입 측정기")
    ap.add_argument("--text", help="검사할 텍스트 한 개")
    ap.add_argument("--files", nargs="+", help="줄당 응답 하나인 텍스트 파일들(모델별 비교)")
    ap.add_argument("--jsonl", help="JSONL 파일")
    ap.add_argument("--field", default="response", help="JSONL에서 읽을 필드명 (기본: response)")
    ap.add_argument("--allow-hanja", action="store_true", help="한자는 혼입에서 제외(가나만 카운트)")
    args = ap.parse_args(argv)

    if not any([args.text, args.files, args.jsonl]):
        ap.print_help()
        return 1

    if args.text is not None:
        rep = Report()
        rep.add(unicodedata.normalize("NFC", args.text), allow_hanja=args.allow_hanja)
        print_report("text", rep)

    if args.jsonl:
        rep = Report()
        for t in read_jsonl(args.jsonl, args.field):
            rep.add(unicodedata.normalize("NFC", t), allow_hanja=args.allow_hanja)
        print_report(args.jsonl, rep)

    if args.files:
        for path in args.files:
            rep = Report()
            for t in read_lines(path):
                rep.add(unicodedata.normalize("NFC", t), allow_hanja=args.allow_hanja)
            print_report(path, rep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
