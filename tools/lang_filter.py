#!/usr/bin/env python3
"""Strip Chinese/Japanese characters that leak into Korean/English VLM output.

TTS(MeloTTS 한국어)는 한국어·영어만 처리하므로, 중국어 한자·일본어 가나가 그대로
넘어가면 합성이 깨진다(docs/진행-순서.md A파트 5번, 파트-분배.md 249행).
정교한 재작문이 아니라 TTS를 보호하는 최소 안전망임 — 낱말 중간에 낀 한자를
지워서 발음 가능한 문자열로 만드는 것까지만 한다.
"""
import re
import unicodedata

# 한자(CJK Unified Ideographs + ExtA), 일본어 가나만 걷어낸다.
# 한글/영문/숫자/구두점/공백은 그대로 둔다.
_STRIP_RANGES = [
    (0x3400, 0x4DBF),   # CJK Ext A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs (한자)
    (0x3040, 0x309F),   # 히라가나
    (0x30A0, 0x30FF),   # 가타카나
    (0xF900, 0xFAFF),   # CJK 호환 한자
]


def _is_strip_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _STRIP_RANGES)


def strip_cjk_leakage(text: str) -> str:
    """중국어 한자·일본어 가나를 제거하고, 생긴 공백을 정리한다."""
    text = unicodedata.normalize('NFC', text)
    out = ''.join(' ' if _is_strip_char(ch) else ch for ch in text)
    out = re.sub(r'[ \t]+', ' ', out)
    out = re.sub(r' ?([.,!?])', r'\1', out)
    return out.strip()


def has_leakage(text: str) -> bool:
    return any(_is_strip_char(ch) for ch in text)


if __name__ == '__main__':
    import sys
    samples = [
        '이 사진에는 테이블에 있는 충전坞가 보입니다. 충전坞에는 여러 개의 충전구가 있습니다.',
        '이 장면은 책상이나 테이블에 있는 모습으로 보입니다. 책상 위에는 전원插座이 여러 개의插座에 연결되어',
        'A black power bank is placed on a white power strip.',
    ]
    for s in (sys.argv[1:] or samples):
        print(f'전: {s}')
        print(f'후: {strip_cjk_leakage(s)}')
        print(f'혼입감지: {has_leakage(s)}')
        print()
