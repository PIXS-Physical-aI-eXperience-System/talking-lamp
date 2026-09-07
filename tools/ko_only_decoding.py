#!/usr/bin/env python3
"""InternVL(HuggingFace transformers) 한국어 출력 언어 혼입 억제 유틸.

문제:
    InternVL3.5-2B는 중국어권 데이터로 학습돼 한국어 답변 중간에 한자/가나가
    섞인다(language contamination). '한국어로만 답하라'는 프롬프트로 누르면
    모델을 낯선 분포로 밀어넣어 같은 말을 반복하는 루프가 늘어난다(트레이드오프).

접근:
    프롬프트로 분포 전체를 왜곡하는 대신, **디코딩 단계에서 한자/가나가 들어간
    토큰만 골라 logit을 -inf로 막는다.** 모델은 그 자리에서 '차선의 한국어 토큰'을
    자연스럽게 고르므로 문장 유창성 손상이 적다. 반복 루프는 no_repeat_ngram_size +
    repetition_penalty 로 별도로 막는다.

사용 (transformers generate):
    from ko_only_decoding import build_suppress_ids, KoOnlyLogitsProcessor
    from transformers import LogitsProcessorList

    suppress_ids = build_suppress_ids(tokenizer)          # 1회 계산 후 캐시
    processors = LogitsProcessorList([KoOnlyLogitsProcessor(suppress_ids)])

    out = model.generate(
        **inputs,
        logits_processor=processors,
        no_repeat_ngram_size=3,     # 반복 루프 방지
        repetition_penalty=1.1,
        do_sample=False,            # 짧은 발화체엔 greedy가 안정적
    )

일부 InternVL 래퍼는 model.chat(...) 안에서 generate를 부른다. 그 경우
generation_config 에 위 인자를 넣거나, model.chat(..., **gen_kwargs) 로 전달한다.
"""
from __future__ import annotations

import argparse
import sys


# --- 코드포인트 판정 (lang_contamination.py 와 동일 기준) -------------------
def _in(cp: int, lo: int, hi: int) -> bool:
    return lo <= cp <= hi


def is_hanja(cp: int) -> bool:
    return (
        _in(cp, 0x4E00, 0x9FFF)
        or _in(cp, 0x3400, 0x4DBF)
        or _in(cp, 0x20000, 0x2A6DF)
        or _in(cp, 0xF900, 0xFAFF)
    )


def is_kana(cp: int) -> bool:
    return (
        _in(cp, 0x3040, 0x309F)      # 히라가나
        or _in(cp, 0x30A0, 0x30FF)   # 가타카나
        or _in(cp, 0x31F0, 0x31FF)
        or _in(cp, 0xFF66, 0xFF9D)   # 반각 가타카나
    )


def has_foreign_cjk(text: str, block_hanja: bool = True, block_kana: bool = True) -> bool:
    for ch in text:
        cp = ord(ch)
        if block_hanja and is_hanja(cp):
            return True
        if block_kana and is_kana(cp):
            return True
    return False


def build_suppress_ids(tokenizer, block_hanja: bool = True, block_kana: bool = True) -> list[int]:
    """토크나이저 vocab을 훑어 한자/가나가 포함된 토큰 id 목록을 만든다.

    모델·토크나이저당 1회만 계산하면 되므로 결과를 캐시해 재사용할 것.
    """
    suppress: list[int] = []
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    for tid in range(vocab_size):
        try:
            piece = tokenizer.decode([tid])
        except Exception:
            continue
        if piece and has_foreign_cjk(piece, block_hanja, block_kana):
            suppress.append(tid)
    return suppress


class KoOnlyLogitsProcessor:
    """suppress_ids 에 해당하는 토큰의 logit을 -inf로 만드는 LogitsProcessor.

    transformers.LogitsProcessor 를 상속하지 않아도 generate()가 요구하는
    __call__(input_ids, scores) 시그니처만 맞추면 동작한다(덕 타이핑).
    """

    def __init__(self, suppress_ids: list[int]):
        self.suppress_ids = suppress_ids
        self._index = None  # torch tensor, 최초 호출 때 scores.device 기준으로 생성

    def __call__(self, input_ids, scores):
        import torch

        if self._index is None:
            self._index = torch.tensor(self.suppress_ids, device=scores.device, dtype=torch.long)
        scores[:, self._index] = float("-inf")
        return scores


def _self_test() -> int:
    """토크나이저 없이 판정 로직만 확인."""
    cases = [
        ("안녕하세요", False),
        ("오늘 날씨가 参 좋네요", True),
        ("キーボード를 정리", True),
        ("책상 위 키보드를 정리할까요?", False),
    ]
    ok = True
    for text, expected in cases:
        got = has_foreign_cjk(text)
        mark = "OK" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"  [{mark}] foreign={got} expected={expected} : {text}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="한국어 전용 디코딩 유틸 자체 테스트")
    ap.add_argument("--self-test", action="store_true", help="판정 로직 테스트")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(_self_test())
    ap.print_help()
