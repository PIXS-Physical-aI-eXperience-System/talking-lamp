#!/usr/bin/env python3
"""bench_vlm.py 보조 모듈 — 혼입 억제 + 경량화(이미지 축소).

bench_vlm.py 와 같은 폴더(vlm_bench/vlm_bench/)에 두고 import 해서 쓴다.
InternVL3_5-2B-HF (AutoModelForImageTextToText + AutoProcessor) 경로 전용.

제공:
  resize_max_side(img, n)            경량화 — 이미지 긴 변을 n px로 축소 → 타일 수↓ → 메모리·속도↓
  get_ko_logits_processors(tok)      혼입 억제 — 한자/가나 토큰 차단 LogitsProcessor (캐시됨)
  sanitize_for_tts(text)             후처리 필터 — TTS 직전 한자/가나 제거 (C 인계물)
  contains_foreign(text)             혼입 여부 빠른 판정
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


# ===== 코드포인트 판정 =====
def _in(cp, lo, hi):
    return lo <= cp <= hi


def is_hanja(cp):
    return (_in(cp, 0x4E00, 0x9FFF) or _in(cp, 0x3400, 0x4DBF)
            or _in(cp, 0x20000, 0x2A6DF) or _in(cp, 0xF900, 0xFAFF))


def is_kana(cp):
    return (_in(cp, 0x3040, 0x309F) or _in(cp, 0x30A0, 0x30FF)
            or _in(cp, 0x31F0, 0x31FF) or _in(cp, 0xFF66, 0xFF9D))


def _is_foreign(ch):
    cp = ord(ch)
    return is_hanja(cp) or is_kana(cp)


def contains_foreign(text):
    return any(_is_foreign(ch) for ch in text)


# ===== 경량화: 이미지 축소 (+ EXIF 회전 보정) =====
def resize_max_side(img, max_side):
    """EXIF 회전을 먼저 바로잡고, 긴 변이 max_side를 넘으면 비율 유지 축소.

    폰 사진은 세로로 찍혀도 가로로 저장되고 EXIF orientation 태그만 90°로
    걸려있는 경우가 많다. 이를 안 풀면 모델이 얼굴을 눕힌 채로 봐서 인식이
    나빠진다. exif_transpose 로 항상 바로 세운다(max_side None이어도 적용).
    """
    from PIL import ImageOps
    img = ImageOps.exif_transpose(img)
    if not max_side:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    s = max_side / float(m)
    return img.resize((max(1, int(w * s)), max(1, int(h * s))))


# ===== 혼입 억제: 디코딩 토큰 차단 =====
def build_suppress_ids(tokenizer, block_hanja=True, block_kana=True):
    suppress = []
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    for tid in range(vocab_size):
        try:
            piece = tokenizer.decode([tid])
        except Exception:
            continue
        if piece:
            for ch in piece:
                cp = ord(ch)
                if (block_hanja and is_hanja(cp)) or (block_kana and is_kana(cp)):
                    suppress.append(tid)
                    break
    return suppress


class KoOnlyLogitsProcessor:
    def __init__(self, suppress_ids):
        self.suppress_ids = suppress_ids
        self._index = None

    def __call__(self, input_ids, scores):
        import torch
        if self._index is None:
            self._index = torch.tensor(self.suppress_ids, device=scores.device, dtype=torch.long)
        scores[:, self._index] = float("-inf")
        return scores


_SUPPRESS_CACHE = {}


def get_ko_logits_processors(tokenizer):
    """한자/가나 차단 LogitsProcessor 리스트 (토크나이저당 1회 계산 후 캐시)."""
    key = id(tokenizer)
    if key not in _SUPPRESS_CACHE:
        import time
        print("[ko_lite] 한자/가나 토큰 목록 생성 중... (최초 1회)")
        t0 = time.perf_counter()
        ids = build_suppress_ids(tokenizer)
        dt = time.perf_counter() - t0
        print(f"[ko_lite] 차단 대상 토큰 {len(ids)}개, 생성 {dt:.1f}s (1회성 오버헤드)")
        _SUPPRESS_CACHE[key] = [KoOnlyLogitsProcessor(ids)]
    return _SUPPRESS_CACHE[key]


# ===== 후처리 필터 (C 인계물) =====
REGEN_RATIO_THRESHOLD = 0.15


@dataclass
class FilterReport:
    was_contaminated: bool = False
    removed: list = field(default_factory=list)
    n_removed: int = 0
    orig_len: int = 0
    needs_regen: bool = False


def sanitize_for_tts(text, regen_threshold=REGEN_RATIO_THRESHOLD):
    text = unicodedata.normalize("NFC", text)
    rep = FilterReport(orig_len=len([c for c in text if not c.isspace()]))
    kept = []
    for ch in text:
        if _is_foreign(ch):
            rep.removed.append(ch)
        else:
            kept.append(ch)
    clean = "".join(kept)
    rep.n_removed = len(rep.removed)
    rep.was_contaminated = rep.n_removed > 0
    if rep.was_contaminated:
        clean = re.sub(r"[ \t]{2,}", " ", clean)
        clean = re.sub(r"\s+([,.!?;:)\]}」』】”’])", r"\1", clean)
        clean = re.sub(r"([(\[{「『【“‘])\s+", r"\1", clean)
        clean = clean.strip()
        rep.needs_regen = (rep.n_removed / (rep.orig_len or 1)) > regen_threshold
    return clean, rep
