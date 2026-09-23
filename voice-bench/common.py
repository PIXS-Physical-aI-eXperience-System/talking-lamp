"""후보 비교에 공용으로 쓰는 것들 — 러너와 오케스트레이터 양쪽에서 import 한다.

의존성 없음(표준 라이브러리만). 러너는 각자 다른 venv에서 돌기 때문에
이 파일은 어떤 venv에서도 import 되어야 한다.
"""
import json
import os
import re
import resource
import sys
import time

RESULT_MARKER = "@@RESULT@@"

# ─── 텍스트 정규화 / CER ────────────────────────────────────────────────

_PUNCT = re.compile(r"[.,!?~…\"'`·:;()\[\]{}\-—]")


def normalize_ko(s: str, drop_space: bool = True) -> str:
    """한국어 CER 비교용 정규화.

    한국어는 띄어쓰기가 모호해서 CER을 잴 때 공백을 빼는 게 관례다.
    띄어쓰기까지 보고 싶으면 drop_space=False 로 따로 재면 된다.
    """
    s = _PUNCT.sub("", s)
    s = s.strip()
    if drop_space:
        s = re.sub(r"\s+", "", s)
    else:
        s = re.sub(r"\s+", " ", s)
    return s


def edit_distance(a: str, b: str) -> int:
    """레벤슈타인 거리. 문장이 짧아서 O(n*m) 로 충분하다."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,        # 삭제
                cur[j - 1] + 1,     # 삽입
                prev[j - 1] + (ca != cb),  # 치환
            ))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str, drop_space: bool = True) -> float:
    """Character Error Rate. 1.0 = 전부 틀림. 참조가 비면 nan."""
    r = normalize_ko(ref, drop_space)
    h = normalize_ko(hyp, drop_space)
    if not r:
        return float("nan")
    return edit_distance(r, h) / len(r)


# ─── 메모리 측정 ────────────────────────────────────────────────────────

def peak_rss_mb() -> float:
    """현재 프로세스의 피크 RSS(MB).

    ru_maxrss 단위가 OS마다 다르다 — macOS는 바이트, Linux는 킬로바이트.
    Jetson(Linux)에서 그대로 돌려야 하므로 여기서 갈라준다.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024


# ─── 문장 세트 / 결과 출력 ──────────────────────────────────────────────

def load_sentences(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def emit(payload: dict) -> None:
    """러너가 부모에게 결과를 넘기는 유일한 통로.

    모델 라이브러리들이 stdout에 진행률을 마구 찍기 때문에,
    마커가 붙은 줄만 부모가 골라 읽는다.
    """
    payload.setdefault("peak_rss_mb", round(peak_rss_mb(), 1))
    print(RESULT_MARKER + json.dumps(payload, ensure_ascii=False), flush=True)


class Timer:
    """with 블록의 경과 시간(초)을 .elapsed 에 담는다."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        return False


def repo_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    return {
        "root": here,
        "sentences": os.path.join(here, "sentences.txt"),
        "ref": os.path.join(here, "ref"),
        "out": os.path.join(here, "out"),
    }
