"""melo 피크 메모리를 줄이는 방법들을 실측 비교한다.

가중치를 줄이는 방향(양자화·헤드 제거)은 평균만 낮추고 피크를 못 낮췄다.
피크를 만드는 건 합성 중의 순간 할당이므로 그쪽을 직접 건드려 본다.
"""
import gc
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # voice-bench/ — 공용 모듈과 산출물이 있는 곳
sys.path.insert(0, ROOT)
from common import load_sentences, repo_paths  # noqa: E402

VARIANT = sys.argv[1]
CYCLES = int(sys.argv[2]) if len(sys.argv) > 2 else 30

import torch  # noqa: E402
if VARIANT in ("threads1", "all"):
    torch.set_num_threads(1)

import melo.text.japanese_bert as jb  # noqa: E402
if VARIANT in ("cpu", "all"):
    # melo는 macOS에서 device="cpu"를 줘도 MPS로 바꿔치기한다. 그걸 막는다.
    _orig = torch.backends.mps.is_available
    torch.backends.mps.is_available = lambda: False

from melo.api import TTS  # noqa: E402


def rss():
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
               capture_output=True, text=True).stdout.strip()) / 1024


tts = TTS(language="KR", device="cpu")
spk = tts.hps.data.spk2id["KR"]
tmp = os.path.join(tempfile.mkdtemp(), "o.wav")
sents = load_sentences(repo_paths()["sentences"])

peak = 0.0
for i in range(CYCLES):
    tts.tts_to_file(sents[i % len(sents)], spk, tmp)
    if VARIANT in ("gc", "all"):
        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    peak = max(peak, rss())

print(f"RESULT\t{VARIANT}\t{peak:.0f}\t{torch.get_num_threads()}")
