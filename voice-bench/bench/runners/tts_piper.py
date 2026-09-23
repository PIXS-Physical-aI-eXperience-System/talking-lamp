"""Piper 계열 러너 (piper-plus, MIT).

라이선스 주의: 원본 rhasspy/piper 는 2025-10 아카이브됐고
후속 OHF-Voice/piper1-gpl 은 GPL-3.0 이다. MIT를 원하면 piper-plus 를 쓸 것.

CLI 인터페이스(stdin 으로 텍스트, -f 로 wav 출력)를 가정한다.
설치한 포크의 CLI가 다르면 --bin / 인자 구성만 고치면 된다.

    python tts_piper.py --bin piper --model ./models/ko_KR-xxx.onnx \
        --out-dir ../out/tts/piper --label piper
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import emit, load_sentences, repo_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="piper")
    ap.add_argument("--model", required=True, help="한국어 .onnx 경로")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sentences = load_sentences(repo_paths()["sentences"])

    import soundfile as sf

    wavs, synth_s, audio_s = [], [], []
    for i, text in enumerate(sentences, 1):
        path = os.path.join(args.out_dir, f"{i:02d}.wav")
        t0 = time.perf_counter()
        subprocess.run(
            [args.bin, "-m", args.model, "-f", path],
            input=text.encode("utf-8"), check=True, capture_output=True,
        )
        synth_s.append(round(time.perf_counter() - t0, 3))
        audio_s.append(round(sf.info(path).duration, 3))
        wavs.append(path)

    # 별도 프로세스라 이 러너의 RSS는 piper 자체를 반영하지 못한다.
    # piper 프로세스 자체의 메모리는 Jetson에서 /usr/bin/time -v 로 따로 잴 것.
    emit({
        "label": args.label,
        "kind": "tts",
        "wavs": wavs,
        "load_s": 0.0,
        "synth_s": synth_s,
        "audio_s": audio_s,
        "rss_note": "CLI 서브프로세스라 peak_rss_mb 는 무의미. 별도 측정 필요",
        "config": {"bin": args.bin, "model": args.model},
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
