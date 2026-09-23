"""Kokoro-82M 러너 (Apache-2.0).

가중치가 327 MB라 음성합성 0.2 GB 예산은 이미 초과다.
'음질 상한이 어디까지인지' 기준선을 잡는 용도로 넣었다 —
이게 MeloTTS/Piper보다 확연히 좋으면 예산 재협상 대상이 하나 늘어난다.

※ 한국어 lang_code 와 voice 이름은 설치한 버전에서 확인이 필요하다.
   기본값으로 실패하면 에러 메시지에 사용 가능한 값이 뜨니 그걸로 바꿔 넣을 것.

    python tts_kokoro.py --lang-code k --voice <voice> --out-dir ../out/tts/kokoro --label kokoro
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import Timer, emit, load_sentences, repo_paths  # noqa: E402

SAMPLE_RATE = 24000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang-code", default="k", help="한국어 코드 — 설치 버전에서 확인 필요")
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    os.makedirs(args.out_dir, exist_ok=True)
    sentences = load_sentences(repo_paths()["sentences"])

    with Timer() as t_load:
        pipeline = KPipeline(lang_code=args.lang_code)

    wavs, synth_s, audio_s = [], [], []
    for i, text in enumerate(sentences, 1):
        path = os.path.join(args.out_dir, f"{i:02d}.wav")
        t0 = time.perf_counter()
        chunks = [audio for _gs, _ps, audio in pipeline(text, voice=args.voice)]
        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        sf.write(path, audio, SAMPLE_RATE)
        synth_s.append(round(time.perf_counter() - t0, 3))
        audio_s.append(round(len(audio) / SAMPLE_RATE, 3))
        wavs.append(path)

    emit({
        "label": args.label,
        "kind": "tts",
        "wavs": wavs,
        "load_s": round(t_load.elapsed, 2),
        "synth_s": synth_s,
        "audio_s": audio_s,
        "config": {"lang_code": args.lang_code, "voice": args.voice},
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
