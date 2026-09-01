"""MeloTTS 한국어 러너 (MIT).

주의: MeloTTS의 한국어 프론트엔드가 한국어 BERT를 같이 물고 올 가능성이 있다.
그러면 음성합성 0.2 GB 예산이 깨진다 — 이 러너가 찍는 peak_rss_mb 로 확인할 것.
그게 이 후보에서 제일 알고 싶은 숫자다.

    python tts_melo.py --out-dir ../out/tts/melo --label melo
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import Timer, emit, load_sentences, repo_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    import soundfile as sf
    from melo.api import TTS

    os.makedirs(args.out_dir, exist_ok=True)
    sentences = load_sentences(repo_paths()["sentences"])
    if args.normalize:
        from ko_normalize import normalize
        sentences = [normalize(s) for s in sentences]

    with Timer() as t_load:
        tts = TTS(language="KR", device=args.device)
        speaker_ids = tts.hps.data.spk2id
        speaker = speaker_ids["KR"]

    wavs, synth_s, audio_s = [], [], []
    for i, text in enumerate(sentences, 1):
        path = os.path.join(args.out_dir, f"{i:02d}.wav")
        t0 = time.perf_counter()
        tts.tts_to_file(text, speaker, path, speed=args.speed)
        synth_s.append(round(time.perf_counter() - t0, 3))
        info = sf.info(path)
        audio_s.append(round(info.duration, 3))
        wavs.append(path)

    emit({
        "label": args.label,
        "kind": "tts",
        "wavs": wavs,
        "load_s": round(t_load.elapsed, 2),
        "synth_s": synth_s,
        "audio_s": audio_s,
        "spoken_text": sentences,
        "config": {"device": args.device, "speed": args.speed,
                   "normalize": args.normalize},
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
