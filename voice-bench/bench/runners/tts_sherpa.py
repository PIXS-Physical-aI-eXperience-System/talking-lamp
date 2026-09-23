"""sherpa-onnx VITS 러너 — 한국어 전용 모델(KSS) 용.

다른 후보들과 결정적으로 다른 점: torch가 없고 onnxruntime만 쓴다.
G2P 데이터(espeak-ng-data)도 모델에 동봉돼 있어 외부 의존성이 없다.
임베디드(라즈베리파이·Jetson)를 대상으로 만들어진 툴킷이라 우리 조건에 맞다.

    python tts_sherpa.py --model-dir ../models/vits-mimic3-ko_KO-kss_low \
        --out-dir ../out/tts/sherpa-kss --label sherpa-kss
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import Timer, emit, load_sentences, repo_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    import sherpa_onnx
    import soundfile as sf

    os.makedirs(args.out_dir, exist_ok=True)
    sentences = load_sentences(repo_paths()["sentences"])
    if args.normalize:
        from ko_normalize import normalize
        sentences = [normalize(s) for s in sentences]

    d = args.model_dir
    onnx = [f for f in os.listdir(d) if f.endswith(".onnx")][0]

    with Timer() as t_load:
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=os.path.join(d, onnx),
                    tokens=os.path.join(d, "tokens.txt"),
                    data_dir=os.path.join(d, "espeak-ng-data"),
                ),
                num_threads=2,
            ),
        )
        tts = sherpa_onnx.OfflineTts(cfg)

    wavs, synth_s, audio_s = [], [], []
    for i, text in enumerate(sentences, 1):
        path = os.path.join(args.out_dir, f"{i:02d}.wav")
        t0 = time.perf_counter()
        audio = tts.generate(text, sid=0, speed=args.speed)
        sf.write(path, audio.samples, audio.sample_rate)
        synth_s.append(round(time.perf_counter() - t0, 3))
        audio_s.append(round(len(audio.samples) / audio.sample_rate, 3))
        wavs.append(path)

    emit({"label": args.label, "kind": "tts", "wavs": wavs,
          "load_s": round(t_load.elapsed, 2), "synth_s": synth_s, "audio_s": audio_s,
          "spoken_text": sentences,
          "config": {"model_dir": d, "normalize": args.normalize}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
