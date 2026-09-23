"""piper 한국어 KSS 모델 러너 (neurlang/piper-onnx-kss-korean).

이 모델은 espeak-ng 대신 goruut G2P를 쓴다. pygoruut 파이썬 래퍼는 매번 92MB
바이너리를 새로 받다가 멈추므로, 바이너리를 미리 받아 서버로 띄우고 HTTP로 직접 부른다.
goruut 기본 설정은 PolicyMaxWords=0 이라 모든 요청이 거부된다 — config로 올려야 한다.

    # 서버 먼저:  models/goruut/goruut-darwin-arm64 -configfile models/goruut/config.json &
    python tts_piper_goruut.py --goruut-port 61294 --out-dir ../out/tts/piper-ko --label piper-ko
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import requests
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import Timer, emit, load_sentences, repo_paths  # noqa: E402

PAD, BOS, EOS = "_", "^", "$"


def phonemize(port: int, text: str) -> str:
    r = requests.post(f"http://127.0.0.1:{port}/tts/phonemize/sentence",
                      json={"Language": "Korean", "Languages": [],
                            "Sentence": text, "IsReverse": False}, timeout=30)
    r.raise_for_status()
    words = r.json().get("Words") or []
    if not words:
        raise RuntimeError(f"goruut가 발음을 못 냄: {text!r} — PolicyMaxWords 설정 확인")
    return " ".join(w["Phonetic"] for w in words)


def to_ids(phonemes: str, pim: dict) -> tuple[list[int], int]:
    """piper 표준 시퀀스: BOS, PAD, (음소, PAD)*, EOS."""
    ids = list(pim[BOS]) + list(pim[PAD])
    skipped = 0
    for ch in phonemes:
        if ch in pim:
            ids.extend(pim[ch])
            ids.extend(pim[PAD])
        else:
            skipped += 1  # 한국어 경음 표시(U+0348) 등 이 모델에 없는 결합 기호
    ids.extend(pim[EOS])
    return ids, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/piper-ko")
    ap.add_argument("--goruut-port", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    d = args.model_dir
    onnx_path = [f for f in os.listdir(d) if f.endswith(".onnx")][0]
    cfg = json.load(open(os.path.join(d, onnx_path + ".json"), encoding="utf-8"))
    pim = cfg["phoneme_id_map"]
    sr = cfg["audio"]["sample_rate"]
    inf = cfg["inference"]
    scales = np.array([inf["noise_scale"], inf["length_scale"], inf["noise_w"]], dtype=np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    sentences = load_sentences(repo_paths()["sentences"])
    if args.normalize:
        from ko_normalize import normalize
        sentences = [normalize(s) for s in sentences]

    with Timer() as t_load:
        sess = ort.InferenceSession(os.path.join(d, onnx_path),
                                    providers=["CPUExecutionProvider"])

    wavs, synth_s, audio_s, skips = [], [], [], 0
    for i, text in enumerate(sentences, 1):
        path = os.path.join(args.out_dir, f"{i:02d}.wav")
        t0 = time.perf_counter()
        ph = phonemize(args.goruut_port, text)
        ids, sk = to_ids(ph, pim)
        skips += sk
        audio = sess.run(None, {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": scales,
        })[0].squeeze()
        sf.write(path, audio, sr)
        synth_s.append(round(time.perf_counter() - t0, 3))
        audio_s.append(round(len(audio) / sr, 3))
        wavs.append(path)

    emit({"label": args.label, "kind": "tts", "wavs": wavs,
          "load_s": round(t_load.elapsed, 2), "synth_s": synth_s, "audio_s": audio_s,
          "spoken_text": sentences,
          "config": {"model": onnx_path, "sample_rate": sr,
                     "normalize": args.normalize, "skipped_phonemes": skips}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
