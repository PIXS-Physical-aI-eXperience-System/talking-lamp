"""facebook/mms-tts-kor 러너 (Meta MMS, VITS 계열).

다른 후보들이 전부 '텍스트→발음' 단계의 외부 의존성(MeCab, pygoruut, espeak)에서
막혔는데, 이 모델은 토크나이저에 그게 내장돼 있어 파이썬 패키지만으로 돌아간다.

라이선스 주의: MMS는 CC-BY-NC 4.0 (비상업). 출품 조건 확인 필요.

    python tts_mms.py --out-dir ../out/tts/mms-kor --label mms-kor
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
    ap.add_argument("--model", default="facebook/mms-tts-kor")
    ap.add_argument("--normalize", action="store_true",
                    help="숫자·영문을 한글 읽기로 바꾼 뒤 합성 (ko_normalize)")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from transformers import VitsModel, AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    sentences = load_sentences(repo_paths()["sentences"])
    if args.normalize:
        from ko_normalize import normalize
        sentences = [normalize(s) for s in sentences]

    with Timer() as t_load:
        model = VitsModel.from_pretrained(args.model)
        tok = AutoTokenizer.from_pretrained(args.model)
        model.eval()
    sr = model.config.sampling_rate

    wavs, synth_s, audio_s = [], [], []
    for i, text in enumerate(sentences, 1):
        path = os.path.join(args.out_dir, f"{i:02d}.wav")
        t0 = time.perf_counter()
        with torch.no_grad():
            audio = model(**tok(text, return_tensors="pt")).waveform[0].cpu().numpy()
        sf.write(path, audio, sr)
        synth_s.append(round(time.perf_counter() - t0, 3))
        audio_s.append(round(len(audio) / sr, 3))
        wavs.append(path)

    emit({"label": args.label, "kind": "tts", "wavs": wavs,
          "load_s": round(t_load.elapsed, 2), "synth_s": synth_s, "audio_s": audio_s,
          "spoken_text": sentences,
          "config": {"model": args.model, "sample_rate": sr,
                     "normalize": args.normalize}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
