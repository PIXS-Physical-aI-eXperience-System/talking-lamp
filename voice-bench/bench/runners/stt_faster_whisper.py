"""faster-whisper(CTranslate2) STT 러너.

같은 스크립트로 base / small / 한국어 파인튜닝 모델을 전부 돌린다.
파인튜닝 모델은 CTranslate2로 변환한 디렉터리 경로를 --model 에 그대로 넣으면 된다.

    python stt_faster_whisper.py --model small --compute-type int8
    python stt_faster_whisper.py --model ./ct2-whisper-small-ko --device cuda --compute-type int8_float16

Jetson에서는 --device cuda 로 바꿔서 돌려야 RSS가 의미 있는 숫자가 된다.
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import Timer, emit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="크기(base/small) 또는 CT2 모델 디렉터리")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (Jetson은 cuda)")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    with Timer() as t_load:
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    wavs = sorted(glob.glob(os.path.join(args.ref_dir, "*.wav")))
    transcripts, ttfbs, infers = [], [], []

    for wav in wavs:
        t0 = time.perf_counter()
        # beam_size=1 — 실시간 대화용이라 그리디가 현실적인 설정이다.
        # 여기서 beam을 키우면 CER은 좋아지지만 지연이 늘어난다.
        segments, _info = model.transcribe(wav, language="ko", beam_size=1)

        first_at = None
        parts = []
        for seg in segments:  # 제너레이터라 여기서 실제 디코딩이 돈다
            if first_at is None:
                first_at = time.perf_counter() - t0
            parts.append(seg.text)

        transcripts.append("".join(parts).strip())
        ttfbs.append(round(first_at if first_at is not None else -1, 3))
        infers.append(round(time.perf_counter() - t0, 3))

    emit({
        "label": args.label,
        "kind": "stt",
        "files": [os.path.basename(w) for w in wavs],
        "transcripts": transcripts,
        "load_s": round(t_load.elapsed, 2),
        "ttfb_s": ttfbs,
        "infer_s": infers,
        "config": {"model": args.model, "device": args.device,
                   "compute_type": args.compute_type},
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
