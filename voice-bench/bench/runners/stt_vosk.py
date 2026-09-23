"""Vosk STT 러너.

후보 중 유일하게 '진짜 스트리밍'이다 — 오디오를 청크 단위로 밀어넣으면서
중간 결과를 받는다. 그래서 TTFB의 하한선을 보여주는 기준선 역할을 한다.
정확도는 낮을 것으로 예상되지만(공개 WER 28.1%), 그 숫자를 직접 확인하는 게 목적이다.

    python stt_vosk.py --model-path ./models/vosk-model-small-ko-0.22
"""
import argparse
import glob
import json
import os
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import Timer, emit  # noqa: E402

CHUNK_FRAMES = 2000  # 16kHz에서 125ms — 스트리밍 지연을 보려면 작게 유지


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    from vosk import KaldiRecognizer, Model, SetLogLevel
    SetLogLevel(-1)

    with Timer() as t_load:
        model = Model(args.model_path)

    wavs = sorted(glob.glob(os.path.join(args.ref_dir, "*.wav")))
    transcripts, ttfbs, infers = [], [], []

    for wav_path in wavs:
        with wave.open(wav_path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                raise SystemExit(f"{wav_path}: 16bit 모노가 아니다. record.py로 다시 녹음할 것")
            rec = KaldiRecognizer(model, wf.getframerate())

            t0 = time.perf_counter()
            first_at = None
            finals = []
            while True:
                data = wf.readframes(CHUNK_FRAMES)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    finals.append(json.loads(rec.Result()).get("text", ""))
                elif first_at is None:
                    # 부분 결과에 글자가 처음 찍힌 순간 = 스트리밍 TTFB
                    if json.loads(rec.PartialResult()).get("partial", "").strip():
                        first_at = time.perf_counter() - t0
            finals.append(json.loads(rec.FinalResult()).get("text", ""))

        transcripts.append(" ".join(x for x in finals if x).strip())
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
        "config": {"model_path": args.model_path},
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
