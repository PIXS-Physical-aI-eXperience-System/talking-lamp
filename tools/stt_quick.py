#!/usr/bin/env python3
"""Quick faster-whisper transcription matching voice-bench's final config
(small, cuda, int8_float16) for the integrated-stack memory test."""
import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True)
    ap.add_argument('--model', default='small')
    ap.add_argument('--expected')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    t0 = time.perf_counter()
    model = WhisperModel(args.model, device='cuda', compute_type='int8_float16')
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    segments, info = model.transcribe(args.wav)
    text = ''.join(s.text for s in segments).strip()
    infer_s = time.perf_counter() - t0
    passed = args.expected is None or text == args.expected.strip()
    result = {'schema_version': 1, 'wav': args.wav, 'text': text,
              'expected': args.expected, 'passed': passed,
              'load_s': load_s, 'infer_s': infer_s}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    raise SystemExit(0 if passed else 3)


if __name__ == '__main__':
    main()
