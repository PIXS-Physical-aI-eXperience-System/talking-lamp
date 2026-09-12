#!/usr/bin/env python3
"""Synthesize the Korean translation produced by translate_vlm_result.py."""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--voice-root', type=Path, required=True)
    parser.add_argument('--model-dir', default='models/melo-ko-onnx')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.voice_root))
    from ko_normalize import normalize
    from runners.tts_melo_onnx import build_synth
    import onnxruntime as ort
    import soundfile as sf

    text = normalize(json.loads(args.input.read_text())['text'])
    ort.set_seed(args.seed)
    synth, sample_rate, providers, load_s, _ = build_synth(
        args.model_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        threads=2, quiet=True, bert_int8=True)
    started = time.perf_counter()
    audio = synth(text)
    synth_s = time.perf_counter() - started
    args.wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.wav, audio, sample_rate)
    result = {
        'schema_version': 1,
        'source': str(args.input),
        'text': text,
        'wav': str(args.wav),
        'sample_rate': sample_rate,
        'audio_s': len(audio) / sample_rate,
        'load_s': load_s,
        'synth_s': synth_s,
        'providers': providers,
        'seed': args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
