#!/usr/bin/env python3
"""Evaluate RAM, VLM, Korean rendering, and TTS artifacts as one gate."""
import argparse
import json
from pathlib import Path

from evaluate_vlm_quality import inspect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ram-summary', type=Path, required=True)
    parser.add_argument('--vlm-dir', type=Path, required=True)
    parser.add_argument('--tts-dir', type=Path, required=True)
    parser.add_argument('--roundtrip-dir', type=Path)
    parser.add_argument('--image-sha256', required=True)
    parser.add_argument('--require-any', action='append', default=[], metavar='A,B')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    groups = [[term.strip() for term in value.split(',') if term.strip()]
              for value in args.require_any]
    if any(not group for group in groups):
        parser.error('require-any groups cannot be empty')

    ram = json.loads(args.ram_summary.read_text())
    attempts = []
    for ram_run in ram['runs']:
        number = ram_run['attempt']
        vlm = json.loads((args.vlm_dir / f'attempt-{number}.json').read_text())
        ko = json.loads((args.vlm_dir / f'attempt-{number}-ko.json').read_text())
        tts = json.loads((args.tts_dir / f'attempt-{number}.json').read_text())
        wav = Path(tts['wav'])
        if not wav.is_file():
            wav = args.tts_dir / wav.name
        vlm_checks = inspect(vlm['runs'][0]['text'], 'en', groups, 2)
        checks = {
            'ram': bool(ram_run['passed']),
            'image': vlm.get('image_sha256') == args.image_sha256,
            'vlm_quality': all(vlm_checks.values()),
            'korean_render': bool(ko.get('passed')) and not ko.get('unknown_items'),
            'korean_script': all(inspect(ko.get('text', ''), 'ko', [], 2).values()),
            'tts_metadata': tts.get('audio_s', 0) > 0 and tts.get('sample_rate', 0) > 0,
            'tts_wav': wav.is_file() and wav.stat().st_size > 44,
        }
        roundtrip = None
        if args.roundtrip_dir:
            roundtrip = json.loads(
                (args.roundtrip_dir / f'attempt-{number}.json').read_text())
            checks['speech_roundtrip'] = bool(roundtrip.get('passed'))
        attempts.append({
            'attempt': number,
            'passed': all(checks.values()),
            'checks': checks,
            'minimum_available_bytes': ram_run['minimum_available_bytes'],
            'vlm_text': vlm['runs'][0]['text'],
            'korean_text': ko['text'],
            'audio_s': tts['audio_s'],
            'wav': str(wav),
            'roundtrip_text': roundtrip.get('text') if roundtrip else None,
        })
    result = {
        'schema_version': 1,
        'passed': bool(attempts) and all(item['passed'] for item in attempts),
        'ram_gate': ram['gate'],
        'worst_minimum_available_bytes': ram['worst_minimum_available_bytes'],
        'attempts': attempts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 3)


if __name__ == '__main__':
    main()
