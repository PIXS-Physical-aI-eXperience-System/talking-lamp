#!/usr/bin/env python3
"""Apply one reproducible RAM/swap gate to ram_probe JSON results."""
import argparse
import json
from pathlib import Path


def evaluate(path, min_available_bytes, max_swap_pages):
    data = json.loads(path.read_text())
    samples = data.get('samples', [])
    minimum = data.get('minimum_available_bytes')
    if minimum is None:
        if not samples:
            raise ValueError(f'{path}: no samples or minimum_available_bytes')
        minimum = min(sample['available_bytes'] for sample in samples)
    swap = sum(data.get('swap_activity_pages', {}).values())
    passed = (data.get('returncode') == 0 and minimum >= min_available_bytes
              and swap <= max_swap_pages)
    return {
        'path': str(path),
        'minimum_available_bytes': minimum,
        'minimum_available_mb': round(minimum / 1_000_000, 1),
        'swap_activity_pages': swap,
        'returncode': data.get('returncode'),
        'passed': passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='+')
    parser.add_argument('--min-available-mb', type=float, default=800)
    parser.add_argument('--max-swap-pages', type=int, default=0)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.min_available_mb < 0 or args.max_swap_pages < 0:
        parser.error('limits must be non-negative')

    floor = int(args.min_available_mb * 1_000_000)
    runs = [evaluate(path, floor, args.max_swap_pages) for path in args.results]
    summary = {
        'schema_version': 1,
        'gate': {'min_available_bytes': floor,
                 'max_swap_pages': args.max_swap_pages},
        'passed': all(run['passed'] for run in runs),
        'worst_minimum_available_bytes': min(
            run['minimum_available_bytes'] for run in runs),
        'runs': runs,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end='')
    raise SystemExit(0 if summary['passed'] else 3)


if __name__ == '__main__':
    main()
