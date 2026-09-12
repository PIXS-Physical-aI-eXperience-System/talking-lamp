#!/usr/bin/env python3
"""Repeat a command in fresh processes and require every RAM run to pass."""
import argparse
import json
import os
from pathlib import Path

from ram_probe import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--min-available-mb', type=float, default=800)
    parser.add_argument('--max-swap-pages', type=int, default=0)
    parser.add_argument('--interval', type=float, default=.05)
    parser.add_argument('--settle', type=float, default=2)
    parser.add_argument('--timeout', type=float, default=300,
                        help='per-attempt timeout in seconds')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if (not command or args.attempts < 1 or args.min_available_mb < 0 or
            args.max_swap_pages < 0 or args.interval <= 0 or args.settle < 0 or
            args.timeout <= 0):
        parser.error('command required; attempts > 0; numeric limits non-negative')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    floor = int(args.min_available_mb * 1_000_000)
    runs = []
    for attempt in range(1, args.attempts + 1):
        previous = os.environ.get('RAM_HARNESS_ATTEMPT')
        os.environ['RAM_HARNESS_ATTEMPT'] = str(attempt)
        try:
            result = measure(command, args.interval, args.settle, args.timeout)
        finally:
            if previous is None:
                os.environ.pop('RAM_HARNESS_ATTEMPT', None)
            else:
                os.environ['RAM_HARNESS_ATTEMPT'] = previous
        swap_pages = sum(result['swap_activity_pages'].values())
        passed = (not result['timed_out'] and result['returncode'] == 0 and
                  result['minimum_available_bytes'] >= floor and
                  swap_pages <= args.max_swap_pages)
        result['attempt'] = attempt
        result['acceptance'] = {
            'min_available_bytes': floor,
            'max_swap_pages': args.max_swap_pages,
            'passed': passed,
        }
        path = args.out_dir / f'attempt-{attempt}.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        runs.append({'attempt': attempt, 'result': str(path), 'passed': passed,
                     'minimum_available_bytes': result['minimum_available_bytes'],
                     'swap_activity_pages': swap_pages,
                     'timed_out': result['timed_out'],
                     'returncode': result['returncode']})
        print(f"attempt {attempt}: {'PASS' if passed else 'FAIL'}, "
              f"available={result['minimum_available_bytes']/1e6:.1f} MB, "
              f"swap_pages={swap_pages}, exit={result['returncode']}", flush=True)

    summary = {
        'schema_version': 1,
        'command': command,
        'gate': {'min_available_bytes': floor,
                 'max_swap_pages': args.max_swap_pages},
        'passed': all(run['passed'] for run in runs),
        'worst_minimum_available_bytes': min(
            run['minimum_available_bytes'] for run in runs),
        'runs': runs,
    }
    (args.out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    raise SystemExit(0 if summary['passed'] else 3)


if __name__ == '__main__':
    main()
