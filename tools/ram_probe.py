#!/usr/bin/env python3
"""Run an existing benchmark in a fresh process; sample Linux system RAM.

No torch import: captures framework startup and memory after process exit.
Values are bytes; system pressure includes other processes and page cache effects.
"""
import argparse
import json
import os
import platform
import signal
import subprocess
import time
from pathlib import Path


def snapshot():
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        mem[key] = int(value.split()[0]) * 1024
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return {
        'monotonic_s': time.monotonic(),
        'total_bytes': mem['MemTotal'],
        'used_bytes': mem['MemTotal'] - mem['MemAvailable'],
        'available_bytes': mem['MemAvailable'],
        'swap_used_bytes': mem['SwapTotal'] - mem['SwapFree'],
        'swap_in_pages': int(vm['pswpin']),
        'swap_out_pages': int(vm['pswpout']),
    }


def measure(command, interval=0.05, settle=1.0, timeout=None):
    before = snapshot()
    samples = [before]
    started = time.monotonic()
    child = subprocess.Popen(command, start_new_session=True)
    timed_out = False

    def signal_group(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass

    try:
        while True:
            samples.append(snapshot())
            if child.poll() is not None:
                break
            if timeout is not None and time.monotonic() - started >= timeout:
                timed_out = True
                signal_group(signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    signal_group(signal.SIGKILL)
                    child.wait()
                break
            time.sleep(interval)
        elapsed = time.monotonic() - started
    finally:
        if child.poll() is None:
            signal_group(signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_group(signal.SIGKILL)
                child.wait()
    time.sleep(settle)
    after = snapshot()
    peak = max(s['used_bytes'] for s in samples)
    lowest_available = min(samples, key=lambda sample: sample['available_bytes'])
    return {
        'schema_version': 1,
        'host': platform.node(),
        'machine': platform.machine(),
        'kernel': platform.release(),
        'command': command,
        'returncode': child.returncode,
        'timed_out': timed_out,
        'wall_s': elapsed,
        'sample_interval_s': interval,
        'before': before,
        'after_exit': after,
        'system_peak_used_bytes': peak,
        'minimum_available_bytes': lowest_available['available_bytes'],
        'minimum_available_at_s': lowest_available['monotonic_s'] - started,
        'system_peak_delta_bytes': peak - before['used_bytes'],
        'swap_activity_pages': {
            key: max(s[key] for s in samples + [after]) - before[key]
            for key in ('swap_in_pages', 'swap_out_pages')
        },
        'samples': samples,
        'limitations': 'Sampled system pressure, not process RSS or CUDA allocations. '
                        'Child must wait for its workers. wall_s includes load and all inference.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=0.05)
    parser.add_argument('--settle', type=float, default=1.0)
    parser.add_argument('--timeout', type=float,
                        help='terminate the isolated command group after this many seconds')
    parser.add_argument('--min-available-mb', type=float, default=0,
                        help='fail when available RAM falls below this decimal-MB floor')
    parser.add_argument('--max-swap-pages', type=int, default=0,
                        help='maximum allowed total swap-in + swap-out pages')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if (not command or args.interval <= 0 or args.settle < 0 or
            (args.timeout is not None and args.timeout <= 0) or
            args.min_available_mb < 0 or args.max_swap_pages < 0):
        parser.error('command required; numeric limits must be non-negative')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = measure(command, args.interval, args.settle, args.timeout)
    available_floor = int(args.min_available_mb * 1_000_000)
    swap_pages = sum(result['swap_activity_pages'].values())
    result['acceptance'] = {
        'min_available_bytes': available_floor,
        'max_swap_pages': args.max_swap_pages,
        'available_pass': result['minimum_available_bytes'] >= available_floor,
        'swap_pass': swap_pages <= args.max_swap_pages,
    }
    result['acceptance']['passed'] = (
        not result['timed_out'] and result['returncode'] == 0 and
        result['acceptance']['available_pass'] and
        result['acceptance']['swap_pass'])
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(f"System peak: {result['system_peak_used_bytes'] / 1e9:.3f} GB; "
          f"minimum available: {result['minimum_available_bytes'] / 1e6:.1f} MB; "
          f"gate={'PASS' if result['acceptance']['passed'] else 'FAIL'}; "
          f"exit={result['returncode']}; result={args.out}")
    if result['returncode']:
        raise SystemExit(result['returncode'] if result['returncode'] >= 0
                         else 128 - result['returncode'])
    raise SystemExit(0 if result['acceptance']['passed'] else 3)


if __name__ == '__main__':
    main()
