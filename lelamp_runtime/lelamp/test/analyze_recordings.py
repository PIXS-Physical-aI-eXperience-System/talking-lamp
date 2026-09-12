import argparse
from pathlib import Path

from lelamp.playback import JOINT_KEYS, load_recording


def analyze_recording(path: Path, max_step: float):
    rows = load_recording(path)
    parsed = {joint: [row[joint] for row in rows] for joint in JOINT_KEYS}

    worst = {}
    ranges = {}
    violations = []
    for joint in JOINT_KEYS:
        ranges[joint] = (min(parsed[joint]), max(parsed[joint]))
        deltas = [abs(v2 - v1) for v1, v2 in zip(parsed[joint], parsed[joint][1:])]
        if not deltas:
            worst[joint] = 0.0
            continue

        max_delta = max(deltas)
        worst[joint] = max_delta

        if max_delta > max_step:
            idx = deltas.index(max_delta)
            start = parsed[joint][idx]
            end = parsed[joint][idx + 1]
            violations.append((joint, idx, max_delta, start, end))

    return {
        'name': path.name,
        'frames': len(rows),
        'joints': worst,
        'ranges': ranges,
        'violations': violations,
        'safe': len(violations) == 0,
    }


def print_summary(report):
    status = "SAFE" if report['safe'] else "CHECK REQUIRED"
    print(f"{report['name']}: frames={report['frames']} status={status}")
    for joint, delta in sorted(report['joints'].items()):
        low, high = report['ranges'][joint]
        print(f"  {joint}: range {low:.3f}..{high:.3f}, max source step {delta:.3f}")

    for joint, idx, delta, start, end in report['violations']:
        print(
            f"    - {joint} frame {idx}->{idx + 1}: {delta:.3f} "
            f"({start:.3f} -> {end:.3f})"
        )


def main():
    parser = argparse.ArgumentParser(description='Analyze movement CSV files for large per-frame jumps')
    default_dir = Path(__file__).resolve().parents[1] / 'recordings'
    parser.add_argument('--dir', type=Path, default=default_dir, help='Directory containing CSV recordings')
    parser.add_argument('--recording', default=None, help='Only analyze one recording name (without extension)')
    parser.add_argument('--max-step', type=float, default=3.0, help='Max allowed step per frame')
    args = parser.parse_args()

    root = args.dir
    if not root.exists():
        raise SystemExit(f'Recordings directory not found: {root}')

    files = [root / f'{args.recording}.csv'] if args.recording else sorted(root.glob('*.csv'))
    if args.recording and not files[0].exists():
        raise SystemExit(f'Recording not found: {files[0]}')

    all_ok = True
    for path in files:
        report = analyze_recording(path, args.max_step)
        print_summary(report)
        all_ok = all_ok and report['safe']

    if not all_ok:
        raise SystemExit(
            'Source-frame jump detected. Playback now interpolates these frames; '
            'inspect the reported joint ranges before a hardware replay.'
        )


if __name__ == '__main__':
    main()
