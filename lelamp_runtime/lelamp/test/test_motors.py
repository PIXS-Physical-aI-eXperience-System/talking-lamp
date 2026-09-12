import argparse
import time


def select_recordings(requested, available, *, play_all):
    """Resolve an explicit list or every named motion except the idle loop."""
    available_set = set(available)
    if play_all:
        return sorted(name for name in available_set if name != "idle")

    unknown = [name for name in requested if name not in available_set]
    if unknown:
        raise ValueError(
            f"Recording {unknown[0]!r} not found. Available: {', '.join(sorted(available_set))}"
        )
    return list(requested)

def test_motors_service():
    from lelamp.service.motors import MotorsService

    parser = argparse.ArgumentParser(description="Test Motors Service")
    parser.add_argument('--id', type=str, required=True, help='ID of the lamp')
    parser.add_argument('--port', type=str, required=True, help='Serial port for the lamp')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        '--recording',
        type=str,
        action='append',
        default=[],
        help='Recording name to test; repeat this option to play a sequence',
    )
    selection.add_argument(
        '--all',
        action='store_true',
        help='Play every named motion in alphabetical order, excluding idle',
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Motor command rate; keep at 30 for smooth playback.",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=30.0,
        help="Frame rate used when the recording was created.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier; 1.0 uses the original recording speed.",
    )
    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=3.0,
        help="Time used to approach the first recorded pose.",
    )
    parser.add_argument(
        "--max-planned-step",
        type=float,
        default=2.0,
        help="Maximum planned step per 30 Hz command.",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="Optional hardware catch-up clamp (default: disabled).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Maximum seconds to wait for playback to finish.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=2.0,
        help="Seconds to hold each ending pose before starting the next motion.",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Keep the motors connected and holding the final pose until Ctrl-C.",
    )
    args = parser.parse_args()
    
    print("Testing Motors Service...")
    
    motors_service = MotorsService(
        port=args.port,
        lamp_id=args.id,
        fps=args.fps,
        source_fps=args.source_fps,
        speed=args.speed,
        transition_seconds=args.transition_seconds,
        max_planned_step=args.max_planned_step,
        max_relative_target=args.max_relative_target,
    )
    recordings = motors_service.get_available_recordings()
    try:
        selected = select_recordings(args.recording, recordings, play_all=args.all)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    motors_service.start()
    try:
        for index, recording_name in enumerate(selected, start=1):
            print(f"Playing {index}/{len(selected)}: {recording_name}")
            motors_service.dispatch("play", recording_name)
            if not motors_service.wait_until_idle(timeout=args.timeout):
                raise TimeoutError(
                    f"Playback did not finish within {args.timeout:g} seconds"
                )
            if motors_service.last_error is not None:
                raise motors_service.last_error
            print(f"Playback completed: {recording_name}")
            if index < len(selected) and args.pause_seconds > 0:
                time.sleep(args.pause_seconds)

        if args.hold:
            print("Holding final pose with torque enabled. Press Ctrl-C to release.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping and releasing motor torque...")
    finally:
        motors_service.stop()
        print("Motors Service test completed!")

if __name__ == "__main__":
    test_motors_service()
