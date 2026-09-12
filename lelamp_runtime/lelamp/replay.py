import argparse
import os

from .follower import LeLampFollowerConfig, LeLampFollower
from .playback import load_recording, park_and_disconnect, play_actions, recording_path, retarget_actions

def main():
    parser = argparse.ArgumentParser(description="Replay recorded actions from CSV file")
    parser.add_argument('--name', type=str, required=True, help='Name of the recording to replay')
    parser.add_argument('--port', type=str, required=True, help='Serial port for the robot')
    parser.add_argument('--id', type=str, required=True, help='ID of the robot')
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Motor command rate; keep at 30 for smooth playback (default: 30)',
    )
    parser.add_argument(
        '--source-fps',
        type=float,
        default=30.0,
        help='Frame rate used when the CSV was recorded (default: 30)',
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Playback speed multiplier (default: 1.0, original recording speed)',
    )
    parser.add_argument(
        '--transition-seconds',
        type=float,
        default=3.0,
        help='Time used to approach the first recorded pose (default: 3.0)',
    )
    parser.add_argument(
        '--max-planned-step',
        type=float,
        default=2.0,
        help='Maximum planned step per 30 Hz command after smoothing (default: 2.0)',
    )
    parser.add_argument(
        '--max-relative-target',
        type=float,
        default=None,
        help='Optional hardware catch-up clamp (default: disabled)',
    )
    args = parser.parse_args()

    recordings_dir = os.path.join(os.path.dirname(__file__), "recordings")
    csv_path = recording_path(recordings_dir, args.name)
    actions = retarget_actions(load_recording(csv_path))

    robot_config = LeLampFollowerConfig(
        port=args.port,
        id=args.id,
        max_relative_target=args.max_relative_target,
    )
    robot = LeLampFollower(robot_config)
    print(
        f"Replaying {len(actions)} source frames from {csv_path} "
        f"at {args.speed:g}x speed"
    )
    try:
        robot.connect(calibrate=False)
        report = play_actions(
            robot,
            actions,
            source_fps=args.source_fps,
            command_fps=args.fps,
            speed=args.speed,
            transition_seconds=args.transition_seconds,
            max_step=args.max_planned_step,
        )
        print(f"Sent {report.frames_sent} interpolated motor commands")
        if report.clipped_frames:
            details = ", ".join(
                f"{joint}={count}" for joint, count in report.clipped_joints.items()
            )
            print(
                f"Safety limit clipped {report.clipped_frames} frames: {details}"
            )
    finally:
        if robot.is_connected:
            print("Moving to sleep pose before torque release")
            park_and_disconnect(robot)

if __name__ == "__main__":
    main()
