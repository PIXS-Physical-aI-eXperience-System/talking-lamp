"""Validated, time-scaled playback planning for LeLamp recording files."""

from __future__ import annotations

import csv
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


JOINT_KEYS = (
    "base_yaw.pos",
    "base_pitch.pos",
    "elbow_pitch.pos",
    "wrist_roll.pos",
    "wrist_pitch.pos",
)

# Relaxed operating home derived from the manually measured vertical reference.
# Physical offsets from straight: yaw 15°, base pitch 30° away from the former
# forward extension, elbow 112°, wrist pitch 12°. The calibration reference remains in
# sim/hardware_alignment.json.
HOME_POSE = {
    "base_yaw.pos": -20.82092022509103,
    "base_pitch.pos": 34.23930021868165,
    "elbow_pitch.pos": 66.02336564752673,
    "wrist_roll.pos": -0.5316578057032331,
    "wrist_pitch.pos": 9.380740740740734,
}

# Captured on the physical lamp with torque disabled on 2026-09-12. Keep the
# elbow four degrees inside its calibrated endpoint so the controller does not
# chatter while pressing against the mechanical stop.
SLEEP_POSE = {
    "base_yaw.pos": -9.53326713,
    "base_pitch.pos": 61.855670103,
    "elbow_pitch.pos": 96.0,
    "wrist_roll.pos": -0.628322861,
    "wrist_pitch.pos": 45.244444444,
}

# The bundled CSVs were recorded as absolute poses on another lamp. Keep the
# official trajectory shape, but make idle-sized gestures subtle enough for
# this assembled lamp to track under load around its local home.
MOTION_SCALES = {
    "base_yaw.pos": 0.35,
    "base_pitch.pos": 0.12,
    "elbow_pitch.pos": 0.15,
    "wrist_roll.pos": 0.35,
    "wrist_pitch.pos": 0.25,
}

MOTION_DIRECTIONS = {
    "base_yaw.pos": 1.0,
    "base_pitch.pos": -1.0,
    "elbow_pitch.pos": 1.0,
    "wrist_roll.pos": 1.0,
    "wrist_pitch.pos": 1.0,
}

DEFAULT_SMOOTHING_ALPHA = 0.15
SLEEP_COMMAND_FPS = 60.0
SLEEP_TRANSITION_SECONDS = 4.0
SLEEP_MAX_STEP = 1.0


class RecordingValidationError(ValueError):
    """Raised when a recording cannot be sent safely to the normalized joints."""


@dataclass(frozen=True)
class PlaybackReport:
    frames_sent: int
    clipped_frames: int
    clipped_joints: dict[str, int]
    interrupted: bool


def recording_path(directory: str | Path, recording_name: str) -> Path:
    """Resolve a recording stem without allowing path traversal or extensions."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", recording_name):
        raise RecordingValidationError(
            f"Invalid recording name {recording_name!r}; use letters, numbers, _ or -"
        )
    path = Path(directory) / f"{recording_name}.csv"
    if not path.is_file():
        raise RecordingValidationError(f"Recording not found: {path}")
    return path


def _validated_pose(
    pose: Mapping[str, object], *, source: str, position_limit: float = 100.0
) -> dict[str, float]:
    missing = [joint for joint in JOINT_KEYS if joint not in pose]
    if missing:
        raise RecordingValidationError(f"{source}: missing columns: {', '.join(missing)}")

    result: dict[str, float] = {}
    for joint in JOINT_KEYS:
        try:
            value = float(pose[joint])
        except (TypeError, ValueError) as exc:
            raise RecordingValidationError(
                f"{source}: {joint} is not a number: {pose[joint]!r}"
            ) from exc
        if not math.isfinite(value) or abs(value) > position_limit:
            raise RecordingValidationError(
                f"{source}: {joint}={value!r} is outside "
                f"[-{position_limit:g}, {position_limit:g}]"
            )
        result[joint] = value
    return result


def load_recording(path: str | Path) -> list[dict[str, float]]:
    """Load a normalized-position CSV after validating every motor command."""
    recording_path = Path(path)
    try:
        with recording_path.open("r", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise RecordingValidationError(
            f"Cannot read recording {recording_path}: {exc}"
        ) from exc

    if not rows:
        raise RecordingValidationError(f"{recording_path}: recording is empty")

    return [
        _validated_pose(row, source=f"{recording_path}: row {index}")
        for index, row in enumerate(rows, start=2)
    ]


def retarget_actions(
    actions: Sequence[Mapping[str, float]],
    *,
    home_pose: Mapping[str, float] = HOME_POSE,
    motion_scales: Mapping[str, float] = MOTION_SCALES,
    motion_directions: Mapping[str, float] = MOTION_DIRECTIONS,
) -> list[dict[str, float]]:
    """Convert foreign absolute poses into motion around the local CAD home."""
    if not actions:
        return []

    home = _validated_pose(home_pose, source="home pose")
    first = _validated_pose(actions[0], source="retarget first frame")
    scales = {
        joint: float(motion_scales[joint])
        for joint in JOINT_KEYS
    }
    directions = {
        joint: float(motion_directions[joint])
        for joint in JOINT_KEYS
    }
    if any(not math.isfinite(scale) or scale < 0 for scale in scales.values()):
        raise ValueError("motion scales must be finite and non-negative")
    if any(direction not in (-1.0, 1.0) for direction in directions.values()):
        raise ValueError("motion directions must be -1 or 1")

    result = []
    for index, action in enumerate(actions):
        pose = _validated_pose(action, source=f"retarget frame {index}")
        retargeted = {
            joint: home[joint]
            + (pose[joint] - first[joint]) * scales[joint] * directions[joint]
            for joint in JOINT_KEYS
        }
        result.append(
            _validated_pose(retargeted, source=f"retargeted frame {index}")
        )
    return result


def smooth_actions(
    actions: Sequence[Mapping[str, float]],
    *,
    alpha: float = DEFAULT_SMOOTHING_ALPHA,
) -> list[dict[str, float]]:
    """Reduce frame-to-frame recording noise without changing frame count."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("smoothing alpha must be in (0, 1]")
    if not actions:
        return []

    state = _validated_pose(actions[0], source="smoothing first frame")
    result = [state.copy()]
    for index, action in enumerate(actions[1:], start=1):
        target = _validated_pose(action, source=f"smoothing frame {index}")
        state = {
            joint: state[joint] + alpha * (target[joint] - state[joint])
            for joint in JOINT_KEYS
        }
        result.append(state)

    # A causal filter naturally trails the final sample. Spread that residual
    # across the clip so both endpoints and the original duration stay exact.
    final_target = _validated_pose(actions[-1], source="smoothing final frame")
    residual = {
        joint: final_target[joint] - result[-1][joint] for joint in JOINT_KEYS
    }
    denominator = max(1, len(result) - 1)
    for index, frame in enumerate(result):
        progress = index / denominator
        for joint in JOINT_KEYS:
            frame[joint] += residual[joint] * progress
    return result


def _validate_timing(source_fps: float, command_fps: float, speed: float) -> None:
    if source_fps <= 0:
        raise ValueError("source_fps must be greater than zero")
    if command_fps <= 0:
        raise ValueError("command_fps must be greater than zero")
    if speed <= 0:
        raise ValueError("speed must be greater than zero")


def resample_actions(
    actions: Sequence[Mapping[str, float]],
    *,
    source_fps: float,
    command_fps: float,
    speed: float,
) -> list[dict[str, float]]:
    """Linearly resample a clip while keeping a steady motor command rate."""
    _validate_timing(source_fps, command_fps, speed)
    if not actions:
        return []

    poses = [
        _validated_pose(action, source=f"frame {index}")
        for index, action in enumerate(actions)
    ]
    if len(poses) == 1:
        return [poses[0].copy()]

    output_intervals = max(
        1,
        int(round((len(poses) - 1) * command_fps / (source_fps * speed))),
    )
    result: list[dict[str, float]] = []
    for output_index in range(output_intervals + 1):
        source_position = (len(poses) - 1) * output_index / output_intervals
        left = min(int(math.floor(source_position)), len(poses) - 2)
        fraction = source_position - left
        right = left + 1
        result.append(
            {
                joint: poses[left][joint]
                + (poses[right][joint] - poses[left][joint]) * fraction
                for joint in JOINT_KEYS
            }
        )
    return result


def transition_actions(
    current_pose: Mapping[str, float],
    target_pose: Mapping[str, float],
    *,
    command_fps: float,
    duration: float,
) -> list[dict[str, float]]:
    if duration < 0:
        raise ValueError("transition_seconds must not be negative")
    start = _validated_pose(current_pose, source="current pose")
    target = _validated_pose(target_pose, source="first recording frame")
    if duration == 0:
        return [target]

    frame_count = max(1, int(round(duration * command_fps)))
    result: list[dict[str, float]] = []
    for frame_index in range(1, frame_count + 1):
        progress = frame_index / frame_count
        eased = progress * progress * (3.0 - 2.0 * progress)
        result.append(
            {
                joint: start[joint] + (target[joint] - start[joint]) * eased
                for joint in JOINT_KEYS
            }
        )
    return result


def limit_action_steps(
    actions: Sequence[Mapping[str, float]],
    *,
    start_pose: Mapping[str, float],
    max_step: float | None,
) -> list[dict[str, float]]:
    """Insert commands so no planned joint step exceeds ``max_step``."""
    if max_step is None:
        return [dict(action) for action in actions]
    if max_step <= 0:
        raise ValueError("max_step must be greater than zero")

    previous = _validated_pose(start_pose, source="step limiter start pose")
    limited: list[dict[str, float]] = []
    for index, action in enumerate(actions):
        target = _validated_pose(action, source=f"step limiter frame {index}")
        largest_delta = max(
            abs(target[joint] - previous[joint]) for joint in JOINT_KEYS
        )
        segments = max(1, int(math.ceil(largest_delta / max_step)))
        for segment in range(1, segments + 1):
            fraction = segment / segments
            limited.append(
                {
                    joint: previous[joint]
                    + (target[joint] - previous[joint]) * fraction
                    for joint in JOINT_KEYS
                }
            )
        previous = target
    return limited


def prepare_playback(
    actions: Sequence[Mapping[str, float]],
    *,
    current_pose: Mapping[str, float],
    source_fps: float = 30.0,
    command_fps: float = 30.0,
    speed: float = 0.5,
    transition_seconds: float = 3.0,
    return_pose: Mapping[str, float] | None = None,
    return_seconds: float | None = None,
    max_step: float | None = None,
    smoothing_alpha: float | None = DEFAULT_SMOOTHING_ALPHA,
) -> list[dict[str, float]]:
    """Build a smooth transition followed by a time-scaled recording."""
    _validate_timing(source_fps, command_fps, speed)
    if transition_seconds < 0:
        raise ValueError("transition_seconds must not be negative")
    source_actions = (
        smooth_actions(actions, alpha=smoothing_alpha)
        if smoothing_alpha is not None
        else actions
    )
    scaled = resample_actions(
        source_actions,
        source_fps=source_fps,
        command_fps=command_fps,
        speed=speed,
    )
    if not scaled:
        return []
    transition = transition_actions(
        current_pose,
        scaled[0],
        command_fps=command_fps,
        duration=transition_seconds,
    )
    planned = transition + scaled[1:]
    if return_pose is not None:
        duration = transition_seconds if return_seconds is None else return_seconds
        planned += transition_actions(
            scaled[-1],
            return_pose,
            command_fps=command_fps,
            duration=duration,
        )
    return limit_action_steps(
        planned,
        start_pose=current_pose,
        max_step=max_step,
    )


def play_actions(
    robot,
    actions: Sequence[Mapping[str, float]],
    *,
    source_fps: float = 30.0,
    command_fps: float = 30.0,
    speed: float = 0.5,
    transition_seconds: float = 3.0,
    return_pose: Mapping[str, float] | None = None,
    return_seconds: float | None = None,
    max_step: float | None = None,
    smoothing_alpha: float | None = DEFAULT_SMOOTHING_ALPHA,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] = lambda: False,
) -> PlaybackReport:
    """Send one smoothly planned clip and return aggregate limiter activity."""
    current_pose = robot.get_observation()
    planned = prepare_playback(
        actions,
        current_pose=current_pose,
        source_fps=source_fps,
        command_fps=command_fps,
        speed=speed,
        transition_seconds=transition_seconds,
        return_pose=return_pose,
        return_seconds=return_seconds,
        max_step=max_step,
        smoothing_alpha=smoothing_alpha,
    )

    clipped_frames = 0
    clipped_joints: Counter[str] = Counter()
    frames_sent = 0
    interrupted = False
    frame_period = 1.0 / command_fps
    for action in planned:
        if should_stop():
            interrupted = True
            break
        # Keep every planned (step-limited) sample, but start its period at
        # the actual send time. A late send or wakeup stretches playback;
        # expired periods must never cause a burst or skipped-pose jump.
        deadline = clock() + frame_period
        sent_action = robot.send_action(action)
        frames_sent += 1
        clipped = [
            joint
            for joint in JOINT_KEYS
            if joint not in sent_action
            or abs(action[joint] - sent_action[joint]) > 1e-6
        ]
        if clipped:
            clipped_frames += 1
            clipped_joints.update(clipped)

        remaining = deadline - clock()
        # If the write itself overran, it may only just have reached the
        # servo. Give it a fresh period before sending the following step.
        sleep(remaining if remaining > 0 else frame_period)

    return PlaybackReport(
        frames_sent=frames_sent,
        clipped_frames=clipped_frames,
        clipped_joints=dict(clipped_joints),
        interrupted=interrupted,
    )


def move_to_sleep(
    robot,
    *,
    command_fps: float = SLEEP_COMMAND_FPS,
    transition_seconds: float = SLEEP_TRANSITION_SECONDS,
    max_step: float = SLEEP_MAX_STEP,
    position_tolerance: float = 2.0,
    settle_timeout: float = 5.0,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> PlaybackReport:
    """Move to the captured sleep pose and confirm it before torque release."""
    report = play_actions(
        robot,
        [SLEEP_POSE],
        source_fps=1.0,
        command_fps=command_fps,
        speed=1.0,
        transition_seconds=transition_seconds,
        max_step=max_step,
        clock=clock,
        sleep=sleep,
    )

    deadline = clock() + settle_timeout
    while True:
        actual = _validated_pose(robot.get_observation(), source="sleep observation")
        max_error = max(
            abs(actual[joint] - SLEEP_POSE[joint]) for joint in JOINT_KEYS
        )
        if max_error <= position_tolerance:
            return report
        if clock() >= deadline:
            raise TimeoutError(
                f"Sleep pose was not reached before torque release; max error={max_error:.2f}"
            )
        robot.send_action(SLEEP_POSE)
        sleep(1.0 / command_fps)


def park_and_disconnect(robot, **kwargs) -> PlaybackReport:
    """Reach the captured sleep pose, then disconnect and release torque."""
    report = move_to_sleep(robot, **kwargs)
    robot.disconnect()
    return report
