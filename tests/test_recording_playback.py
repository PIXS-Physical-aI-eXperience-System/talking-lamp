import csv
import sys
from pathlib import Path

import pytest


RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "lelamp_runtime"
sys.path.insert(0, str(RUNTIME_ROOT))

from lelamp.playback import (  # noqa: E402
    HOME_POSE,
    JOINT_KEYS,
    MOTION_SCALES,
    RecordingValidationError,
    SLEEP_POSE,
    load_recording,
    park_and_disconnect,
    play_actions,
    prepare_playback,
    recording_path,
    retarget_actions,
    resample_actions,
)
from lelamp.motor_tuning import position_p_coefficient  # noqa: E402
from lelamp.test.test_motors import select_recordings  # noqa: E402


def _pose(value: float) -> dict[str, float]:
    return {joint: value for joint in JOINT_KEYS}


def test_loaded_pitch_joints_use_factory_position_gain():
    assert position_p_coefficient("base_pitch") == 32
    assert position_p_coefficient("elbow_pitch") == 32
    assert position_p_coefficient("base_yaw") == 16
    assert position_p_coefficient("wrist_roll") == 16
    assert position_p_coefficient("wrist_pitch") == 16


def test_retarget_actions_starts_at_home_and_scales_relative_motion():
    source = [_pose(20.0), _pose(30.0)]

    result = retarget_actions(source)

    assert result[0] == HOME_POSE
    expected = {
        joint: HOME_POSE[joint] + 10.0 * MOTION_SCALES[joint]
        for joint in JOINT_KEYS
    }
    expected["base_pitch.pos"] = (
        HOME_POSE["base_pitch.pos"] - 10.0 * MOTION_SCALES["base_pitch.pos"]
    )
    assert result[1] == pytest.approx(expected)


def test_bundled_idle_is_retargeted_to_a_subtle_local_motion():
    idle_path = RUNTIME_ROOT / "lelamp" / "recordings" / "idle.csv"

    actions = retarget_actions(load_recording(idle_path))
    spans = {
        joint: max(frame[joint] for frame in actions)
        - min(frame[joint] for frame in actions)
        for joint in JOINT_KEYS
    }

    assert 10.0 <= spans["base_yaw.pos"] <= 40.0
    assert 1.0 <= spans["base_pitch.pos"] <= 3.0
    assert 2.0 <= spans["elbow_pitch.pos"] <= 7.0
    assert 10.0 <= spans["wrist_roll.pos"] <= 30.0
    assert 5.0 <= spans["wrist_pitch.pos"] <= 20.0


def test_original_speed_smoothing_keeps_shy_duration_without_step_insertion():
    shy_path = RUNTIME_ROOT / "lelamp" / "recordings" / "shy.csv"
    actions = retarget_actions(load_recording(shy_path))

    planned = prepare_playback(
        actions,
        current_pose=HOME_POSE,
        source_fps=30.0,
        command_fps=30.0,
        speed=1.0,
        transition_seconds=0.0,
        max_step=2.0,
    )

    assert len(planned) == len(actions)
    assert max(
        abs(planned[index][joint] - planned[index - 1][joint])
        for index in range(1, len(planned))
        for joint in JOINT_KEYS
    ) <= 2.0


def test_park_and_disconnect_reaches_sleep_pose_before_releasing_torque():
    class FakeRobot:
        def __init__(self):
            self.current = _pose(0.0)
            self.events = []

        def get_observation(self):
            return self.current.copy()

        def send_action(self, action):
            self.current = action.copy()
            self.events.append(("move", action.copy()))
            return action

        def disconnect(self):
            self.events.append(("disconnect", self.current.copy()))

    now = 0.0

    def clock():
        return now

    def sleep(delay):
        nonlocal now
        now += delay

    robot = FakeRobot()
    park_and_disconnect(
        robot,
        command_fps=4.0,
        transition_seconds=1.0,
        max_step=30.0,
        clock=clock,
        sleep=sleep,
    )

    assert robot.events[-1] == ("disconnect", pytest.approx(SLEEP_POSE))


def test_sleep_elbow_stays_clear_of_the_calibrated_endpoint():
    assert SLEEP_POSE["elbow_pitch.pos"] == pytest.approx(96.0)


def test_default_sleep_transition_sends_240_frames_in_four_seconds():
    class FakeRobot:
        def __init__(self):
            self.current = HOME_POSE.copy()
            self.actions = []

        def get_observation(self):
            return self.current.copy()

        def send_action(self, action):
            self.current = action.copy()
            self.actions.append(action.copy())
            return action

        def disconnect(self):
            pass

    now = 0.0

    def clock():
        return now

    def sleep(delay):
        nonlocal now
        now += delay

    robot = FakeRobot()
    park_and_disconnect(robot, clock=clock, sleep=sleep)

    assert len(robot.actions) == 240
    assert now == pytest.approx(4.0)


def test_prepare_playback_returns_smoothly_to_requested_pose():
    current = _pose(0.0)
    home = _pose(5.0)

    result = prepare_playback(
        [_pose(10.0), _pose(20.0)],
        current_pose=current,
        source_fps=1.0,
        command_fps=4.0,
        speed=1.0,
        transition_seconds=1.0,
        return_pose=home,
        return_seconds=1.0,
        max_step=3.0,
    )

    assert result[-1] == pytest.approx(home)
    previous = current
    for frame in result:
        assert max(abs(frame[joint] - previous[joint]) for joint in JOINT_KEYS) <= 3.0
        previous = frame


def test_select_recordings_can_run_all_non_idle_motions_in_stable_order():
    available = ["wake_up", "idle", "nod", "curious"]

    assert select_recordings([], available, play_all=True) == [
        "curious",
        "nod",
        "wake_up",
    ]


def test_select_recordings_preserves_requested_order_and_rejects_unknown():
    available = ["curious", "nod"]

    assert select_recordings(["nod", "curious"], available, play_all=False) == [
        "nod",
        "curious",
    ]
    with pytest.raises(ValueError, match="missing"):
        select_recordings(["missing"], available, play_all=False)


def test_half_speed_keeps_command_rate_and_halves_each_step():
    actions = [_pose(0.0), _pose(10.0), _pose(20.0)]

    result = resample_actions(
        actions,
        source_fps=2.0,
        command_fps=4.0,
        speed=0.5,
    )

    assert [frame["base_yaw.pos"] for frame in result] == pytest.approx(
        [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    )


def test_prepare_playback_approaches_first_frame_from_measured_pose():
    current = _pose(0.0)
    recorded = [_pose(30.0), _pose(40.0)]

    result = prepare_playback(
        recorded,
        current_pose=current,
        source_fps=1.0,
        command_fps=4.0,
        speed=1.0,
        transition_seconds=1.0,
    )

    yaw = [frame["base_yaw.pos"] for frame in result]
    assert yaw[:4] == pytest.approx([4.6875, 15.0, 25.3125, 30.0])
    assert yaw[-1] == pytest.approx(40.0)


def test_prepare_playback_subdivides_commands_that_exceed_max_step():
    result = prepare_playback(
        [_pose(30.0), _pose(40.0)],
        current_pose=_pose(0.0),
        source_fps=1.0,
        command_fps=4.0,
        speed=1.0,
        transition_seconds=1.0,
        max_step=3.0,
    )

    previous = _pose(0.0)
    for frame in result:
        assert max(abs(frame[joint] - previous[joint]) for joint in JOINT_KEYS) <= 3.0
        previous = frame
    assert previous["base_yaw.pos"] == pytest.approx(40.0)


def test_load_recording_rejects_missing_joint_column(tmp_path):
    path = tmp_path / "broken.csv"
    columns = ["timestamp", *JOINT_KEYS[:-1]]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: "0" for column in columns})

    with pytest.raises(RecordingValidationError, match="missing columns"):
        load_recording(path)


@pytest.mark.parametrize("bad_value", ["nan", "inf", "101"])
def test_load_recording_rejects_unsafe_position_values(tmp_path, bad_value):
    path = tmp_path / "unsafe.csv"
    columns = ["timestamp", *JOINT_KEYS]
    row = {column: "0" for column in columns}
    row["base_pitch.pos"] = bad_value
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(RecordingValidationError, match="base_pitch.pos"):
        load_recording(path)


@pytest.mark.parametrize(
    ("source_fps", "command_fps", "speed", "transition_seconds"),
    [
        (0.0, 30.0, 0.5, 2.0),
        (30.0, 0.0, 0.5, 2.0),
        (30.0, 30.0, 0.0, 2.0),
        (30.0, 30.0, 0.5, -1.0),
    ],
)
def test_prepare_playback_rejects_invalid_timing(
    source_fps, command_fps, speed, transition_seconds
):
    with pytest.raises(ValueError):
        prepare_playback(
            [_pose(0.0), _pose(1.0)],
            current_pose=_pose(0.0),
            source_fps=source_fps,
            command_fps=command_fps,
            speed=speed,
            transition_seconds=transition_seconds,
        )


def test_play_actions_starts_from_measured_pose_and_reaches_last_frame():
    class FakeRobot:
        def __init__(self):
            self.sent = []

        def get_observation(self):
            return _pose(0.0)

        def send_action(self, action):
            self.sent.append(action.copy())
            return action

    now = 0.0

    def clock():
        return now

    def sleep(delay):
        nonlocal now
        now += delay

    robot = FakeRobot()
    report = play_actions(
        robot,
        [_pose(30.0), _pose(40.0)],
        source_fps=1.0,
        command_fps=4.0,
        speed=1.0,
        transition_seconds=1.0,
        clock=clock,
        sleep=sleep,
    )

    assert robot.sent[0]["base_yaw.pos"] == pytest.approx(4.6875)
    assert robot.sent[-1]["base_yaw.pos"] == pytest.approx(40.0)
    assert report.frames_sent == len(robot.sent)
    assert report.clipped_frames == 0
    assert now == pytest.approx(len(robot.sent) / 4.0)


def test_play_actions_reports_safety_clipping_once_in_its_summary():
    class ClippingRobot:
        def get_observation(self):
            return _pose(0.0)

        def send_action(self, action):
            sent = action.copy()
            sent["base_pitch.pos"] -= 1.0
            return sent

    elapsed = 0.0

    def clock():
        return elapsed

    def sleep(delay):
        nonlocal elapsed
        elapsed += delay

    report = play_actions(
        ClippingRobot(),
        [_pose(10.0)],
        command_fps=2.0,
        transition_seconds=1.0,
        clock=clock,
        sleep=sleep,
    )

    assert report.frames_sent == 2
    assert report.clipped_frames == 2
    assert report.clipped_joints == {"base_pitch.pos": 2}


def test_play_actions_stops_before_sending_another_motor_command():
    class FakeRobot:
        def __init__(self):
            self.sent = []

        def get_observation(self):
            return _pose(0.0)

        def send_action(self, action):
            self.sent.append(action.copy())
            return action

    elapsed = 0.0
    stop = False

    def clock():
        return elapsed

    def sleep(delay):
        nonlocal elapsed, stop
        elapsed += delay
        stop = True

    robot = FakeRobot()
    report = play_actions(
        robot,
        [_pose(10.0)],
        command_fps=30.0,
        transition_seconds=3.0,
        clock=clock,
        sleep=sleep,
        should_stop=lambda: stop,
    )

    assert len(robot.sent) == 1
    assert report.frames_sent == 1
    assert report.interrupted


def test_recording_path_accepts_only_a_recording_in_the_given_directory(tmp_path):
    expected = tmp_path / "nod.csv"
    expected.write_text("unused")

    assert recording_path(tmp_path, "nod") == expected
    with pytest.raises(RecordingValidationError, match="Invalid recording name"):
        recording_path(tmp_path, "../nod")
    with pytest.raises(RecordingValidationError, match="not found"):
        recording_path(tmp_path, "missing")
