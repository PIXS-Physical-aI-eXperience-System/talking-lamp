"""Exercise real service startup/parking with only the follower and clock faked."""

import importlib
from functools import partial
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lelamp_runtime"))

from lelamp.playback import HOME_POSE, SLEEP_POSE, park_and_disconnect  # noqa: E402


@pytest.fixture(params=["motors", "animation"])
def service_case(request, monkeypatch):
    class Robot:
        def __init__(self):
            self.is_connected = False
            self.bus = self
            self.pose = HOME_POSE.copy()
            self.connect_error = None
            self.fail_before_open = False
            self.read_error = None
            self.park_error = None
            self.released_at = None

        def connect(self, *, calibrate):
            assert calibrate is False
            if self.fail_before_open:
                raise self.connect_error
            self.is_connected = True
            if self.connect_error:
                raise self.connect_error

        def get_observation(self):
            if not self.is_connected:
                raise OSError("bus is closed")
            error, self.read_error = self.read_error, None
            if error:
                raise error
            return self.pose.copy()

        def send_action(self, action):
            if self.park_error:
                raise self.park_error
            self.pose = action.copy()
            return action

        def disconnect(self):
            self.released_at = self.pose.copy()
            self.is_connected = False

    robot = Robot()
    follower = SimpleNamespace(
        LeLampFollower=lambda config: robot, LeLampFollowerConfig=SimpleNamespace
    )
    monkeypatch.setitem(sys.modules, "lelamp.follower", follower)
    module = importlib.import_module(f"lelamp.service.motors.{request.param}_service")
    monkeypatch.setattr(module, "LeLampFollower", follower.LeLampFollower)
    monkeypatch.setattr(module, "LeLampFollowerConfig", SimpleNamespace)
    now = 0.0

    def sleep(delay):
        nonlocal now
        now += delay

    monkeypatch.setattr(module, "park_and_disconnect", partial(
        park_and_disconnect, clock=lambda: now, sleep=sleep
    ))
    cls = module.AnimationService if request.param == "animation" else module.MotorsService
    return cls(port="fake", lamp_id="test"), robot, request.param


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_partial_connection_failure_parks_before_releasing(service_case, error_type):
    service, robot, _ = service_case
    robot.connect_error = error_type("connect interrupted after opening bus")
    with pytest.raises(error_type):
        service.start()
    assert robot.released_at == pytest.approx(SLEEP_POSE)
    assert not robot.is_connected
    assert service.robot is None
    assert not service._running.is_set()


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("service_case", ["animation"], indirect=True)
def test_first_animation_observation_failure_parks(service_case, error_type):
    service, robot, _ = service_case
    robot.read_error = error_type("first observation failed")
    with pytest.raises(error_type):
        service.start()
    assert robot.released_at == pytest.approx(SLEEP_POSE)
    assert not robot.is_connected
    assert service.robot is None


def test_worker_start_interrupt_still_parks(service_case, monkeypatch):
    import threading

    service, robot, _ = service_case

    def interrupt_start(thread):
        raise KeyboardInterrupt("worker startup interrupted")

    monkeypatch.setattr(threading.Thread, "start", interrupt_start)
    with pytest.raises(KeyboardInterrupt):
        service.start()
    assert robot.released_at == pytest.approx(SLEEP_POSE)
    assert not service._running.is_set()
    assert service.robot is None


def test_failed_startup_parking_retains_recoverable_ownership(service_case):
    service, robot, _ = service_case
    robot.connect_error = OSError("startup failed")
    robot.park_error = TimeoutError("parking failed")
    with pytest.raises(TimeoutError, match="parking failed") as caught:
        service.start()
    assert isinstance(caught.value.__context__, OSError)
    assert service.robot is robot
    assert robot.is_connected
    assert robot.released_at is None
    with pytest.raises(RuntimeError, match="owned"):
        service.start()
    assert service.robot is robot
    robot.park_error = None
    service.stop()
    assert robot.released_at == pytest.approx(SLEEP_POSE)
    assert service.robot is None


def test_connection_failure_before_open_needs_no_parking(service_case):
    service, robot, _ = service_case
    robot.fail_before_open = True
    robot.connect_error = OSError("port open failed")
    with pytest.raises(OSError, match="port open failed"):
        service.start()
    assert robot.released_at is None
    assert not robot.is_connected
    assert service.robot is None


@pytest.mark.parametrize("service_case", ["animation"], indirect=True)
def test_animation_restart_clears_stale_playback_before_worker_starts(
    service_case, monkeypatch
):
    service, robot, _ = service_case
    service._current_recording = "stale"
    service._current_actions = [{"base_yaw.pos": 99.0}]
    service._current_frame_index = 7
    service._event_queue = [("play", "stale")]
    observed = []

    def inspect_start(thread):
        observed.append((
            service._current_recording,
            service._current_actions.copy(),
            service._current_frame_index,
            service._event_queue.copy(),
        ))

    monkeypatch.setattr("threading.Thread.start", inspect_start)
    service.start()
    try:
        assert observed == [(None, [], 0, [("play", service.idle_recording)])]
        assert robot.pose == pytest.approx(HOME_POSE)
    finally:
        service.stop()
