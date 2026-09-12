"""Physical boundary tests use a robot double; no serial ports are opened."""
import importlib
import sys

import numpy as np
import pytest

from motion.config import JOINT_NAMES, REST_POSE
from motion.hardware_alignment import HardwareAlignment
from motion.runtime import MotionRuntime


class FakeRobot:
    def __init__(self):
        self.is_connected = False
        self.connect_args = []
        self.observation = {f"{j}.pos": 0.0 for j in JOINT_NAMES}
        self.reads = 0
        self.actions = []

    def connect(self, *, calibrate):
        self.connect_args.append(calibrate)
        self.is_connected = True

    def get_observation(self):
        self.reads += 1
        return self.observation.copy()

    def send_action(self, action):
        self.actions.append(action)
        return action


def make_backend(**kwargs):
    module = importlib.import_module("motion.hardware_backend")
    robot = FakeRobot()
    parked = []
    backend = module.FeetechBackend(
        port="fake", lamp_id="test", robot_factory=lambda **kw: robot,
        park=parked.append, **kwargs,
    )
    return backend, robot, parked


def test_import_does_not_load_lerobot():
    importlib.import_module("motion.hardware_backend")
    assert not any(name == "lerobot" or name.startswith("lerobot.") for name in sys.modules)


def test_connect_measures_immediately_without_calibration():
    backend, robot, _ = make_backend()
    assert robot.connect_args == [False]
    assert robot.reads == 1
    expected = HardwareAlignment.load().normalized_to_radians(np.zeros(5))
    np.testing.assert_allclose(backend.measured(), expected)
    backend.measured()[:] = 123
    np.testing.assert_allclose(backend.measured(), expected)


def test_send_converts_clamps_and_decimates_reads():
    backend, robot, _ = make_backend(feedback_hz=20)
    alignment = HardwareAlignment.load()
    command = alignment.normalized_to_radians(np.array([120, -110, 30, 40, 50]))
    for _ in range(100):
        backend.send(command)
        backend.measured()
    assert len(robot.actions) == 100
    assert robot.reads == 21
    np.testing.assert_allclose(list(robot.actions[-1].values()), [100, -100, 30, 40, 50])
    assert backend.sent_ticks == 100
    assert backend.clamped_ticks == 100
    assert backend.clamped_joints == {"base_yaw": 100, "base_pitch": 100}


@pytest.mark.parametrize("q", [np.zeros(4), np.full(5, np.nan), np.full(5, np.inf)])
def test_invalid_command_never_reaches_robot(q):
    backend, robot, _ = make_backend()
    with pytest.raises(ValueError):
        backend.send(q)
    assert robot.actions == []


def test_close_parks_once_and_rejects_further_commands():
    backend, robot, parked = make_backend()
    backend.close()
    backend.close()
    assert parked == [robot]
    with pytest.raises(RuntimeError):
        backend.send(REST_POSE)


def test_failed_initial_read_still_parks():
    module = importlib.import_module("motion.hardware_backend")
    robot = FakeRobot()
    robot.observation = {}
    parked = []
    with pytest.raises((KeyError, ValueError)):
        module.FeetechBackend(port="fake", lamp_id="test", robot_factory=lambda **kw: robot,
                              park=parked.append)
    assert parked == [robot]


def test_runtime_initial_measurement_seeds_motion_but_preserves_idle_target():
    initial = REST_POSE + 0.2
    rt = MotionRuntime(initial_pose=initial)
    np.testing.assert_allclose(rt.traj.pos, initial)
    np.testing.assert_allclose(rt.track.q, initial)
    np.testing.assert_allclose(rt.rest_pose, REST_POSE)
    first = rt.step()
    assert np.max(np.abs(first.q_cmd - initial)) < 0.003
    assert np.max(np.abs(first.q_blend - REST_POSE)) < 0.05


def test_missing_hardware_dependency_has_actionable_error(monkeypatch):
    import builtins
    module = importlib.import_module("motion.hardware_backend")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("lelamp."):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="PYTHONPATH"):
        module.FeetechBackend(port="fake", lamp_id="test")


def test_real_factory_disables_per_send_position_read_and_degree_mode(monkeypatch):
    from types import SimpleNamespace
    module = importlib.import_module("motion.hardware_backend")
    configs = []
    robot = FakeRobot()

    def follower(config):
        configs.append(config)
        return robot

    monkeypatch.setitem(sys.modules, "lelamp.follower.lelamp_follower",
                        SimpleNamespace(LeLampFollower=follower))
    monkeypatch.setitem(sys.modules, "lelamp.follower.config_lelamp_follower",
                        SimpleNamespace(LeLampFollowerConfig=SimpleNamespace))
    monkeypatch.setitem(sys.modules, "lelamp.playback",
                        SimpleNamespace(park_and_disconnect=lambda r: None))
    module.FeetechBackend(port="fake", lamp_id="existing")
    assert vars(configs[0]) == dict(port="fake", id="existing", use_degrees=False,
                                   max_relative_target=None, cameras={},
                                   disable_torque_on_disconnect=True)


def test_fractional_feedback_rate_is_not_rounded_to_command_divisor():
    backend, robot, _ = make_backend(feedback_hz=12.5)
    for _ in range(200):
        backend.send(REST_POSE)
    assert robot.reads == 26


@pytest.mark.parametrize("rate", [0, -1, 101, float("nan"), float("inf")])
def test_invalid_feedback_rate_rejected_before_connect(rate):
    module = importlib.import_module("motion.hardware_backend")
    robot = FakeRobot()
    with pytest.raises(ValueError):
        module.FeetechBackend(port="fake", lamp_id="test", feedback_hz=rate,
                              robot_factory=lambda **kw: robot, park=lambda r: None)
    assert robot.connect_args == []


def test_parking_failure_propagates_without_releasing_torque():
    module = importlib.import_module("motion.hardware_backend")
    robot = FakeRobot()

    def failed_park(robot):
        raise TimeoutError("sleep pose not reached")

    backend = module.FeetechBackend(port="fake", lamp_id="test",
                                   robot_factory=lambda **kw: robot, park=failed_park)
    with pytest.raises(TimeoutError, match="sleep pose"):
        backend.close()
    assert robot.is_connected
