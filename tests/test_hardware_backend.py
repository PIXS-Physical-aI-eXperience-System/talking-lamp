"""Physical boundary tests use a robot double; no serial ports are opened."""
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from motion.config import ACC_LIMIT, CONTROL_DT, JERK_LIMIT, JOINT_NAMES, REST_POSE, VEL_LIMIT
from motion.hardware_alignment import HardwareAlignment
from motion.primitives import Primitive, PrimitiveLibrary
from motion.runtime import MotionRuntime
from motion.sim_backend import MujocoKinematicsBackend


class FakeRobot:
    def __init__(self):
        self.is_connected = False
        self.connect_args = []
        self.observation = {f"{j}.pos": 0.0 for j in JOINT_NAMES}
        self.reads = 0
        self.actions = []
        homing_offsets = (-653, -1406, 1767, -1491, 2003)
        alignment = HardwareAlignment.load()
        self.calibration = {
            joint: SimpleNamespace(
                id=index,
                drive_mode=0,
                homing_offset=homing_offset,
                range_min=int(raw_range[0]),
                range_max=int(raw_range[1]),
            )
            for index, (joint, homing_offset, raw_range) in enumerate(
                zip(JOINT_NAMES, homing_offsets, alignment.raw_ranges), start=1
            )
        }

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
        port="fake", lamp_id="lelamp", robot_factory=lambda **kw: robot,
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


def test_wrong_lamp_id_is_rejected_before_constructing_robot():
    module = importlib.import_module("motion.hardware_backend")
    constructed = []
    with pytest.raises(ValueError, match="calibrated for 'lelamp'"):
        module.FeetechBackend(
            port="fake",
            lamp_id="another-lamp",
            robot_factory=lambda **kw: constructed.append(kw),
            park=lambda robot: None,
        )
    assert constructed == []


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [("id", 42), ("homing_offset", -1405), ("range_max", 3072)],
)
def test_saved_calibration_mismatch_is_rejected_before_connect(field, wrong_value):
    module = importlib.import_module("motion.hardware_backend")
    robot = FakeRobot()
    setattr(robot.calibration["base_pitch"], field, wrong_value)
    with pytest.raises(ValueError, match="base_pitch"):
        module.FeetechBackend(
            port="fake",
            lamp_id="lelamp",
            robot_factory=lambda **kw: robot,
            park=lambda robot: None,
        )
    assert robot.connect_args == []


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
        module.FeetechBackend(port="fake", lamp_id="lelamp", robot_factory=lambda **kw: robot,
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
        module.FeetechBackend(port="fake", lamp_id="lelamp")


def test_real_factory_disables_per_send_position_read_and_degree_mode(monkeypatch):
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
    module.FeetechBackend(port="fake", lamp_id="lelamp")
    assert vars(configs[0]) == dict(port="fake", id="lelamp", use_degrees=False,
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

    backend = module.FeetechBackend(port="fake", lamp_id="lelamp",
                                   robot_factory=lambda **kw: robot, park=failed_park)
    with pytest.raises(TimeoutError, match="sleep pose"):
        backend.close()
    assert robot.is_connected


@pytest.mark.parametrize("direction", [-1, 1])
def test_tracking_plus_expression_brakes_within_calibrated_command_limits(direction):
    alignment = HardwareAlignment.load()
    limits = alignment.joint_limits
    initial = REST_POSE.copy()
    initial[0] = limits[0, 1 if direction > 0 else 0] - direction * 0.12
    offsets = np.zeros((4, 5))
    offsets[1:3, 0] = direction * 0.65
    clip = Primitive("boundary", np.array([0.0, 0.2, 3.0, 3.2]), offsets, loop=True)
    backend, robot, _ = make_backend()
    sim = MujocoKinematicsBackend(q0=initial)
    physical = MotionRuntime(
        backend=backend, initial_pose=initial,
        primitives=PrimitiveLibrary(_cache={"boundary": clip}),
    )
    simulated = MotionRuntime(
        backend=sim, initial_pose=initial,
        primitives=PrimitiveLibrary(_cache={"boundary": clip}),
    )
    physical.play_primitive("boundary")
    simulated.play_primitive("boundary")
    commands = [initial.copy()] * 3
    outside_target = False
    for k in range(650):
        target = np.array([-0.3, direction * 0.8, 0.3])
        if k == 300:
            physical.barge_in()
            simulated.barge_in()
        physical.track.observe_point(target)
        simulated.track.observe_point(target)
        state = physical.step()
        sim_state = simulated.step()
        actual = alignment.normalized_to_radians(np.array(list(robot.actions[-1].values())))
        commands.append(actual)
        outside_target |= bool(np.any(state.q_blend < limits[:, 0])
                               or np.any(state.q_blend > limits[:, 1]))
        np.testing.assert_allclose(actual, state.q_cmd, atol=1e-12)
        np.testing.assert_allclose(actual, physical.traj.pos, atol=1e-12)
        np.testing.assert_allclose(actual, sim_state.q_cmd, atol=1e-12)
        np.testing.assert_allclose(actual, sim.measured(), atol=1e-12)

    assert outside_target  # L1+L2 composition actually exercises the boundary.
    commands = np.array(commands)
    assert np.all(commands >= limits[:, 0] - 1e-12)
    assert np.all(commands <= limits[:, 1] + 1e-12)
    velocity = np.diff(commands, axis=0) / CONTROL_DT
    acceleration = np.diff(velocity, axis=0) / CONTROL_DT
    assert np.all(np.abs(velocity) <= VEL_LIMIT + 1e-8)
    assert np.all(np.abs(acceleration) <= ACC_LIMIT + 1e-8)
    if physical.traj.backend == "ruckig":
        jerk = np.diff(acceleration, axis=0) / CONTROL_DT
        assert np.all(np.abs(jerk) <= JERK_LIMIT + 1e-7)
    assert backend.clamped_ticks == 0
