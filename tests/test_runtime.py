import numpy as np
import pytest

from motion import MotionRuntime
from motion.config import ACC_LIMIT, CONTROL_DT, REST_POSE, VEL_LIMIT
from motion.trajectory import TrajectoryGenerator
from motion.hardware_alignment import HardwareAlignment

RUCKIG = TrajectoryGenerator(np.zeros(5)).backend == "ruckig"


@pytest.fixture
def rt():
    return MotionRuntime(dt=CONTROL_DT)


def test_idle_only_stays_near_rest(rt):
    for _ in range(600):
        rt.step()
    assert np.max(np.abs(rt.traj.pos - REST_POSE)) < np.deg2rad(8)


def test_commands_never_break_kinematic_limits(rt):
    rt.play_primitive("wake_up")
    vmax = np.zeros(5)
    amax = np.zeros(5)
    last_v = np.zeros(5)
    for _ in range(1500):
        s = rt.step()
        vmax = np.maximum(vmax, np.abs(s.vel))
        amax = np.maximum(amax, np.abs(s.vel - last_v) / CONTROL_DT)
        last_v = s.vel
    assert np.all(vmax <= VEL_LIMIT + 1e-3)
    assert np.all(amax <= ACC_LIMIT + (0.5 if RUCKIG else ACC_LIMIT.max()))


def test_primitive_returns_toward_rest_after_playing(rt):
    for _ in range(50):
        rt.step()
    rt.play_primitive("nod")
    moved = False
    for _ in range(900):
        s = rt.step()
        if np.max(np.abs(s.q_cmd - REST_POSE)) > np.deg2rad(6):
            moved = True
    assert moved
    assert np.max(np.abs(rt.traj.pos - REST_POSE)) < np.deg2rad(10)


def test_tracking_points_head_at_target(rt):
    for _ in range(50):
        rt.step()
    target = np.array([0.42, 0.15, 0.33])
    for k in range(500):
        if k % 3 == 0:
            rt.track.observe_point(target)
        rt.step()
    pose = rt.kin.head_pose(rt.traj.pos)
    to = target - pose.pos
    to /= np.linalg.norm(to)
    assert np.dot(pose.forward, to) > np.cos(np.deg2rad(10 if RUCKIG else 18))


def test_barge_in_drops_primitive_fast(rt):
    rt.play_primitive("wake_up")
    for _ in range(120):
        rt.step()
    assert rt.primitive.busy
    rt.barge_in()
    dropped_at = None
    for i in range(60):
        rt.step()
        if not rt.primitive.busy and dropped_at is None:
            dropped_at = i
    assert dropped_at is not None and dropped_at < 20  # < 0.2 s


def test_task_light_then_clear(rt):
    for _ in range(30):
        rt.step()
    res = rt.place_task_light(np.array([0.24, 0.0, 0.0]))
    assert res.pos_err < 0.04
    for _ in range(300):
        rt.step()
    held = rt.kin.head_pose(rt.traj.pos)
    aim = np.array([0.24, 0.0, 0.0]) - held.pos
    aim /= np.linalg.norm(aim)
    assert np.dot(held.forward, aim) > np.cos(np.deg2rad(12))
    rt.clear_task_light()
    for _ in range(400):
        rt.step()
    assert np.max(np.abs(rt.traj.pos - REST_POSE)) < np.deg2rad(12)


def test_reach_puts_head_on_the_point(rt):
    for _ in range(30):
        rt.step()
    point = np.array([0.30, 0.10, 0.22])
    res = rt.reach_to(point)
    assert res.pos_err < 0.02
    for _ in range(400):
        rt.step()
    assert np.linalg.norm(rt.kin.head_position(rt.traj.pos) - point) < 0.03


def test_reflex_overrides_idle_but_yields_to_task_light(rt):
    # priorities: idle(0) < track(10) < primitive(20) < task_light(30)
    prios = [ly.priority for ly in rt.blender.layers]
    assert prios == sorted(prios)
    assert rt.blender.layers[-1].name == "task_light"


def test_initial_pose_outside_calibration_is_rejected_without_sending():
    initial = REST_POSE.copy()
    initial[0] = HardwareAlignment.load().joint_limits[0, 1] + 0.01
    with pytest.raises(ValueError, match="position limits"):
        MotionRuntime(initial_pose=initial)


@pytest.mark.parametrize("endpoint", [0, 1])
def test_measured_calibrated_endpoint_is_a_valid_initial_pose(endpoint):
    initial = HardwareAlignment.load().joint_limits[:, endpoint]
    runtime = MotionRuntime(initial_pose=initial)
    np.testing.assert_array_equal(runtime.traj.pos, initial)
    assert np.max(np.abs(runtime.step().q_cmd - initial)) < .003


@pytest.mark.parametrize("dt", [.02, .005, 0, -.01, float("nan"), float("inf")])
def test_invalid_period_does_not_advance_runtime_or_layers(rt, dt):
    rt.play_primitive("nod")
    rt.track.observe_point([.4, .1, .3])
    before = rt.traj.state
    with pytest.raises(ValueError, match="control period"):
        rt.step(dt=dt)
    assert rt.t == 0.0
    assert rt.primitive.env.level == 0.0
    assert rt.track.env.level == 0.0
    assert rt.backend.measured() is None
    for old, new in zip(before, rt.traj.state):
        np.testing.assert_array_equal(old, new)


def test_replaying_cached_primitive_with_zero_scale_emits_only_idle_motion(rt):
    idle = MotionRuntime()
    rt.play_primitive("nod")
    rt.play_primitive("nod", scale=np.zeros(5))
    for _ in range(80):
        np.testing.assert_allclose(rt.step().q_cmd, idle.step().q_cmd, atol=1e-12)


def test_reading_step_trace_keeps_commands_envelopes_and_filter_unchanged(rt):
    unlogged = MotionRuntime()
    for runtime in (rt, unlogged):
        runtime.play_primitive("nod")
        runtime.track.observe_point([.4, .1, .3])
    logged_authority = []
    for _ in range(120):
        state = rt.step()
        logged_authority.append({n: np.linalg.norm(v) for n, v in state.trace.authority.items()})
        np.testing.assert_array_equal(state.trace.q, state.q_blend)
        np.testing.assert_array_equal(state.q_cmd, unlogged.step().q_cmd)
        assert rt.primitive.env.level == unlogged.primitive.env.level
        assert rt.track.env.level == unlogged.track.env.level
        np.testing.assert_array_equal(rt.track.track.x, unlogged.track.track.x)
        np.testing.assert_array_equal(rt.track.track.P, unlogged.track.track.P)
    assert any(entry.get("primitive", 0) > 0 for entry in logged_authority)
