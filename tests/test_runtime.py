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


def test_orientation_anchor_overrides_tracking_yaw_but_not_other_tracking_joints(rt):
    """Removing the yaw-only anchor must let tracking control base yaw again."""
    rt.observe_point([.4, .3, .3])
    rt.acquire_orientation("speech-1", .5, now=0.0)

    states = rt.run(1.5)

    assert states[-1].trace.per_layer["orientation"][0] == 1.0
    assert states[-1].trace.per_layer["orientation"][1:].sum() == 0.0
    assert states[-1].q_cmd[0] == pytest.approx(.5, abs=.06)


def test_orientation_layer_order_keeps_task_light_yaw_authority(rt):
    """Moving TaskLight below orientation would prevent its full pose from winning."""
    assert [layer.name for layer in rt.blender.layers] == [
        "idle", "track", "orientation", "primitive", "task_light",
    ]

    rt.acquire_orientation("speech-1", .5, now=0.0)
    result = rt.place_task_light([.24, .1, 0.0])
    assert result.pos_err < .04
    states = rt.run(.6)

    assert abs(rt.task_light.q_hold[0] - .5) > .1
    assert states[-1].trace.per_layer["orientation"][0] == 1.0
    assert states[-1].trace.per_layer["task_light"][0] == 1.0
    assert states[-1].q_blend[0] == pytest.approx(rt.task_light.q_hold[0])


def test_anchored_motion_scales_only_yaw_to_fit_safe_range(rt):
    """Unscaled positive headshake yaw would cross the speaker-safe upper bound."""
    rt.acquire_orientation(
        "speech-1", rt.traj.position_limits[0, 1] - np.deg2rad(6), now=0.0,
    )

    info = rt.play_primitive("headshake")

    assert 0.0 <= info.yaw_scale < 1.0
    for state in rt.run(4.0):
        assert state.q_blend[0] <= rt.traj.position_limits[0, 1] - np.deg2rad(5) + 1e-9


def test_unanchored_motion_keeps_existing_commands_and_reports_full_yaw_scale(rt):
    """Adding anchor support must leave unanchored primitive playback byte-identical."""
    reference = MotionRuntime(dt=CONTROL_DT)

    info = rt.play_primitive("headshake")
    reference.play_primitive("headshake")

    assert info.yaw_scale == 1.0
    for _ in range(400):
        np.testing.assert_array_equal(rt.step().q_cmd, reference.step().q_cmd)


def test_primitive_layer_requires_a_complete_yaw_anchor_context(rt):
    """Supplying only one half of the safety context would make scaling ambiguous."""
    with pytest.raises(ValueError, match="provided together"):
        rt.primitive.play("headshake", t=0.0, yaw_anchor=.1)
    with pytest.raises(ValueError, match="provided together"):
        rt.primitive.play("headshake", t=0.0, yaw_limits=(-.5, .5))


def test_releasing_orientation_restores_tracking_behavior(rt):
    """Leaving the anchor active after release would continue suppressing tracking yaw."""
    reference = MotionRuntime()
    for runtime in (rt, reference):
        runtime.observe_point([.4, .1, .3])
    rt.acquire_orientation("speech-1", .5, now=0.0)
    rt.release_orientation()

    for _ in range(120):
        np.testing.assert_array_equal(rt.step().q_cmd, reference.step().q_cmd)


def test_releasing_orientation_resets_snapshot_and_reacquires_same_speech_id(rt):
    """A stale coordinator session would reject this same-ID orientation retry."""
    rt.acquire_orientation("speech-1", .5, now=0.0)

    rt.release_orientation()

    released = rt.orientation_snapshot()
    assert released.state == "idle"
    assert released.speech_id is None
    assert released.target_yaw is None
    assert released.clamped is False
    assert released.code == "idle"

    reacquired = rt.acquire_orientation("speech-1", -.4, now=.1)
    assert reacquired.state == "orienting"
    assert reacquired.target_yaw == pytest.approx(-.4)
    assert rt.step().trace.per_layer["orientation"][0] == 1.0


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


def test_primitive_progress_is_read_only_and_does_not_change_commands(rt):
    reference = MotionRuntime()
    assert rt.primitive.active_name is None
    assert rt.primitive.started_at is None
    assert rt.primitive.progress(rt.t) == 0.
    for runtime in (rt, reference):
        runtime.play_primitive("nod")
    assert rt.primitive.active_name == "nod"
    assert rt.primitive.started_at == 0.
    duration = rt.primitive.clip.duration
    assert rt.primitive.progress(-1.) == 0.
    assert rt.primitive.progress(duration / 2) == pytest.approx(.5)
    assert rt.primitive.progress(duration * 2) == 1.
    with pytest.raises(AttributeError):
        rt.primitive.active_name = "other"
    for i in range(900):
        assert 0. <= rt.primitive.progress(rt.t) <= 1.
        np.testing.assert_array_equal(rt.step().q_cmd, reference.step().q_cmd)
    assert rt.primitive.active_name is None
    assert rt.primitive.started_at is None


def test_runtime_tracking_controls_preserve_layer_behavior(rt):
    reference = MotionRuntime()
    rt.observe_point([.4, .1, .3])
    reference.track.observe_point([.4, .1, .3])
    np.testing.assert_array_equal(rt.step().q_cmd, reference.step().q_cmd)
    rt.observe_bearing([1., .2, .1])
    reference.track.observe_bearing([0., 0., 0.], [1., .2, .1])
    np.testing.assert_array_equal(rt.step().q_cmd, reference.step().q_cmd)
    rt.clear_tracking()
    reference.track.clear()
    np.testing.assert_array_equal(rt.step().q_cmd, reference.step().q_cmd)


def test_anchor_refits_release_tail_without_restarting_or_scaling_body(rt):
    """An interrupted clip still contributes yaw while its envelope releases."""
    rt.play_primitive("headshake")
    rt.run(1.5)
    original_body = rt.primitive.clip.offsets[:, 1:].copy()
    started = rt.primitive.started_at
    rt.barge_in()
    lo, hi = rt.orientation_safe_yaw_limits()
    rt.acquire_orientation("tail", hi - .001, now=1.)
    for _ in range(10):
        state = rt.step()
        assert lo - 1e-9 <= state.q_blend[0] <= hi + 1e-9
        assert lo - 1e-9 <= state.q_cmd[0] <= hi + 1e-9
    assert rt.primitive.started_at == started
    np.testing.assert_array_equal(rt.primitive.clip.offsets[:, 1:], original_body)


def test_anchor_refit_restores_original_yaw_when_room_returns(rt):
    """Refitting an already scaled source would permanently shrink expression."""
    rt.play_primitive("headshake")
    original = rt.primitive.clip.offsets.copy()
    rt.acquire_orientation("edge", rt.orientation_safe_yaw_limits()[1] - .001, now=0.)
    rt.step()
    assert np.max(np.abs(rt.primitive.clip.offsets[:, 0])) < .01
    rt.acquire_orientation("middle", .5, now=.01)
    state = rt.step()
    np.testing.assert_array_equal(rt.primitive.clip.offsets, original)
    assert state.q_blend[0] < .5


def test_autonomous_return_refits_primitive_release_before_next_blend():
    """The coordinator can change the absolute target after the previous hardware tick."""
    from motion.orientation import OrientationConfig
    hi = HardwareAlignment.load().joint_limits[0, 1] - np.deg2rad(5)
    rt = MotionRuntime(orientation_cfg=OrientationConfig(center_yaw=hi - .001, disconnect_hold=.01))
    rt.acquire_orientation("old", .5, now=0.)
    rt.play_primitive("headshake")
    rt.run(1.5)
    rt.barge_in()
    rt.orientation_control.disconnected(now=2.)
    snapshot = rt.orientation_control.observe(now=2.02, current_yaw=rt.traj.pos[0], velocity=rt.traj.vel[0])
    assert snapshot.state == "returning"
    for _ in range(10):
        state = rt.step()
        assert state.q_blend[0] <= hi + 1e-9
        assert state.q_cmd[0] <= hi + 1e-9
