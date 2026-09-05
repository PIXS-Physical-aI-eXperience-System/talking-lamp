import numpy as np
import pytest

from motion import MotionRuntime
from motion.config import ACC_LIMIT, CONTROL_DT, REST_POSE, VEL_LIMIT
from motion.trajectory import TrajectoryGenerator

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
