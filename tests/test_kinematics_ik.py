import numpy as np
import pytest

from motion.config import NJ, REST_POSE
from motion.ik import IKSolver
from motion.kinematics import ArmKinematics


@pytest.fixture(scope="module")
def kin():
    return ArmKinematics()


def test_fk_matches_finite_diff_jacobian(kin):
    q = REST_POSE.copy()
    Jp, _ = kin.jacobian(q)
    p0 = kin.head_position(q)
    eps = 1e-6
    Jnum = np.zeros((3, NJ))
    for i in range(NJ):
        dq = q.copy()
        dq[i] += eps
        Jnum[:, i] = (kin.head_position(dq) - p0) / eps
    assert np.allclose(Jp, Jnum, atol=1e-4)


def test_joint_limits_loaded(kin):
    lo, hi = kin.joint_limits[:, 0], kin.joint_limits[:, 1]
    assert np.all(lo < hi)
    assert np.all(kin.clamp(REST_POSE) == REST_POSE)


def test_ik_reaches_reachable_targets(kin):
    solver = IKSolver(kin)
    rng = np.random.default_rng(0)
    hits = 0
    trials = 50
    for _ in range(trials):
        q_true = np.array([rng.uniform(lo, hi) for lo, hi in kin.joint_limits])
        target = kin.head_position(q_true)
        res = solver.solve(target, q0=REST_POSE)
        if res.pos_err < 3e-3:
            hits += 1
    assert hits >= trials - 2  # rare self-folded random configs may miss


def test_ik_covers_desk_work_zone(kin):
    """The head must reach a dome of standoff points above the work area."""
    solver = IKSolver(kin)
    misses = total = 0
    for x in np.linspace(0.08, 0.30, 5):
        for y in np.linspace(-0.15, 0.15, 5):
            for z in (0.14, 0.24):
                total += 1
                if solver.solve(np.array([x, y, z]), q0=REST_POSE).pos_err > 5e-3:
                    misses += 1
    assert misses <= total // 20  # <=5%


def test_ik_look_at_aims_head(kin):
    """With a look-at point, the head forward axis should point at it."""
    solver = IKSolver(kin)
    aim = np.array([0.25, 0.05, 0.0])
    res = solver.solve(np.array([0.12, 0.0, 0.30]), q0=REST_POSE, aim_point=aim)
    pose = kin.head_pose(res.q)
    to_aim = aim - pose.pos
    to_aim /= np.linalg.norm(to_aim)
    assert np.dot(pose.forward, to_aim) > np.cos(np.deg2rad(6))


def test_ik_respects_joint_limits(kin):
    solver = IKSolver(kin)
    # a target far outside the workspace - solver must not escape limits
    res = solver.solve(np.array([2.0, 2.0, 2.0]), q0=REST_POSE)
    assert np.all(res.q >= kin.joint_limits[:, 0] - 1e-9)
    assert np.all(res.q <= kin.joint_limits[:, 1] + 1e-9)


def test_ik_streaming_step_converges(kin):
    solver = IKSolver(kin)
    q = REST_POSE.copy()
    q_true = kin.clamp(REST_POSE + np.array([0.4, 0.2, -0.3, 0.1, 0.1]))
    target = kin.head_position(q_true)
    for _ in range(300):
        q = solver.step(target, q, dt=0.01)
    assert np.linalg.norm(kin.head_position(q) - target) < 5e-3
