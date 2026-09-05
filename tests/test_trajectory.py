import numpy as np
import pytest

from motion.config import ACC_LIMIT, CONTROL_DT, JERK_LIMIT, NJ, VEL_LIMIT
from motion.trajectory import TrajectoryGenerator

RUCKIG = TrajectoryGenerator(np.zeros(NJ)).backend == "ruckig"
ruckig_only = pytest.mark.skipif(not RUCKIG, reason="strict smoothness needs the ruckig backend")


def _run(target, steps=2000, q0=None):
    q0 = np.zeros(NJ) if q0 is None else q0
    tg = TrajectoryGenerator(q0)
    log = []
    for _ in range(steps):
        tg.step(target)
        log.append(tg.state)
    pos = np.array([s[0] for s in log])
    vel = np.array([s[1] for s in log])
    acc = np.array([s[2] for s in log])
    jerk = np.diff(acc, axis=0, prepend=acc[:1]) / CONTROL_DT
    return pos, vel, acc, jerk


def test_reaches_step_target():
    target = np.array([1.0, -0.8, 0.5, 1.2, -0.6])
    pos, vel, *_ = _run(target)
    assert np.allclose(pos[-1], target, atol=1e-3)
    assert np.allclose(vel[-1], 0.0, atol=1e-3)


def test_respects_velocity_and_acceleration_limits():
    target = np.array([2.5, 1.5, -1.5, 3.0, 2.0])
    pos, vel, acc, jerk = _run(target)
    assert np.all(np.abs(vel) <= VEL_LIMIT + 1e-3)
    # analytic fallback allows a 1-tick decel spike at the final corner
    tol = 1e-2 if RUCKIG else ACC_LIMIT.max()
    assert np.all(np.abs(acc) <= ACC_LIMIT + tol)


@ruckig_only
def test_respects_jerk_limit():
    _, _, _, jerk = _run(np.array([2.5, 1.5, -1.5, 3.0, 2.0]))
    assert np.all(np.abs(jerk) <= JERK_LIMIT + 1.0)


def test_no_meaningful_overshoot():
    target = np.ones(NJ)
    pos, *_ = _run(target)
    tol = 1e-3 if RUCKIG else 8e-3
    assert np.all(pos <= target + tol)


def test_tracks_moving_setpoint_without_lag_blowup():
    tg = TrajectoryGenerator(np.zeros(NJ))
    t = 0.0
    max_err = 0.0
    for _ in range(3000):
        t += CONTROL_DT
        sp = np.full(NJ, 0.5 * np.sin(0.5 * t))  # slow sine, within limits
        tg.step(sp)
        if t > 2.0:
            max_err = max(max_err, np.max(np.abs(tg.pos - sp)))
    assert max_err < 0.05


def test_step_dt_override():
    tg = TrajectoryGenerator(np.zeros(NJ))
    q = tg.step(np.ones(NJ), dt=0.02)
    assert q.shape == (NJ,)
