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


@ruckig_only
def test_loaded_pitch_acceleration_changes_gently_enough_for_physical_arm():
    """Keep pitch setpoints from exciting the assembled lamp's visible flex."""
    _, _, acceleration, _ = _run(np.array([0.0, 1.5, -1.5, 0.0, 0.0]))
    pitch_acceleration_step = np.abs(np.diff(acceleration[:, 1:3], axis=0))
    assert np.max(pitch_acceleration_step) <= 0.25 + 1e-7


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


@pytest.mark.parametrize("dt", [.02, .005, 0, -.01, float("nan"), float("inf")])
def test_changed_step_period_is_rejected_before_advancing(dt):
    tg = TrajectoryGenerator(np.zeros(NJ))
    before = tg.state
    with pytest.raises(ValueError, match="control period"):
        tg.step(np.ones(NJ), dt=dt)
    for old, new in zip(before, tg.state):
        np.testing.assert_array_equal(old, new)


def test_explicit_configured_period_matches_default_elapsed_state():
    default = TrajectoryGenerator(np.zeros(NJ), dt=.02)
    explicit = TrajectoryGenerator(np.zeros(NJ), dt=.02)
    for _ in range(30):
        default.step(np.ones(NJ))
        explicit.step(np.ones(NJ), dt=.02)
    for a, b in zip(default.state, explicit.state):
        np.testing.assert_array_equal(a, b)


def test_repeated_target_reversals_keep_actual_commands_inside_position_limits():
    # Simultaneous changes on several joints can make a synchronized Ruckig
    # replan overshoot an endpoint even though every target is inside it.
    rng = np.random.default_rng(7)
    tg = TrajectoryGenerator(np.zeros(NJ), position_limits=np.tile([-.2, .2], (NJ, 1)))
    positions = [tg.pos.copy()] * 3
    for k in range(1000):
        if k % 20 == 0:
            target = rng.choice([-1, 1], NJ) * .2
        positions.append(tg.step(target))
    positions = np.array(positions)
    assert np.all(np.abs(positions) <= .2 + 1e-10)
    velocity = np.diff(positions, axis=0) / CONTROL_DT
    acceleration = np.diff(velocity, axis=0) / CONTROL_DT
    assert np.all(np.abs(velocity) <= VEL_LIMIT + 1e-8)
    assert np.all(np.abs(acceleration) <= ACC_LIMIT + 1e-8)
    if RUCKIG:
        jerk = np.diff(acceleration, axis=0) / CONTROL_DT
        assert np.all(np.abs(jerk) <= JERK_LIMIT + 1e-7)
