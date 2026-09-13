"""Online jerk-limited trajectory generator (the "100 Hz 궤적 생성기").

Streaming, not point-to-point: the setpoint may move every tick (the blender
output does). Each joint is driven to a kinematically feasible state that
respects per-joint velocity / acceleration / jerk limits. Optional position
bounds also constrain the complete path, including during target reversals.

Uses `ruckig` (the reference online trajectory generator) when available, and
falls back to a self-contained acceleration-limited tracker otherwise, so the
stack still runs on a machine without the wheel. Same interface either way:

    tg = TrajectoryGenerator(q0)
    q_cmd = tg.step(target)          # call at CONTROL_HZ
    pos, vel, acc = tg.state
"""

from __future__ import annotations

import os

import numpy as np

from .config import ACC_LIMIT, CONTROL_DT, JERK_LIMIT, NJ, VEL_LIMIT

if os.environ.get("TALKING_LAMP_NO_RUCKIG"):
    _HAVE_RUCKIG = False
else:
    try:  # pragma: no cover - import guard
        from ruckig import InputParameter, Ruckig, Trajectory

        _HAVE_RUCKIG = True
    except Exception:  # pragma: no cover
        _HAVE_RUCKIG = False


class TrajectoryGenerator:
    def __init__(
        self,
        q0: np.ndarray,
        *,
        vel_limit: np.ndarray = VEL_LIMIT,
        acc_limit: np.ndarray = ACC_LIMIT,
        jerk_limit: np.ndarray = JERK_LIMIT,
        dt: float = CONTROL_DT,
        position_limits: np.ndarray | None = None,
    ) -> None:
        self.pos = np.asarray(q0, float).copy()
        self.vel = np.zeros(NJ)
        self.acc = np.zeros(NJ)
        self.vmax = np.asarray(vel_limit, float)
        self.amax = np.asarray(acc_limit, float)
        self.jmax = np.asarray(jerk_limit, float)
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("control period must be finite and positive")
        self.position_limits = None if position_limits is None else np.asarray(
            position_limits, float
        ).copy()
        self._validate_position(self.pos)
        self._impl = _Ruckig(self) if _HAVE_RUCKIG else _Analytic(self)

    def _validate_position(self, q: np.ndarray) -> None:
        if self.position_limits is not None:
            lo, hi = self.position_limits.T
            if np.any(q < lo) or np.any(q > hi):
                raise ValueError("Initial/reset pose is outside joint position limits")

    @property
    def backend(self) -> str:
        return "ruckig" if _HAVE_RUCKIG else "analytic"

    def reset(self, q: np.ndarray) -> None:
        q = np.asarray(q, float)
        self._validate_position(q)
        self.pos = q.copy()
        self.vel[:] = 0.0
        self.acc[:] = 0.0
        self._impl.sync()

    def step(self, target: np.ndarray, dt: float | None = None) -> np.ndarray:
        h = self.validate_dt(dt)
        target = np.asarray(target, float)
        if self.position_limits is not None:
            # Brake to a feasible setpoint, rather than clip the trajectory's
            # output and silently invalidate its velocity/acceleration state.
            target = np.clip(target, *self.position_limits.T)
        self._impl.step(target, h)
        return self.pos.copy()

    def validate_dt(self, dt: float | None) -> float:
        """Both backends advance exactly one configured control period."""
        h = self.dt if dt is None else float(dt)
        if not np.isfinite(h) or not np.isclose(h, self.dt, rtol=0, atol=1e-12):
            raise ValueError(f"dt must match the fixed control period ({self.dt:g} s)")
        return self.dt

    @property
    def state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.pos.copy(), self.vel.copy(), self.acc.copy()


class _Ruckig:
    def __init__(self, tg: TrajectoryGenerator) -> None:
        self.tg = tg
        self.otg = Ruckig(NJ, tg.dt)
        self.inp = InputParameter(NJ)
        self.sync()
        self.inp.max_velocity = tg.vmax.tolist()
        self.inp.max_acceleration = tg.amax.tolist()
        self.inp.max_jerk = tg.jmax.tolist()
        self.inp.target_velocity = [0.0] * NJ
        self.inp.target_acceleration = [0.0] * NJ

    def sync(self) -> None:
        self._trajectory = None
        self._target = None
        self._elapsed = 0.0

    def step(self, target: np.ndarray, h: float) -> None:
        self.inp.current_position = self.tg.pos.tolist()
        self.inp.current_velocity = self.tg.vel.tolist()
        self.inp.current_acceleration = self.tg.acc.tolist()
        if self._target is None or not np.array_equal(target, self._target):
            self.inp.target_position = target.tolist()
            candidate = Trajectory(NJ)
            result = self.otg.calculate(self.inp, candidate)
            if result < 0:
                raise RuntimeError(f"Ruckig trajectory calculation failed: {result}")
            feasible = True
            if self.tg.position_limits is not None:
                # Community Ruckig does not enforce position limits. Check
                # every continuous-time extremum before accepting a replan;
                # braking/synchronization can overshoot a bounded target.
                feasible = all(
                    lo - 1e-10 <= extent.min and extent.max <= hi + 1e-10
                    for extent, (lo, hi) in zip(
                        candidate.position_extrema, self.tg.position_limits
                    )
                )
            if feasible:
                self._trajectory = candidate
                self._target = target.copy()
                self._elapsed = 0.0
            elif self._trajectory is None:
                raise RuntimeError("No trajectory inside joint position limits")
            # An unsafe replacement never modifies the commanded state. Keep
            # advancing the previous feasible path, retrying from its next
            # state so position, velocity and acceleration stay continuous.
        self._elapsed += h
        if self._elapsed >= self._trajectory.duration:
            self.tg.pos = self._target.copy()
            self.tg.vel = np.zeros(NJ)
            self.tg.acc = np.zeros(NJ)
        else:
            p, v, a = self._trajectory.at_time(self._elapsed)
            self.tg.pos, self.tg.vel, self.tg.acc = np.array(p), np.array(v), np.array(a)


class _Analytic:
    """Fallback used only when ruckig is absent.

    Acceleration-limited trapezoidal-velocity profile with a spring near the
    target and a stopping-distance reserve at joint boundaries. Velocity,
    acceleration and configured position bounds are enforced. Jerk is *not*
    bounded, so this is a degraded mode; install ruckig for smooth jerk.
    """

    _MARGIN = 0.95  # brake a little early to cover the discrete step

    def __init__(self, tg: TrajectoryGenerator) -> None:
        self.tg = tg

    def sync(self) -> None:
        pass

    def step(self, target: np.ndarray, h: float) -> None:
        tg = self.tg
        e = target - tg.pos
        v_cap = np.sqrt(2.0 * tg.amax * np.abs(e) * self._MARGIN)
        v_star = np.clip(np.sign(e) * np.minimum(tg.vmax, v_cap), -tg.vmax, tg.vmax)
        a_cmd = np.clip((v_star - tg.vel) / h, -tg.amax, tg.amax)

        # near the target, swap the bang-bang profile for a critically-damped
        # spring so it doesn't limit-cycle around the setpoint
        near = np.abs(e) < 0.02
        if np.any(near):
            w = 0.35 / h
            a_spring = np.clip(w * w * e - 2.0 * w * tg.vel, -tg.amax, tg.amax)
            a_cmd = np.where(near, a_spring, a_cmd)

        new_vel = np.clip(tg.vel + a_cmd * h, -tg.vmax, tg.vmax)
        if tg.position_limits is not None:
            lo, hi = tg.position_limits.T
            # Reserve one full integration step plus the continuous stopping
            # distance: h*v + v**2/(2*amax) <= distance to the endpoint.
            # From a feasible state, the resulting speed cap can always be
            # reached within one acceleration-limited step. Applying it after
            # the spring also prevents a near-target spring overshoot.
            ah = tg.amax * h
            v_hi = np.sqrt(ah**2 + 2 * tg.amax * np.maximum(hi - tg.pos, 0)) - ah
            v_lo = np.sqrt(ah**2 + 2 * tg.amax * np.maximum(tg.pos - lo, 0)) - ah
            new_vel = np.clip(new_vel, -v_lo, v_hi)
        tg.acc = (new_vel - tg.vel) / h
        tg.vel = new_vel
        tg.pos = tg.pos + tg.vel * h
