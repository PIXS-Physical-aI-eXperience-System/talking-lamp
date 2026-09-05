"""Online jerk-limited trajectory generator (the "100 Hz 궤적 생성기").

Streaming, not point-to-point: the setpoint may move every tick (the blender
output does). Each joint is driven to a kinematically feasible state that
respects per-joint velocity / acceleration / jerk limits, with no overshoot.

Uses `ruckig` (the reference online trajectory generator) when available, and
falls back to a self-contained analytic third-order tracker otherwise, so the
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
        from ruckig import InputParameter, OutputParameter, Ruckig

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
    ) -> None:
        self.pos = np.asarray(q0, float).copy()
        self.vel = np.zeros(NJ)
        self.acc = np.zeros(NJ)
        self.vmax = np.asarray(vel_limit, float)
        self.amax = np.asarray(acc_limit, float)
        self.jmax = np.asarray(jerk_limit, float)
        self.dt = float(dt)
        self._impl = _Ruckig(self) if _HAVE_RUCKIG else _Analytic(self)

    @property
    def backend(self) -> str:
        return "ruckig" if _HAVE_RUCKIG else "analytic"

    def reset(self, q: np.ndarray) -> None:
        self.pos = np.asarray(q, float).copy()
        self.vel[:] = 0.0
        self.acc[:] = 0.0
        self._impl.sync()

    def step(self, target: np.ndarray, dt: float | None = None) -> np.ndarray:
        h = self.dt if dt is None else float(dt)
        self._impl.step(np.asarray(target, float), h)
        return self.pos.copy()

    @property
    def state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.pos.copy(), self.vel.copy(), self.acc.copy()


class _Ruckig:
    def __init__(self, tg: TrajectoryGenerator) -> None:
        self.tg = tg
        self.otg = Ruckig(NJ, tg.dt)
        self.inp = InputParameter(NJ)
        self.out = OutputParameter(NJ)
        self.sync()
        self.inp.max_velocity = tg.vmax.tolist()
        self.inp.max_acceleration = tg.amax.tolist()
        self.inp.max_jerk = tg.jmax.tolist()
        self.inp.target_velocity = [0.0] * NJ
        self.inp.target_acceleration = [0.0] * NJ

    def sync(self) -> None:
        self.inp = getattr(self, "inp", InputParameter(NJ))
        self.inp.current_position = self.tg.pos.tolist()
        self.inp.current_velocity = self.tg.vel.tolist()
        self.inp.current_acceleration = self.tg.acc.tolist()

    def step(self, target: np.ndarray, h: float) -> None:
        self.inp.target_position = target.tolist()
        self.otg.update(self.inp, self.out)
        self.out.pass_to_input(self.inp)
        self.tg.pos = np.array(self.out.new_position)
        self.tg.vel = np.array(self.out.new_velocity)
        self.tg.acc = np.array(self.out.new_acceleration)


class _Analytic:
    """Fallback used only when ruckig is absent.

    Acceleration-limited trapezoidal-velocity profile: provably overshoot-free
    (``v_cap = sqrt(2·amax·d)`` is the exact brake-to-rest speed) and velocity/
    acceleration bounded. Jerk is *not* bounded - it spikes to amax/dt at the two
    profile corners - so this is a degraded mode; install ruckig for smooth jerk.
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
        tg.acc = (new_vel - tg.vel) / h
        tg.vel = new_vel
        tg.pos = tg.pos + tg.vel * h

        settled = (np.abs(target - tg.pos) < 1e-3) & (np.abs(tg.vel) < 5e-3)
        tg.pos = np.where(settled, target, tg.pos)
        tg.vel = np.where(settled, 0.0, tg.vel)
        tg.acc = np.where(settled, 0.0, tg.acc)
