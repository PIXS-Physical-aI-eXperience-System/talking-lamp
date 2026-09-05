"""Inverse kinematics for the 5-DOF lamp arm.

Damped least squares (Levenberg-Marquardt) with:

* selectable task: head **position** only (3-DOF) or position + **look-at**
  direction (5-DOF), which is all a 5-joint arm can honour anyway;
* adaptive damping that grows near singularities (SVD-free, uses J Jᵀ);
* null-space bias pulling unused freedom toward ``REST_POSE``;
* hard joint-limit clamping every iteration.

Two entry points:

``solve(target, q0)``      - iterate to convergence, for one-shot goals (S1 light
                             placement, "look here").
``step(target, q, dt)``    - a single damped step, for streaming targets fed at
                             100 Hz by the reflex layer (face / sound tracking).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import REST_POSE
from .kinematics import ArmKinematics


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float          # metres
    aim_err: float          # radians (0 if no look-at target)
    iters: int
    converged: bool


class IKSolver:
    def __init__(
        self,
        kin: ArmKinematics | None = None,
        *,
        damping: float = 5e-3,
        rest_pose: np.ndarray | None = None,
        nullspace_gain: float = 0.02,
    ) -> None:
        self.kin = kin or ArmKinematics()
        self.base_damping = float(damping)
        self.rest_pose = np.asarray(rest_pose if rest_pose is not None else REST_POSE, float)
        self.nullspace_gain = float(nullspace_gain)

    # ------------------------------------------------------------------
    def _task_error(
        self, q: np.ndarray, target_pos: np.ndarray, aim_point: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        pose = self.kin.head_pose(q)
        e_pos = target_pos - pose.pos
        Jp, Jr = self.kin.jacobian(q)

        if aim_point is None:
            return e_pos, Jp, float(np.linalg.norm(e_pos)), 0.0

        to_target = aim_point - pose.pos
        n = np.linalg.norm(to_target)
        if n < 1e-6:
            return e_pos, Jp, float(np.linalg.norm(e_pos)), 0.0
        desired = to_target / n
        current = pose.forward
        # rotation error vector = current x desired (small-angle), mapped by Jr
        e_rot = np.cross(current, desired)
        ang = float(np.arctan2(np.linalg.norm(e_rot), np.dot(current, desired)))
        e = np.concatenate([e_pos, e_rot])
        J = np.vstack([Jp, Jr])
        return e, J, float(np.linalg.norm(e_pos)), ang

    def _dls_step(self, q: np.ndarray, e: np.ndarray, J: np.ndarray) -> np.ndarray:
        nq = len(q)
        # freeze DOFs sitting on a limit and pushed further into it
        lo, hi = self.kin.joint_limits[:, 0], self.kin.joint_limits[:, 1]
        grad = J.T @ e
        free = ~(((q <= lo + 1e-6) & (grad < 0)) | ((q >= hi - 1e-6) & (grad > 0)))
        Jf = J[:, free]
        if Jf.shape[1] == 0:
            return np.zeros(nq)

        # SVD pseudo-inverse with singular-value-wise damping (robust near singular)
        U, s, Vt = np.linalg.svd(Jf, full_matrices=False)
        lam2 = self.base_damping**2
        s_inv = s / (s**2 + lam2 + np.where(s < 0.04, (0.04 - s) ** 2 * 25, 0.0))
        dq_f = Vt.T @ (s_inv * (U.T @ e))

        # null-space bias toward the rest pose (only on free DOFs)
        if self.nullspace_gain > 0:
            Jf_pinv = Vt.T @ np.diag(s_inv) @ U.T
            null = np.eye(Jf.shape[1]) - Jf_pinv @ Jf
            dq_f += null @ (self.nullspace_gain * (self.rest_pose[free] - q[free]))

        dq = np.zeros(nq)
        dq[free] = dq_f
        return dq

    # ------------------------------------------------------------------
    def solve(
        self,
        target_pos,
        q0=None,
        *,
        aim_point=None,
        pos_tol: float = 2e-3,
        aim_tol: float = np.deg2rad(2.0),
        max_iters: int = 120,
        max_step: float = 0.3,
        restarts: int = 6,
    ) -> IKResult:
        """Iterate DLS to convergence. Tries several seeds (given q0, rest pose,
        random) and returns the best result - the arm's wide joint ranges create
        local minima that a single seed can get stuck in."""
        target_pos = np.asarray(target_pos, float)
        aim = None if aim_point is None else np.asarray(aim_point, float)
        rng = np.random.default_rng(0xB0A)
        lo, hi = self.kin.joint_limits[:, 0], self.kin.joint_limits[:, 1]

        seeds = []
        if q0 is not None:
            seeds.append(np.asarray(q0, float))
        seeds.append(self.rest_pose)
        while len(seeds) < 1 + restarts:
            seeds.append(rng.uniform(lo, hi))

        best: IKResult | None = None
        for seed in seeds:
            res = self._run(seed, target_pos, aim, pos_tol, aim_tol, max_iters, max_step)
            if best is None or (res.pos_err + res.aim_err) < (best.pos_err + best.aim_err):
                best = res
            if best.converged:
                break
        return best

    def _run(self, seed, target_pos, aim, pos_tol, aim_tol, max_iters, max_step) -> IKResult:
        q = self.kin.clamp(np.asarray(seed, float).copy())
        pe = ae = np.inf
        for i in range(1, max_iters + 1):
            e, J, pe, ae = self._task_error(q, target_pos, aim)
            if pe < pos_tol and ae < aim_tol:
                return IKResult(q, pe, ae, i, True)
            dq = self._dls_step(q, e, J)
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q = self.kin.clamp(q + dq)
        return IKResult(q, pe, ae, max_iters, pe < pos_tol and ae < aim_tol)

    def step(
        self,
        target_pos,
        q,
        dt: float,
        *,
        aim_point=None,
        gain: float = 10.0,
        max_step: float = 0.12,
        pos_weight: float = 1.0,
        aim_weight: float = 1.0,
    ) -> np.ndarray:
        """One partial DLS step toward a (possibly moving) target. Returns new q.

        ``gain`` is a first-order tracking bandwidth in 1/s: each call closes
        ~``gain*dt`` of the remaining error, so the joint path stays smooth when
        the target jumps. ``pos_weight`` / ``aim_weight`` trade the position and
        look-at objectives (for pure "turn to look at you", drop ``pos_weight``).
        """
        target_pos = np.asarray(target_pos, float)
        aim = None if aim_point is None else np.asarray(aim_point, float)
        q = np.asarray(q, float)
        e, J, _, _ = self._task_error(q, target_pos, aim)
        if e.shape[0] == 6:
            w = np.array([pos_weight] * 3 + [aim_weight] * 3)
            e, J = e * w, J * w[:, None]
        dq_full = self._dls_step(q, e, J)
        dq = min(gain * dt, 1.0) * dq_full
        n = np.linalg.norm(dq)
        if n > max_step:
            dq *= max_step / n
        return self.kin.clamp(q + dq)
