"""Concrete motion layers L0-L3 for the blender.

L0 ``IdleLayer``      priority  0   breathing / micro-gaze, always the canvas
L1 ``TrackLayer``     priority 10   Kalman-tracked face / sound follow via IK
L2 ``PrimitiveLayer`` priority 20   expressive clips triggered by behaviour tags
L3 ``TaskLightLayer`` priority 30   hold an IK pose that aims the head at a spot

Every layer owns its crossfade envelope, so the blender stays stateless and a
barge-in just means "tell L2/L3 to release".
"""

from __future__ import annotations

import numpy as np

from .blender import BlendContext, LayerOutput
from .config import NJ, REST_POSE
from .idle import IdleConfig, IdleMotion
from .ik import IKResult, IKSolver
from .kalman import TargetTrack, TrackConfig
from .kinematics import ArmKinematics
from .primitives import Primitive, PrimitiveLibrary


class Envelope:
    """Linear attack/release gain in [0, 1]."""

    def __init__(self, attack: float = 0.2, release: float = 0.3) -> None:
        self.attack = max(attack, 1e-3)
        self.release = max(release, 1e-3)
        self.level = 0.0
        self._target = 0.0

    def open(self, release: float | None = None) -> None:
        self._target = 1.0
        if release is not None:
            self.release = max(release, 1e-3)

    def close(self, release: float | None = None) -> None:
        self._target = 0.0
        if release is not None:
            self.release = max(release, 1e-3)

    def step(self, dt: float) -> float:
        if self._target > self.level:
            self.level = min(self._target, self.level + dt / self.attack)
        elif self._target < self.level:
            self.level = max(self._target, self.level - dt / self.release)
        return self.level

    @property
    def closed(self) -> bool:
        return self.level <= 1e-4 and self._target == 0.0


# ---------------------------------------------------------------------------
class IdleLayer:
    """L0. Additive breathing/micro-gaze offset. The blender's canvas is already
    REST_POSE, so idle only needs to supply the *motion*, not the pose - that way
    it never fights a higher layer by pulling back toward rest."""

    name = "idle"
    priority = 0

    def __init__(self, cfg: IdleConfig | None = None, rest_pose=REST_POSE, seed: int = 7) -> None:
        self.idle = IdleMotion(cfg, seed=seed)
        self.rest_pose = np.asarray(rest_pose, float)
        self.weight = np.ones(NJ)

    def set_rest_pose(self, q) -> None:
        self.rest_pose = np.asarray(q, float)

    def update(self, ctx: BlendContext) -> LayerOutput:
        return LayerOutput(
            value=self.idle.offset(ctx.t),
            weight=self.weight,
            gain=1.0,
            additive=True,
        )


# ---------------------------------------------------------------------------
class TrackLayer:
    """L1 reflex follow. Feed it measurements from D (vision) / C (audio DOA);
    it keeps a smooth head-aim even through detection gaps."""

    name = "track"
    priority = 10

    def __init__(
        self,
        kin: ArmKinematics,
        *,
        track_cfg: TrackConfig | None = None,
        lead_time: float = 0.08,
        follow_gain: float = 12.0,
        joints: np.ndarray | None = None,
    ) -> None:
        self.kin = kin
        # strong pull to REST_POSE so "look straight ahead" doesn't crank the base
        self.ik = IKSolver(kin, nullspace_gain=0.06)
        self.track = TargetTrack(track_cfg or TrackConfig())
        self.lead_time = lead_time
        self.follow_gain = follow_gain
        self.env = Envelope(attack=0.3, release=0.6)
        self.q = np.asarray(REST_POSE, float).copy()
        # which joints L1 is allowed to move (default: all but keep it gentle)
        self.weight = np.ones(NJ) if joints is None else np.asarray(joints, float)
        self._neutral_head = kin.head_position(REST_POSE)
        self._lean = 0.06   # 0 = pure look (head stays put), 1 = head chases target

    # -- inputs --------------------------------------------------------
    def observe_point(self, p, meas_std: float | None = None) -> None:
        self.track.update_point(p, meas_std=meas_std)

    def observe_bearing(self, origin, direction) -> None:
        self.track.update_bearing(origin, direction)

    def clear(self) -> None:
        self.track.reset()

    def seed_pose(self, q) -> None:
        self.q = np.asarray(q, float).copy()

    # -- layer --------------------------------------------------------
    def update(self, ctx: BlendContext) -> LayerOutput:
        dt = ctx.dt
        self.track.predict(dt)
        if self.track.active:
            self.env.open()
            aim = self.track.predicted_position(self.lead_time)
            # keep our IK state on a short leash to the real joints (so it can't
            # drift) but let it lead by a little, so convergence isn't bottlenecked
            # by the trajectory generator's lag
            self.q = ctx.q_current + np.clip(self.q - ctx.q_current, -0.25, 0.25)
            head_goal = self._neutral_head + self._lean * (aim - self._neutral_head)
            # follow harder when the estimate is crisp, gently when it's stale
            rate = self.follow_gain * (0.35 + 0.65 * self.track.confidence)
            self.q = self.ik.step(
                head_goal, self.q, dt, aim_point=aim, gain=rate,
                pos_weight=1.0, aim_weight=1.0, max_step=0.15,
            )
        else:
            self.env.close()
        g = self.env.step(dt)
        return LayerOutput(value=self.q.copy(), weight=self.weight, gain=g, additive=False)


# ---------------------------------------------------------------------------
class PrimitiveLayer:
    """L2 expressive clips. `play("nod")` from a behaviour tag; additive offset."""

    name = "primitive"
    priority = 20

    def __init__(self, library: PrimitiveLibrary | None = None, dt: float = 0.01) -> None:
        self.lib = library or PrimitiveLibrary()
        self.dt = dt
        self.clip: Primitive | None = None
        self.t0 = 0.0
        self.env = Envelope(attack=0.18, release=0.3)
        self._weight = np.zeros(NJ)
        self._interrupted = False

    def play(self, name: str, t: float, **load_kw) -> None:
        self.clip = self.lib.get(name, **load_kw).resampled(self.dt)
        self.t0 = t
        self._interrupted = False
        # claim only the joints this clip actually moves
        span = np.ptp(self.clip.offsets, axis=0)
        self._weight = np.clip(span / (np.deg2rad(3.0)), 0.0, 1.0)
        self.env.attack = 0.18
        self.env.open()

    def stop(self) -> None:
        self.env.close(release=0.3)

    def interrupt(self) -> None:
        self._interrupted = True
        self.env.close(release=0.12)

    @property
    def busy(self) -> bool:
        return self.clip is not None and not self.env.closed

    def update(self, ctx: BlendContext) -> LayerOutput:
        if self.clip is None:
            return LayerOutput.inactive()
        rel = ctx.t - self.t0
        if not self.clip.loop and rel >= self.clip.duration and not self._interrupted:
            self.env.close()
        offset = self.clip.sample(rel)
        g = self.env.step(ctx.dt)
        if self.env.closed:
            self.clip = None
        return LayerOutput(value=offset, weight=self._weight, gain=g, additive=True)


# ---------------------------------------------------------------------------
class TaskLightLayer:
    """L3 functional pose: light a desk point from up-and-back (so the beam clears
    the user's hand shadow, per S1) and hold the pose."""

    name = "task_light"
    priority = 30

    def __init__(self, kin: ArmKinematics, *, standoff: float = 0.20) -> None:
        self.kin = kin
        self.ik = IKSolver(kin, nullspace_gain=0.08)
        self.standoff = standoff
        self.env = Envelope(attack=0.4, release=0.5)
        self.q_hold = np.asarray(REST_POSE, float).copy()
        self._active = False

    def place(self, target_point, q_seed=None) -> IKResult:
        target_point = np.asarray(target_point, float)
        # stand off up and back toward the base: a raking light angle, and a pose
        # the arm can hit without swinging the base around
        back = np.array([-target_point[0], -target_point[1], 0.0])
        n = np.linalg.norm(back)
        offset = np.array([0.0, 0.0, 1.0]) if n < 1e-6 else (
            0.55 * back / n + np.array([0.0, 0.0, 1.0])
        )
        offset = offset / np.linalg.norm(offset) * self.standoff
        approach = target_point + offset
        res = self.ik.solve(
            approach, q0=q_seed if q_seed is not None else REST_POSE,
            aim_point=target_point, restarts=4,
        )
        self.q_hold = res.q
        self._active = True
        self.env.open()
        return res

    def reach(self, point, q_seed=None) -> IKResult:
        """Put the head *at* `point` (head-shell centre), no standoff, no aim -
        the "touch it" pose."""
        res = self.ik.solve(
            np.asarray(point, float),
            q0=q_seed if q_seed is not None else REST_POSE,
            restarts=4,
        )
        self.q_hold = res.q
        self._active = True
        self.env.open()
        return res

    def clear(self) -> None:
        self._active = False
        self.env.close()

    def interrupt(self) -> None:
        self.env.close(release=0.15)
        self._active = False

    @property
    def busy(self) -> bool:
        return not self.env.closed

    def update(self, ctx: BlendContext) -> LayerOutput:
        g = self.env.step(ctx.dt)
        if g <= 1e-4 and not self._active:
            return LayerOutput.inactive()
        return LayerOutput(value=self.q_hold.copy(), weight=np.ones(NJ), gain=g, additive=False)
