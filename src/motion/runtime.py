"""MotionRuntime - the 100 Hz loop that turns layers into servo commands.

    layers ──▶ MotionBlender ──▶ q_blend ──▶ TrajectoryGenerator ──▶ q_cmd ──▶ backend

The blender output can jump (a primitive fires, a track appears); the trajectory
generator absorbs that into a velocity/accel/jerk-limited path. The backend is
either the MuJoCo sim (dynamics or kinematic) or, later, the Feetech bus.

Wiring for the rest of the team:
* D (vision) / C (audio):  ``rt.track.observe_point(xyz)`` / ``observe_bearing(...)``
* A (cognition) via B:     ``rt.play_primitive("nod")``  (behaviour tag)
* D (vision) via B:        ``rt.place_task_light(desk_xyz)``      (S1)
* B (barge-in):            ``rt.barge_in()``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .blender import BlendContext, MotionBlender
from .config import CONTROL_DT, REST_POSE
from .idle import IdleConfig
from .kinematics import ArmKinematics
from .layers import IdleLayer, PrimitiveLayer, TaskLightLayer, TrackLayer
from .primitives import PrimitiveLibrary
from .trajectory import TrajectoryGenerator


class Backend(Protocol):
    def send(self, q_cmd: np.ndarray) -> None: ...
    def measured(self) -> np.ndarray | None: ...


@dataclass
class NullBackend:
    """No hardware - the trajectory command is the truth."""

    _last: np.ndarray | None = None

    def send(self, q_cmd: np.ndarray) -> None:
        self._last = np.asarray(q_cmd, float).copy()

    def measured(self) -> np.ndarray | None:
        return self._last


@dataclass
class StepState:
    t: float
    q_blend: np.ndarray
    q_cmd: np.ndarray
    q_meas: np.ndarray
    vel: np.ndarray


class MotionRuntime:
    def __init__(
        self,
        *,
        kin: ArmKinematics | None = None,
        backend: Backend | None = None,
        rest_pose: np.ndarray = REST_POSE,
        idle_cfg: IdleConfig | None = None,
        dt: float = CONTROL_DT,
        primitives: PrimitiveLibrary | None = None,
    ) -> None:
        self.kin = kin or ArmKinematics()
        self.dt = float(dt)
        self.rest_pose = np.asarray(rest_pose, float)

        self.idle = IdleLayer(idle_cfg, rest_pose=self.rest_pose)
        self.track = TrackLayer(self.kin)
        self.primitive = PrimitiveLayer(primitives or PrimitiveLibrary(), dt=self.dt)
        self.task_light = TaskLightLayer(self.kin)
        self.blender = MotionBlender(
            [self.idle, self.track, self.primitive, self.task_light], self.rest_pose
        )

        self.backend = backend or NullBackend()
        self.traj = TrajectoryGenerator(self.rest_pose, dt=self.dt)
        self.track.seed_pose(self.rest_pose)
        self.t = 0.0

    # -- team-facing controls ----------------------------------------
    def play_primitive(self, name: str, **load_kw) -> None:
        self.primitive.play(name, self.t, **load_kw)

    def place_task_light(self, desk_point, *, seed_from_current: bool = True):
        seed = self.traj.pos if seed_from_current else None
        return self.task_light.place(desk_point, q_seed=seed)

    def reach_to(self, point, *, seed_from_current: bool = True):
        """Move the head to touch `point` (head-shell centre at the point)."""
        seed = self.traj.pos if seed_from_current else None
        return self.task_light.reach(point, q_seed=seed)

    def clear_task_light(self) -> None:
        self.task_light.clear()

    def barge_in(self) -> None:
        """C detected the user talking over the lamp - drop expressive + task
        motion fast and let reflex/idle take the head to a neutral listening pose."""
        self.primitive.interrupt()
        self.task_light.interrupt()

    # -- loop -------------------------------------------------------
    def step(self, dt: float | None = None) -> StepState:
        h = self.dt if dt is None else float(dt)
        self.t += h
        ctx = BlendContext(q_current=self.traj.pos.copy(), t=self.t, dt=h)
        trace = self.blender.compute(ctx)
        q_cmd = self.traj.step(trace.q, h)
        self.backend.send(q_cmd)
        meas = self.backend.measured()
        q_meas = self.traj.pos if meas is None else np.asarray(meas, float)
        return StepState(self.t, trace.q, q_cmd, q_meas, self.traj.vel.copy())

    def run(
        self, duration: float, on_step: Callable[[StepState], None] | None = None
    ) -> list[StepState]:
        n = int(round(duration / self.dt))
        log = []
        for _ in range(n):
            s = self.step()
            if on_step is not None:
                on_step(s)
            log.append(s)
        return log
