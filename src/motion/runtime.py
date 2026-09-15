"""MotionRuntime - the 100 Hz loop that turns layers into servo commands.

    layers ──▶ MotionBlender ──▶ q_blend ──▶ TrajectoryGenerator ──▶ q_cmd ──▶ backend

The blender output can jump (a primitive fires, a track appears); the trajectory
generator absorbs that into a velocity/accel/jerk-limited path. The backend is
either the MuJoCo sim (dynamics or kinematic) or the Feetech bus.

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

from .blender import BlendContext, BlendTrace, MotionBlender
from .config import CONTROL_DT, NJ, REST_POSE
from .idle import IdleConfig
from .hardware_alignment import HardwareAlignment
from .kinematics import ArmKinematics
from .layers import IdleLayer, PrimitiveLayer, PrimitivePlayInfo, TaskLightLayer, TrackLayer
from .orientation import (
    BaseYawOrientationLayer,
    OrientationConfig,
    OrientationCoordinator,
    OrientationSnapshot,
)
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

    def close(self) -> None:
        """Idempotent lifecycle hook; no physical connection to park."""


@dataclass
class StepState:
    t: float
    q_blend: np.ndarray
    q_cmd: np.ndarray
    q_meas: np.ndarray
    vel: np.ndarray
    trace: BlendTrace


class MotionRuntime:
    def __init__(
        self,
        *,
        kin: ArmKinematics | None = None,
        backend: Backend | None = None,
        rest_pose: np.ndarray = REST_POSE,
        initial_pose: np.ndarray | None = None,
        idle_cfg: IdleConfig | None = None,
        orientation_cfg: OrientationConfig | None = None,
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
        position_limits = HardwareAlignment.load().joint_limits
        orientation_cfg = orientation_cfg or OrientationConfig()
        self.orientation = BaseYawOrientationLayer(orientation_cfg, position_limits)
        self.orientation_control = OrientationCoordinator(self.orientation, orientation_cfg)
        self.blender = MotionBlender(
            [self.idle, self.track, self.orientation, self.primitive, self.task_light],
            self.rest_pose,
        )

        self.backend = backend or NullBackend()
        initial = self.rest_pose if initial_pose is None else np.asarray(initial_pose, float)
        if initial.shape != (NJ,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_pose must contain five finite radians")
        # Preserve the exact measured range, including valid initial poses
        # at its endpoints; the generated MJCF rounds these same limits.
        self.traj = TrajectoryGenerator(initial, dt=self.dt, position_limits=position_limits)
        self.track.seed_pose(initial)
        self.t = 0.0

    # -- team-facing controls ----------------------------------------
    def play_primitive(self, name: str, **load_kw) -> PrimitivePlayInfo:
        if self.orientation.target_yaw is not None:
            return self.primitive.play(
                name,
                self.t,
                yaw_anchor=self.orientation.target_yaw,
                yaw_limits=self.orientation.safe_yaw_limits,
                **load_kw,
            )
        return self.primitive.play(name, self.t, **load_kw)

    def observe_point(self, point) -> None:
        self.track.observe_point(point)

    def observe_bearing(self, direction) -> None:
        """Observe a bearing in the lamp base frame (origin at the base)."""
        self.track.observe_bearing(np.zeros(3), direction)

    def clear_tracking(self) -> None:
        self.track.clear()

    def acquire_orientation(self, speech_id: str, target_yaw: float, *, now: float):
        snapshot = self.orientation_control.acquire(
            speech_id,
            target_yaw,
            now=now,
            current_yaw=float(self.traj.pos[0]),
            task_light_busy=self.task_light.busy,
            current_velocity=float(self.traj.vel[0]),
        )
        if snapshot.target_yaw is not None:
            self.primitive.refit_yaw(
                yaw_anchor=snapshot.target_yaw, yaw_limits=self.orientation.safe_yaw_limits,
            )
        return snapshot

    def return_center(self, *, now: float, motion_busy: bool):
        return self.orientation_control.return_center(
            now=now,
            current_yaw=float(self.traj.pos[0]),
            motion_busy=motion_busy,
        )

    def release_orientation(self) -> None:
        self.orientation_control.release()

    def orientation_snapshot(self) -> OrientationSnapshot:
        return self.orientation_control._snapshot()

    def orientation_safe_yaw_limits(self) -> tuple[float, float]:
        return self.orientation.safe_yaw_limits

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
        # Reject an unsupported period before advancing time or any layer.
        h = self.traj.validate_dt(dt)
        anchor = self.orientation.target_yaw
        self.primitive.refit_yaw(
            yaw_anchor=anchor,
            yaw_limits=self.orientation.safe_yaw_limits if anchor is not None else None,
        )
        self.t += h
        ctx = BlendContext(q_current=self.traj.pos.copy(), t=self.t, dt=h)
        trace = self.blender.compute(ctx)
        limits = None
        if anchor is not None and not self.task_light.busy:
            limits = self.traj.position_limits.copy()
            lo, hi = self.orientation.safe_yaw_limits
            # Initial hardware poses can lie within the mechanical margin.
            # Permit a continuous recovery from that pose into the safe range.
            limits[0] = min(lo, self.traj.pos[0]), max(hi, self.traj.pos[0])
        q_cmd = self.traj.step(trace.q, h, position_limits=limits)
        self.backend.send(q_cmd)
        meas = self.backend.measured()
        q_meas = self.traj.pos if meas is None else np.asarray(meas, float)
        return StepState(self.t, trace.q, q_cmd, q_meas, self.traj.vel.copy(), trace)

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
