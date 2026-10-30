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

from dataclasses import dataclass, replace
from typing import Callable, Protocol

import numpy as np

from .blender import BlendContext, BlendTrace, MotionBlender
from .config import CONTROL_DT, NJ, REST_POSE
from .idle import IdleConfig
from .hardware_alignment import HardwareAlignment
from .kinematics import ArmKinematics
from .layers import AwayPoseLayer, FaceFollowLayer, IdleLayer, NodHoldLayer, PrimitiveLayer, PrimitivePlayInfo, TaskLightLayer, TrackLayer
from .orientation import (
    BaseYawOrientationLayer,
    OrientationConfig,
    OrientationCoordinator,
    OrientationSnapshot,
)
from .primitives import DEFAULT_SCALE, PrimitiveLibrary
from .trajectory import TrajectoryGenerator

# The first physical nod at intensity 0.5 moved the wrist about 4.4 deg, which
# read as too small. Only nod uses this; other clips keep the shared scale.
NOD_WRIST_GAIN = 2.0


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
        hold_initial_pose: bool = False,
        idle_cfg: IdleConfig | None = None,
        orientation_cfg: OrientationConfig | None = None,
        dt: float = CONTROL_DT,
        primitives: PrimitiveLibrary | None = None,
    ) -> None:
        self.kin = kin or ArmKinematics()
        self.dt = float(dt)
        self.rest_pose = np.asarray(rest_pose, float)
        initial = self.rest_pose if initial_pose is None else np.asarray(initial_pose, float)
        if initial.shape != (NJ,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_pose must contain five finite radians")

        self.idle = IdleLayer(idle_cfg, rest_pose=self.rest_pose)
        self.track = TrackLayer(self.kin)
        self.primitive = PrimitiveLayer(primitives or PrimitiveLibrary(), dt=self.dt)
        self.task_light = TaskLightLayer(self.kin)
        position_limits = HardwareAlignment.load().joint_limits
        orientation_cfg = orientation_cfg or OrientationConfig()
        if hold_initial_pose:
            # The pose the arm was in at start-up is the user's chosen rest
            # pose. Return turns there too, not to the fixed REST_POSE yaw:
            # once that sent the head 17 deg away from the user after every
            # sound turn (2026-10-30).
            lo = position_limits[0, 0] + orientation_cfg.yaw_margin
            hi = position_limits[0, 1] - orientation_cfg.yaw_margin
            orientation_cfg = replace(
                orientation_cfg, center_yaw=float(np.clip(initial[0], lo, hi)))
        self.orientation = BaseYawOrientationLayer(orientation_cfg, position_limits)
        self.orientation_control = OrientationCoordinator(self.orientation, orientation_cfg)
        self.face_follow = FaceFollowLayer(initial, position_limits)
        self.nod_hold = NodHoldLayer()
        self.away_pose = AwayPoseLayer(self.kin)
        self.blender = MotionBlender(
            [self.idle, self.track, self.orientation, self.face_follow,
             self.away_pose, self.nod_hold, self.primitive, self.task_light],
            initial if hold_initial_pose else self.rest_pose,
        )

        self.backend = backend or NullBackend()
        # Preserve the exact measured range, including valid initial poses
        # at its endpoints; the generated MJCF rounds these same limits.
        self.traj = TrajectoryGenerator(initial, dt=self.dt, position_limits=position_limits)
        self.measured_pose = initial.copy()
        self.track.seed_pose(initial)
        self.t = 0.0

    # -- team-facing controls ----------------------------------------
    def play_primitive(self, name: str, **load_kw) -> PrimitivePlayInfo:
        nod_moves = False
        if name == "nod":
            scale = np.asarray(load_kw.get("scale", DEFAULT_SCALE), float).copy()
            if scale.shape != (NJ,):
                raise ValueError("nod scale must contain five joints")
            scale[:4] = 0.0
            scale[4] *= NOD_WRIST_GAIN
            nod_moves = bool(scale[4])
            load_kw["scale"] = scale
        if self.orientation.target_yaw is not None:
            info = self.primitive.play(
                name,
                self.t,
                yaw_anchor=self.orientation.target_yaw,
                yaw_limits=self.orientation.safe_yaw_limits,
                **load_kw,
            )
        else:
            info = self.primitive.play(name, self.t, **load_kw)
        if nod_moves:
            self.nod_hold.hold(self.measured_pose)
        else:
            self.nod_hold.release()
        return info

    def observe_point(self, point) -> None:
        self.track.observe_point(point)

    def observe_bearing(self, direction) -> None:
        """Observe a bearing in the lamp base frame (origin at the base)."""
        self.track.observe_bearing(np.zeros(3), direction)

    def clear_tracking(self) -> None:
        self.track.clear()

    def face_step(self, yaw_deg: float, wrist_deg: float) -> None:
        self.face_follow.step(self.measured_pose, yaw_deg, wrist_deg)

    def clear_face_follow(self) -> None:
        self.face_follow.clear()

    def acquire_orientation(self, speech_id: str, target_yaw: float, *, now: float):
        snapshot = self.orientation_control.acquire(
            speech_id,
            target_yaw,
            now=now,
            current_yaw=float(self.measured_pose[0]),
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
            current_yaw=float(self.measured_pose[0]),
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

    def hold_away_pose(self) -> float | None:
        """Take up the S4 away pose. Returns the predicted torque, or None if refused."""
        self.face_follow.clear()
        return self.away_pose.hold()

    def clear_away_pose(self) -> None:
        self.away_pose.clear()

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
        if self.primitive.active_name != "nod":
            self.nod_hold.release()
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
        if q_meas.shape != (NJ,) or not np.all(np.isfinite(q_meas)):
            raise ValueError("backend measurement must contain five finite radians")
        self.measured_pose = q_meas.copy()
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
