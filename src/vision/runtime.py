"""What D publishes, frame by frame. No ROS here -- the node is a thin shell.

The whole design turns on one fact: **the camera rides on the head**, so the
camera pose is the arm pose run through E's FK and the mount calibration. D can
turn pixels into ``lamp_base`` points only while it knows that pose.

It learns it one of two ways:

- **Joint states** (``update_joints``). The Pi reads the servos back and the
  bridge publishes them at 5 Hz. The pose is known in any posture, for as long
  as they keep arriving.
- **The rest pose** (``enter_rest``), the fallback when no joint states arrive:
  after ``/lamp/return_center`` the angles are REST_POSE, so the pose is known
  there and nowhere else. Any move voids it (``leave_rest``).

That splits D's two jobs:

- **S1 (task light)** needs absolute positions. ``ObjectAverager`` keeps a
  rolling average while the pose is known, so when the trigger comes the answer
  is already settled (see ``tracking`` for why averaging is needed at all).
  Averages are in ``lamp_base``, so a head that moves to a new known pose keeps
  them; only frames taken mid-move are left out, because the 5 Hz pose lags a
  deliberate motion by centimetres on the desk.
- **S2 (look at the user)** needs an absolute point only for its first aim.
  After that ``FaceFollower`` works from the error in the image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from .camera import Intrinsics
from .follow import FaceFollower, FollowConfig
from .geometry import Pose, face_point
from .head_camera import HeadCameraMount
from .pipeline import ObjectTarget, VisionFrame, VisionPipeline
from .tracking import ObjectAverager, PresenceGate


@dataclass(frozen=True)
class RuntimeConfig:
    average_window_s: float = 6.0
    presence_on_s: float = 0.5
    presence_off_s: float = 5.0
    # YOLOX-s costs about 610 ms a frame on the Jetson's CPU and YuNet 13 ms, so
    # running both every frame would drop face tracking to ~1.6 Hz. Objects on a
    # desk do not move, so they get their own slow rate and faces keep the full
    # frame rate. 0.5 s still fills the averager within one breath period.
    object_period_s: float = 0.5
    # While following, the head is busy with the user and objects run mostly
    # to keep the cognition labels fresh.
    object_period_following_s: float = 2.0
    # Joint states come at 5 Hz. Older than this, the Pi or the bridge has
    # stopped and the camera pose is unknown.
    joints_stale_s: float = 0.6
    # Faster than this on any joint, the head is in a deliberate motion and a
    # pose up to 0.2 s old is centimetres off on the desk, so the frame is not
    # averaged. The idle sway peaks near 0.06 rad/s and stays below it.
    moving_rad_s: float = 0.15
    follow: FollowConfig = field(default_factory=FollowConfig)


@dataclass
class RuntimeOutput:
    """Everything the node might publish for one frame. ``None`` == nothing to say."""

    frame: VisionFrame
    track_point: np.ndarray | None = None      # -> /lamp/track_point
    presence: bool | None = None               # -> /lamp/vision/presence, on change only
    labels: list[str] | None = None            # -> /lamp/vision/labels, on change only
    note: str = ""


class VisionRuntime:
    def __init__(self, cam: Intrinsics, mount: HeadCameraMount | None = None,
                 kin=None, pipeline: VisionPipeline | None = None,
                 cfg: RuntimeConfig | None = None):
        self.cfg = cfg or RuntimeConfig()
        self.mount = mount
        self.kin = kin
        self.pipeline = pipeline or VisionPipeline(cam, Pose.look_at([0, 0, 0.35], [0.5, 0, 0]))
        self.averager = ObjectAverager(window_s=self.cfg.average_window_s)
        self.presence = PresenceGate(self.cfg.presence_on_s, self.cfg.presence_off_s)
        self.at_rest = False
        self.off_rest_since = "not yet centred"
        self.joints_at: float | None = None     # when update_joints last ran
        self.moving = False
        self.following = False
        self.follower: FaceFollower | None = None
        self.head_pos = np.array([0.18, 0.0, 0.35])
        self._labels: list[str] = []
        self._last_objects_at = -1e9

    # -- posture -------------------------------------------------------------

    def update_joints(self, q, qd, stamp: float) -> Pose:
        """The arm pose as the Pi read it back. Supersedes the rest pose."""
        if self.kin is None or self.mount is None:
            raise RuntimeError("joint states need both a mount calibration and E's kinematics")
        head_R, head_t = self.kin.head(np.asarray(q, float))
        self.head_pos = np.asarray(head_t, float)
        self.pipeline.pose = self.mount.camera_pose(head_R, head_t)
        self.joints_at = stamp
        self.moving = bool(np.max(np.abs(np.asarray(qd, float))) > self.cfg.moving_rad_s)
        self.at_rest = False
        return self.pipeline.pose

    def joints_live(self, stamp: float) -> bool:
        return self.joints_at is not None and stamp - self.joints_at <= self.cfg.joints_stale_s

    def pose_known(self, stamp: float) -> bool:
        if self.joints_at is not None:
            return self.joints_live(stamp)
        return self.at_rest

    def _pose_unknown_why(self, stamp: float) -> str:
        if self.joints_at is not None:
            return (f"관절 각도가 {stamp - self.joints_at:.1f} s 째 오지 않는다 "
                    "(Pi 모션 데몬이나 모션 브릿지 확인)")
        return f"머리가 휴식 자세에 있지 않다 ({self.off_rest_since}). return_center 먼저"

    def enter_rest(self, base_yaw: float | None = None) -> Pose:
        """Call once ``/lamp/return_center`` has finished and the head is settled.

        Fixes the camera pose from E's FK of the rest pose, so desk
        back-projection is meaningful again. Only for when no joint states
        arrive: it switches the runtime back to trusting the rest pose.
        """
        if self.kin is None or self.mount is None:
            raise RuntimeError("rest pose needs both a mount calibration and E's kinematics")
        q = self.kin.rest_q(base_yaw)
        head_R, head_t = self.kin.head(q)
        self.head_pos = np.asarray(head_t, float)
        self.pipeline.pose = self.mount.camera_pose(head_R, head_t)
        self.averager.reset()
        self.at_rest = True
        self.joints_at = None
        self.moving = False
        self.follower = None
        return self.pipeline.pose

    def leave_rest(self, why: str = "head moved") -> None:
        """The head is no longer where D thinks it is: absolute positions are void."""
        self.at_rest = False
        self.off_rest_since = why
        self.averager.reset()

    def set_following(self, on: bool) -> None:
        """S2 on/off. B turns this on at the wake word and off when the turn ends."""
        self.following = on
        if not on:
            self.follower = None

    # -- per frame -----------------------------------------------------------

    def due_for_objects(self, stamp: float) -> bool:
        period = (self.cfg.object_period_following_s if self.following
                  else self.cfg.object_period_s)
        return stamp - self._last_objects_at >= period

    def process(self, bgr: np.ndarray, stamp: float | None = None) -> RuntimeOutput:
        stamp = time.monotonic() if stamp is None else stamp
        objects = self.due_for_objects(stamp)
        frame = self.pipeline.process(bgr, stamp, objects=objects)
        return self.consume(frame, stamp, had_objects=objects)

    def consume(self, frame: VisionFrame, stamp: float,
                had_objects: bool = True) -> RuntimeOutput:
        """Geometry already done. Split from ``process`` so it tests without a model.

        ``had_objects`` says whether the object detector actually ran for this
        frame. On a faces-only frame an empty object list means "not looked
        for", not "nothing there", so neither the averages nor the labels may
        be updated from it.
        """
        out = RuntimeOutput(frame=frame)
        if had_objects:
            self._last_objects_at = stamp
            if self.pose_known(stamp) and not self.moving:
                self.averager.add(frame)
            labels = frame.labels_ko()
            if labels != self._labels:
                self._labels = labels
                out.labels = labels

        out.presence = self.presence.update(bool(frame.faces), stamp)

        if self.following:
            out.track_point, out.note = self._follow(frame, stamp)
        return out

    def _follow(self, frame: VisionFrame, stamp: float) -> tuple[np.ndarray | None, str]:
        face = _nearest_raw_face(frame)
        if face is None:
            if self.follower is not None and self.follower.lost(stamp):
                return None, "face lost"
            return None, ""

        if self.follower is None:
            # First aim needs an absolute point, so a known camera pose.
            if not self.pose_known(stamp):
                return None, "cannot start following: camera pose unknown (no joint states, not at rest)"
            target = face_point(face.right_eye, face.left_eye, self.pipeline.cam,
                                self.pipeline.pose, face.box)
            if target is None:
                return None, "implausible face depth"
            gaze = None if self.mount is None else self.mount.gaze_uv(
                self.pipeline.cam, float(np.linalg.norm(target - self.head_pos)))
            self.follower = FaceFollower(self.pipeline.cam, self.head_pos, target,
                                         self.cfg.follow, gaze_uv=gaze, stamp=stamp)
            if self.joints_at is None:
                self.leave_rest("looking at the user")      # the rest pose is about to go
            return target, "first aim"

        upd = self.follower.update(face, stamp)
        if upd is None:
            return None, ""
        return upd.target, f"correction {upd.error_deg:.1f} deg"

    # -- S1 ------------------------------------------------------------------

    def task_light_target(self, stamp: float | None = None) -> tuple[ObjectTarget | None, str]:
        """The point to send to ``/lamp/place_task_light``, or why there is none."""
        stamp = time.monotonic() if stamp is None else stamp
        best = self.averager.best(stamp)
        if best is not None:
            return best, ""
        if not self.pose_known(stamp):
            return None, self._pose_unknown_why(stamp)
        if not self.averager.tracks:
            if self.moving:
                return None, "머리가 움직이는 중이라 위치를 쌓지 않는다"
            return None, "책상 위에 조명 대상이 보이지 않는다"
        return None, "평균이 아직 덜 쌓였다 (머리가 멈춘 채로 몇 초 더 필요)"


def _nearest_raw_face(frame: VisionFrame):
    """Most confident face, with its eye landmarks, which the follow loop needs."""
    return max(frame.face_detections, key=lambda f: f.conf, default=None)
