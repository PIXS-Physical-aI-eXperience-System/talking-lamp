"""What D publishes, frame by frame. No ROS here -- the node is a thin shell.

The whole design turns on one fact: **the camera rides on the head, and the
head's joint angles never leave the Pi.** So D can only convert pixels into
``lamp_base`` points at a posture whose angles it already knows, and the only
such posture is the rest pose that ``/lamp/return_center`` restores.

That splits D's two jobs cleanly:

- **S1 (task light)** needs absolute positions, so it runs **only at rest**.
  ``ObjectAverager`` keeps a rolling average the whole time the head sits
  there, so when the trigger comes the answer is already settled and there is
  no capture delay (see ``tracking`` for why averaging is needed at all).
- **S2 (look at the user)** needs no absolute position: ``FaceFollower``
  measures the error in the image and nudges the target E is already tracking.
  It keeps working after the head has left the rest pose, which S1 cannot.

Publishing a track point makes the head move, so the runtime marks itself off
rest at that moment and drops the averages: they were computed from a pose that
no longer holds. Getting back to absolute positions needs another
``/lamp/return_center``.
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
    # While following, object positions are void anyway (the head is off the
    # rest pose). Objects then run only to keep the cognition labels fresh.
    object_period_following_s: float = 2.0
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
        self.following = False
        self.follower: FaceFollower | None = None
        self.head_pos = np.array([0.18, 0.0, 0.35])
        self._labels: list[str] = []
        self._last_objects_at = -1e9

    # -- posture -------------------------------------------------------------

    def enter_rest(self, base_yaw: float | None = None) -> Pose:
        """Call once ``/lamp/return_center`` has finished and the head is settled.

        Fixes the camera pose from E's FK of the rest pose, so desk
        back-projection is meaningful again.
        """
        if self.kin is None or self.mount is None:
            raise RuntimeError("rest pose needs both a mount calibration and E's kinematics")
        q = self.kin.rest_q(base_yaw)
        head_R, head_t = self.kin.head(q)
        self.head_pos = np.asarray(head_t, float)
        self.pipeline.pose = self.mount.camera_pose(head_R, head_t)
        self.averager.reset()
        self.at_rest = True
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
            if self.at_rest:
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
            # First aim needs an absolute point, which only the rest pose gives.
            if not self.at_rest:
                return None, "cannot start following away from the rest pose"
            target = face_point(face.right_eye, face.left_eye, self.pipeline.cam,
                                self.pipeline.pose, face.box)
            if target is None:
                return None, "implausible face depth"
            gaze = None if self.mount is None else self.mount.gaze_uv(
                self.pipeline.cam, float(np.linalg.norm(target - self.head_pos)))
            self.follower = FaceFollower(self.pipeline.cam, self.head_pos, target,
                                         self.cfg.follow, gaze_uv=gaze, stamp=stamp)
            self.leave_rest("looking at the user")
            return target, "first aim from the rest pose"

        upd = self.follower.update(face, stamp)
        if upd is None:
            return None, ""
        return upd.target, f"correction {upd.error_deg:.1f} deg"

    # -- S1 ------------------------------------------------------------------

    def task_light_target(self, stamp: float | None = None) -> tuple[ObjectTarget | None, str]:
        """The point to send to ``/lamp/place_task_light``, or why there is none."""
        stamp = time.monotonic() if stamp is None else stamp
        if not self.at_rest:
            return None, f"머리가 휴식 자세에 있지 않다 ({self.off_rest_since}) — return_center 먼저"
        best = self.averager.best(stamp)
        if best is None:
            if not self.averager.tracks:
                return None, "책상 위에 조명 대상이 보이지 않는다"
            return None, "평균이 아직 덜 쌓였다 (휴식 자세로 몇 초 더 필요)"
        return best, ""


def _nearest_raw_face(frame: VisionFrame):
    """Most confident face, with its eye landmarks, which the follow loop needs."""
    return max(frame.face_detections, key=lambda f: f.conf, default=None)
