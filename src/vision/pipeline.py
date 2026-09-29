"""One camera frame -> desk objects and faces in ``lamp_base``.

This is D's output boundary. What leaves here maps one-to-one onto the ROS
interfaces E already defined (jetson_ws/src/lamp_interfaces):

- ``ObjectTarget.pos`` -> ``PointStamped`` goal of ``/lamp/place_task_light`` (S1)
- ``FaceTarget.pos``   -> ``PointStamped`` on ``/lamp/track_point`` (S2, S4)
- ``VisionFrame.labels_ko()`` -> A's cognition prompt (topic not yet defined)

All positions are metres in ``lamp_base`` and every result carries a
``time.monotonic()`` stamp, per the common convention in docs/파트-분배.md 2.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np

from .camera import Intrinsics
from .detector import Detection, FaceDetection
from .geometry import FRAME_ID, Pose, desk_point, face_point
from .labels import to_korean


@dataclass(frozen=True)
class Workspace:
    """Where on the desk a task-light target is allowed to be.

    Anything outside is dropped: the ray hit the desk somewhere the lamp cannot
    light (behind it, under its own base, off the far edge). This also removes
    most detections of the lamp's own body, whose boxes project to points at or
    behind the base.
    """
    min_radius_m: float = 0.12     # lamp base footprint plus margin
    max_radius_m: float = 0.80     # beyond this the light is too weak (500 lx design point is 0.5 m)
    min_x_m: float = 0.0           # in front of the base only

    def contains(self, p: np.ndarray) -> bool:
        r = math.hypot(p[0], p[1])
        return self.min_radius_m <= r <= self.max_radius_m and p[0] >= self.min_x_m


@dataclass(frozen=True)
class ObjectTarget:
    label: str
    conf: float
    pos: np.ndarray            # (x, y, z) in lamp_base, z == desk height
    box: tuple[float, float, float, float]
    stamp: float
    frame_id: str = FRAME_ID


@dataclass(frozen=True)
class FaceTarget:
    conf: float
    pos: np.ndarray            # eye midpoint in lamp_base
    box: tuple[float, float, float, float]
    stamp: float
    frame_id: str = FRAME_ID


@dataclass
class VisionFrame:
    stamp: float
    objects: list[ObjectTarget] = field(default_factory=list)
    faces: list[FaceTarget] = field(default_factory=list)
    # Every detection before any filtering. S1 only wants desk objects inside
    # the workspace; the cognition prompt wants the whole scene (a chair, a
    # monitor, a person) — A's end-to-end example includes "의자".
    detections: list[Detection] = field(default_factory=list)
    # Raw face detections, which keep the eye landmarks that FaceTarget drops.
    # The S2 follow loop works in the image and needs them.
    face_detections: list[FaceDetection] = field(default_factory=list)
    # Detections that did not become targets, with the reason. Kept for tuning.
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def labels_ko(self) -> list[str]:
        """Korean names of everything in view, most confident first, for cognition."""
        names = [d.label for d in sorted(self.detections, key=lambda d: -d.conf)]
        if self.faces and "person" not in names:
            names.append("person")
        return to_korean(names)

    def best_object(self) -> ObjectTarget | None:
        return max(self.objects, key=lambda o: o.conf, default=None)

    def nearest_face(self) -> FaceTarget | None:
        return min(self.faces, key=lambda f: float(np.linalg.norm(f.pos)), default=None)


class VisionPipeline:
    def __init__(self, cam: Intrinsics, pose: Pose, objects=None, faces=None,
                 workspace: Workspace | None = None):
        self.cam = cam
        self.pose = pose
        self.object_detector = objects
        self.face_detector = faces
        self.workspace = workspace or Workspace()

    def locate(self, dets: list[Detection], face_dets: list[FaceDetection],
               stamp: float | None = None) -> VisionFrame:
        """Geometry only. Split from ``process`` so it can be tested without a model."""
        vf = VisionFrame(stamp=time.monotonic() if stamp is None else stamp,
                         detections=list(dets), face_detections=list(face_dets))
        for d in dets:
            if not d.task_light_class:
                vf.rejected.append((d.label, "not a task-light class"))
                continue
            p = desk_point(d.ground_uv, self.cam, self.pose)
            if p is None:
                vf.rejected.append((d.label, "ray misses desk"))
                continue
            if not self.workspace.contains(p):
                vf.rejected.append((d.label, "outside workspace"))
                continue
            vf.objects.append(ObjectTarget(d.label, d.conf, p, d.box, vf.stamp))
        for f in face_dets:
            p = face_point(f.right_eye, f.left_eye, self.cam, self.pose, f.box)
            if p is None:
                vf.rejected.append(("face", "implausible depth"))
                continue
            vf.faces.append(FaceTarget(f.conf, p, f.box, vf.stamp))
        return vf

    def process(self, bgr: np.ndarray, stamp: float | None = None,
                objects: bool = True, faces: bool = True) -> VisionFrame:
        """``objects``/``faces`` skip a detector for this frame.

        The two cost very different amounts on the Jetson's CPU -- YOLOX-s
        about 610 ms against YuNet's 13 ms -- so the caller runs the object
        detector at its own slower rate and keeps faces at full frame rate.
        """
        stamp = time.monotonic() if stamp is None else stamp
        h, w = bgr.shape[:2]
        if (w, h) != (self.cam.width, self.cam.height):
            self.cam = self.cam.scaled(w, h)
        dets = self.object_detector(bgr) if objects and self.object_detector else []
        face_dets = self.face_detector(bgr) if faces and self.face_detector else []
        return self.locate(dets, face_dets, stamp)
