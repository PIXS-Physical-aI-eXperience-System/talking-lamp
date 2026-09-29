"""2D detections -> 3D points in the ``lamp_base`` frame.

``lamp_base`` is the team-wide frame from docs/파트-분배.md 2.4: origin at the
centre of the lamp base's underside, +x forward (toward the user), +y left,
+z up, metres. The base sits on the desk, so **the desk surface is z = 0** in
this frame. That is what makes a single camera enough for objects: a pixel
defines a ray, and the ray meets the desk at exactly one point.

Faces are not on the desk, so the desk plane cannot place them. Their depth
comes from the apparent distance between the eyes instead (``face_point``).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .camera import Intrinsics

FRAME_ID = "lamp_base"

# Adult interpupillary distance. Population mean is ~63 mm with a spread of a
# few mm, so a single-face depth is good to roughly +/-8 %. Turning the head
# shortens the apparent eye distance and makes the face read as farther away.
IPD_M = 0.063
# Fallback when eye landmarks are unusable: bizygomatic face width.
FACE_WIDTH_M = 0.14
# Seated user at a desk lamp. Outside this the estimate is treated as noise.
FACE_DEPTH_RANGE_M = (0.25, 2.5)
# Frontal faces have eye separation ~0.45 of face width (63 / 140 mm). Turning
# the head shrinks the eye separation much faster than the box, so a low ratio
# means a profile view where the eye method overestimates distance. Measured on
# the head camera: a user in near-profile gave 41 px eyes in a 200 px box
# (0.2), and the eye method read ~1 m for a face ~0.5 m away.
PROFILE_RATIO = 0.30


@dataclass(frozen=True)
class Pose:
    """Camera pose in ``lamp_base``: ``p_base = R @ p_cam + t``.

    ``t`` is the optical centre in metres; the columns of ``R`` are the camera's
    +x (right), +y (down) and +z (forward) axes expressed in ``lamp_base``.
    """

    R: np.ndarray
    t: np.ndarray
    source: str = "registered"
    rms_px: float | None = None

    @classmethod
    def look_at(cls, position, target, source: str = "assumed") -> "Pose":
        """Camera at ``position`` aimed at ``target``, image kept level (no roll)."""
        pos = np.asarray(position, dtype=float)
        fwd = np.asarray(target, dtype=float) - pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        if np.linalg.norm(right) < 1e-9:
            raise ValueError("camera looks straight up or down; roll is undefined")
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        return cls(np.column_stack([right, down, fwd]), pos, source)

    def to_base(self, p_cam) -> np.ndarray:
        return np.asarray(p_cam, dtype=float) @ self.R.T + self.t

    def to_cam(self, p_base) -> np.ndarray:
        return (np.asarray(p_base, dtype=float) - self.t) @ self.R

    @property
    def height(self) -> float:
        return float(self.t[2])

    @property
    def tilt_deg(self) -> float:
        """How far below horizontal the optical axis points."""
        fwd = self.R[:, 2]
        return math.degrees(math.asin(-fwd[2]))

    def to_dict(self) -> dict:
        return {"R": self.R.tolist(), "t": self.t.tolist(), "source": self.source,
                "rms_px": self.rms_px, "frame_id": FRAME_ID}

    @classmethod
    def from_dict(cls, d: dict) -> "Pose":
        return cls(np.array(d["R"], dtype=float), np.array(d["t"], dtype=float),
                   d.get("source", "registered"), d.get("rms_px"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Pose":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def desk_point(uv, cam: Intrinsics, pose: Pose, desk_z: float = 0.0) -> np.ndarray | None:
    """Where the ray through pixel ``uv`` meets the desk plane ``z = desk_z``.

    Returns ``None`` if the ray points at or above the horizon, which happens
    for detections that are not on the desk at all (a wall, a monitor top).
    """
    d = pose.R @ cam.pixel_to_ray(uv)
    if d[2] > -1e-6:
        return None
    s = (desk_z - pose.t[2]) / d[2]
    if s <= 0:
        return None
    return pose.t + s * d


def face_depth_from_eyes(right_eye_uv, left_eye_uv, cam: Intrinsics) -> float:
    """Distance along the optical axis to the eye line, from eye separation.

    Uses undistorted normalized coordinates, so it stays valid near the edge of
    a wide-angle image where raw pixel distances are squashed by distortion.
    """
    a, b = cam.pixel_to_normalized([right_eye_uv, left_eye_uv])
    sep = float(np.linalg.norm(a - b))
    if sep <= 1e-9:
        return math.inf
    return IPD_M / sep


def face_depth_from_box(x1: float, x2: float, v: float, cam: Intrinsics) -> float:
    a, b = cam.pixel_to_normalized([[x1, v], [x2, v]])
    w = float(abs(a[0] - b[0]))
    return FACE_WIDTH_M / w if w > 1e-9 else math.inf


def face_depth(right_eye_uv, left_eye_uv, cam: Intrinsics,
               box: tuple[float, float, float, float] | None = None) -> float | None:
    """Depth of the eye line along the optical axis, or ``None`` if implausible.

    Eye separation by default; face width when the eyes look too close
    together for the box (a profile view) or when the eye estimate is out of
    range.
    """
    z = face_depth_from_eyes(right_eye_uv, left_eye_uv, cam)
    lo, hi = FACE_DEPTH_RANGE_M
    if box is not None:
        x1, y1, x2, y2 = box
        z_box = face_depth_from_box(x1, x2, (y1 + y2) / 2, cam)
        # Eyes-to-box ratio in depth terms: z_box / z == eye_sep / box_width * (FACE_WIDTH / IPD)
        ratio = (z_box / z) * (IPD_M / FACE_WIDTH_M) if math.isfinite(z) and z > 0 else 0.0
        if ratio < PROFILE_RATIO or not lo <= z <= hi:
            z = z_box
    return z if lo <= z <= hi else None


def face_point_cam(right_eye_uv, left_eye_uv, cam: Intrinsics,
                   box: tuple[float, float, float, float] | None = None) -> np.ndarray | None:
    """Eye midpoint in the camera frame. Needs no camera pose."""
    z = face_depth(right_eye_uv, left_eye_uv, cam, box)
    if z is None:
        return None
    mid = cam.pixel_to_normalized((np.asarray(right_eye_uv, float) + np.asarray(left_eye_uv, float)) / 2)
    return np.array([mid[0] * z, mid[1] * z, z])


def face_point(right_eye_uv, left_eye_uv, cam: Intrinsics, pose: Pose,
               box: tuple[float, float, float, float] | None = None) -> np.ndarray | None:
    """3D point between the eyes, in ``lamp_base``. ``None`` if implausible.

    The eye midpoint rather than the box centre: it is what the lamp should
    look at, and YuNet's box centre sits lower, near the nose tip.
    """
    p_cam = face_point_cam(right_eye_uv, left_eye_uv, cam, box)
    return None if p_cam is None else pose.to_base(p_cam)
