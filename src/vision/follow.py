"""S2 face following without knowing the head's joint angles.

E's L1 layer takes a point in ``lamp_base`` (``/lamp/track_point``) and turns
the head to aim its +x axis at it. With the camera on the head, D cannot turn a
face pixel into a ``lamp_base`` point directly: that needs the head pose at the
moment of capture, and the joint angles never leave the Pi.

So this closes the loop in the image instead (image-based visual servoing on
top of E's point tracking):

- The target D last published is taken as where the head is looking. That is
  only true once the head has arrived, so **corrections are made only when the
  head has settled**, detected from the image alone: the face's offset stops
  changing over a few frames.
- The face's angular offset from the head's line of sight in the image is the
  error. With a mount calibration that line of sight is ``mount.gaze_uv``;
  without one, the image centre.
- The target is rotated by a fraction of that error around the head and
  published again.

Correcting while the head is still moving stacks new corrections on top of
ones it has not executed yet, and the head overshoots; a closed-loop test with a
lagging head showed a 1.9 deg overshoot doing exactly that. Waiting for the
settle makes each correction start from a known pose, so with a gain below one
there is nothing to overshoot, whatever E's lag is.

Approximations, and why they are safe here: the head position is taken as the
rest-pose head position (it moves only a few cm while tracking), and the
camera frame is rebuilt as level (no roll). Both only change the effective loop
gain slightly; the error itself is measured in the image and goes to zero at
convergence regardless.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .camera import Intrinsics
from .detector import FaceDetection
from .geometry import face_depth


@dataclass(frozen=True)
class FollowConfig:
    gain: float = 0.8              # fraction of the measured angle corrected per update
    deadband_deg: float = 2.0      # errors below this are left alone (no jitter)
    max_step_deg: float = 25.0     # largest single correction
    settle_frames: int = 3         # frames the error must hold still before correcting
    settle_deg: float = 0.5        # "still": spread of those frames' errors; above L0 idle sway
    min_interval_s: float = 0.20   # floor between corrections, so the head has started moving
    lost_after_s: float = 1.0      # E's tracker drops the target after 1 s without updates
    distance_gain: float = 0.5     # how fast the target distance follows the face distance


@dataclass
class FollowUpdate:
    target: np.ndarray             # new aim point in lamp_base
    error_deg: float               # angular error that triggered it
    stamp: float


class FaceFollower:
    def __init__(self, cam: Intrinsics, head_pos, initial_target, cfg: FollowConfig | None = None,
                 gaze_uv=None, stamp: float | None = None):
        self.cam = cam
        self.head_pos = np.asarray(head_pos, float)
        self.target = np.asarray(initial_target, float)
        self.cfg = cfg or FollowConfig()
        self.gaze_uv = None if gaze_uv is None else np.asarray(gaze_uv, float)
        self._last_sent = -math.inf
        self._last_seen = -math.inf if stamp is None else stamp
        self._recent: list[tuple[float, float]] = []   # (yaw, pitch) of the last few frames

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.target - self.head_pos))

    def lost(self, stamp: float) -> bool:
        return stamp - self._last_seen > self.cfg.lost_after_s

    def _gaze_normalized(self) -> np.ndarray:
        if self.gaze_uv is None:
            return np.zeros(2)                       # principal point
        return self.cam.pixel_to_normalized(self.gaze_uv)

    def error(self, face: FaceDetection) -> tuple[float, float]:
        """(yaw, pitch) of the face relative to the head's line of sight, radians.

        Camera convention: +yaw to the image right, +pitch to the image down.
        """
        mid = (np.asarray(face.right_eye, float) + np.asarray(face.left_eye, float)) / 2
        m = self.cam.pixel_to_normalized(mid)
        g = self._gaze_normalized()
        return math.atan(m[0]) - math.atan(g[0]), math.atan(m[1]) - math.atan(g[1])

    def settled(self) -> bool:
        n = self.cfg.settle_frames
        if len(self._recent) < n:
            return False
        e = np.degrees(np.array(self._recent[-n:]))
        return float(np.ptp(e[:, 0])) < self.cfg.settle_deg and float(np.ptp(e[:, 1])) < self.cfg.settle_deg

    def update(self, face: FaceDetection | None, stamp: float) -> FollowUpdate | None:
        """Feed every frame's best face. Returns a new target to publish, or ``None``."""
        if face is None:
            self._recent.clear()
            return None
        self._last_seen = stamp
        yaw, pitch = self.error(face)
        self._recent = (self._recent + [(yaw, pitch)])[-self.cfg.settle_frames:]
        if stamp - self._last_sent < self.cfg.min_interval_s or not self.settled():
            return None

        err = math.hypot(yaw, pitch)
        if math.degrees(err) < self.cfg.deadband_deg:
            return None
        step = min(err * self.cfg.gain, math.radians(self.cfg.max_step_deg))
        yaw, pitch = yaw * step / err, pitch * step / err

        # Level frame around the current line of sight (camera right / down).
        d = self.target - self.head_pos
        dist = float(np.linalg.norm(d))
        d /= dist
        right = np.cross(d, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(d, right)
        new_d = d + math.tan(yaw) * right + math.tan(pitch) * down
        new_d /= np.linalg.norm(new_d)

        z = face_depth(face.right_eye, face.left_eye, self.cam, face.box)
        if z is not None:
            dist += self.cfg.distance_gain * (z - dist)

        self.target = self.head_pos + dist * new_d
        self._last_sent = stamp
        self._recent.clear()          # the head is about to move; wait for a fresh settle
        return FollowUpdate(self.target.copy(), math.degrees(err), stamp)
