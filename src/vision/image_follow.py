"""Small camera-centering corrections for the assembled lamp.

The camera mount's absolute head transform is not reliable enough for the
first S2 aim. This controller uses the measured image response instead. It
only proposes bounded base-yaw and wrist-pitch changes; the motion owner must
enforce joint limits, hold the other joints, and stop when feedback is stale.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ImageFollowConfig:
    yaw_px_per_deg: float = 35.0
    wrist_px_per_deg: float = 18.0
    # 1 deg every 0.6 s was too slow to keep a returning user in view after
    # the S4 greeting (2026-10-30). The motion layer accepts up to 2 deg.
    max_step_deg: float = 2.0
    deadband_px: float = 30.0
    min_interval_s: float = 0.3
    max_match_px: float = 300.0
    min_confidence: float = 0.6


@dataclass(frozen=True)
class ImageCorrection:
    yaw_deg: float
    wrist_deg: float
    face_center: tuple[float, float]


class ImageFaceFollower:
    def __init__(self, cfg: ImageFollowConfig | None = None) -> None:
        self.cfg = cfg or ImageFollowConfig()
        self._center: tuple[float, float] | None = None
        self._last_step = -math.inf

    def reset(self) -> None:
        self._center = None
        self._last_step = -math.inf

    def propose(self, faces, width: int, height: int, stamp: float,
                *, moving: bool = False) -> ImageCorrection | None:
        if width <= 0 or height <= 0 or not math.isfinite(stamp):
            raise ValueError("frame size and stamp must be valid")
        candidates = [face for face in faces if face.conf >= self.cfg.min_confidence]
        if not candidates:
            self._center = None
            return None

        def center(face):
            x0, y0, x1, y1 = face.box
            return ((x0 + x1) / 2, (y0 + y1) / 2)

        if self._center is None:
            face = max(candidates, key=lambda item:
                       (item.box[2] - item.box[0]) * (item.box[3] - item.box[1]))
        else:
            face = min(candidates, key=lambda item: math.dist(center(item), self._center))
            if math.dist(center(face), self._center) > self.cfg.max_match_px:
                self._center = None
                return None
        self._center = center(face)
        if moving or stamp - self._last_step < self.cfg.min_interval_s:
            return None

        dx = width / 2 - self._center[0]
        dy = height / 2 - self._center[1]
        if abs(dx) < self.cfg.deadband_px:
            dx = 0.0
        if abs(dy) < self.cfg.deadband_px:
            dy = 0.0
        if dx == dy == 0.0:
            return None
        # The live micro-step measured that positive base yaw shifts the image
        # right, while negative wrist pitch shifts it down.
        yaw = max(-self.cfg.max_step_deg, min(self.cfg.max_step_deg,
                  dx / self.cfg.yaw_px_per_deg))
        wrist = max(-self.cfg.max_step_deg, min(self.cfg.max_step_deg,
                    -dy / self.cfg.wrist_px_per_deg))
        self._last_step = stamp
        return ImageCorrection(yaw, wrist, self._center)
