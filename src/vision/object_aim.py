"""S1 without 3D: point the head at the book the way S2 points it at a face.

Placing the task light from a desk point needs the camera mount, and the mount
on the lamp is not trustworthy: tilting only the wrist moved the same book from
0.13 m to 0.33 m on the desk (2026-10-30). Centering in the image needs none of
that. The head tilts down, sweeps left and right until a task-light object is
in view, then takes the same small yaw/wrist steps the face follow uses until
the object sits in the middle of the frame. The light rides on the head, so a
centred book is a lit book.

It tilts down until it sees a book, or all the way before choosing anything
else, and then prefers a book and otherwise the object lowest in the frame,
i.e. nearest the lamp. From a home pose already bowed to 57 deg, the full 55 deg
more looked past the book in front of the base (2026-10-30 21:25), so a book
seen on the way down is taken there. On the first
try (2026-10-30) it locked on the user's laptop, fully in view next to them,
while the book in front of the base sat half out of the bottom of the frame,
and so it lit the user.

Steps are relative and bounded like ``FaceFollowLayer``'s (yaw 12 deg, wrist
55 deg down from home), and the offsets here mirror that clamp so a target past the
limit ends as "as centred as it gets" instead of pushing forever.

Once centred, ``reach_point`` gives the desk point for ``/lamp/place_task_light``
so the arm reaches out toward the book instead of only bowing the head. The
direction comes from the base yaw the book was centred at, which the joints
measure without the camera mount; the distance is fixed at 15 cm, the middle of
the 12-18 cm the task-light pose can hold.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .image_follow import ImageCorrection, ImageFaceFollower, ImageFollowConfig


REACH_M = 0.15
# The task-light IK puts the base this far left of the desk point it aims at
# (E's model, 2026-10-30: 21.2 deg at 0.12-0.18 m, the same for any azimuth).
# Subtracting it keeps the base where the book was centred.
REACH_YAW_OFFSET_DEG = 21.2
# Turn the reach this much from where the book was centred; negative is the
# lamp's right (base_yaw is counter-clockwise from above). On the first reach
# (2026-10-30 21:32) the user asked for about 10 deg to the lamp's right.
REACH_TRIM_DEG = -10.0


def reach_point(base_yaw_rad: float, distance: float = REACH_M) -> tuple[float, float, float]:
    """Desk point whose task-light pose turns the base ``REACH_TRIM_DEG`` from
    ``base_yaw_rad``."""
    az = base_yaw_rad + math.radians(REACH_TRIM_DEG - REACH_YAW_OFFSET_DEG)
    return (distance * math.cos(az), distance * math.sin(az), 0.0)


@dataclass(frozen=True)
class AimConfig:
    max_yaw_deg: float = 12.0          # FaceFollowLayer.MAX_YAW_FROM_HOME_DEG
    max_wrist_deg: float = 55.0        # FaceFollowLayer.MAX_WRIST_DOWN_FROM_HOME_DEG
    step_deg: float = 2.0
    search_interval_s: float = 0.5
    settle_frames: int = 2             # object passes in a row with nothing to correct
    # Object passes in a row without the book before looking around again. The
    # detector misses a book it saw a second ago often enough that going back
    # to the sweep on one miss lost it for good (2026-10-30).
    lost_frames: int = 3
    # Objects are detected less surely than faces, and the frame from YOLOX
    # comes about once a second, so match wider and accept lower scores.
    follow: ImageFollowConfig = ImageFollowConfig(
        deadband_px=60.0, max_match_px=500.0, min_confidence=0.3, min_interval_s=0.6)


class ObjectAimer:
    """searching -> centering -> centered."""

    def __init__(self, cfg: AimConfig | None = None) -> None:
        self.cfg = cfg or AimConfig()
        self.follower = ImageFaceFollower(self.cfg.follow)
        self.reset()

    def reset(self) -> None:
        self.state = "searching"
        self.label = ""
        self.yaw = 0.0                 # offset from home this aimer has asked for
        self.wrist = 0.0
        self._dir = 1.0
        self._last_step = -math.inf
        self._quiet = 0
        self._missed = 0
        self.follower.reset()

    def _clamp(self, yaw: float, wrist: float) -> tuple[float, float]:
        """Clip a step to what the motion layer will still move, and track it."""
        new_yaw = max(-self.cfg.max_yaw_deg, min(self.cfg.max_yaw_deg, self.yaw + yaw))
        new_wrist = max(-self.cfg.max_wrist_deg, min(self.cfg.max_wrist_deg, self.wrist + wrist))
        step = (new_yaw - self.yaw, new_wrist - self.wrist)
        self.yaw, self.wrist = new_yaw, new_wrist
        return step

    def update(self, detections, width: int, height: int, stamp: float,
               *, moving: bool = False) -> ImageCorrection | None:
        """``detections``: this frame's object detections, or None when the
        object detector did not run on it. Returns a step to send, or None."""
        if self.state == "centered":
            return None
        if self.state == "searching" and self.wrist < self.cfg.max_wrist_deg:
            books = [d for d in detections or () if d.task_light_class and d.label == "book"]
            if books:                                   # a book on the way down
                return self._centre(books, width, height, stamp, moving)
            return self._search(stamp, moving)          # look down before choosing
        if detections is not None:
            targets = [d for d in detections if d.task_light_class]
            if targets:
                self._missed = 0
                return self._centre(targets, width, height, stamp, moving)
            if self.state == "centering":
                self._missed += 1
                if self._missed < self.cfg.lost_frames:
                    return None                 # hold still and look again
                self.state = "searching"        # lost it: look around again
                self.follower.reset()
        if self.state == "searching":
            return self._search(stamp, moving)
        return None

    def _centre(self, targets, width, height, stamp, moving):
        if self.state != "centering":
            # Choose once: a book first, then whatever sits lowest in the frame.
            # After that the follower keeps to the one nearest where it was.
            targets = [max(targets, key=lambda d: (d.label == "book", d.box[3]))]
            self.follower.reset()
        self.state = "centering"
        corr = self.follower.propose(targets, width, height, stamp, moving=moving)
        centre = self.follower._center
        best = min(targets, key=lambda d: math.dist(
            ((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2), centre))
        self.label = best.label
        if corr is not None:
            yaw, wrist = self._clamp(corr.yaw_deg, corr.wrist_deg)
            if yaw or wrist:
                self._quiet = 0
                return ImageCorrection(yaw, wrist, corr.face_center)
            # Wants to go past the limit: this is as centred as it gets.
        elif moving or stamp - self.follower._last_step < self.cfg.follow.min_interval_s:
            return None
        self._quiet += 1
        if self._quiet >= self.cfg.settle_frames:
            self.state = "centered"
        return None

    def _search(self, stamp, moving):
        if moving or stamp - self._last_step < self.cfg.search_interval_s:
            return None
        self._last_step = stamp
        if self.wrist < self.cfg.max_wrist_deg:          # look down first
            _, wrist = self._clamp(0.0, self.cfg.step_deg)
            return ImageCorrection(0.0, wrist, (0.0, 0.0))
        if abs(self.yaw + self._dir * self.cfg.step_deg) > self.cfg.max_yaw_deg:
            self._dir = -self._dir                       # sweep back the other way
        yaw, _ = self._clamp(self._dir * self.cfg.step_deg, 0.0)
        return ImageCorrection(yaw, 0.0, (0.0, 0.0))
