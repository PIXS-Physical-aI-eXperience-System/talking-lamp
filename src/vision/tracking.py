"""Steadying the per-frame results over time.

Two problems a single frame cannot solve:

**The head is never quite still.** E's L0 idle layer adds a few degrees of
breathing sway (``motion.idle``: 2.4 deg breath, 1.3 deg gaze, 0.6 deg wander)
on top of whatever pose the arm holds, including the rest pose. D assumes the
rest pose when it back-projects onto the desk, so that sway lands directly in
the answer: replaying the idle offsets through E's FK moves a desk point by
1.1 cm at the median and up to 17.7 cm at the worst phase.

The sway is periodic and close to zero mean, so averaging a track over a breath
period (4.2 s) cancels most of it. D's camera runs continuously, so the average
is already built up by the time S1 fires -- no capture delay is added.
``ObjectAverager`` keeps that rolling average per object.

**A face flickers.** YuNet misses a frame when the user blinks or turns away,
and a lone missed frame must not read as "left the desk" (S4).
``PresenceGate`` applies hysteresis: present quickly, away slowly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from .pipeline import ObjectTarget, VisionFrame


@dataclass
class ObjectTrack:
    """One physical object, seen over several frames."""

    label: str
    samples: deque = field(default_factory=deque)   # (stamp, pos, conf, box)

    @property
    def pos(self) -> np.ndarray:
        return np.mean([s[1] for s in self.samples], axis=0)

    @property
    def conf(self) -> float:
        return float(np.mean([s[2] for s in self.samples]))

    @property
    def last_seen(self) -> float:
        return self.samples[-1][0]

    @property
    def span(self) -> float:
        """Seconds between the oldest and newest sample held."""
        return self.samples[-1][0] - self.samples[0][0]

    @property
    def scatter(self) -> float:
        """RMS distance of the samples from their mean, metres.

        Mostly idle sway. A value far above that means the association is
        merging two objects, or the object is being moved.
        """
        p = np.array([s[1] for s in self.samples])
        return float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))

    def target(self) -> ObjectTarget:
        return ObjectTarget(self.label, self.conf, self.pos, self.samples[-1][3], self.last_seen)


class ObjectAverager:
    """Rolling average of each desk object's position while the head is at rest.

    Only valid at rest: away from it the camera pose is unknown, so the caller
    must ``reset()`` whenever the head leaves the rest pose.
    """

    def __init__(self, window_s: float = 6.0, assoc_radius_m: float = 0.12,
                 min_samples: int = 8, min_span_s: float = 4.2, stale_s: float = 1.0):
        self.window_s = window_s
        self.assoc_radius_m = assoc_radius_m
        self.min_samples = min_samples
        self.min_span_s = min_span_s          # one breath period of E's idle layer
        self.stale_s = stale_s
        self.tracks: list[ObjectTrack] = []

    def reset(self) -> None:
        self.tracks.clear()

    def add(self, frame: VisionFrame) -> None:
        for o in frame.objects:
            best, best_d = None, self.assoc_radius_m
            for tr in self.tracks:
                if tr.label != o.label:
                    continue
                d = float(np.linalg.norm(tr.pos - o.pos))
                if d < best_d:
                    best, best_d = tr, d
            if best is None:
                best = ObjectTrack(o.label)
                self.tracks.append(best)
            best.samples.append((o.stamp, np.asarray(o.pos, float), o.conf, o.box))
        self._prune(frame.stamp)

    def _prune(self, stamp: float) -> None:
        for tr in self.tracks:
            while tr.samples and stamp - tr.samples[0][0] > self.window_s:
                tr.samples.popleft()
        self.tracks = [tr for tr in self.tracks if tr.samples]

    def ready(self, stamp: float) -> list[ObjectTrack]:
        """Tracks averaged over enough time to have cancelled the idle sway."""
        return [tr for tr in self.tracks
                if len(tr.samples) >= self.min_samples
                and tr.span >= self.min_span_s
                and stamp - tr.last_seen <= self.stale_s]

    def best(self, stamp: float) -> ObjectTarget | None:
        """The object S1 should light: the most confident settled track."""
        ready = self.ready(stamp)
        if not ready:
            return None
        return max(ready, key=lambda tr: tr.conf).target()


class PresenceGate:
    """Is the user at the desk? (S4)

    Asymmetric on purpose: leaning out of frame for a moment is not leaving,
    but coming back should be noticed at once. ``off_s`` is the **[미정]**
    threshold in docs/파트-분배.md 4.4; 5 s is a starting value, not a measured
    one -- long enough to ride out a stretch or a turn to the window.
    """

    def __init__(self, on_s: float = 0.5, off_s: float = 5.0, present: bool = False):
        self.on_s = on_s
        self.off_s = off_s
        self.present = present
        self._since = math.inf       # last stamp at which the raw reading agreed

    def update(self, seen: bool, stamp: float) -> bool | None:
        """Returns the new state only when it changes, else ``None``."""
        if math.isinf(self._since):
            self._since = stamp      # start the clock at the first frame, not at -inf
        if seen == self.present:
            self._since = stamp      # raw agrees with the state: no pending change
            return None
        if stamp - self._since >= (self.on_s if seen else self.off_s):
            self.present = seen
            self._since = stamp
            return seen
        return None
