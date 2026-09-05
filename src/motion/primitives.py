"""Motion primitives - short canned clips the cognition layer triggers by tag.

Sourced from the LeLamp recordings (`lelamp_runtime/lelamp/recordings/*.csv`,
30 Hz, degrees, in LeLamp's *calibrated* joint space). That space does not line
up with our MuJoCo model (different zero/sign), so clips are used **relative**:
the delta from frame 0, converted to radians, optionally sign/scale-mapped per
joint, and added by the blender as an offset on top of the current base pose.

    prim = Primitive.load("nod")
    off = prim.sample(t)        # (5,) radian offset, 0 at t<=0 and t>=duration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import JOINT_NAMES, NJ, RECORDINGS_DIR

# LeLamp calibrated space -> our model. Tune against the sim; +1 = same sense.
DEFAULT_SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
DEFAULT_SCALE = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

CLIP_NAMES = (
    "nod", "headshake", "curious", "excited", "happy_wiggle",
    "sad", "shy", "shock", "scanning", "wake_up", "idle",
)


@dataclass
class Primitive:
    name: str
    times: np.ndarray            # (T,) seconds from 0
    offsets: np.ndarray          # (T, 5) radian delta from frame 0
    loop: bool = False

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    @classmethod
    def load(
        cls,
        name: str,
        *,
        sign: np.ndarray | None = None,
        scale: np.ndarray | None = None,
        loop: bool | None = None,
        recordings_dir: Path | None = None,
    ) -> "Primitive":
        path = Path(recordings_dir or RECORDINGS_DIR) / f"{name}.csv"
        raw = np.genfromtxt(path, delimiter=",", names=True)
        # genfromtxt turns the header "base_yaw.pos" into the field "base_yawpos"
        deg = np.stack([raw[f"{j}pos"] for j in JOINT_NAMES], axis=1)
        t = raw["timestamp"].astype(float)
        t = t - t[0]

        rad = np.deg2rad(deg.astype(float))
        rad = rad - rad[0]  # relative to first frame
        rad *= (sign if sign is not None else DEFAULT_SIGN)
        rad *= (scale if scale is not None else DEFAULT_SCALE)

        return cls(
            name=name,
            times=t,
            offsets=rad,
            loop=(name == "idle") if loop is None else loop,
        )

    def sample(self, t: float) -> np.ndarray:
        if t <= 0:
            return np.zeros(NJ)
        if self.loop:
            t = t % self.duration
        elif t >= self.duration:
            return np.zeros(NJ)
        return np.array([np.interp(t, self.times, self.offsets[:, i]) for i in range(NJ)])

    def resampled(self, dt: float) -> "Primitive":
        n = max(int(round(self.duration / dt)) + 1, 2)
        tt = np.linspace(0.0, self.duration, n)
        off = np.stack([np.interp(tt, self.times, self.offsets[:, i]) for i in range(NJ)], axis=1)
        return Primitive(self.name, tt, off, self.loop)


@dataclass
class PrimitiveLibrary:
    recordings_dir: Path = field(default_factory=lambda: RECORDINGS_DIR)
    _cache: dict[str, Primitive] = field(default_factory=dict)

    def get(self, name: str, **kw) -> Primitive:
        if name not in self._cache:
            self._cache[name] = Primitive.load(
                name, recordings_dir=self.recordings_dir, **kw
            )
        return self._cache[name]

    def available(self) -> list[str]:
        return sorted(p.stem for p in Path(self.recordings_dir).glob("*.csv"))
