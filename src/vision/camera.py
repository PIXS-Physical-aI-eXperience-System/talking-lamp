"""Pinhole camera model with Brown-Conrady distortion, in pure numpy.

Kept free of OpenCV so the geometry can be tested on a machine without cv2.
The parameter layout matches OpenCV's ``calibrateCamera`` output exactly
(``K`` and ``dist = [k1, k2, p1, p2, k3]``), so a calibration produced on the
Jetson loads here unchanged.

Pixel coordinates are OpenCV's: origin at the top-left pixel centre, +u right,
+v down. The camera frame is OpenCV's optical frame: +x right, +y down,
+z forward (out of the lens).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # k1, k2, p1, p2, k3 — OpenCV's 5-term model
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    # Where the numbers came from. "fov" means an uncalibrated guess.
    source: str = "calibrated"
    rms: float | None = field(default=None, compare=False)

    @classmethod
    def from_fov(cls, width: int, height: int, diag_fov_deg: float) -> "Intrinsics":
        """Rough intrinsics from a catalogue field of view, no distortion.

        Only for running the pipeline before a calibration exists. A 120 deg
        wide-angle lens has strong barrel distortion that this ignores, so
        points near the image edge will be off by several centimetres.
        """
        half_diag = math.hypot(width, height) / 2
        f = half_diag / math.tan(math.radians(diag_fov_deg) / 2)
        return cls(width, height, f, f, (width - 1) / 2, (height - 1) / 2, source="fov")

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)

    def scaled(self, width: int, height: int) -> "Intrinsics":
        """Same lens at another capture resolution (same aspect ratio assumed)."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(
            width, height, self.fx * sx, self.fy * sy,
            (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5,
            self.dist, self.source, self.rms,
        )

    # -- distortion ---------------------------------------------------------

    def _distort_norm(self, xy: np.ndarray) -> np.ndarray:
        k1, k2, p1, p2, k3 = self.dist
        x, y = xy[..., 0], xy[..., 1]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([xd, yd], axis=-1)

    def _undistort_norm(self, xyd: np.ndarray, iters: int = 20) -> np.ndarray:
        # Fixed-point iteration, the same scheme cv2.undistortPoints uses.
        if not any(self.dist):
            return xyd
        k1, k2, p1, p2, k3 = self.dist
        xy = xyd.copy()
        for _ in range(iters):
            x, y = xy[..., 0], xy[..., 1]
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            xy = np.stack([(xyd[..., 0] - dx) / radial, (xyd[..., 1] - dy) / radial], axis=-1)
        return xy

    # -- projection ---------------------------------------------------------

    def pixel_to_normalized(self, uv) -> np.ndarray:
        """Pixels -> undistorted normalized image coords (x/z, y/z)."""
        uv = np.asarray(uv, dtype=float)
        xyd = np.stack([(uv[..., 0] - self.cx) / self.fx, (uv[..., 1] - self.cy) / self.fy], axis=-1)
        return self._undistort_norm(xyd)

    def pixel_to_ray(self, uv) -> np.ndarray:
        """Pixels -> unit viewing rays in the camera frame."""
        xy = self.pixel_to_normalized(uv)
        ray = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], axis=-1)
        return ray / np.linalg.norm(ray, axis=-1, keepdims=True)

    def project(self, p_cam) -> np.ndarray:
        """Camera-frame points (z > 0) -> distorted pixels. Used by tests."""
        p = np.asarray(p_cam, dtype=float)
        xy = self._distort_norm(p[..., :2] / p[..., 2:3])
        return np.stack([xy[..., 0] * self.fx + self.cx, xy[..., 1] * self.fy + self.cy], axis=-1)

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "width": self.width, "height": self.height,
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "dist": list(self.dist), "source": self.source, "rms": self.rms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        dist = tuple(float(v) for v in d.get("dist", [0.0] * 5))
        dist = (dist + (0.0,) * 5)[:5]
        return cls(int(d["width"]), int(d["height"]), float(d["fx"]), float(d["fy"]),
                   float(d["cx"]), float(d["cy"]), dist, d.get("source", "calibrated"), d.get("rms"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Intrinsics":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
