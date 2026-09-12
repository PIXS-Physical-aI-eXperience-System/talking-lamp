"""Constant-velocity Kalman filter for reflex-layer target tracking (L1).

Vision (face box centre projected to 3D by D) and audio DOA (a bearing from C)
arrive noisy, late, and intermittently. L1 needs a smooth, always-available
estimate of *where to look* at 100 Hz. This filter:

* runs a 3D constant-velocity model (state = [pos(3), vel(3)]);
* predicts every tick, updates only when a measurement is supplied;
* inflates covariance during measurement gaps and reports ``confidence`` so the
  blender can fade the layer out when the track goes stale;
* accepts either a 3D point (vision) or a unit bearing at assumed range (audio).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackConfig:
    process_std: float = 0.8          # m/s^2, how much real accel we allow
    meas_std: float = 0.03            # m, vision measurement noise
    bearing_range: float = 0.8        # m, assumed distance for audio-only bearings
    bearing_meas_std: float = 0.15    # m, effective noise for a bearing measurement
    max_coast: float = 1.0            # s without a measurement before track drops
    init_vel_std: float = 0.5


@dataclass
class TargetTrack:
    cfg: TrackConfig = field(default_factory=TrackConfig)
    x: np.ndarray = field(default_factory=lambda: np.zeros(6))
    P: np.ndarray = field(default_factory=lambda: np.eye(6))
    initialised: bool = False
    time_since_meas: float = np.inf

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        self.initialised = False
        self.time_since_meas = np.inf

    def _init_from(self, pos: np.ndarray) -> None:
        self.x = np.concatenate([pos, np.zeros(3)])
        self.P = np.diag(
            [self.cfg.meas_std**2] * 3 + [self.cfg.init_vel_std**2] * 3
        )
        self.initialised = True
        self.time_since_meas = 0.0

    # -- predict / update ----------------------------------------------
    def predict(self, dt: float) -> None:
        if not self.initialised:
            return
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        q = self.cfg.process_std**2
        # white-noise-acceleration Q
        Q = np.zeros((6, 6))
        Q[:3, :3] = q * dt**4 / 4 * np.eye(3)
        Q[:3, 3:] = q * dt**3 / 2 * np.eye(3)
        Q[3:, :3] = q * dt**3 / 2 * np.eye(3)
        Q[3:, 3:] = q * dt**2 * np.eye(3)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.time_since_meas += dt
        if self.time_since_meas > self.cfg.max_coast:
            self.initialised = False

    def update_point(self, pos: np.ndarray, meas_std: float | None = None) -> None:
        pos = np.asarray(pos, float)
        if not self.initialised:
            self._init_from(pos)
            return
        r = (meas_std if meas_std is not None else self.cfg.meas_std) ** 2
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        R = r * np.eye(3)
        y = pos - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.time_since_meas = 0.0

    def update_bearing(self, origin: np.ndarray, direction: np.ndarray) -> None:
        """Audio DOA: a ray from `origin` along unit `direction`. Treated as a
        point measurement at the configured assumed range."""
        direction = np.asarray(direction, float)
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        pt = np.asarray(origin, float) + self.cfg.bearing_range * direction
        self.update_point(pt, meas_std=self.cfg.bearing_meas_std)

    # -- outputs ------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    def predicted_position(self, lead_time: float) -> np.ndarray:
        return self.x[:3] + lead_time * self.x[3:]

    @property
    def confidence(self) -> float:
        """0..1 - collapses as the position covariance grows during a gap."""
        if not self.initialised:
            return 0.0
        pos_var = np.trace(self.P[:3, :3]) / 3.0
        c = self.cfg.meas_std**2 / (self.cfg.meas_std**2 + pos_var)
        return float(np.clip(c, 0.0, 1.0))

    @property
    def active(self) -> bool:
        return self.initialised
