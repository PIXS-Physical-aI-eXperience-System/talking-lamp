"""L0 idle motion - the always-on "it's alive" layer.

A small joint-space offset (a few degrees) added on top of whatever base pose is
active. Built from incommensurate sinusoids so it never reads as a loop:

* breathing - a slow vertical bob shared by base_pitch / elbow_pitch / wrist_pitch
  in the proportions that keep the head height moving but its aim roughly fixed;
* micro-gaze - tiny, slower drift on base_yaw and wrist_roll;
* optional 1/f-ish wander so it doesn't sit dead still between breaths.

Deterministic given a seed, so sim runs are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import NJ


@dataclass
class IdleConfig:
    breath_period: float = 4.2          # s
    breath_amp_deg: float = 2.4         # peak head-bob in "pitch-equivalent" degrees
    gaze_period: float = 11.0           # s
    gaze_amp_deg: float = 1.3
    wander_amp_deg: float = 0.6
    enabled: bool = True


class IdleMotion:
    # how the breath bob distributes over the pitch joints (keeps aim ~steady)
    _BREATH_MIX = np.array([0.0, 1.0, -0.7, 0.0, -0.3])
    _GAZE_MIX = np.array([1.0, 0.0, 0.0, 0.6, 0.0])

    def __init__(self, cfg: IdleConfig | None = None, seed: int = 7) -> None:
        self.cfg = cfg or IdleConfig()
        rng = np.random.default_rng(seed)
        self._phase = rng.uniform(0, 2 * np.pi, 4)
        # a few extra detuned components for the wander term
        self._wander_w = rng.uniform(0.15, 0.5, (3, NJ))
        self._wander_p = rng.uniform(0, 2 * np.pi, (3, NJ))

    def offset(self, t: float) -> np.ndarray:
        if not self.cfg.enabled:
            return np.zeros(NJ)
        c = self.cfg
        w_breath = 2 * np.pi / c.breath_period
        w_gaze = 2 * np.pi / c.gaze_period

        breath = np.deg2rad(c.breath_amp_deg) * np.sin(w_breath * t + self._phase[0])
        # asymmetric breath: slower inhale, quicker exhale -> add 2nd harmonic
        breath += np.deg2rad(0.25 * c.breath_amp_deg) * np.sin(2 * w_breath * t + self._phase[1])

        gaze = np.deg2rad(c.gaze_amp_deg) * np.sin(w_gaze * t + self._phase[2])
        gaze_y = np.deg2rad(0.6 * c.gaze_amp_deg) * np.sin(0.73 * w_gaze * t + self._phase[3])

        out = breath * self._BREATH_MIX + gaze * self._GAZE_MIX
        out[3] += gaze_y

        wander = np.zeros(NJ)
        for k in range(3):
            wander += np.sin(self._wander_w[k] * t + self._wander_p[k]) / (k + 1)
        out = out + np.deg2rad(c.wander_amp_deg) * wander / 1.83

        return out
