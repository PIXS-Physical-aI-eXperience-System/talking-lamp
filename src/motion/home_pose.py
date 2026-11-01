"""The arm's home pose, kept across restarts.

The daemon used to take the pose the arm was in at start-up as home, so
someone set the arm by hand before every start. Started at boot, the arm lies
wherever it fell when the power went off. The first start without a saved
pose saves the start-up pose; later starts rise to it
(``StartupRiseLayer``). To teach a new home pose, stop the daemon, set the
arm, delete the file, and start the daemon again.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .config import HOLD_TORQUE_MARGIN, NJ, SERVO_RATED_TORQUE_NM
from .hardware_alignment import HardwareAlignment
from .kinematics import ArmKinematics


def check(pose) -> str:
    """Why this pose cannot be home, or "" if it can."""
    q = np.asarray(pose, float)
    if q.shape != (NJ,) or not np.all(np.isfinite(q)):
        return "not five finite radians"
    lo, hi = HardwareAlignment.load().joint_limits.T
    if np.any(q < lo) or np.any(q > hi):
        return "outside the joint limits"
    torque = float(ArmKinematics().holding_torque(q).max())
    if torque > SERVO_RATED_TORQUE_NM * HOLD_TORQUE_MARGIN:
        return f"needs {torque:.2f} N*m to hold"
    return ""


def load(path: Path) -> np.ndarray | None:
    """The saved pose, or None when there is none or it cannot be used."""
    try:
        q = np.asarray(json.loads(path.read_text())["pose_rad"], float)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Home pose {path} unreadable ({exc}); using the start-up pose",
              file=sys.stderr, flush=True)
        return None
    reason = check(q)
    if reason:
        print(f"Home pose {path} rejected ({reason}); using the start-up pose",
              file=sys.stderr, flush=True)
        return None
    return q


def save(path: Path, pose) -> bool:
    q = np.asarray(pose, float)
    reason = check(q)
    if reason:
        print(f"Start-up pose not saved as home ({reason})", file=sys.stderr, flush=True)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"pose_rad": [float(v) for v in q],
                               "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    os.replace(tmp, path)
    return True
