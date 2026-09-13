"""Run the physical lamp with monotonic 100 Hz deadlines.

Run ``python -m motion.hardware_run --help`` for connection options.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import sys
import time
from typing import Callable

from .config import CONTROL_DT, CONTROL_HZ, REST_POSE
from .hardware_backend import FeetechBackend
from .primitives import PrimitiveLibrary
from .runtime import MotionRuntime
from .trajectory import TrajectoryGenerator


# Accept ordinary OS wakeup jitter without treating every late wake as a miss.
# A send may start at most 1 ms after its grid deadline, so adjacent starts
# can be 9 ms apart, but substantially overdue trajectory steps are discarded.
DEADLINE_JITTER_SECONDS = 0.001


@dataclass
class HardwareRunReport:
    sent_ticks: int
    deadline_misses: int
    clamped_ticks: int
    clamped_joints: dict[str, int]
    interrupted: bool


def run_hardware(
    *, port: str, lamp_id: str, primitive: str | None = None,
    duration: float = 10.0, feedback_hz: float = 20.0,
    allow_analytic_fallback: bool = False,
    backend_factory: Callable = FeetechBackend,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> HardwareRunReport:
    """Run until duration or Ctrl-C, always parking an opened backend.

    Misses count expired command slots, skipped to avoid burst catch-up. Each
    sent command advances the trajectory by one fixed 10 ms step, even after an
    overrun. Feedback and parking time are included/excluded, respectively, in
    the scheduled run duration. Up to 1 ms of deadline lateness is tolerated;
    later slots are skipped before sending, including after a late wakeup.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and greater than zero")
    if not math.isfinite(feedback_hz) or not 0 < feedback_hz <= CONTROL_HZ:
        raise ValueError(f"feedback_hz must be in (0, {CONTROL_HZ:g}]")
    if TrajectoryGenerator(REST_POSE).backend != "ruckig" and not allow_analytic_fallback:
        raise RuntimeError(
            "Physical motion requires Ruckig for jerk limits. Install ruckig or "
            "deliberately pass --allow-analytic-fallback for acceleration-only motion."
        )
    library = PrimitiveLibrary()
    if primitive is not None:
        library.get(primitive)  # Validate recordings before enabling hardware.
    backend = backend_factory(port=port, lamp_id=lamp_id, feedback_hz=feedback_hz)
    missed = 0
    interrupted = False
    try:
        runtime = MotionRuntime(backend=backend, initial_pose=backend.measured(), primitives=library)
        if primitive is not None:
            runtime.play_primitive(primitive)
        start = clock()
        slots = math.ceil(duration / CONTROL_DT - 1e-9)
        slot = 0
        while slot < slots:
            deadline = start + slot * CONTROL_DT
            remaining = deadline - clock()
            while remaining > 0:
                sleep(remaining)
                remaining = deadline - clock()
            now = clock()
            if now >= start + duration:
                missed += slots - slot
                break
            # Reconcile after waking and before advancing the trajectory. This
            # same policy covers both sleep overshoot and slow previous sends.
            # Keep the original grid; re-enter the wait if its next usable
            # deadline is still in the future.
            next_slot = min(slots, math.ceil(
                (now - start - DEADLINE_JITTER_SECONDS) / CONTROL_DT - 1e-9
            ))
            if next_slot > slot:
                missed += next_slot - slot
                slot = next_slot
                continue
            runtime.step()
            slot += 1
    except KeyboardInterrupt:
        interrupted = True
    finally:
        backend.close()
    return HardwareRunReport(
        backend.sent_ticks, missed, backend.clamped_ticks,
        dict(backend.clamped_joints), interrupted,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Feetech serial device")
    parser.add_argument("--lamp-id", required=True, help="Existing LeRobot calibration id")
    parser.add_argument("--primitive", help="Primitive name; omitted runs idle motion")
    parser.add_argument("--duration", type=float, default=10.0, help="Run seconds, excluding parking")
    parser.add_argument("--feedback-hz", type=float, default=20.0, help="Position reads/s at 100 commands/s")
    parser.add_argument("--allow-analytic-fallback", action="store_true",
                        help="Deliberately allow motion without Ruckig jerk limits")
    args = parser.parse_args(argv)
    try:
        report = run_hardware(**vars(args))
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Hardware motion failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"sent_ticks={report.sent_ticks} deadline_misses={report.deadline_misses} "
        f"clamped_ticks={report.clamped_ticks} clamped_joints={report.clamped_joints} "
        f"interrupted={report.interrupted}"
    )
    return 130 if report.interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
