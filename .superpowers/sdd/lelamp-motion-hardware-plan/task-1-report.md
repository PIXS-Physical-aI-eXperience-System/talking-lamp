# Task 1 report: calibrated primitive offsets

## Files changed

- `src/motion/primitives.py`
  - Treat recording values as normalized LeRobot servo commands.
  - Convert them through `HardwareAlignment` before calculating relative
    simulation-radian offsets.
  - Apply the verified playback direction map (base pitch reversed) and motion
    scales: `[0.35, 0.12, 0.15, 0.35, 0.25]`.
  - Preserve explicit `sign` and `scale` overrides.
- `tests/test_primitives.py`
  - Prove each bundled primitive has the same offset as
    `playback.retarget_actions` after both are converted to simulation radians.
  - Cover the reversed base-pitch result and explicit override behavior.
- `src/motion/README.md`
  - Replace the obsolete identity sign/scale limitation with the calibrated
    mapping limitation and recalibration trigger.

## TDD evidence

Added the parity regression before changing production code. It failed against
the old degree conversion and identity map: 12 new parity/base-pitch failures.
After implementing the calibrated conversion and playback map, the same test
file passed.

## Verification

| Command | Result |
| --- | --- |
| `PYTHONPATH= .venv/bin/python -m pytest tests/test_primitives.py` | 28 passed |
| `PYTHONPATH= .venv/bin/python -m pytest tests/test_hardware_alignment.py` | 6 passed |
| `make lint` | passed (`pyflakes src/motion tests sim/*.py`) |
| `PYTHONPATH= .venv/bin/python -m compileall -q src/motion tests` | passed |
| `make test` | 94 passed |

## Self-review

- The conversion preserves the motion stack's `(5,)` simulation-radian
  contract and leaves timing, looping, and interpolation unchanged.
- Default direction and scale values exactly match the verified playback map;
  the base-pitch regression asserts the reversal explicitly.
- The parity test exercises every bundled clip using the real playback
  retargeter and calibration object, so a wrong scale, direction, or
  normalized-to-radian conversion fails.
- Explicit overrides replace their corresponding defaults and are verified
  against the unmodified calibrated offset.
- `git diff --check` is clean.

## Concerns

No blocking concerns. The default calibration is specific to the measured
lamp and requires recalibration after changes to the mechanism, servos, or
head load; this is documented in the motion README.
