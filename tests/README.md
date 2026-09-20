# Automated tests

This directory is the repository's pytest suite. Tests here must run without
physical audio, RGB, motor, or serial devices.

- `unit/`: isolated control and math behavior.
- `model/`: kinematics, calibration, and MuJoCo model contracts.
- `integration/`: runtime and hardware-boundary behavior using fakes or local
  recording data. These tests never open a real serial port.
- `simulation/`: headless simulator command and tool behavior.

Run the suite from the repository root with `make test`. Manual checks that
operate real LeLamp hardware live under `lelamp_runtime/lelamp/test/`.
