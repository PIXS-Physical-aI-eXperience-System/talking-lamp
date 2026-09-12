# motion/ - Talking Lamp motion stack (part E)

L0-L3 layered motion, IK, online trajectory generation, Kalman target tracking,
and the 100 Hz runtime that blends them into 5-axis servo commands. Covers
[진행-순서.md](../../docs/진행-순서.md) part E steps 6-10 and the "움직임" half of
[파트-분배.md](../../docs/파트-분배.md) 4.5.

```
layers ──▶ MotionBlender ──▶ q_blend ──▶ TrajectoryGenerator ──▶ q_cmd ──▶ backend
  L0 idle        priority-composited        vel/accel/jerk           MuJoCo sim
  L1 track       (gain·weight per joint)    calibrated joint bounds or Feetech bus
  L2 primitive
  L3 task_light
```

## Install / run

From the repo root (the `.venv` there already has the deps):

```bash
make test            # full suite (ruckig backend)
make test-fallback   # same, analytic trajectory backend
make demo            # scripted end-to-end demo -> sim/out/
```

`make` sets `PYTHONPATH=` so a sourced ROS environment can't shadow the venv.
Running pytest directly (`.venv/bin/python -m pytest tests`) also works - the
repo's `addopts` blocks the ROS pytest plugins - but `make` is the safe path.
First-time setup: `make venv`.

## Physical lamp

Use an environment with this project's NumPy, MuJoCo, and Ruckig dependencies
plus the LeRobot dependencies from `lelamp_runtime/pyproject.toml`. From the
repository root, expose both source trees (no LeRobot import is needed for
simulation or `--help`):

```bash
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" python -m motion.hardware_run \
  --port /dev/ttyACM0 --lamp-id lelamp \
  --primitive nod --duration 10 --feedback-hz 20
```

Use the serial port and the checked-in `lelamp` calibration profile. The runner
rejects a different ID, saved calibration, or motor-resident calibration before
enabling torque. Connection uses `calibrate=False` and normalized servo positions. Omit `--primitive` for
idle motion. The runtime seeds its trajectory and tracking pose from the first
physical measurement while keeping `REST_POSE` as its idle target. The trajectory
generator bounds blended targets to the exact calibrated joint ranges. Ruckig
replans are checked at every continuous-time position extremum; an unsafe replan
is retried while the previous feasible trajectory continues. This keeps the
command and planner state consistent at endpoints. The analytic fallback reserves
braking distance within those same bounds. All outbound radian commands also pass
through `HardwareAlignment` and a defensive clamp to calibrated servo endpoints.

The runner uses monotonic 10 ms deadlines. It skips expired command slots to
avoid sending a burst after an overrun, and reports `sent_ticks`,
`deadline_misses` (skipped slots), `clamped_ticks`, and per-joint clamp counts.
Each sent command advances the trajectory by 10 ms; overload therefore slows
motion instead of increasing a command's trajectory step. Position feedback is
cached, with a default read every five sends (20 Hz nominal); `measured()` does
not perform a bus read. `--feedback-hz` accepts rates in `(0, 100]`, with lower
rates reducing serial traffic. Real hardware timing still needs measurement.

Ruckig is required before the serial connection opens. The deliberate override
`--allow-analytic-fallback` permits the degraded acceleration-limited tracker,
which does **not** bound jerk. Completion, Ctrl-C, and runtime exceptions call
`lelamp.playback.park_and_disconnect`: reach the captured `SLEEP_POSE`, confirm
it, then disconnect and release torque. Parking can take several additional
seconds beyond `--duration`. Playback and parking keep every planned step after a
late send or sleep, stretching their duration without a catch-up burst. If parking
fails, the existing helper raises and keeps torque engaged; the runner does not
force a disconnect. The legacy motor services retain the follower on startup or
cleanup failure until it can be safely parked and disconnected.

## Modules

| module | what |
| --- | --- |
| `config.py` | joint order (the contract with B/A/D), radian limits, control rate, `REST_POSE` |
| `kinematics.py` | `ArmKinematics` - FK + site Jacobian for the 5 joints, backed by `sim/world.xml` |
| `ik.py` | `IKSolver` - damped least squares, SVD selective damping, joint-limit freezing, null-space bias. `solve()` for one-shot goals, `step()` for 100 Hz streaming |
| `trajectory.py` | `TrajectoryGenerator` - online jerk-limited OTG via **ruckig**, with a self-contained accel-limited fallback (`TALKING_LAMP_NO_RUCKIG=1`) |
| `kalman.py` | `TargetTrack` - constant-velocity KF, point (vision) or bearing (audio DOA) updates, coasts + reports `confidence` through detection gaps |
| `primitives.py` | `Primitive` / `PrimitiveLibrary` - LeLamp `recordings/*.csv` loaded as **relative** radian-offset clips (sign/scale-mappable) |
| `idle.py` | `IdleMotion` - breathing + micro-gaze, incommensurate sinusoids |
| `blender.py` | `MotionBlender` + `BlendContext` - priority compositing, `gain·weight` authority, additive vs absolute layers |
| `layers.py` | `IdleLayer` `TrackLayer` `PrimitiveLayer` `TaskLightLayer` + `Envelope` |
| `runtime.py` | `MotionRuntime` - owns the loop and the 4 layers; team-facing API below |
| `hardware_backend.py` | `FeetechBackend` - lazy physical follower adapter, calibrated commands, cached feedback, clamp counts, safe parking |
| `hardware_run.py` | physical CLI with Ruckig gate and monotonic 100 Hz deadlines |
| `sim_backend.py` | `MujocoDynamicsBackend` (servo lag, gravity) / `MujocoKinematicsBackend` (exact) |

## Team-facing API (`MotionRuntime`)

| caller | call | scenario |
| --- | --- | --- |
| D (vision) | `rt.track.observe_point(xyz)` | face position -> L1 follow (S2) |
| C (audio) | `rt.track.observe_bearing(origin, dir)` | sound DOA -> L1 turn (S6) |
| A via B | `rt.play_primitive("nod")` | behaviour tag -> L2 clip (S2) |
| D via B | `rt.place_task_light(desk_xyz)` | light a work spot -> L3 (S1) |
| - | `rt.reach_to(xyz)` | put the head *on* a point (touch it) |
| B | `rt.barge_in()` | user talks over the lamp -> drop L2/L3 fast |
| loop | `rt.step()` at 100 Hz | returns `StepState(t, q_blend, q_cmd, q_meas, vel, trace)` |

All joint arrays are `(5,)` radians in `config.JOINT_NAMES` order - the interface
E exposes to the rest of the team ("절대 관절 각도" convention).

`dt` is fixed at construction in both `MotionRuntime` and `TrajectoryGenerator`.
A per-step `dt` may repeat that configured period; a different or non-finite value
raises before time, layers, or trajectory state advance. `StepState.trace` is the
blend trace used for that command; logging should read it without calling
`blender.compute()` again. Explicit primitive `sign`, `scale`, and `loop` options
reload the clip and leave the cached default unchanged.

## Known limitations / TODO

- **Kinematics is the `build_arm.py` stopgap model**, not a CAD re-export -
  see `sim/README.md`. Geometry inherits its approximations; joint endpoints use
  the measured hardware calibration.
- **Kinematic limits** (`config.VEL/ACC/JERK_LIMIT`) are conservative guesses;
  retune once the real head weight is measured.
- **Primitive calibration** follows the measured servo-to-simulation mapping
  and the verified playback direction/scale map (including reversed base
  pitch). Recalibrate these values after changing the mechanism, servos, or
  head load.
- **Head "forward" axis** comes from the CAD site frame; "look straight ahead"
  can still cock the base ~25°. Fine for faces, revisit if it reads wrong.
- The analytic fallback bounds velocity, acceleration, and calibrated position,
  but does not bound jerk; install `ruckig` (a declared dep) for smooth jerk.
- No self-collision geometry yet (proxy boxes are inertia-only).
