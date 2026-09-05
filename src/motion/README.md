# motion/ - Talking Lamp motion stack (part E)

L0-L3 layered motion, IK, online trajectory generation, Kalman target tracking,
and the 100 Hz runtime that blends them into 5-axis servo commands. Covers
[진행-순서.md](../../docs/진행-순서.md) part E steps 6-10 and the "움직임" half of
[파트-분배.md](../../docs/파트-분배.md) 4.5.

```
layers ──▶ MotionBlender ──▶ q_blend ──▶ TrajectoryGenerator ──▶ q_cmd ──▶ backend
  L0 idle        priority-composited        vel/accel/jerk           MuJoCo sim
  L1 track       (gain·weight per joint)    limited, no overshoot    or Feetech bus
  L2 primitive
  L3 task_light
```

## Install / run

From the repo root (the `.venv` there already has the deps):

```bash
make test            # 50 tests (ruckig backend)
make test-fallback   # same, analytic trajectory backend
make demo            # scripted end-to-end demo -> sim/out/
```

`make` sets `PYTHONPATH=` so a sourced ROS environment can't shadow the venv.
Running pytest directly (`.venv/bin/python -m pytest tests`) also works - the
repo's `addopts` blocks the ROS pytest plugins - but `make` is the safe path.
First-time setup: `make venv`.

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
| loop | `rt.step()` at 100 Hz | returns `StepState(t, q_blend, q_cmd, q_meas, vel)` |

All joint arrays are `(5,)` radians in `config.JOINT_NAMES` order - the interface
E exposes to the rest of the team ("절대 관절 각도" convention).

## Known limitations / TODO

- **Kinematics is the `build_arm.py` stopgap model**, not a CAD re-export -
  see `sim/README.md`. IK/limits inherit its approximations.
- **Kinematic limits** (`config.VEL/ACC/JERK_LIMIT`) are conservative guesses;
  retune once the real head weight is measured.
- **Primitive sign/scale** (`primitives.DEFAULT_SIGN/SCALE`) are identity -
  eyeball each clip in the viewer and set the per-joint map.
- **Head "forward" axis** comes from the CAD site frame; "look straight ahead"
  can still cock the base ~25°. Fine for faces, revisit if it reads wrong.
- Trajectory generator's analytic fallback allows a 1-tick decel spike at the
  final corner - install `ruckig` (a declared dep) for clean jerk.
- No self-collision geometry yet (proxy boxes are inertia-only).
