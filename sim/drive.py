"""Drive the lamp to a point - live in the MuJoCo viewer, or headless.

Runs the full motion stack (MotionRuntime) at 100 Hz on the dynamics backend and
sends the head where you tell it.

    make drive                       # viewer + hold-key control in this terminal
    make drive ARGS="--mode reach"   # ... start in reach mode
    make goto ARGS="0.30 0.12 0.02"  # (any terminal) jump the target to a point
    make goto ARGS="0.30 0.12 0 light"
    make goto ARGS="home"

    make drive ARGS="--point 0.3 0.1 0 --headless"   # one-shot, no window
    make drive ARGS="-i --headless"                  # type 'x y z' lines, no window

Coordinates are metres. Origin = base bottom-centre, +x forward, +z up.
Desk surface z~=0; a seated face is around [0.4, 0, 0.35].

Modes:  track  aim the head at the point (S2 / S6)
        light  work-light pose above+behind it (S1)
        reach  put the head *on* the point (touch it)

CONTROL (keep this terminal focused, not the viewer window):
    W / S   forward / back        A / D   left / right        R / F   up / down
    (arrow keys + PageUp/Down work too)   hold a key = keep moving, release = stop
    + / -   faster / slower    1 / 2 / 3   mode    0   home    SPACE  stop   Q  quit
The green ball marks the target; in the viewer window you can also Ctrl+right-drag
it. `Esc` in the viewer restores the free camera if a stray key locked it.
"""

from __future__ import annotations

from _bootstrap import strip_ros_paths

strip_ros_paths()

import argparse
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import _keys  # noqa: E402
from motion import MotionRuntime  # noqa: E402
from motion.config import CONTROL_DT, REST_POSE  # noqa: E402
from motion.sim_backend import MujocoDynamicsBackend  # noqa: E402

TARGET_FILE = Path(__file__).resolve().parent / ".target"
HOME_TARGET = np.array([0.42, 0.0, 0.34])


def _parse_target_line(line: str, state: dict) -> None:
    """Accepts 'x y z', 'x y z <mode>', a bare mode word, or 'home'."""
    toks = line.split()
    if not toks:
        return
    if toks[-1] in ("track", "light", "reach"):
        state["mode"] = toks[-1]
        toks = toks[:-1]
    if toks == ["home"]:
        state["target"] = HOME_TARGET.copy()
    elif len(toks) == 3:
        try:
            state["target"] = np.array([float(v) for v in toks])
        except ValueError:
            print(f"  ? can't read '{line}'")
    elif toks:
        print(f"  ? need 'x y z' or 'x y z <track|light|reach>', got '{line}'")


def _aim_error(rt, be, target) -> float:
    pose = rt.kin.head_pose(be.measured())
    to = target - pose.pos
    return float(np.degrees(np.arccos(np.clip(np.dot(pose.forward, to / np.linalg.norm(to)), -1, 1))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["track", "light", "reach"], default="track")
    ap.add_argument("--point", type=float, nargs=3, metavar=("X", "Y", "Z"))
    ap.add_argument("-i", "--interactive", action="store_true", help="read 'x y z' lines from stdin")
    ap.add_argument("--headless", action="store_true", help="no viewer window")
    ap.add_argument("--no-file", action="store_true", help="ignore sim/.target")
    args = ap.parse_args()

    be = MujocoDynamicsBackend(q0=REST_POSE, control_dt=CONTROL_DT)
    rt = MotionRuntime(backend=be, dt=CONTROL_DT)
    model, data = be.model, be.data
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    mocap = model.body_mocapid[target_body]

    state = {
        "target": np.array(args.point) if args.point else data.mocap_pos[mocap].copy(),
        "mode": args.mode,
        "running": True,
        "vel": np.zeros(3),
        "vel_until": 0.0,
    }
    placed = None
    placed_mode = [None]
    file_mtime = 0.0

    def poll_file() -> None:
        nonlocal file_mtime
        if args.no_file:
            return
        try:
            m = TARGET_FILE.stat().st_mtime
        except OSError:
            return
        if m != file_mtime:
            file_mtime = m
            _parse_target_line(TARGET_FILE.read_text().strip(), state)

    def tick() -> None:
        nonlocal placed
        target = state["target"]
        data.mocap_pos[mocap] = target
        mode = state["mode"]
        if mode == "track":
            rt.track.observe_point(target)
        elif placed is None or np.linalg.norm(target - placed) > 0.01 or mode != placed_mode[0]:
            (rt.reach_to if mode == "reach" else rt.place_task_light)(target)
            placed = target.copy()
            placed_mode[0] = mode
        rt.step()

    def report() -> None:
        t = state["target"]
        print(f"[{state['mode']}] target {t.round(3)}  ->  joints(deg) "
              f"{np.degrees(be.measured()).round(1)}  head "
              f"{rt.kin.head_pose(be.measured()).pos.round(3)}  aim off {_aim_error(rt, be, t):.1f}°")

    # ---- headless ----
    if args.headless:
        for _ in range(int(6.0 / CONTROL_DT)):
            tick()
        report()
        if args.interactive:
            print("type:  x y z  |  x y z <track|light|reach>  |  home  |  q")
            for line in sys.stdin:
                if line.strip() in ("q", "quit", "exit"):
                    break
                _parse_target_line(line.strip(), state)
                for _ in range(int(2.0 / CONTROL_DT)):
                    tick()
                report()
        return

    # ---- viewer ----
    if args.interactive:
        def _lines() -> None:
            for line in sys.stdin:
                if line.strip() in ("q", "quit", "exit"):
                    state["running"] = False
                    return
                _parse_target_line(line.strip(), state)
        threading.Thread(target=_lines, daemon=True).start()
        print("type:  x y z  |  x y z <track|light|reach>  |  home  |  q")
    elif _keys.available():
        threading.Thread(target=_keys.run, args=(state,), daemon=True).start()
        print("hold  W/S A/D R/F  (or arrows) to move the target,  1/2/3 mode,  0 home,  Q quit")
    else:
        print('no tty for key control - use  make goto ARGS="x y z"  from another terminal')

    import mujoco.viewer as _viewer

    last_print = 0.0
    with _viewer.launch_passive(model, data) as viewer:
        i = 0
        while viewer.is_running() and state["running"]:
            tic = time.perf_counter()
            i += 1

            pert = viewer.perturb
            if pert.select == target_body and (pert.active or pert.active2):
                state["target"] = np.array(pert.refpos, dtype=float)
            else:
                if time.monotonic() < state["vel_until"]:
                    state["target"] = state["target"] + state["vel"] * CONTROL_DT
                if i % 5 == 0:
                    poll_file()

            tick()
            viewer.sync()

            now = time.monotonic()
            if now - last_print > 0.25 and (np.any(state["vel"]) or i % 400 == 0):
                last_print = now
                print(f"\r  [{state['mode']}] target {state['target'].round(3)}      ",
                      end="", flush=True)

            slack = CONTROL_DT - (time.perf_counter() - tic)
            if slack > 0:
                time.sleep(slack)
    print()


if __name__ == "__main__":
    main()
