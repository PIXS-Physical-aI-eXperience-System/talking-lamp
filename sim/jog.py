"""Steer the target with held keys - run in its OWN terminal.

Two-terminal setup:

    terminal 1:   make drive      # shows the lamp, follows sim/.target
    terminal 2:   make jog        # <- click here, then hold keys

jog writes sim/.target ~60x/s; drive polls it and moves the lamp.

    W / S   forward / back   (+x / -x)        R / F   up / down (+z / -z)
    A / D   left / right     (+y / -y)        arrows + PageUp/Down also work
    hold a key -> moves at constant speed;  release -> stops after ~0.15 s
    + / -   faster / slower        1 / 2 / 3   mode track / light / reach
    0   home        SPACE   stop        Q  quit

Needs a real terminal (raw mode). `make jog ARGS=--debug` echoes every key byte.
"""

from __future__ import annotations

from _bootstrap import strip_ros_paths

strip_ros_paths()

import sys
import time
from pathlib import Path

import numpy as np

import _keys

TARGET_FILE = Path(__file__).resolve().parent / ".target"
HZ = 60.0


def main() -> None:
    debug = "--debug" in sys.argv
    if not _keys.available():
        sys.exit("jog needs a real terminal (got a pipe/redirect). Run it directly, not piped.")

    cur = np.array([0.42, 0.0, 0.34])
    try:
        cur = np.array([float(v) for v in TARGET_FILE.read_text().split()[:3]])
    except (OSError, ValueError):
        pass

    state = {
        "target": cur, "mode": "track", "running": True,
        "vel": np.zeros(3), "vel_until": 0.0, "speed": 0.15, "debug": debug,
    }

    import threading
    threading.Thread(target=_keys.run, args=(state,), daemon=True).start()

    print("jog ready - hold W/A/S/D  R/F  (or arrows).  +/- speed  1/2/3 mode  0 home  Q quit")
    dt = 1.0 / HZ
    last_line = None
    while state["running"]:
        if time.monotonic() < state["vel_until"]:
            state["target"] = state["target"] + state["vel"] * dt
        t = state["target"]
        line = f"{t[0]:.4f} {t[1]:.4f} {t[2]:.4f} {state['mode']}"
        if line != last_line:
            TARGET_FILE.write_text(line + "\n")
            last_line = line
            print(f"\r  [{state['mode']}] {t.round(3)}   {state.get('speed', 0.15):.2f} m/s     ",
                  end="", flush=True)
        time.sleep(dt)
    print("\njog stopped")


if __name__ == "__main__":
    main()
