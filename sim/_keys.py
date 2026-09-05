"""Raw-terminal held-key reader used by drive.py / jog.py.

Runs in a background thread, reads the controlling terminal in cbreak mode, and
keeps a small `state` dict updated:

    state["vel"]        (3,) m/s velocity of the target while a key is held
    state["vel_until"]  monotonic time the velocity expires (refreshed by repeats)
    state["target"]     jumped on 0 / home
    state["mode"]       set on 1 / 2 / 3
    state["running"]    cleared on q / Ctrl-C

Keys: W/S +x/-x, A/D +y/-y, R/F +z/-z (arrows + PageUp/Down too), +/- speed,
1/2/3 mode, 0 home, SPACE stop, Q quit.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty

import numpy as np

HOME = np.array([0.42, 0.0, 0.34])

_KEYVEL = {
    "w": (1, 0, 0), "s": (-1, 0, 0),
    "a": (0, 1, 0), "d": (0, -1, 0),
    "r": (0, 0, 1), "f": (0, 0, -1),
    "\x1b[A": (1, 0, 0), "\x1b[B": (-1, 0, 0),
    "\x1b[D": (0, 1, 0), "\x1b[C": (0, -1, 0),
    "\x1b[5~": (0, 0, 1), "\x1b[6~": (0, 0, -1),
}
_MODEKEY = {"1": "track", "2": "light", "3": "reach"}


def available() -> bool:
    return sys.stdin.isatty()


def _read_tokens(timeout: float) -> list[str]:
    """Block up to `timeout` for input, then drain and split the whole buffer
    into key tokens (a bare char, or a full CSI escape sequence like '\\x1b[A')."""
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return []
    try:
        data = os.read(sys.stdin.fileno(), 4096).decode("latin-1", "ignore")
    except OSError:
        return []
    toks: list[str] = []
    i = 0
    while i < len(data):
        if data[i] == "\x1b" and data[i + 1: i + 2] == "[":
            j = i + 2
            while j < len(data) and not (data[j].isalpha() or data[j] == "~"):
                j += 1
            toks.append(data[i: j + 1])
            i = j + 1
        else:
            toks.append(data[i])
            i += 1
    return toks


def run(state: dict, speed0: float = 0.15, tick: float = 0.02) -> None:
    """Blocking key loop; call in a daemon thread. Restores the terminal on exit."""
    speed = speed0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while state.get("running", True):
            for tok in _read_tokens(tick):
                if state.get("debug"):
                    sys.stderr.write(f"[key {tok!r}]\n")
                    sys.stderr.flush()
                low = tok.lower()
                if tok in ("q", "\x03"):
                    state["running"] = False
                    return
                if tok == " ":
                    state["vel"] = np.zeros(3)
                elif tok == "0":
                    state["target"] = HOME.copy()
                    state["vel"] = np.zeros(3)
                elif tok in ("+", "="):
                    speed = min(speed * 1.3, 0.6)
                elif tok in ("-", "_"):
                    speed = max(speed / 1.3, 0.03)
                elif tok in _MODEKEY:
                    state["mode"] = _MODEKEY[tok]
                elif tok in _KEYVEL or low in _KEYVEL:
                    key = tok if tok in _KEYVEL else low
                    fast = len(tok) == 1 and tok.isupper()
                    state["vel"] = np.array(_KEYVEL[key], float) * speed * (2.5 if fast else 1.0)
                    state["vel_until"] = time.monotonic() + 0.14
            state["speed"] = speed
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
