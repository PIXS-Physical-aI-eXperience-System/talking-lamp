"""Interactive MuJoCo viewer for the Talking Lamp scene.

Needs a display (run on your desktop, not a headless server):

    .venv/bin/python sim/view.py

Drag the sliders in the "Control" panel to move the 5 servos, or press the
keyboard shortcuts shown in the viewer help (F1).
"""

from __future__ import annotations

from _bootstrap import strip_ros_paths
strip_ros_paths()  # drop ROS PYTHONPATH leaks before importing numpy/mujoco

from pathlib import Path

import mujoco
import mujoco.viewer

import lamp


def main() -> None:
    model, data = lamp.load()
    print("joints:", ", ".join(f"{n} (servo {lamp.SERVO_ID[n]})" for n in lamp.JOINT_NAMES))
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    import os

    os.chdir(Path(__file__).resolve().parent)
    main()
