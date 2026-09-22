"""Test bootstrap.

Puts ``src/`` on the path, and drops any ``/opt/ros`` entries that a sourced ROS
environment leaks in through ``PYTHONPATH`` (they shadow venv packages and pull
in broken pytest plugins). See the repo Makefile / README for the wrapper that
also clears this before pytest's plugin autoload runs.
"""

import sys
from pathlib import Path

sys.path[:] = [p for p in sys.path if "/opt/ros" not in p]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
workspace_src = Path(__file__).resolve().parents[1] / "jetson_ws/src"
for package in ("lamp_device_bridge", "lamp_motion_bridge", "lamp_interaction"):
    sys.path.insert(0, str(workspace_src / package))
