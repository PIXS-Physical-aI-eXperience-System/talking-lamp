"""Call ``strip_ros_paths()`` first in any sim entry script.

A sourced ROS environment prepends ``/opt/ros/<distro>/lib/pythonX/site-packages``
to ``PYTHONPATH``; those shadow the venv's numpy/mujoco/yaml and break things.
"""

import sys


def strip_ros_paths() -> None:
    sys.path[:] = [p for p in sys.path if "/opt/ros" not in p]
