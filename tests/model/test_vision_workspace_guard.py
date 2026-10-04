"""D's workspace and E's task-light guard must agree.

D drops desk points outside ``Workspace``; E refuses a task-light pose the
shoulder cannot hold. If D's range is wider, S1 picks a target, sends it, and
the arm refuses: the user sees nothing happen. If it is narrower, the lamp
ignores objects it could have lit. This runs E's actual guard over D's edges.
"""

import math

import numpy as np
import pytest

from vision.pipeline import Workspace

pytest.importorskip("mujoco")
layers = pytest.importorskip("motion.layers")
if not hasattr(layers, "PlaceResult"):
    pytest.skip("needs E's task-light guard (fix/motion-task-light-guard)", allow_module_level=True)

from motion.kinematics import ArmKinematics  # noqa: E402

WS = Workspace()


@pytest.fixture(scope="module")
def light():
    return layers.TaskLightLayer(ArmKinematics())


def _at(r, bearing_deg):
    a = math.radians(bearing_deg)
    return np.array([r * math.cos(a), r * math.sin(a), 0.0])


@pytest.mark.parametrize("bearing", [WS.min_bearing_deg, -45.0, 0.0, 30.0, WS.max_bearing_deg])
@pytest.mark.parametrize("r", [WS.min_radius_m, WS.max_radius_m])
def test_every_edge_of_the_workspace_is_placed(light, r, bearing):
    placed = light.place(_at(r, bearing))
    assert placed.ok, placed.message


def test_the_workspace_is_not_far_from_the_guard_edge(light):
    """Not needlessly narrow: a few centimetres further out is already refused."""
    placed = light.place(_at(WS.max_radius_m + 0.03, 0.0))
    assert not placed.ok and placed.code == "over_torque"
