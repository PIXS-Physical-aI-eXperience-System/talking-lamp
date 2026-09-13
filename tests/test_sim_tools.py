"""Exercise the simulator controls without a viewer or a graphics context."""

import importlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motion.runtime import MotionRuntime


def test_demo_logging_updates_layers_once_per_command(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "sim"))
    demo = importlib.import_module("motion_demo")
    counts = {"step": 0, "compute": 0}

    def make_runtime(**kwargs):
        runtime = MotionRuntime(**kwargs)
        step, compute = runtime.step, runtime.blender.compute

        def counted_step():
            counts["step"] += 1
            return step()

        def counted_compute(ctx):
            counts["compute"] += 1
            return compute(ctx)

        runtime.step = counted_step
        runtime.blender.compute = counted_compute
        return runtime

    monkeypatch.setattr(demo, "MotionRuntime", make_runtime)
    monkeypatch.setattr(demo, "OUT", tmp_path)
    monkeypatch.setattr(demo.mujoco, "Renderer", lambda *args: SimpleNamespace(
        update_scene=lambda *args, **kwargs: None,
        render=lambda: np.zeros((2, 2, 3), dtype=np.uint8),
    ))
    monkeypatch.setattr(demo, "_timeline", lambda *args: None)
    demo.main()
    assert counts["step"] > 0
    assert counts["compute"] == counts["step"]


@pytest.mark.parametrize("mode", ["light", "reach"])
def test_drive_releases_task_pose_when_switching_to_tracking(monkeypatch, mode):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "sim"))
    drive = importlib.import_module("drive")
    snapshots = []

    def make_runtime(**kwargs):
        runtime = MotionRuntime(**kwargs)
        step = runtime.step

        def recorded_step():
            state = step()
            if round(state.t * 100) in (600, 800, 1000, 1200):
                snapshots.append((
                    runtime.task_light.busy, state, runtime.track.env.level,
                    runtime.task_light.q_hold.copy(),
                ))
            return state

        runtime.step = recorded_step
        return runtime

    point = ".24 0 0" if mode == "light" else ".30 .10 .22"
    monkeypatch.setattr(drive, "MotionRuntime", make_runtime)
    monkeypatch.setattr(drive.sys, "argv", [
        "drive", "--headless", "--interactive", "--no-file", "--mode", mode,
        "--point", *point.split(),
    ])
    # Re-entering the same task point after tracking must place it again.
    monkeypatch.setattr(drive.sys, "stdin", io.StringIO(
        f".42 -.18 .34 track\n{point} {mode}\n.42 -.18 .34 track\n"
    ))
    drive.main()
    assert len(snapshots) == 4
    assert [s[0] for s in snapshots] == [True, False, True, False]
    for _, state, track_gain, held in snapshots[1::2]:
        assert track_gain == pytest.approx(1.0)
        assert np.linalg.norm(state.q_blend - held) > .05
        np.testing.assert_array_equal(state.trace.per_layer["task_light"], np.zeros(5))
    assert np.linalg.norm(snapshots[1][1].q_cmd - snapshots[0][1].q_cmd) > .05
