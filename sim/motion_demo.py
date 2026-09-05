"""End-to-end demo of the motion stack driving the MuJoCo lamp.

Scripted timeline (idle -> wake -> follow a moving face -> nod -> place the work
light -> barge-in) run through `MotionRuntime` at 100 Hz on the dynamics
backend. Writes:

    sim/out/motion_<phase>.png   snapshots at each phase
    sim/out/motion_timeline.png  joint angles + per-layer authority over time

    .venv/bin/python sim/motion_demo.py
"""

from __future__ import annotations

from _bootstrap import strip_ros_paths
strip_ros_paths()  # drop ROS PYTHONPATH leaks before importing numpy/mujoco

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motion import MotionRuntime, JOINT_NAMES  # noqa: E402
from motion.blender import BlendContext  # noqa: E402
from motion.sim_backend import MujocoDynamicsBackend  # noqa: E402
from motion.config import REST_POSE  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
DT = 0.01


def moving_face(t: float) -> np.ndarray:
    """A person leaning / shifting in their chair in front of the lamp."""
    return np.array([
        0.42 + 0.03 * np.sin(0.5 * t),
        0.18 * np.sin(0.35 * t),
        0.34 + 0.02 * np.sin(0.8 * t),
    ])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    be = MujocoDynamicsBackend(q0=REST_POSE, control_dt=DT)
    rt = MotionRuntime(backend=be, dt=DT)
    renderer = mujoco.Renderer(be.model, 720, 960)
    cam = mujoco.mj_name2id(be.model, mujoco.mjtObj.mjOBJ_CAMERA, "closeup")

    phases: list[tuple[str, float]] = [
        ("idle", 2.5),
        ("wake_up", 3.5),
        ("follow_face", 6.0),
        ("nod", 3.5),
        ("task_light", 4.5),
        ("barge_in", 3.0),
    ]
    track_during = {"follow_face", "nod", "task_light", "barge_in"}

    log_t, log_q, log_auth = [], [], []
    layer_names = [ly.name for ly in rt.blender.layers]
    t = 0.0

    for name, dur in phases:
        if name == "wake_up":
            rt.play_primitive("wake_up")
        elif name == "follow_face":
            rt.primitive.stop()  # let the wake clip release
        elif name == "nod":
            rt.play_primitive("nod")
        elif name == "task_light":
            rt.primitive.stop()
            rt.place_task_light(np.array([0.34, -0.06, 0.0]))
        elif name == "barge_in":
            rt.barge_in()

        steps = int(dur / DT)
        for _ in range(steps):
            if name in track_during and int(round(t / DT)) % 3 == 0:
                rt.track.observe_point(moving_face(t))
            s = rt.step()
            t += DT

            # recompute per-layer authority for logging (cheap)
            ctx = BlendContext(q_current=rt.traj.pos.copy(), t=t, dt=DT)
            trace = rt.blender.compute(ctx)
            auth = np.array([
                float(np.linalg.norm(trace.authority.get(n, np.zeros(5)))) for n in layer_names
            ])
            log_t.append(t)
            log_q.append(s.q_cmd.copy())
            log_auth.append(auth)

        renderer.update_scene(be.data, camera=cam)
        (OUT / f"motion_{name}.png").write_bytes(_png(renderer.render()))
        print(f"  {name:12} t={t:5.1f}s  head={rt.kin.head_position(be.measured()).round(3)}")

    _timeline(np.array(log_t), np.array(log_q), np.array(log_auth), layer_names)
    print("wrote sim/out/motion_timeline.png")


def _timeline(t, q, auth, layer_names) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for i, n in enumerate(JOINT_NAMES):
        ax1.plot(t, np.degrees(q[:, i]), label=n, lw=1.4)
    ax1.set_ylabel("joint angle (deg)")
    ax1.legend(ncol=5, fontsize=8, loc="upper center")
    ax1.grid(alpha=0.3)
    ax1.set_title("Talking Lamp motion stack - scripted demo")

    for i, n in enumerate(layer_names):
        ax2.plot(t, auth[:, i], label=f"L{i} {n}", lw=1.4)
    ax2.set_ylabel("layer authority\n|Δq| (rad)")
    ax2.set_xlabel("time (s)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "motion_timeline.png", dpi=110)


def _png(rgb: np.ndarray) -> bytes:
    import struct
    import zlib

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, payload):
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(
            ">I", zlib.crc32(tag + payload) & 0xFFFFFFFF
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


if __name__ == "__main__":
    main()
