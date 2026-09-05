"""Headless sanity check for the Talking Lamp scene.

    .venv/bin/python sim/check.py

Loads the model, prints the joint map, sweeps each joint through its range
(pure kinematics) and reports head-tip travel + the reachable box, then holds
the home pose with the position servos to check it is stable. Writes a few
frames to sim/out/ if a GL backend is available. No display needed.
"""

from __future__ import annotations

from _bootstrap import strip_ros_paths
strip_ros_paths()  # drop ROS PYTHONPATH leaks before importing numpy/mujoco

import os
import struct
import zlib
from pathlib import Path

import mujoco
import numpy as np

import lamp

OUT = Path(__file__).resolve().parent / "out"


def main() -> None:
    model, data = lamp.load()
    print(f"model: nq={model.nq} nv={model.nv} nu={model.nu}  timestep={model.opt.timestep}")

    ranges = lamp.joint_ranges(model)
    qadr = lamp.qpos_order(model)
    print("\njoint           servo   range (deg)")
    for name, (lo, hi) in zip(lamp.JOINT_NAMES, ranges):
        print(f"  {name:<13} id {lamp.SERVO_ID[name]}   [{np.degrees(lo):7.1f}, {np.degrees(hi):7.1f}]")

    home = lamp.head_pose(model, data)
    print(f"\nhome head tip (m): {home.round(4)}")

    print("\nper-joint sweep (kinematic, others at home):")
    pts = [home[None]]
    for i, name in enumerate(lamp.JOINT_NAMES):
        seg = []
        for val in np.linspace(*ranges[i], 60):
            data.qpos[:] = 0.0
            data.qpos[qadr[i]] = val
            mujoco.mj_forward(model, data)
            seg.append(lamp.head_pose(model, data))
        seg = np.array(seg)
        pts.append(seg)
        travel = float(np.linalg.norm(seg - home, axis=1).max())
        print(f"  {name:<13} max head-tip travel = {travel * 1000:5.0f} mm")

    allpts = np.concatenate(pts)
    print("\napprox reachable box from single-joint sweeps (m):")
    for ax, lab in enumerate("xyz"):
        print(f"  {lab}  {allpts[:, ax].min():+.3f} .. {allpts[:, ax].max():+.3f}")

    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(3000):
        mujoco.mj_step(model, data)  # ctrl stays 0 -> hold home
    drift = float(np.linalg.norm(lamp.head_pose(model, data) - home))
    print(f"\nhome-pose hold: head drift after 6 s = {drift * 1000:.1f} mm "
          f"({'OK' if drift < 0.01 else 'CHECK'})")

    render_poses(model, data)
    print("\nOK")


def render_poses(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "closeup")
    try:
        renderer = mujoco.Renderer(model, 720, 960)
    except Exception as exc:  # noqa: BLE001 - missing GL backend is expected on bare servers
        print(f"\n(render skipped: {type(exc).__name__}: {exc})")
        return
    OUT.mkdir(exist_ok=True)
    ranges = lamp.joint_ranges(model)
    qadr = lamp.qpos_order(model)
    poses = {
        "home": np.zeros(5),
        "reach_forward": np.array([0.0, ranges[1, 1] * 0.6, ranges[2, 0] * 0.5, 0.0, 0.3]),
        "look_left": np.array([1.0, 0.2, -0.6, 0.5, 0.2]),
    }
    for name, q in poses.items():
        data.qpos[:] = 0.0
        for i in range(5):
            data.qpos[qadr[i]] = np.clip(q[i], *ranges[i])
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        (OUT / f"{name}.png").write_bytes(_png(renderer.render()))
        print(f"  wrote sim/out/{name}.png")


def _png(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
