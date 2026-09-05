"""Rebuild the LeLamp arm as a correct 5-DOF *serial* chain.

Why this exists
---------------
The vendored MuJoCo/URDF export (`LeLamp/simulation/robot.xml`, `robot.urdf`)
has a broken kinematic tree: the five hinges are split into two branches off the
root instead of one serial chain, so the arm does not articulate (see
`sim/README.md`). The rest pose (all joints at 0) is geometrically correct
though - every body's world transform matches the CAD.

This script loads the broken model, reads each joint's world anchor + axis and
each structural mesh's world transform *at the home pose*, then emits a fresh
MJCF where:

  * the 5 hinges form a serial chain base -> yaw -> pitch -> elbow -> roll ->
    wrist-pitch -> head, each placed at its true home anchor with its true axis
    and the vendored travel limits;
  * every link frame is world-aligned at home, so mesh geoms keep the exact
    transform they have now (pos = world_pos - link_anchor, quat = world_quat)
    and the home render is identical to upstream;
  * each link gets a simple box inertial proxy (mass from the matching vendored
    body) - good enough for trajectory / blender work, refine from CAD later.

Servo meshes and PCB clutter are dropped; only the 5 structural shells + base.

Regenerate:
    .venv/bin/python sim/build_arm.py

Replace this with a clean onshape-to-robot re-export once OnShape API access is
set up (LeLamp CAD doc 16c9706360b5ad34f9c8db49).
"""

from __future__ import annotations

from _bootstrap import strip_ros_paths
strip_ros_paths()  # drop ROS PYTHONPATH leaks before importing numpy/mujoco

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_SCENE = ROOT / "LeLamp" / "simulation" / "scene.xml"
DST = ROOT / "sim" / "lelamp_arm.xml"
MESHDIR = "../LeLamp/simulation/assets"

# serial chain: (joint name in vendored file, readable axis, servo id)
CHAIN = [
    ("1", "base_yaw", 1),
    ("2", "base_pitch", 2),
    ("3", "elbow_pitch", 3),
    ("4", "wrist_roll", 4),
    ("5", "wrist_pitch", 5),
]

# structural mesh -> index of the chain link it rigidly belongs to
# (0 = world-fixed base, 1 = yaw turntable, ... 5 = head)
MESH_LINK = {
    "lamp_base": 0,
    "lamp_base_cover": 0,
    "lamparm__base_elbow": 2,
    "lamparm__elbow_wrist": 3,
    "lamparm__wrist_head": 4,   # the copy attached to lamparm__wrist_head_2
    "diffuser": 5,
    "lamphead": 5,
}
# vendored body whose mass seeds each link's inertial proxy
LINK_MASS_FROM_BODY = {
    0: "scs215_v5",
    1: "lamparm__wrist_head",
    2: "lamparm__base_elbow",
    3: "lamparm__elbow_wrist",
    4: "lamparm__wrist_head_2",
    5: "diffuser",
}


def fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in np.asarray(v).ravel())


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SRC_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # --- joint anchors / axes / ranges at home ---
    anchor, axis, jrange = {}, {}, {}
    for jname, _, _ in CHAIN:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        anchor[jname] = data.xanchor[jid].copy()
        axis[jname] = data.xaxis[jid].copy()
        jrange[jname] = model.jnt_range[jid].copy()

    # link i (1..5) sits at anchor of CHAIN[i-1]; link 0 sits at origin
    link_origin = [np.zeros(3)] + [anchor[CHAIN[i][0]] for i in range(5)]

    # --- structural mesh world transforms at home ---
    # Compose the *raw* geom local pose (from the vendored XML) with its parent
    # body's compiled world pose. Using data.geom_xpos here would double-apply
    # MuJoCo's mesh recentring once our file is recompiled.
    import xml.etree.ElementTree as ET

    def parse_vec(s, default):
        return np.array([float(x) for x in s.split()]) if s else np.array(default, float)

    tree = ET.parse(ROOT / "LeLamp" / "simulation" / "robot.xml")
    # skip the misplaced duplicate wrist_head geom (the one not on wrist_head_2)
    keep_body = {"lamparm__wrist_head": "lamparm__wrist_head_2"}

    mesh_world = {}  # mesh name -> (pos, quat) of the geom frame in world at home
    for body in tree.iter("body"):
        bname = body.get("name")
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        bpos, bquat = data.xpos[bid], data.xquat[bid]
        for g in body.findall("geom"):
            mname = g.get("mesh")
            if mname not in MESH_LINK or mname in mesh_world or g.get("class") != "visual":
                continue
            if mname in keep_body and bname != keep_body[mname]:
                continue
            lpos = parse_vec(g.get("pos"), [0, 0, 0])
            lquat = parse_vec(g.get("quat"), [1, 0, 0, 0])
            wpos = np.zeros(3)
            mujoco.mju_rotVecQuat(wpos, lpos, bquat)
            wpos += bpos
            wquat = np.zeros(4)
            mujoco.mju_mulQuat(wquat, bquat, lquat)
            mesh_world[mname] = (wpos, wquat)

    link_meshes: dict[int, list[str]] = {i: [] for i in range(6)}
    for mname, li in MESH_LINK.items():
        if mname in mesh_world:
            link_meshes[li].append(mname)

    # head reference point = world AABB centre of the lamphead shell at home
    head_center = _mesh_world_aabb_center(model, data, "lamphead")

    link_mass = {i: float(model.body_mass[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LINK_MASS_FROM_BODY[i])
    ]) for i in range(6)}

    # --- emit MJCF ---
    used_meshes = sorted(mesh_world)
    out = []
    w = out.append
    w('<?xml version="1.0" ?>')
    w("<!-- GENERATED by sim/build_arm.py - serial-chain rebuild of the LeLamp arm.")
    w("     Kinematics from LeLamp/simulation/robot.xml home pose; do not edit by hand. -->")
    w('<mujoco model="lelamp_arm">')
    w(f'  <compiler angle="radian" meshdir="{MESHDIR}" autolimits="true"/>')
    w('  <default>')
    w('    <default class="sts3215">')
    w('      <!-- damping/frictionloss/armature from the vendored STS3215 class;')
    w('           kp/kv bumped for a usable position-hold at 12 V (retune vs real servo) -->')
    w('      <joint damping="0.60" frictionloss="0.052" armature="0.028"/>')
    w('      <position kp="120" kv="8" forcerange="-6.0 6.0"/>')
    w('    </default>')
    w('    <default class="visual">')
    w('      <geom type="mesh" contype="0" conaffinity="0" group="2" mass="0"/>')
    w('    </default>')
    w('    <default class="proxy">')
    w('      <!-- inertia only: consecutive links share endpoints so their boxes')
    w('           overlap; enabling contact would weld the arm. Add real collision')
    w('           geometry from CAD later. -->')
    w('      <geom type="box" group="3" contype="0" conaffinity="0" rgba="0.3 0.5 0.9 0.12"/>')
    w('    </default>')
    w('  </default>')

    w('  <asset>')
    for mname in used_meshes:
        w(f'    <mesh name="{mname}" file="{mname}.stl"/>')
    w('  </asset>')

    w('  <worldbody>')
    w('    <body name="base" pos="0 0 0" childclass="sts3215">')
    _emit_geoms(w, 2, link_meshes[0], mesh_world, link_origin[0])
    _emit_proxy(w, 2, link_origin[0], link_origin[1], link_mass[0])

    indent = 3
    for i, (jname, axis_name, sid) in enumerate(CHAIN, start=1):
        rel = link_origin[i] - link_origin[i - 1]
        pad = "  " * indent
        w(f'{pad}<body name="{axis_name}" pos="{fmt(rel)}">')
        lo, hi = jrange[jname]
        w(f'{pad}  <joint name="{axis_name}" axis="{fmt(axis[jname])}" '
          f'range="{lo:.6g} {hi:.6g}" class="sts3215"/>')
        _emit_geoms(w, indent + 1, link_meshes[i], mesh_world, link_origin[i])
        child = link_origin[i + 1] if i < 5 else head_center
        _emit_proxy(w, indent + 1, link_origin[i], child, link_mass[i])
        if i == 5:
            tip = head_center - link_origin[5]
            w(f'{pad}  <site name="head" pos="{fmt(tip)}" size="0.01" rgba="1 0.9 0.3 1"/>')
        indent += 1

    for _ in range(6):
        indent -= 1
        w("  " * (indent + 1) + "</body>")
    w('  </worldbody>')

    w('  <actuator>')
    for jname, axis_name, sid in CHAIN:
        w(f'    <position class="sts3215" name="{axis_name}" joint="{axis_name}" inheritrange="1"/>')
    w('  </actuator>')
    w('</mujoco>')

    DST.write_text("\n".join(out) + "\n")
    print(f"wrote {DST.relative_to(ROOT)}")
    _verify()


def _mesh_world_aabb_center(model, data, mesh_name):
    mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts = model.mesh_vert[va:va + vn].reshape(-1, 3)
    for g in range(model.ngeom):
        if model.geom_dataid[g] == mid and model.geom_group[g] == 2:
            wv = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            return (wv.min(0) + wv.max(0)) / 2
    raise RuntimeError(f"no visual geom for mesh {mesh_name}")


def _emit_geoms(w, indent, meshes, mesh_world, link_origin):
    pad = "  " * indent
    for mname in meshes:
        wpos, wq = mesh_world[mname]
        w(f'{pad}<geom class="visual" mesh="{mname}" '
          f'pos="{fmt(wpos - link_origin)}" quat="{fmt(wq)}"/>')


def _emit_proxy(w, indent, a, b, mass):
    pad = "  " * indent
    seg = np.asarray(b) - np.asarray(a)
    length = max(float(np.linalg.norm(seg)), 0.03)
    center = (np.asarray(a) + np.asarray(b)) / 2 - np.asarray(a)
    half = f"0.02 0.02 {length / 2:.6g}"
    # orient local z along the segment
    z = seg / np.linalg.norm(seg) if np.linalg.norm(seg) > 1e-9 else np.array([0, 0, 1.0])
    x = np.cross([0, 1.0, 0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.array([x, y, z]).T.ravel())
    w(f'{pad}<geom class="proxy" size="{half}" pos="{fmt(center)}" quat="{fmt(q)}" mass="{mass:.6g}"/>')


def _verify():
    from lamp import load, head_pose, JOINT_NAMES

    m, d = load()
    base = head_pose(m, d).copy()
    print("  articulation check (sweep each joint over its full range, head-tip travel):")
    ok = True
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = m.jnt_qposadr[jid]
        span = 0.0
        for val in np.linspace(*m.jnt_range[jid], 30):
            d.qpos[:] = 0.0
            d.qpos[qadr] = val
            mujoco.mj_forward(m, d)
            span = max(span, float(np.linalg.norm(head_pose(m, d) - base)))
        flag = "" if span > 0.03 else "  <-- barely moves, check chain"
        if flag:
            ok = False
        print(f"    {name:<13} {span * 1000:6.0f} mm{flag}")
    print("  OK - serial chain articulates" if ok else "  WARNING: chain still wrong")


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.chdir(Path(__file__).resolve().parent)
    main()
