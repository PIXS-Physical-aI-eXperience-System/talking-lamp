"""MuJoCo backends for MotionRuntime.

``MujocoDynamicsBackend`` - drive the position servos and step physics, so you
see real servo lag / settling / gravity sag. Uses the same ``sim/world.xml``.

``MujocoKinematicsBackend`` - just set qpos to the command; exact playback for
checking geometry, IK targets, self-collision.

Both expose the live ``model``/``data`` so a viewer or renderer can draw them.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .config import WORLD_XML


class _Base:
    def __init__(self, world_xml: str | None = None, q0: np.ndarray | None = None) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(world_xml or WORLD_XML))
        self.data = mujoco.MjData(self.model)
        names = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")
        self._qadr = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in names]
        )
        self._act = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names]
        )
        if q0 is not None:
            self.data.qpos[self._qadr] = q0
            self.data.ctrl[self._act] = q0
        mujoco.mj_forward(self.model, self.data)

    def measured(self) -> np.ndarray:
        return self.data.qpos[self._qadr].copy()


class MujocoKinematicsBackend(_Base):
    def send(self, q_cmd: np.ndarray) -> None:
        self.data.qpos[self._qadr] = np.asarray(q_cmd, float)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)


class MujocoDynamicsBackend(_Base):
    def __init__(self, world_xml: str | None = None, q0: np.ndarray | None = None,
                 control_dt: float = 0.01) -> None:
        super().__init__(world_xml, q0)
        self.substeps = max(1, int(round(control_dt / self.model.opt.timestep)))

    def send(self, q_cmd: np.ndarray) -> None:
        self.data.ctrl[self._act] = np.asarray(q_cmd, float)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
