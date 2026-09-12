"""Forward kinematics and Jacobians for the 5-DOF lamp arm, backed by MuJoCo.

We deliberately do not hand-derive analytic FK: the arm geometry comes straight
from CAD (``sim/lelamp_arm.xml``) and MuJoCo already gives exact, fast site
kinematics. This module just narrows MuJoCo down to "the 5 actuated joints" so
the IK / trajectory code never touches raw ``mjData`` indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import HEAD_SITE, JOINT_NAMES, NJ, WORLD_XML


@dataclass(frozen=True)
class HeadPose:
    pos: np.ndarray          # (3,) world position of the head site
    rot: np.ndarray          # (3, 3) world orientation of the head site
    forward: np.ndarray      # (3,) unit vector the head "looks" along (site +x)


class ArmKinematics:
    """Stateless-ish FK/Jacobian helper. Owns a private mjModel/mjData."""

    def __init__(self, world_xml: str | None = None) -> None:
        path = str(world_xml or WORLD_XML)
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)

        self._jnt_qpos = np.array(
            [self.model.jnt_qposadr[self._jid(n)] for n in JOINT_NAMES]
        )
        self._jnt_dof = np.array(
            [self.model.jnt_dofadr[self._jid(n)] for n in JOINT_NAMES]
        )
        self._site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, HEAD_SITE)
        if self._site < 0:
            raise ValueError(f"site {HEAD_SITE!r} not found in {path}")

        lo = np.array([self.model.jnt_range[self._jid(n)][0] for n in JOINT_NAMES])
        hi = np.array([self.model.jnt_range[self._jid(n)][1] for n in JOINT_NAMES])
        self.joint_limits = np.stack([lo, hi], axis=1)  # (5, 2)

    def _jid(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} not found")
        return jid

    # -- forward kinematics -------------------------------------------------
    def _apply(self, q: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[self._jnt_qpos] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # needed for jacobians

    def head_pose(self, q: np.ndarray) -> HeadPose:
        self._apply(np.asarray(q, float))
        pos = self.data.site_xpos[self._site].copy()
        rot = self.data.site_xmat[self._site].reshape(3, 3).copy()
        return HeadPose(pos=pos, rot=rot, forward=rot[:, 0].copy())

    def head_position(self, q: np.ndarray) -> np.ndarray:
        return self.head_pose(q).pos

    # -- jacobian ---------------------------------------------------------
    def jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (Jp, Jr): 3x5 positional and 3x5 rotational site Jacobians."""
        self._apply(np.asarray(q, float))
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._site)
        return jacp[:, self._jnt_dof].copy(), jacr[:, self._jnt_dof].copy()

    # -- convenience ----------------------------------------------------
    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    @property
    def n_joints(self) -> int:
        return NJ
