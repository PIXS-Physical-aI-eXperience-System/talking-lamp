"""The camera rides on the lamp head: camera pose = head pose x mount.

The camera board is fixed to the top rim of the head, so it moves with all
five joints. Its pose in ``lamp_base`` is therefore not a constant but

    T_base_cam(q) = T_base_head(q) @ T_head_cam

where ``T_base_head(q)`` is E's forward kinematics of the ``head`` site
(``motion.kinematics.ArmKinematics.head_pose``) and ``T_head_cam`` is a fixed
mount offset measured once with the ChArUco board.

The head's measured joint angles do not leave the Pi (only base_yaw is
published, on ``/lamp/orientation_status``). So D only evaluates the pose at a
posture whose joint angles it already knows: the rest pose, which E's
``/lamp/return_center`` action returns the arm to. ``REST_POSE`` is a constant
in ``motion.config``; base_yaw may differ and is taken from the status topic.

``HeadCameraMount`` is pure numpy. Anything that needs FK imports E's
``motion.kinematics`` (MuJoCo) lazily, so the rest of ``vision`` stays
importable on a machine without it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .camera import Intrinsics
from .geometry import Pose


@dataclass(frozen=True)
class HeadCameraMount:
    """Camera optical frame expressed in the head-site frame.

    ``R`` columns are the camera's +x (right), +y (down), +z (forward) axes in
    head-site coordinates; ``t`` is the optical centre in metres. E's IK aims
    the head-site **+x** axis at targets, so for a lens parallel to the head's
    face, ``R[:, 2]`` should be close to (1, 0, 0).
    """

    R: np.ndarray
    t: np.ndarray
    source: str = "charuco"
    rms_px: float | None = None

    @classmethod
    def from_poses(cls, cam_pose: Pose, head_R: np.ndarray, head_t: np.ndarray,
                   source: str = "charuco") -> "HeadCameraMount":
        """Mount from one camera pose measured while the head pose was known."""
        R = head_R.T @ cam_pose.R
        t = head_R.T @ (cam_pose.t - head_t)
        return cls(R, t, source, cam_pose.rms_px)

    def camera_pose(self, head_R: np.ndarray, head_t: np.ndarray, source: str = "fk") -> Pose:
        return Pose(head_R @ self.R, head_t + head_R @ self.t, source, self.rms_px)

    @property
    def axis_offset_deg(self) -> float:
        """Angle between the lens axis and the head's aiming axis (+x)."""
        return math.degrees(math.acos(float(np.clip(self.R[0, 2], -1.0, 1.0))))

    def gaze_uv(self, cam: Intrinsics, distance_m: float) -> np.ndarray:
        """Pixel where the head's aim point appears, ``distance_m`` along its +x.

        When E has converged on a tracking target, the target sits here in the
        image, not at the image centre: the lens is a few centimetres off the
        head's axis, so the two lines of sight differ by parallax.
        """
        p_cam = self.R.T @ (np.array([distance_m, 0.0, 0.0]) - self.t)
        return cam.project(p_cam)

    def to_dict(self) -> dict:
        return {"R": self.R.tolist(), "t": self.t.tolist(), "source": self.source,
                "rms_px": self.rms_px, "frame": "head site -> camera optical"}

    @classmethod
    def from_dict(cls, d: dict) -> "HeadCameraMount":
        return cls(np.array(d["R"], float), np.array(d["t"], float),
                   d.get("source", "charuco"), d.get("rms_px"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HeadCameraMount":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# -- FK glue (needs E's motion package and MuJoCo) ---------------------------

def joints_in_order(names, values, order) -> np.ndarray:
    """``values`` keyed by ``names``, rearranged into ``order``.

    A JointState carries its own names; matching on them rather than on
    position keeps a reordering on the Pi from silently swapping two joints.
    """
    by_name = dict(zip(names, values))
    if len(by_name) != len(names) or len(names) != len(values):
        raise ValueError(f"joint names and values do not pair up: {list(names)}")
    missing = [n for n in order if n not in by_name]
    if missing:
        raise ValueError(f"joint state is missing {missing}")
    return np.array([float(by_name[n]) for n in order])


class HeadKinematics:
    """Thin wrapper over E's FK so vision never touches MuJoCo directly."""

    def __init__(self, world_xml: str | Path | None = None):
        from motion.config import JOINT_NAMES, REST_POSE
        from motion.kinematics import ArmKinematics
        self._kin = ArmKinematics(None if world_xml is None else str(world_xml))
        self.rest_pose = np.asarray(REST_POSE, float).copy()
        self.joint_names = tuple(JOINT_NAMES)

    def rest_q(self, base_yaw: float | None = None) -> np.ndarray:
        """REST_POSE, with base_yaw replaced by the measured value if given."""
        q = self.rest_pose.copy()
        if base_yaw is not None:
            q[0] = float(base_yaw)
        return q

    def head(self, q) -> tuple[np.ndarray, np.ndarray]:
        hp = self._kin.head_pose(np.asarray(q, float))
        return hp.rot, hp.pos

    def camera_pose(self, mount: HeadCameraMount, q) -> Pose:
        R, t = self.head(q)
        return mount.camera_pose(R, t)

    def camera_pose_at_rest(self, mount: HeadCameraMount, base_yaw: float | None = None) -> Pose:
        return self.camera_pose(mount, self.rest_q(base_yaw))


def fit_mount(observations, cam: Intrinsics, head_R: np.ndarray, head_t: np.ndarray,
              source: str = "charuco-fit", max_frames: int = 250) -> tuple["HeadCameraMount", float]:
    """Mount that best reproduces the board over many frames, in pixels.

    **Averaging per-frame camera poses does not work.** The mean of a set of
    poses is a pose that matches no instant, and back-projected onto the desk
    at the shallow angle the head camera sees it, that mean landed 2.6 cm out
    where each individual frame was 0.7 mm out.

    So nothing is averaged. This fits one fixed mount that, run through the
    rest-pose FK, reprojects the known board corners closest to where they
    were actually seen across every frame. Frame-to-frame noise is then spread
    evenly instead of biasing the answer toward whichever frame was picked.

    Minimising the error **in pixels** rather than in metres on the desk plane
    is deliberate. Fitting the desk error directly was tried on the real lamp
    and is nearly degenerate: it converged on a camera 12 cm above the desk
    (the real one is 33 cm) and was no more accurate, because many poses put
    the rays through the same points of one plane. Pixels keep the fit
    well conditioned, and measured desk accuracy came out at 1.2 cm.

    ``observations`` is a list of ``(obj_base, uv)``: board corners in
    ``lamp_base`` metres and the pixels they appeared at. Returns the mount and
    its RMS reprojection error in pixels.

    Levenberg-Marquardt with a numerical Jacobian over the 6 parameters, rather
    than scipy: ``vision``'s geometry stays numpy-only so it runs anywhere the
    detectors do.
    """
    if not observations:
        raise ValueError("no board observations")
    if len(observations) > max_frames:                  # even spread over the window
        idx = np.linspace(0, len(observations) - 1, max_frames).round().astype(int)
        observations = [observations[i] for i in idx]
    obj = np.concatenate([np.asarray(o, float) for o, _ in observations])
    uv = np.concatenate([np.asarray(p, float) for _, p in observations])

    def residual(p):
        pose = Pose(head_R @ _rodrigues(p[:3]), head_t + head_R @ p[3:6])
        return (cam.project(pose.to_cam(obj)) - uv).ravel()

    seed = _seed_mount(*observations[len(observations) // 2], cam, head_R, head_t)
    p = np.concatenate([_log_rodrigues(seed.R), seed.t])
    r = seed_r = residual(p)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(80):
        J = np.empty((r.size, 6))
        for k in range(6):
            step = 1e-6 if k < 3 else 1e-7
            q = p.copy(); q[k] += step
            J[:, k] = (residual(q) - r) / step
        JtJ, Jtr = J.T @ J, J.T @ r
        for _ in range(12):                              # back off until the step helps
            try:
                dp = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ)), -Jtr)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            r2 = residual(p + dp)
            if float(r2 @ r2) < cost:
                p, r, cost = p + dp, r2, float(r2 @ r2)
                lam = max(lam * 0.3, 1e-9)
                break
            lam *= 10
        else:
            break
        if np.linalg.norm(dp[:3]) < 1e-10 and np.linalg.norm(dp[3:]) < 1e-10:
            break
    rms = float(np.sqrt(np.mean(np.sum(r.reshape(-1, 2) ** 2, axis=1))))
    if float(seed_r @ seed_r) < cost:      # never return worse than the frame we started from
        return seed, float(np.sqrt(np.mean(np.sum(seed_r.reshape(-1, 2) ** 2, axis=1))))
    return HeadCameraMount(_rodrigues(p[:3]), p[3:6], source, rms), rms


def _rodrigues(r: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def _log_rodrigues(R: np.ndarray) -> np.ndarray:
    c = float(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    theta = math.acos(c)
    if theta < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (theta / (2 * math.sin(theta)))


def _seed_mount(obj_base, uv, cam: Intrinsics, head_R, head_t) -> HeadCameraMount:
    """One frame's mount, as a starting point: a real pose, not an average."""
    import cv2
    ok, rvec, tvec = cv2.solvePnP(np.asarray(obj_base, float), np.asarray(uv, float),
                                  cam.K, np.array(cam.dist), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("seed solvePnP failed")
    R_cb, _ = cv2.Rodrigues(rvec)                  # lamp_base -> camera
    R = R_cb.T
    return HeadCameraMount.from_poses(Pose(R, -R @ tvec.ravel()), head_R, head_t)
