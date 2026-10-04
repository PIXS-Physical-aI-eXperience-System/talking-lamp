"""Head-mounted camera: mount algebra and the S2 image-space follow loop.

The follow loop is checked in closed loop against a simulated head that only
moves part of the way toward each new target per step, the way E's L1 lags
behind its input. It must converge on the face without overshooting.
"""

import math

import numpy as np
import pytest

from vision import Intrinsics
from vision.detector import FaceDetection
from vision.follow import FaceFollower, FollowConfig
from vision.geometry import IPD_M
from vision.head_camera import HeadCameraMount, joints_in_order

CAM = Intrinsics(1920, 1080, 1050.0, 1050.0, 959.5, 539.5, (-0.30, 0.09, 0.001, -0.0005, -0.01))

# Lens parallel to the head's aiming axis (+x), 3 cm above it: head +x -> cam +z,
# head +y (left) -> cam -x, head +z (up) -> cam -y.
MOUNT = HeadCameraMount(
    np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
    np.array([0.0, 0.0, 0.03]),
)


def head_frame(fwd):
    fwd = np.asarray(fwd, float) / np.linalg.norm(fwd)
    left = np.cross([0.0, 0.0, 1.0], fwd)
    left /= np.linalg.norm(left)
    return np.column_stack([fwd, left, np.cross(fwd, left)])


def random_rotation(seed):
    q = np.random.default_rng(seed).normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# -- mount -------------------------------------------------------------------

def test_mount_round_trip():
    head_R, head_t = random_rotation(7), np.array([0.18, -0.01, 0.35])
    mount = HeadCameraMount(random_rotation(3), np.array([0.01, -0.02, 0.04]))
    cam_pose = mount.camera_pose(head_R, head_t)
    back = HeadCameraMount.from_poses(cam_pose, head_R, head_t)
    assert np.allclose(back.R, mount.R, atol=1e-12)
    assert np.allclose(back.t, mount.t, atol=1e-12)


def test_parallel_mount_has_no_axis_offset():
    assert MOUNT.axis_offset_deg == pytest.approx(0.0, abs=1e-9)


def test_gaze_sits_below_centre_for_a_lens_above_the_axis():
    """3 cm above the axis, a target 0.6 m out appears ~2.9 deg below centre."""
    ideal = Intrinsics(1920, 1080, 1050.0, 1050.0, 959.5, 539.5)
    u, v = MOUNT.gaze_uv(ideal, 0.6)
    assert u == pytest.approx(959.5, abs=1e-6)
    assert math.degrees(math.atan((v - 539.5) / 1050.0)) == pytest.approx(
        math.degrees(math.atan(0.03 / 0.6)), abs=0.01)


def test_head_kinematics_at_rest():
    pytest.importorskip("mujoco")
    from vision.head_camera import HeadKinematics
    kin = HeadKinematics()
    q = kin.rest_q(base_yaw=0.1)
    assert q[0] == 0.1 and np.allclose(q[1:], kin.rest_pose[1:])
    R, t = kin.head(q)
    pose = kin.camera_pose(MOUNT, q)
    assert np.allclose(pose.t, t + R @ MOUNT.t)
    assert 0.25 < t[2] < 0.45        # the head sits ~35 cm above the desk at rest


# -- follow loop -------------------------------------------------------------

HEAD_POS = np.array([0.18, 0.0, 0.35])
FACE = np.array([0.70, 0.15, 0.40])      # seated user, eye line 40 cm above the desk


def render_face(look_dir):
    """What the head camera sees when the head aims along ``look_dir``."""
    pose = MOUNT.camera_pose(head_frame(look_dir), HEAD_POS)
    side = head_frame(FACE - HEAD_POS)[:, 1]          # face turned toward the lamp
    re, le = FACE - side * IPD_M / 2, FACE + side * IPD_M / 2
    uv_r, uv_l = CAM.project(pose.to_cam(re)), CAM.project(pose.to_cam(le))
    w = abs(uv_l[0] - uv_r[0]) / 0.45
    cx, cy = (uv_r + uv_l) / 2
    return FaceDetection(0.9, (cx - w / 2, cy - 0.3 * w, cx + w / 2, cy + 0.9 * w),
                         tuple(uv_r), tuple(uv_l))


def angle_deg(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(a @ b, -1, 1))))


def run_loop(lag: float, steps: int = 80, use_gaze: bool = True):
    start = HEAD_POS + 0.5 * np.array([0.9, -0.3, -0.2])          # looking well off the face
    fol = FaceFollower(CAM, HEAD_POS, start, FollowConfig(),
                       gaze_uv=MOUNT.gaze_uv(CAM, 0.6) if use_gaze else None)
    look = start - HEAD_POS
    errs = []
    for k in range(steps):
        stamp = 0.1 * k
        upd = fol.update(render_face(look), stamp)
        cmd = fol.target - HEAD_POS
        look = look / np.linalg.norm(look) + lag * (cmd / np.linalg.norm(cmd) - look / np.linalg.norm(look))
        errs.append(angle_deg(look, FACE - HEAD_POS))
        del upd
    return errs, fol


def test_follow_converges_on_the_face():
    errs, fol = run_loop(lag=1.0)
    assert errs[0] > 20
    assert errs[-1] < 2.5                               # deadband is 2 deg
    assert np.linalg.norm(fol.target - FACE) < 0.12     # distance from face width, roughly right


def test_follow_with_a_lagging_head_does_not_oscillate():
    errs, _ = run_loop(lag=0.3)
    assert errs[-1] < 2.5
    # Monotone apart from tiny ripples: no overshoot-and-come-back.
    rises = [b - a for a, b in zip(errs, errs[1:]) if b > a]
    assert max(rises, default=0.0) < 0.5


def test_gaze_correction_removes_parallax_bias():
    with_gaze, _ = run_loop(lag=1.0, use_gaze=True)
    centred, _ = run_loop(lag=1.0, use_gaze=False)
    assert with_gaze[-1] < centred[-1]


def test_follow_is_fast_enough_for_s2():
    """Turning to look at whoever called should finish within a couple of seconds."""
    errs, _ = run_loop(lag=0.3)
    t_done = next(k for k, e in enumerate(errs) if e < 2.5) * 0.1
    assert t_done < 3.0


def test_no_correction_while_the_head_is_moving():
    fol = FaceFollower(CAM, HEAD_POS, FACE, FollowConfig(), gaze_uv=MOUNT.gaze_uv(CAM, 0.6))
    base = FACE - HEAD_POS
    # The face drifts across the image frame to frame, as it does while the head turns.
    for k, dy in enumerate([0.20, 0.15, 0.10, 0.05]):
        assert fol.update(render_face(base + np.array([0.0, dy, 0.0])), 0.1 * k) is None


def test_deadband_settle_and_rate_limit():
    fol = FaceFollower(CAM, HEAD_POS, FACE, FollowConfig(), gaze_uv=MOUNT.gaze_uv(CAM, 0.6))
    on = render_face(FACE - HEAD_POS)
    assert all(fol.update(on, 0.1 * k) is None for k in range(5))    # on target: nothing to do
    off = render_face(FACE - HEAD_POS + np.array([0.0, 0.15, 0.0]))
    out = [fol.update(off, 1.0 + 0.1 * k) for k in range(3)]
    assert out[:2] == [None, None] and out[2] is not None            # corrects once settled
    assert fol.update(off, 1.25) is None                             # history reset after sending
    assert fol.update(None, 5.0) is None and fol.lost(5.0)


# -- fitting the mount from many frames --------------------------------------

def test_fit_mount_recovers_a_known_mount_despite_head_sway():
    """The real lamp never holds still, so the fit sees each frame from a
    slightly different head pose while the FK it is given says "rest pose".
    It must still land on the mount, not on whichever frame came first."""
    from vision.head_camera import fit_mount
    rng = np.random.default_rng(4)
    head_R, head_t = head_frame([1.0, 0.0, -0.6]), np.array([0.18, 0.0, 0.35])
    board = np.array([[x, y, 0.0] for x in np.linspace(0.45, 0.60, 4)
                      for y in np.linspace(-0.12, 0.12, 6)])
    from vision.head_camera import _rodrigues
    obs = []
    for _ in range(40):
        # A few degrees of sway about the head, as motion.idle produces.
        swayed = _rodrigues(rng.normal(0, math.radians(3.0), 3)) @ head_R
        pose = MOUNT.camera_pose(swayed, head_t)
        obs.append((board, CAM.project(pose.to_cam(board))))
    fitted, rms = fit_mount(obs, CAM, head_R, head_t)
    assert np.linalg.norm(fitted.t - MOUNT.t) < 0.02          # within 2 cm
    assert angle_deg(fitted.R[:, 2], MOUNT.R[:, 2]) < 2.0


def test_fit_mount_is_exact_when_the_head_is_perfectly_still():
    from vision.head_camera import fit_mount
    head_R, head_t = head_frame([1.0, 0.1, -0.6]), np.array([0.18, 0.0, 0.35])
    board = np.array([[x, y, 0.0] for x in np.linspace(0.45, 0.60, 4)
                      for y in np.linspace(-0.12, 0.12, 6)])
    pose = MOUNT.camera_pose(head_R, head_t)
    obs = [(board, CAM.project(pose.to_cam(board)))] * 3
    fitted, rms = fit_mount(obs, CAM, head_R, head_t)
    assert rms < 0.5
    assert np.allclose(fitted.t, MOUNT.t, atol=1e-3)


def test_joint_states_are_matched_by_name_not_position():
    order = ("base_yaw", "base_pitch", "elbow_pitch")
    q = joints_in_order(["elbow_pitch", "base_yaw", "base_pitch"], [3.0, 1.0, 2.0], order)
    assert list(q) == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("names, values", [
    (["base_yaw", "base_pitch"], [1.0, 2.0]),                   # elbow missing
    (["base_yaw", "base_pitch", "elbow_pitch"], [1.0, 2.0]),    # one value short
    (["base_yaw", "base_yaw", "elbow_pitch"], [1.0, 2.0, 3.0]),  # duplicate name
])
def test_an_incomplete_joint_state_is_refused(names, values):
    with pytest.raises(ValueError):
        joints_in_order(names, values, ("base_yaw", "base_pitch", "elbow_pitch"))
