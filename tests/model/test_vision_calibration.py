"""Desk registration round trip on a rendered ChArUco board.

The board image is warped into a synthetic camera at a known pose; the
registration must recover that pose. This pins down the board-frame convention
in ``DeskPlacement.board_in_base``: a wrong axis there sends the task light to
the mirror-image spot, and nothing else in the pipeline would notice.

Needs OpenCV, so it runs on the Jetson and is skipped on machines without cv2.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vision import Intrinsics, Pose, desk_point  # noqa: E402
from vision import calibration as cal  # noqa: E402

CAM = Intrinsics(1920, 1080, 1050.0, 1050.0, 959.5, 539.5)
PLACEMENT = cal.DeskPlacement(near_edge_m=0.20, lateral_m=0.0, square_m=cal.SQUARE_M)
TRUE_POSE = Pose.look_at([-0.05, 0.20, 0.30], [0.30, 0.0, 0.0])


def render_board(cam: Intrinsics, pose: Pose, placement: cal.DeskPlacement) -> np.ndarray:
    pps, margin = 210, 40
    board = cal.board_image(pps, margin)
    sq = placement.square_m
    # board-image pixel -> board metres (z = 0 plane)
    S = np.array([[sq / pps, 0, -margin * sq / pps],
                  [0, sq / pps, -margin * sq / pps],
                  [0, 0, 1.0]])
    R_bb, t_bb = placement.board_in_base()
    R_cb = pose.R.T @ R_bb
    t_cb = pose.R.T @ (t_bb - pose.t)
    H = cam.K @ np.column_stack([R_cb[:, 0], R_cb[:, 1], t_cb]) @ S
    img = cv2.warpPerspective(board, H, (cam.width, cam.height), borderValue=255)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_board_is_found_in_the_render():
    found = cal.detect(render_board(CAM, TRUE_POSE, PLACEMENT))
    assert found is not None
    assert len(found[0]) >= 20


def test_registration_recovers_the_camera_pose():
    pose = cal.register_desk(render_board(CAM, TRUE_POSE, PLACEMENT), CAM, PLACEMENT)
    assert pose.rms_px < 1.0
    assert np.allclose(pose.t, TRUE_POSE.t, atol=0.005)          # within 5 mm
    assert np.allclose(pose.R, TRUE_POSE.R, atol=0.01)


def test_registered_pose_places_desk_points_correctly():
    pose = cal.register_desk(render_board(CAM, TRUE_POSE, PLACEMENT), CAM, PLACEMENT)
    for target in ([0.45, 0.0, 0.0], [0.35, 0.12, 0.0], [0.55, -0.15, 0.0]):
        uv = CAM.project(TRUE_POSE.to_cam(target))
        p = desk_point(uv, CAM, pose)
        assert p is not None
        assert np.linalg.norm(p - target) < 0.01


def test_board_far_edge_is_the_print_top():
    """The print's top-left corner must land far from the lamp, on its left."""
    R, t = PLACEMENT.board_in_base()
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert t[0] > PLACEMENT.near_edge_m and t[1] > 0


def test_mount_calibration_recovers_the_head_camera_offset():
    """Board photo at the rest pose -> head-to-camera mount, through E's real FK."""
    pytest.importorskip("mujoco")
    from vision.head_camera import HeadCameraMount, HeadKinematics

    true_mount = HeadCameraMount(
        np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),   # lens along head +x
        np.array([0.0, 0.0, 0.03]),                                        # 3 cm above the axis
    )
    kin = HeadKinematics()
    q = kin.rest_q()
    head_R, head_t = kin.head(q)
    cam_pose = true_mount.camera_pose(head_R, head_t)

    # Lay the board where the camera actually looks at rest.
    d = cam_pose.R[:, 2]
    hit = cam_pose.t + (-cam_pose.t[2] / d[2]) * d
    placement = cal.DeskPlacement(near_edge_m=hit[0] - 2.5 * cal.SQUARE_M, lateral_m=hit[1])

    measured = cal.register_desk(render_board(CAM, cam_pose, placement), CAM, placement)
    mount = HeadCameraMount.from_poses(measured, head_R, head_t)
    assert np.linalg.norm(mount.t - true_mount.t) < 0.005       # within 5 mm
    assert np.allclose(mount.R, true_mount.R, atol=0.01)
    assert mount.axis_offset_deg < 1.0


# -- printable PDF -----------------------------------------------------------

def test_board_pdf_is_a4_at_true_size():
    """The print must not need a scale dialog: the page is A4 and the squares
    are the physical size asked for, with room for any printer's dead border."""
    pdf = cal.board_pdf(35.0).decode("latin-1")
    assert pdf.startswith("%PDF-") and pdf.rstrip().endswith("%%EOF")
    box = pdf.split("/MediaBox [")[1].split("]")[0].split()
    width, height = float(box[2]), float(box[3])
    assert (width / 72 * 25.4, height / 72 * 25.4) == pytest.approx((297.0, 210.0), abs=0.01)
    # 7 x 5 squares of 35 mm leaves >= 15 mm of margin on every side
    assert (297.0 - 7 * 35.0) / 2 >= 15.0 and (210.0 - 5 * 35.0) / 2 >= 15.0


def test_board_pdf_refuses_a_size_that_would_be_clipped():
    with pytest.raises(ValueError):
        cal.board_pdf(45.0)


def test_board_backing_thickness_shifts_the_camera_height():
    """A clipboard under the print raises the board, and ignoring that pushes
    the whole calibration down by the same amount."""
    flat = cal.DeskPlacement(0.20, 0.0, 0.035)
    raised = cal.DeskPlacement(0.20, 0.0, 0.035, height_m=0.003)
    assert raised.board_in_base()[1][2] - flat.board_in_base()[1][2] == pytest.approx(0.003)
