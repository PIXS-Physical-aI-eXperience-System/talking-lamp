"""Vision geometry: pixels -> lamp_base points, without any camera or model.

Every test builds a synthetic camera, projects a known 3D point to a pixel,
then checks the pipeline recovers that point. Distortion is included so the
undistortion path is exercised the way the 120 deg lens will exercise it.
"""

import math

import numpy as np
import pytest

from vision import Intrinsics, Pose, VisionPipeline, desk_point, face_point, to_korean
from vision.detector import Detection, decode, nms
from vision.geometry import IPD_M

# Roughly the IMX415 at 1080p with a wide-angle barrel distortion.
CAM = Intrinsics(1920, 1080, 1050.0, 1050.0, 959.5, 539.5, (-0.30, 0.09, 0.001, -0.0005, -0.01))
# Recommended mount (vision-bench/README.md 5): 30 cm up, beside the lamp,
# looking at the work area 0.45 m in front of the base.
POSE = Pose.look_at([-0.05, 0.20, 0.30], [0.45, 0.0, 0.0])


def test_distortion_round_trip():
    xy = np.array([[0.0, 0.0], [0.4, -0.3], [-0.8, 0.5], [0.9, 0.45]])
    back = CAM._undistort_norm(CAM._distort_norm(xy))
    assert np.allclose(back, xy, atol=1e-6)


def test_from_fov_matches_catalogue():
    cam = Intrinsics.from_fov(1920, 1080, 120.0)
    assert cam.source == "fov"
    # half diagonal 1101.5 px over tan(60 deg)
    assert cam.fx == pytest.approx(1101.45 / math.sqrt(3), rel=1e-3)


def test_scaled_keeps_the_same_rays():
    half = CAM.scaled(960, 540)
    ray_full = CAM.pixel_to_ray([1500.0, 800.0])
    ray_half = half.pixel_to_ray([(1500.0 + 0.5) / 2 - 0.5, (800.0 + 0.5) / 2 - 0.5])
    assert np.allclose(ray_full, ray_half, atol=1e-9)


def test_look_at_is_level_and_points_down():
    assert np.allclose(POSE.R.T @ POSE.R, np.eye(3), atol=1e-12)
    assert np.linalg.det(POSE.R) == pytest.approx(1.0)
    assert POSE.R[2, 0] == pytest.approx(0.0, abs=1e-12)   # camera x axis horizontal: no roll
    assert 25 < POSE.tilt_deg < 40


@pytest.mark.parametrize("target", [[0.45, 0.0, 0.0], [0.30, 0.15, 0.0], [0.60, -0.20, 0.0]])
def test_desk_point_recovers_the_object(target):
    uv = CAM.project(POSE.to_cam(target))
    p = desk_point(uv, CAM, POSE)
    assert p is not None
    assert np.allclose(p, target, atol=1e-6)


def test_ray_above_horizon_misses_the_desk():
    uv = CAM.project(POSE.to_cam([2.0, 0.0, 1.0]))   # a point high on a far wall
    assert desk_point(uv, CAM, POSE) is None


def test_desk_error_grows_as_the_camera_drops():
    """Why the mount must be high: 1 px of box error, at 5 cm vs 30 cm."""
    target = np.array([0.45, 0.0, 0.0])
    errs = {}
    for h in (0.05, 0.30):
        pose = Pose.look_at([-0.05, 0.0, h], target)
        uv = CAM.project(pose.to_cam(target))
        p = desk_point(uv + np.array([0.0, 1.0]), CAM, pose)
        errs[h] = float(np.linalg.norm(p - target))
    assert errs[0.05] > 3 * errs[0.30]


def test_face_depth_from_eye_distance():
    eyes_mid = np.array([0.55, 0.05, 0.40])       # seated user, eyes 40 cm above the desk
    # Eye line perpendicular to the viewing direction, as for a face looking at the lamp.
    view = eyes_mid - POSE.t
    side = np.cross(view, [0, 0, 1.0])
    side /= np.linalg.norm(side)
    re, le = eyes_mid - side * IPD_M / 2, eyes_mid + side * IPD_M / 2
    uv_r, uv_l = CAM.project(POSE.to_cam(re)), CAM.project(POSE.to_cam(le))
    p = face_point(uv_r, uv_l, CAM, POSE)
    assert p is not None
    # The eye-distance model assumes the eye line is parallel to the image plane.
    # Off-axis faces violate that slightly; a few centimetres is expected.
    assert np.linalg.norm(p - eyes_mid) < 0.04


def _face_pixels(eyes_mid, eye_sep_m, face_width_m):
    """Project a synthetic face: eyes ``eye_sep_m`` apart inside a box ``face_width_m`` wide."""
    view = eyes_mid - POSE.t
    side = np.cross(view, [0, 0, 1.0])
    side /= np.linalg.norm(side)
    re, le = eyes_mid - side * eye_sep_m / 2, eyes_mid + side * eye_sep_m / 2
    l, r = eyes_mid - side * face_width_m / 2, eyes_mid + side * face_width_m / 2
    uv_re, uv_le = CAM.project(POSE.to_cam(re)), CAM.project(POSE.to_cam(le))
    u = sorted([CAM.project(POSE.to_cam(l))[0], CAM.project(POSE.to_cam(r))[0]])
    v = uv_re[1]
    return uv_re, uv_le, (u[0], v - 80, u[1], v + 120)


def test_frontal_face_keeps_the_eye_estimate():
    eyes_mid = np.array([0.55, 0.05, 0.40])
    re, le, box = _face_pixels(eyes_mid, IPD_M, 0.14)
    p = face_point(re, le, CAM, POSE, box)
    assert np.linalg.norm(p - eyes_mid) < 0.04


def test_profile_face_falls_back_to_box_width():
    """Head turned ~65 deg: eyes look 0.027 m apart, box still ~0.14 m wide."""
    eyes_mid = np.array([0.55, 0.05, 0.40])
    re, le, box = _face_pixels(eyes_mid, 0.027, 0.14)
    eyes_only = face_point(re, le, CAM, POSE)                 # the old behaviour
    with_box = face_point(re, le, CAM, POSE, box)
    true_range = np.linalg.norm(eyes_mid - POSE.t)
    assert np.linalg.norm(eyes_only - POSE.t) > 1.8 * true_range   # eyes alone: badly too far
    assert np.linalg.norm(with_box - eyes_mid) < 0.05              # box fallback: close


def test_face_too_close_or_far_is_rejected():
    assert face_point([900, 500], [901, 500], CAM, POSE) is None      # 1 px apart: metres away
    assert face_point([500, 500], [1400, 500], CAM, POSE) is None     # whole frame: centimetres away


def test_pipeline_filters_by_class_and_workspace():
    pipe = VisionPipeline(CAM, POSE)
    def det(label, target):
        u, v = CAM.project(POSE.to_cam(target))
        return Detection(label, 0.9, (u - 40, v - 80, u + 40, v))
    vf = pipe.locate([
        det("book", [0.45, 0.0, 0.0]),        # kept
        det("mouse", [0.40, 0.1, 0.0]),       # on the desk, but not something to light
        det("laptop", [0.05, 0.0, 0.0]),      # under the lamp base
        det("keyboard", [1.20, 0.0, 0.0]),    # beyond reach
    ], [], stamp=12.5)
    assert [o.label for o in vf.objects] == ["book"]
    assert np.allclose(vf.objects[0].pos, [0.45, 0.0, 0.0], atol=1e-6)
    assert vf.objects[0].frame_id == "lamp_base" and vf.objects[0].stamp == 12.5
    reasons = dict(vf.rejected)
    assert reasons == {"mouse": "not a task-light class", "laptop": "outside workspace",
                       "keyboard": "outside workspace"}
    # Cognition sees the whole scene, not just the light targets.
    assert vf.labels_ko() == ["책", "마우스", "노트북", "키보드"]


def test_labels_for_cognition():
    assert to_korean(["book", "keyboard", "book", "tv", "toothbrush"]) == ["책", "키보드", "모니터"]


def test_nms_keeps_the_stronger_of_two_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    assert nms(boxes, np.array([0.5, 0.9, 0.7])) == [1, 2]


def test_decode_places_grid_cells():
    size = 64
    n = sum((size // s) ** 2 for s in (8, 16, 32))
    raw = np.zeros((1, n, 85))
    out = decode(raw, size)[0]
    # First cell of the stride-8 grid, zero offsets: centre (0, 0), size exp(0)*8.
    assert out[0, :4].tolist() == [0.0, 0.0, 8.0, 8.0]
    # Last cell of the stride-32 grid in a 2x2 map: centre (32, 32).
    assert out[-1, :2].tolist() == [32.0, 32.0]
