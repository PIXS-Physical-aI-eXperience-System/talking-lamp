"""D's publishing logic: when a target is valid, and when it must not be sent.

The property that matters most is negative. Desk back-projection is only
meaningful at the rest pose, so the runtime must refuse to produce a task-light
target at any other posture rather than quietly returning a point computed from
a camera pose that no longer holds.
"""

import numpy as np
import pytest

from vision import Intrinsics, Pose
from vision.detector import Detection, FaceDetection
from vision.pipeline import VisionPipeline
from vision.runtime import RuntimeConfig, VisionRuntime
from vision.tracking import ObjectAverager, PresenceGate
from vision.geometry import IPD_M

CAM = Intrinsics(1920, 1080, 705.0, 705.0, 1014.5, 503.6)
HEAD = np.array([0.184, 0.0, 0.347])
POSE = Pose.look_at(HEAD, [0.184 + 0.8, 0.0, 0.347 - 0.58])      # ~36 deg down, as at rest


def box_at(p_base, pose=POSE, w=120.0, h=80.0):
    """A detection box whose bottom centre projects to the desk point ``p_base``."""
    u, v = CAM.project(pose.to_cam(np.asarray(p_base, float)))
    return (u - w / 2, v - h, u + w / 2, v)


def face_at(p_base, pose=POSE, conf=0.9):
    side = np.array([0.0, 1.0, 0.0])
    uv_r = CAM.project(pose.to_cam(p_base - side * IPD_M / 2))
    uv_l = CAM.project(pose.to_cam(p_base + side * IPD_M / 2))
    w = abs(uv_l[0] - uv_r[0]) / 0.45
    cx, cy = (uv_r + uv_l) / 2
    return FaceDetection(conf, (cx - w / 2, cy - 0.3 * w, cx + w / 2, cy + 0.9 * w),
                         tuple(uv_r), tuple(uv_l))


def make_runtime(at_rest=True, **kw):
    rt = VisionRuntime(CAM, pipeline=VisionPipeline(CAM, POSE), cfg=RuntimeConfig(**kw))
    if at_rest:
        rt.at_rest = True                 # as enter_rest() leaves it, without MuJoCo
    return rt


def feed(rt, dets, faces=(), t0=0.0, n=20, dt=0.3):
    """Run n frames of the same scene through the geometry and the runtime."""
    outs = []
    for k in range(n):
        stamp = t0 + k * dt
        outs.append(rt.consume(rt.pipeline.locate(list(dets), list(faces), stamp), stamp))
    return outs


BOOK = np.array([0.45, 0.10, 0.0])


def book_det(conf=0.8, pos=BOOK):
    return Detection("book", conf, box_at(pos))


# -- the rest-pose rule ------------------------------------------------------

def test_task_light_target_after_enough_frames():
    rt = make_runtime()
    feed(rt, [book_det()], n=20)
    target, why = rt.task_light_target(6.0)
    assert why == ""
    assert target.label == "book"
    assert np.linalg.norm(target.pos - BOOK) < 0.02


def test_no_task_light_target_away_from_the_rest_pose():
    rt = make_runtime()
    feed(rt, [book_det()], n=20)
    assert rt.task_light_target(6.0)[0] is not None
    rt.leave_rest("E가 모션 재생 중")
    target, why = rt.task_light_target(6.0)
    assert target is None
    assert "휴식 자세" in why and "모션 재생" in why


def test_averages_are_dropped_when_the_head_leaves_rest():
    """Stale averages must not survive: they came from a pose that no longer holds."""
    rt = make_runtime()
    feed(rt, [book_det()], n=20)
    rt.leave_rest()
    rt.at_rest = True                      # back at rest, but nothing re-measured yet
    assert rt.task_light_target(6.0)[0] is None


def test_nothing_accumulates_while_off_rest():
    rt = make_runtime(at_rest=False)
    feed(rt, [book_det()], n=20)
    assert rt.averager.tracks == []


def test_target_needs_a_full_breath_period_of_samples():
    """Fewer frames than one idle cycle is not enough to have cancelled the sway."""
    rt = make_runtime()
    feed(rt, [book_det()], n=6, dt=0.3)            # 1.8 s < 4.2 s
    target, why = rt.task_light_target(1.8)
    assert target is None and "평균" in why


def test_reason_when_the_desk_is_empty():
    rt = make_runtime()
    feed(rt, [Detection("mouse", 0.9, box_at([0.45, 0.0, 0.0]))], n=20)
    target, why = rt.task_light_target(6.0)
    assert target is None and "조명 대상" in why


def test_the_most_confident_object_wins():
    rt = make_runtime()
    feed(rt, [book_det(0.5), Detection("laptop", 0.9, box_at([0.5, -0.2, 0.0]))], n=20)
    target, _ = rt.task_light_target(6.0)
    assert target.label == "laptop"


# -- averaging cancels the sway ---------------------------------------------

def test_averaging_beats_any_single_frame_under_sway():
    """Positions jittered like E's idle layer: the mean is closer than the frames."""
    rng = np.random.default_rng(0)
    av = ObjectAverager()
    from vision.pipeline import ObjectTarget, VisionFrame
    errs = []
    for k in range(20):
        noisy = BOOK + np.array([*rng.normal(0, 0.03, 2), 0.0])
        errs.append(np.linalg.norm(noisy - BOOK))
        f = VisionFrame(stamp=0.3 * k)
        f.objects.append(ObjectTarget("book", 0.8, noisy, (0, 0, 1, 1), 0.3 * k))
        av.add(f)
    assert np.linalg.norm(av.best(5.7).pos - BOOK) < np.median(errs) / 2


def test_two_objects_of_one_class_stay_separate():
    rt = make_runtime()
    far = np.array([0.45, -0.25, 0.0])
    feed(rt, [book_det(0.8, BOOK), book_det(0.7, far)], n=20)
    ready = rt.averager.ready(6.0)
    assert len(ready) == 2
    assert min(np.linalg.norm(t.pos - far) for t in ready) < 0.02


# -- S4 presence -------------------------------------------------------------

def test_presence_reports_only_changes():
    rt = make_runtime()
    outs = feed(rt, [], [face_at([0.7, 0.0, 0.42])], n=10, dt=0.3)
    changes = [o.presence for o in outs if o.presence is not None]
    assert changes == [True]


def test_a_blink_does_not_read_as_leaving():
    gate = PresenceGate(on_s=0.5, off_s=5.0, present=True)
    assert all(gate.update(False, t) is None for t in np.arange(0.0, 4.5, 0.1))
    assert gate.update(True, 4.6) is None          # back before the timeout: no event at all
    assert gate.present


def test_leaving_is_reported_after_the_timeout():
    gate = PresenceGate(on_s=0.5, off_s=5.0, present=True)
    assert gate.update(False, 0.0) is None
    assert gate.update(False, 4.9) is None
    assert gate.update(False, 5.1) is False


# -- S2 follow ---------------------------------------------------------------

FACE = np.array([0.70, 0.15, 0.42])


def test_following_starts_from_the_rest_pose_and_then_leaves_it():
    rt = make_runtime()
    rt.set_following(True)
    out = feed(rt, [], [face_at(FACE)], n=1)[0]
    assert out.track_point is not None
    assert np.linalg.norm(out.track_point - FACE) < 0.10
    assert not rt.at_rest                          # the head is about to move


def test_following_cannot_start_off_rest():
    rt = make_runtime(at_rest=False)
    rt.set_following(True)
    out = feed(rt, [], [face_at(FACE)], n=1)[0]
    assert out.track_point is None
    assert "rest pose" in out.note


def test_no_track_point_while_not_following():
    rt = make_runtime()
    outs = feed(rt, [], [face_at(FACE)], n=5)
    assert all(o.track_point is None for o in outs)


def test_labels_are_published_on_change_only():
    rt = make_runtime()
    outs = feed(rt, [book_det()], n=5)
    assert outs[0].labels == ["책"]
    assert all(o.labels is None for o in outs[1:])
    more = rt.consume(rt.pipeline.locate(
        [book_det(), Detection("laptop", 0.9, box_at([0.5, -0.2, 0.0]))], [], 9.0), 9.0)
    assert more.labels == ["노트북", "책"]


@pytest.mark.parametrize("seen", [True, False])
def test_presence_gate_is_stable_when_it_agrees(seen):
    gate = PresenceGate(present=seen)
    assert all(gate.update(seen, t) is None for t in np.arange(0.0, 20.0, 0.5))


# -- detector rates ----------------------------------------------------------

def test_objects_run_at_their_own_rate_not_every_frame():
    """YOLOX-s costs ~610 ms; faces must not wait for it."""
    rt = make_runtime()
    assert rt.due_for_objects(0.0)
    rt.consume(rt.pipeline.locate([book_det()], [], 0.0), 0.0, had_objects=True)
    assert not rt.due_for_objects(0.3)
    assert rt.due_for_objects(0.6)


def test_a_faces_only_frame_does_not_clear_the_labels():
    """An empty object list there means 'not looked for', not 'nothing there'."""
    rt = make_runtime()
    rt.consume(rt.pipeline.locate([book_det()], [], 0.0), 0.0, had_objects=True)
    assert rt._labels == ["책"]
    out = rt.consume(rt.pipeline.locate([], [face_at(FACE)], 0.1), 0.1, had_objects=False)
    assert out.labels is None and rt._labels == ["책"]


def test_a_faces_only_frame_does_not_feed_the_averages():
    rt = make_runtime()
    for k in range(20):
        rt.consume(rt.pipeline.locate([book_det()], [], 0.3 * k), 0.3 * k, had_objects=True)
    before = len(rt.averager.tracks[0].samples)
    rt.consume(rt.pipeline.locate([], [], 6.1), 6.1, had_objects=False)
    assert len(rt.averager.tracks[0].samples) == before


def test_objects_slow_down_while_following():
    rt = make_runtime()
    rt.set_following(True)
    rt.consume(rt.pipeline.locate([book_det()], [], 0.0), 0.0, had_objects=True)
    assert not rt.due_for_objects(1.0)          # 0.5 s rate does not apply while following
    assert rt.due_for_objects(2.1)
