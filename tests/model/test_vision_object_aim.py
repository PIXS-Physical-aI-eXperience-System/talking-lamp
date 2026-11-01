"""S1 in image space: tilt down, sweep, centre the book."""

from vision.detector import Detection
from vision.object_aim import AimConfig, ObjectAimer

W, H = 1920, 1080


def book(cx, cy, label="book", conf=0.5):
    return Detection(label, conf, (cx - 80, cy - 60, cx + 80, cy + 60))


def run_search(aimer, seconds, detections=()):
    steps = []
    for k in range(int(seconds * 10)):
        step = aimer.update(list(detections), W, H, k * 0.1)
        if step is not None:
            steps.append((step.yaw_deg, step.wrist_deg))
    return steps


def test_tilts_down_then_sweeps_both_ways_without_a_target():
    aimer = ObjectAimer()
    steps = run_search(aimer, 50)
    wrist = [w for _, w in steps]
    assert sum(wrist) == AimConfig().max_wrist_deg          # down to the limit, no further
    yaws = [y for y, _ in steps if y]
    assert any(y > 0 for y in yaws) and any(y < 0 for y in yaws)
    assert abs(aimer.yaw) <= AimConfig().max_yaw_deg
    assert aimer.state == "searching"


def test_ignores_things_the_light_is_not_for():
    aimer = ObjectAimer()
    run_search(aimer, 3, [book(300, 900, "person"), book(200, 200, "cup")])
    assert aimer.state == "searching"


def lowered():
    """An aimer that has already tilted all the way down."""
    aimer = ObjectAimer()
    run_search(aimer, 15)
    assert aimer.wrist == AimConfig().max_wrist_deg
    return aimer


def test_looks_down_before_choosing_even_with_a_target_in_view():
    aimer = ObjectAimer()
    step = aimer.update([book(W / 2, 300, "laptop")], W, H, 0.0)
    assert (step.yaw_deg, step.wrist_deg) == (0.0, 2.0)
    assert aimer.state == "searching"


def test_takes_a_book_seen_on_the_way_down():
    aimer = ObjectAimer()
    run_search(aimer, 5)
    assert 0 < aimer.wrist < AimConfig().max_wrist_deg
    aimer.update([book(W / 2, 100)], W, H, 100.0)
    assert aimer.state == "centering" and aimer.label == "book"


def test_prefers_the_book_in_front_to_the_users_laptop():
    aimer = lowered()
    laptop, front_book = book(1500, 500, "laptop"), book(700, 1000)
    aimer.update([laptop, front_book], W, H, 100.0)
    assert aimer.label == "book"
    aimer = lowered()
    aimer.update([book(1500, 500, "laptop"), book(700, 1000, "keyboard")], W, H, 100.0)
    assert aimer.label == "keyboard"            # lowest in the frame


def test_centres_a_book_below_and_to_the_left():
    aimer = lowered()
    step = aimer.update([book(400, 1000)], W, H, 100.0)
    assert aimer.state == "centering"
    assert step.yaw_deg > 0          # positive base yaw shifts the image right
    assert step.wrist_deg == 0       # already at the wrist limit
    assert aimer.label == "book"


def test_centres_upward_when_the_book_is_above_the_middle():
    aimer = lowered()
    step = aimer.update([book(W / 2, 200)], W, H, 100.0)
    assert step.wrist_deg < 0        # negative wrist pitch tilts the camera up


def test_reports_centred_once_nothing_is_left_to_correct():
    aimer = lowered()
    for k in range(3):
        aimer.update([book(W / 2 + 10, H / 2 - 10)], W, H, 100.0 + k)
    assert aimer.state == "centered"
    assert aimer.update([book(200, 200)], W, H, 5.0) is None    # holds once done


def test_a_book_past_the_wrist_limit_ends_as_centred_as_it_gets():
    aimer = lowered()
    for k in range(40):
        aimer.update([book(W / 2, H - 50, "laptop")], W, H, 100.0 + k)
    assert aimer.wrist == AimConfig().max_wrist_deg
    assert aimer.state == "centered"


def test_losing_the_book_goes_back_to_searching():
    aimer = lowered()
    aimer.update([book(400, 1000)], W, H, 100.0)
    aimer.update([], W, H, 101.0)
    aimer.update([], W, H, 102.0)
    assert aimer.state == "centering"           # a missed detection or two is normal
    aimer.update([], W, H, 103.0)
    assert aimer.state == "searching"


def test_reach_point_turns_the_base_by_the_trim_from_where_the_book_was_centred():
    import math
    import numpy as np
    from motion.kinematics import ArmKinematics
    from motion.layers import TaskLightLayer
    from vision.object_aim import REACH_TRIM_DEG, reach_point

    layer = TaskLightLayer(ArmKinematics())
    for yaw_deg in (-13.0, -1.0, 11.0):
        placed = layer.place(reach_point(math.radians(yaw_deg)))
        assert placed.ok, placed.code
        assert abs(np.degrees(placed.ik.q[0]) - (yaw_deg + REACH_TRIM_DEG)) < 1.0
