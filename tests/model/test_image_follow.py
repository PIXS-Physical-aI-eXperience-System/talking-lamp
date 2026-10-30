from dataclasses import dataclass

from vision.image_follow import ImageFaceFollower


@dataclass(frozen=True)
class Face:
    conf: float
    box: tuple[float, float, float, float]


def test_face_at_upper_left_proposes_the_measured_safe_directions():
    follower = ImageFaceFollower()
    step = follower.propose([Face(.9, (780, 180, 900, 340))], 1920, 1080, 1.0)
    assert step is not None
    assert step.yaw_deg == 2.0
    assert step.wrist_deg == -2.0
    assert follower.propose([Face(.9, (790, 190, 910, 350))], 1920, 1080, 1.25) is None
    assert follower.propose([Face(.9, (790, 190, 910, 350))], 1920, 1080, 1.31) is not None


def test_follower_keeps_the_same_person_when_another_face_appears():
    follower = ImageFaceFollower()
    near = Face(.8, (750, 180, 950, 380))
    far = Face(.99, (1100, 250, 1150, 310))
    first = follower.propose([near, far], 1920, 1080, 1.0)
    second = follower.propose([far, near], 1920, 1080, 3.0)
    assert first.face_center == second.face_center == (850.0, 280.0)


def test_follower_does_not_command_while_moving_or_after_target_jump():
    follower = ImageFaceFollower()
    face = Face(.9, (750, 180, 950, 380))
    assert follower.propose([face], 1920, 1080, 1.0, moving=True) is None
    assert follower.propose([Face(.9, (1200, 180, 1400, 380))],
                            1920, 1080, 3.0) is None


def test_centered_face_needs_no_command():
    follower = ImageFaceFollower()
    assert follower.propose([Face(.9, (900, 480, 1020, 600))],
                            1920, 1080, 1.0) is None


def test_repeated_image_feedback_moves_a_face_into_the_center_deadband():
    follower = ImageFaceFollower()
    x, y = 837.0, 266.0
    for tick in range(25):
        face = Face(.9, (x - 60, y - 80, x + 60, y + 80))
        step = follower.propose([face], 1920, 1080, 2.0 * tick)
        if step is None:
            break
        x += step.yaw_deg * follower.cfg.yaw_px_per_deg
        y -= step.wrist_deg * follower.cfg.wrist_px_per_deg
    assert abs(x - 960) <= follower.cfg.deadband_px
    assert abs(y - 540) <= follower.cfg.deadband_px
