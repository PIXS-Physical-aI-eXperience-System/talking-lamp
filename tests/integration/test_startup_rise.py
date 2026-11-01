import json

import numpy as np

from motion import MotionRuntime
from motion import home_pose
from motion.config import CONTROL_DT, REST_POSE
from motion.idle import IdleConfig


def _slumped():
    pose = REST_POSE.copy()
    pose[1] -= np.deg2rad(25)
    pose[2] -= np.deg2rad(40)
    return pose


def test_a_boot_start_rises_slowly_to_the_saved_home_pose():
    start = _slumped()
    runtime = MotionRuntime(initial_pose=start, hold_initial_pose=True, home_pose=REST_POSE,
                            idle_cfg=IdleConfig(enabled=False))
    np.testing.assert_allclose(runtime.step().q_blend, start, atol=np.deg2rad(0.5))
    seconds = runtime.startup_rise.seconds
    assert seconds >= np.deg2rad(40) / 0.25 - 1e-6
    states = runtime.run(seconds + 2.0)
    speeds = np.abs(np.diff([s.q_cmd for s in states], axis=0)) / CONTROL_DT
    assert speeds.max() < 0.35                       # rad/s, not the 2.5 rad/s limit
    np.testing.assert_allclose(runtime.traj.pos, REST_POSE, atol=.01)
    assert not runtime.startup_rise.busy


def test_face_follow_and_return_use_the_home_pose_not_the_start_pose():
    runtime = MotionRuntime(initial_pose=_slumped(), hold_initial_pose=True,
                            home_pose=REST_POSE, idle_cfg=IdleConfig(enabled=False))
    np.testing.assert_allclose(runtime.face_follow.home_pose, REST_POSE)
    assert runtime.orientation.cfg.center_yaw == REST_POSE[0]


def test_without_a_home_pose_the_start_pose_is_home_and_nothing_rises():
    start = _slumped()
    runtime = MotionRuntime(initial_pose=start, hold_initial_pose=True,
                            idle_cfg=IdleConfig(enabled=False))
    assert not runtime.startup_rise.busy
    runtime.run(1.0)
    np.testing.assert_allclose(runtime.traj.pos, start, atol=1e-3)


def test_home_pose_file_is_saved_once_and_then_loaded(tmp_path):
    path = tmp_path / "state" / "home_pose.json"
    assert home_pose.load(path) is None
    assert home_pose.save(path, REST_POSE)
    np.testing.assert_allclose(home_pose.load(path), REST_POSE)


def test_a_broken_or_unsafe_home_pose_file_is_not_used(tmp_path):
    path = tmp_path / "home_pose.json"
    path.write_text("{not json")
    assert home_pose.load(path) is None
    path.write_text(json.dumps({"pose_rad": [0.0] * 4}))
    assert home_pose.load(path) is None
    far = REST_POSE.copy()
    far[0] = 100.0
    path.write_text(json.dumps({"pose_rad": list(far)}))
    assert home_pose.load(path) is None
