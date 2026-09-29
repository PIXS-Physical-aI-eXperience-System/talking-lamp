import numpy as np
import pytest

from motion.blender import BlendContext, LayerOutput, MotionBlender
from motion.config import NJ, REST_POSE
from motion.idle import IdleConfig
from motion.kinematics import ArmKinematics
from motion.layers import Envelope, IdleLayer, PrimitiveLayer, TaskLightLayer, TrackLayer


class ConstLayer:
    def __init__(self, name, priority, value, weight, gain=1.0, additive=False):
        self.name, self.priority = name, priority
        self._out = LayerOutput(np.asarray(value, float), np.asarray(weight, float), gain, additive)

    def update(self, ctx):
        return self._out


def ctx(q=None, t=0.0):
    return BlendContext(q_current=np.asarray(q if q is not None else REST_POSE, float), t=t, dt=0.01)


def test_higher_priority_full_weight_overrides():
    lo = ConstLayer("lo", 0, np.zeros(NJ), np.ones(NJ))
    hi = ConstLayer("hi", 10, np.full(NJ, 0.5), np.ones(NJ))
    b = MotionBlender([hi, lo], rest_pose=np.zeros(NJ))
    assert np.allclose(b.compute(ctx(np.zeros(NJ))).q, 0.5)


def test_partial_weight_blends():
    lo = ConstLayer("lo", 0, np.zeros(NJ), np.ones(NJ))
    hi = ConstLayer("hi", 10, np.ones(NJ), np.full(NJ, 0.25))
    b = MotionBlender([hi, lo], rest_pose=np.zeros(NJ))
    assert np.allclose(b.compute(ctx(np.zeros(NJ))).q, 0.25)


def test_additive_layer_rides_on_top():
    base = ConstLayer("base", 0, np.full(NJ, 0.3), np.ones(NJ))
    add = ConstLayer("add", 10, np.full(NJ, 0.1), np.ones(NJ), additive=True)
    b = MotionBlender([base, add], rest_pose=np.zeros(NJ))
    assert np.allclose(b.compute(ctx()).q, 0.4)


def test_zero_gain_layer_is_noop():
    base = ConstLayer("base", 0, np.full(NJ, 0.2), np.ones(NJ))
    ghost = ConstLayer("ghost", 10, np.ones(NJ), np.ones(NJ), gain=0.0)
    b = MotionBlender([base, ghost], rest_pose=np.zeros(NJ))
    assert np.allclose(b.compute(ctx()).q, 0.2)


def test_envelope_attack_release():
    e = Envelope(attack=0.1, release=0.2)
    e.open()
    for _ in range(10):
        e.step(0.01)
    assert e.level == pytest.approx(1.0, abs=1e-6)
    e.close()
    for _ in range(10):
        e.step(0.01)
    assert e.level == pytest.approx(0.5, abs=0.05)


def test_idle_layer_is_small_and_additive():
    ly = IdleLayer(IdleConfig())
    out = ly.update(ctx(t=3.3))
    assert out.additive
    assert np.max(np.abs(out.value)) < np.deg2rad(6)


def test_primitive_layer_plays_and_finishes():
    ly = PrimitiveLayer()
    ly.play("nod", t=0.0)
    assert ly.busy
    peak = 0.0
    for i in range(1200):
        out = ly.update(ctx(t=i * 0.01))
        peak = max(peak, np.max(np.abs(out.value * out.gain)))
    assert peak > np.deg2rad(2)
    assert not ly.busy  # clip ended, envelope closed


def test_primitive_interrupt_is_fast():
    ly = PrimitiveLayer()
    ly.play("wake_up", t=0.0)
    for i in range(100):
        ly.update(ctx(t=i * 0.01))
    ly.interrupt()
    t = 1.0
    while ly.busy and t < 2.0:
        t += 0.01
        ly.update(ctx(t=t))
    assert t - 1.0 < 0.25  # released within the interrupt release time


def test_task_light_layer_solves_and_holds():
    kin = ArmKinematics()
    ly = TaskLightLayer(kin)
    # 0.20 m out, not the 0.25 this used to use: past about 0.25 m the pose
    # needs more than a servo's rated holding torque and is now refused.
    res = ly.place(np.array([0.20, 0.0, 0.0]))
    assert res.ok, res.message
    assert res.ik.pos_err < 0.035
    assert res.ik.aim_err < np.deg2rad(3)
    out = ly.update(ctx())
    assert not out.additive and out.gain > 0
    assert np.allclose(out.value, res.ik.q)


def test_track_layer_fades_out_when_track_lost():
    kin = ArmKinematics()
    ly = TrackLayer(kin)
    for i in range(200):
        ly.observe_point(np.array([0.4, 0.0, 0.35]))
        ly.update(ctx(q=ly.q, t=i * 0.01))
    assert ly.update(ctx(q=ly.q)).gain > 0.5
    ly.clear()
    for i in range(200):
        ly.update(ctx(q=ly.q, t=2.0 + i * 0.01))
    assert ly.update(ctx(q=ly.q)).gain < 0.05


def test_task_light_refuses_a_pose_the_servos_cannot_hold():
    """Reaching far out over the desk is what cooked a servo on the real lamp:
    the arm goes there, holds, overheats and cuts its own torque with nothing
    reported. The layer must refuse instead, and not move."""
    kin = ArmKinematics()
    ly = TaskLightLayer(kin)
    before = ly.q_hold.copy()
    res = ly.place(np.array([0.45, 0.0, 0.0]))
    assert not res.ok and res.code == "over_torque"
    assert res.torque_nm > res.limit_nm
    assert np.allclose(ly.q_hold, before)          # the arm was never sent there
    assert not ly.busy
    assert "exceeds" in res.message


def test_task_light_refuses_a_point_the_arm_cannot_reach():
    kin = ArmKinematics()
    ly = TaskLightLayer(kin)
    res = ly.place(np.array([1.5, 0.0, 0.0]))
    assert not res.ok and res.code == "unreachable"
    assert not ly.busy


def test_holding_torque_grows_as_the_arm_reaches_out():
    """The reason the limit bites: it is the moment arm, not the payload."""
    kin = ArmKinematics()
    ly = TaskLightLayer(kin)
    near = ly.place(np.array([0.15, 0.0, 0.0]))
    far = ly.place(np.array([0.45, 0.0, 0.0]))
    assert near.ok and not far.ok
    assert far.torque_nm > 2 * near.torque_nm


def test_reach_is_held_to_the_same_limit_as_place():
    """`reach` puts the head right at a point with no standoff, so it can ask
    for even more torque than `place` does. Same guard."""
    kin = ArmKinematics()
    ly = TaskLightLayer(kin)
    assert ly.reach(np.array([0.25, 0.0, 0.30])).ok
    far = ly.reach(np.array([0.45, 0.0, 0.15]))
    assert not far.ok and far.code == "over_torque"
