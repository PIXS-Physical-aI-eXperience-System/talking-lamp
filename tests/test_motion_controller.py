"""Controller tests use real layers/trajectory and only fake the motor clock."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from motion.catalog import MotionCatalog
from motion.config import RECORDINGS_DIR
from motion.controller import MotionController
from motion.idle import IdleConfig
from motion.orientation import OrientationConfig
from motion.protocol import Request
from motion.runtime import MotionRuntime, NullBackend


def request(kind, ident="a", expires_at=1000., **payload):
    return Request(ident, kind, payload, expires_at)


def play(ident="a", name="nod", **kw):
    return request("motion.play", ident, **dict(name=name, replace_current=False,
                   intensity=1., repeat=1, **kw))


def orient(ident="orientation", speech_id="speech-1", target_yaw=.4):
    return request("orientation.acquire", ident, speech_id=speech_id, target_yaw=target_yaw)


@pytest.fixture
def controller():
    catalog = MotionCatalog.load(RECORDINGS_DIR / "catalog.toml")
    runtime = MotionRuntime(primitives=catalog.library(), idle_cfg=IdleConfig(enabled=False))
    return MotionController(runtime, catalog)


def test_latest_tracking_value_replaces_pending_value(controller):
    first = controller.submit(request("track.point", point=[.2, 0., .3]))
    last = controller.submit(request("track.point", "b", point=[.3, .1, .4]))
    assert controller.runtime.t == 0
    controller.tick_once(now=1.)
    np.testing.assert_allclose(controller.runtime.track.track.position, [.3, .1, .4])
    assert first.completed.result().code == "superseded"
    assert last.completed.result().state == "completed"


def test_duplicate_motion_id_cannot_start_twice(controller):
    first = controller.submit(play("same"))
    assert controller.submit(play("same")) is first
    controller.tick_once(now=1.)
    assert controller.submit(play("same")) is first
    assert first.accepted.result().state == "accepted"
    assert not first.completed.done()


def test_expired_request_never_reaches_runtime(controller):
    ticket = controller.submit(play("late", expires_at=1.))
    controller.tick_once(now=1.1)
    assert ticket.completed.result().code == "expired"
    assert not controller.runtime.primitive.busy


def test_fifo_consumes_at_most_one_and_busy_rejects(controller):
    first = controller.submit(play())
    second = controller.submit(play("b"))
    controller.tick_once(now=1.)
    assert not second.accepted.done()
    controller.tick_once(now=1.01)
    assert second.completed.result().code == "busy"
    assert not first.completed.done()


def test_replace_cancels_previous_ticket(controller):
    first = controller.submit(play())
    controller.tick_once(now=1.)
    replacement = controller.submit(request("motion.play", "b", name="curious",
                    replace_current=True, intensity=1., repeat=1))
    controller.tick_once(now=1.01)
    assert first.completed.result().state == "cancelled"
    assert replacement.accepted.result().state == "accepted"
    assert controller.snapshot().active_motion == "curious"


def test_queue_bound_rejects_without_calling_runtime(controller):
    for i in range(32):
        controller.submit(request("system.heartbeat", str(i)))
    ticket = controller.submit(play("overflow"))
    assert ticket.accepted.result().code == "queue_full"
    assert ticket.completed.result().code == "queue_full"
    assert controller.runtime.t == 0


def test_tracking_ttl_clears_applied_target(controller):
    controller.submit(request("track.point", expires_at=1.02, point=[.4, 0., .3]))
    controller.tick_once(now=1.)
    assert controller.runtime.track.track.active
    controller.tick_once(now=1.02)
    assert not controller.runtime.track.track.active


def test_bearing_and_expired_point_slots_are_independent(controller):
    point = controller.submit(request("track.point", expires_at=.5, point=[.4, 0., .3]))
    bearing = controller.submit(request("track.bearing", "b", direction=[1., 0., 0.]))
    controller.tick_once(now=1.)
    assert point.completed.result().code == "expired"
    assert bearing.completed.result().state == "completed"
    np.testing.assert_allclose(controller.runtime.track.track.position, [.8, 0., 0.])


def test_unknown_motion_fails_without_replacing_active(controller):
    first = controller.submit(play())
    controller.tick_once(now=1.)
    bad = controller.submit(request("motion.play", "bad", name="not_in_catalog",
                   replace_current=True, intensity=1., repeat=1))
    controller.tick_once(now=1.01)
    assert bad.completed.result().code == "unknown_motion"
    assert not first.completed.done()
    assert controller.snapshot().active_motion == "nod"


@pytest.mark.parametrize("intensity", [0., .4, 1.])
def test_intensity_matches_existing_runtime_output(controller, intensity):
    from motion.primitives import DEFAULT_SCALE
    reference = MotionRuntime(idle_cfg=IdleConfig(enabled=False))
    reference.play_primitive("nod", scale=DEFAULT_SCALE * intensity, loop=False)
    controller.submit(request("motion.play", name="nod", replace_current=False,
                              intensity=intensity, repeat=1))
    for i in range(80):
        controller.tick_once(now=1. + i / 100)
        np.testing.assert_array_equal(controller.runtime.traj.pos, reference.step().q_cmd)


def test_repeat_completes_only_after_last_clip_and_five_settled_ticks(controller):
    ticket = controller.submit(request("motion.play", name="nod", replace_current=False,
                                       intensity=0., repeat=3))
    starts = []
    inactive_ticks = 0
    for i in range(3000):
        controller.tick_once(now=1. + i / 100)
        start = controller.runtime.primitive.started_at
        if start is not None and (not starts or start != starts[-1]):
            starts.append(start)
        inactive_ticks = inactive_ticks + 1 if not controller.runtime.primitive.busy else 0
        if ticket.completed.done():
            break
    assert len(starts) == 3
    assert starts == sorted(set(starts))
    assert inactive_ticks == 5
    assert ticket.completed.result().state == "completed"


def test_motion_completion_waits_for_trajectory_velocity(controller):
    ticket = controller.submit(play())
    settled = 0
    saw_release_in_motion = False
    for i in range(3000):
        if i == 640:
            controller.submit(request("track.point", "track", point=[.3, .4, .3]))
        controller.tick_once(now=1. + i / 100)
        inactive = not controller.runtime.primitive.busy
        slow = np.max(np.abs(controller.runtime.traj.vel)) < controller.settling_tolerance
        saw_release_in_motion |= inactive and not slow
        settled = settled + 1 if inactive and slow else 0
        assert ticket.completed.done() == (settled >= 5)
        if ticket.completed.done():
            break
    assert saw_release_in_motion
    assert ticket.completed.result().state == "completed"


def test_cancel_targets_id_and_interrupt_cancels_active(controller):
    active = controller.submit(play())
    controller.tick_once(now=1.)
    wrong = controller.submit(request("motion.cancel", "wrong", request_id="other"))
    controller.tick_once(now=1.01)
    assert wrong.completed.result().code == "not_found"
    assert not active.completed.done()
    cancel = controller.submit(request("motion.cancel", "cancel", request_id="a"))
    controller.tick_once(now=1.02)
    assert active.completed.result().state == "cancelled"
    assert cancel.completed.result().state == "completed"
    controller.submit(request("motion.interrupt", "interrupt"))
    controller.tick_once(now=1.03)
    assert controller.snapshot().state == "safe_wait"


def test_task_light_completion_then_targeted_cancel_releases_hold(controller):
    ticket = controller.submit(request("task_light.place", point=[.24, 0., 0.]))
    for i in range(2000):
        controller.tick_once(now=1. + i / 100)
        if ticket.completed.done():
            break
    assert ticket.completed.result().state == "completed"
    assert controller.runtime.task_light.busy
    cancel = controller.submit(request("task_light.cancel", "c", request_id="a"))
    controller.tick_once(now=22.)
    assert cancel.completed.result().state == "completed"
    for i in range(60):
        controller.tick_once(now=22.01 + i / 100)
    assert not controller.runtime.task_light.busy


def test_disconnect_bypasses_full_fifo_and_cancels_pending(controller):
    active = controller.submit(play())
    controller.tick_once(now=1.)
    tracking = controller.submit(request("track.point", "track", point=[.4, 0., .3]))
    queued = [controller.submit(play(str(i))) for i in range(32)]
    controller.remote_disconnected()
    assert not active.completed.done()  # network callback never touches runtime
    controller.tick_once(now=1.01)
    assert active.completed.result().state == "cancelled"
    assert tracking.completed.result().state == "cancelled"
    assert all(t.completed.result().state == "cancelled" for t in queued)
    assert not controller.runtime.track.track.active
    assert controller.snapshot().state == "safe_wait"


def test_readonly_queries_and_completed_id_cache(controller):
    listing = controller.submit(request("motion.list"))
    controller.tick_once(now=1.)
    assert listing.completed.result().data["motions"] == controller.catalog.names()
    status = controller.submit(request("motion.status", "status"))
    controller.tick_once(now=1.01)
    assert status.completed.result().data["state"] == "idle"
    assert controller.submit(request("motion.list")) is listing


def test_inflight_id_survives_completed_cache_eviction(controller):
    active = controller.submit(play())
    controller.tick_once(now=1.)
    for i in range(300):
        controller.submit(request("track.point", f"p{i}", point=[.4, 0., .3]))
    assert controller.submit(play()) is active
    assert len(controller._recent) <= 256


def test_cross_thread_submit_and_snapshot_never_step_runtime(controller):
    with ThreadPoolExecutor(max_workers=4) as pool:
        tickets = list(pool.map(lambda _: controller.submit(play()), range(40)))
        statuses = list(pool.map(lambda _: controller.snapshot(), range(10)))
    assert all(t is tickets[0] for t in tickets)
    assert all(s.sent_ticks == 0 for s in statuses)
    assert controller.runtime.backend.measured() is None
    controller.tick_once(now=1.)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="owner"):
            pool.submit(controller.tick_once, now=1.01).result()


class Clock:
    def __init__(self, overshoot=0.):
        self.now = 0.
        self.overshoot = overshoot
    def __call__(self):
        return self.now
    def sleep(self, duration):
        self.now += duration + self.overshoot
        self.overshoot = 0.


@pytest.mark.parametrize("cost, overshoot, expected, misses", [
    (.002, 0., [0., .01, .02, .03], 0),
    (.025, 0., [0., .03, .06, .09], 6),
    (.002, .027, [0., .04, .05, .06], 3),
    (.002, .0005, [0., .0105, .02, .03], 0),
])
def test_run_preserves_deadline_grid_without_bursts(controller, cost, overshoot, expected, misses):
    clock = Clock(overshoot)
    times = []
    class Backend(NullBackend):
        def send(self, q):
            times.append(clock())
            clock.now += cost
            super().send(q)
            if len(times) == 4:
                controller.stop("test")
    controller.runtime.backend = Backend()
    controller.clock = clock
    controller.sleep = clock.sleep
    controller.run()
    np.testing.assert_allclose(times, expected, atol=1e-12)
    assert controller.snapshot().sent_ticks == 4
    assert controller.snapshot().deadline_misses == misses
    assert controller.runtime.t == pytest.approx(.04)
    assert controller.snapshot().state == "stopped"


def test_fault_finishes_all_tickets_and_rejects_new_work(controller):
    class BrokenBackend(NullBackend):
        def send(self, q):
            raise OSError("serial failure")
    controller.runtime.backend = BrokenBackend()
    controller.clock = Clock()
    active = controller.submit(play())
    queued = controller.submit(play("b"))
    controller.run()
    assert controller.snapshot().state == "fault"
    assert "serial failure" in controller.snapshot().fault
    for ticket in (active, queued, controller.submit(play("new"))):
        assert ticket.completed.result().code == "fault"


def test_stop_is_idempotent_and_resolves_pending_without_sending(controller):
    ticket = controller.submit(play())
    controller.stop("shutdown")
    controller.stop("shutdown")
    controller.run()
    assert ticket.completed.result().state == "cancelled"
    assert controller.runtime.backend.measured() is None
    assert controller.submit(play("new")).completed.result().code == "stopped"


def test_scheduler_failure_faults_and_rejects_new_work(controller):
    controller.clock = Clock()
    def broken_sleep(seconds):
        raise OSError("scheduler failure")
    controller.sleep = broken_sleep
    controller.run()
    assert controller.snapshot().state == "fault"
    assert "scheduler failure" in controller.snapshot().fault
    assert controller.submit(play()).completed.result().code == "fault"


def test_cancelled_transport_future_does_not_abort_runtime(controller):
    ticket = controller.submit(play())
    ticket.accepted.cancel()
    ticket.completed.cancel()
    controller.tick_once(now=1.)
    assert controller.runtime.primitive.busy
    assert controller.snapshot().fault is None


def test_zero_intensity_idle_recording_is_finite(controller):
    ticket = controller.submit(request("motion.play", name="idle", replace_current=False,
                                       intensity=0., repeat=1))
    controller.tick_once(now=1.)
    assert controller.runtime.primitive.clip.loop is False
    for i in range(10000):
        controller.tick_once(now=1.01 + i / 100)
        if ticket.completed.done():
            break
    assert ticket.completed.done()
    assert ticket.completed.result().state == "completed"


def test_task_light_interrupt_cancels_ticket_and_releases_layer(controller):
    task = controller.submit(request("task_light.place", point=[.24, 0., 0.]))
    controller.tick_once(now=1.)
    interrupt = controller.submit(request("motion.interrupt", "i"))
    controller.tick_once(now=1.01)
    assert task.completed.result().state == "cancelled"
    assert interrupt.completed.result().state == "completed"
    for i in range(20):
        controller.tick_once(now=1.02 + i / 100)
    assert not controller.runtime.task_light.busy


def test_completed_active_request_reenters_recent_cache_after_tracking_churn(controller):
    active = controller.submit(play())
    controller.tick_once(now=1.)
    for i in range(300):
        controller.submit(request("track.point", f"p{i}", point=[.4, 0., .3]))
    controller.remote_disconnected()
    controller.tick_once(now=1.01)
    assert active.completed.result().state == "cancelled"
    assert controller.submit(play()) is active


@pytest.mark.parametrize("kind, payload", [
    ("track.point", {"point": [.4, .1, .3]}),
    ("track.bearing", {"direction": [1., .1, .2]}),
])
def test_latest_tracking_observation_replaces_its_sources_longer_ttl(controller, kind, payload):
    controller.submit(request(kind, "long", expires_at=10., **payload))
    controller.tick_once(now=1.)
    controller.submit(request(kind, "short", expires_at=1.02, **payload))
    controller.tick_once(now=1.01)
    assert controller.runtime.track.track.active
    controller.tick_once(now=1.02)
    assert not controller.runtime.track.track.active


@pytest.mark.parametrize("kind, payload, other_kind, other_payload", [
    ("track.point", {"point": [.4, .1, .3]},
     "track.bearing", {"direction": [1., .1, .2]}),
    ("track.bearing", {"direction": [1., .1, .2]},
     "track.point", {"point": [.4, .1, .3]}),
])
def test_shorter_tracking_ttl_preserves_other_source_until_its_own_expiry(
    controller, kind, payload, other_kind, other_payload,
):
    controller.submit(request(kind, "long", expires_at=10., **payload))
    controller.submit(request(other_kind, "other", expires_at=1.05, **other_payload))
    controller.tick_once(now=1.)
    controller.submit(request(kind, "short", expires_at=1.02, **payload))
    controller.tick_once(now=1.01)
    controller.tick_once(now=1.02)
    assert controller.runtime.track.track.active
    controller.tick_once(now=1.05)
    assert not controller.runtime.track.track.active


def test_finishing_old_hold_preserves_new_same_id_ticket_after_cache_eviction(controller):
    original = request("task_light.place", "same", point=[.24, 0., 0.])
    old_ticket = controller.submit(original)
    for i in range(2000):
        controller.tick_once(now=1. + i / 100)
        if old_ticket.completed.done():
            break
    assert old_ticket.completed.done()
    assert old_ticket.completed.result().state == "completed"
    assert controller.runtime.task_light.busy
    for i in range(300):
        controller.submit(request("system.heartbeat", f"churn{i}"))
        controller.tick_once(now=21. + i / 100)
    replacement = request("task_light.place", "same", point=[.24, .15, 0.])
    new_ticket = controller.submit(replacement)
    assert new_ticket is not old_ticket
    controller.tick_once(now=25.)
    assert new_ticket.accepted.result().state == "accepted"
    assert not new_ticket.completed.done()
    assert controller.submit(replacement) is new_ticket
    controller.stop("shutdown")
    controller.run()
    assert new_ticket.completed.done()
    assert new_ticket.completed.result().state == "cancelled"
    assert old_ticket.completed.result().state == "completed"


def test_orientation_ticket_completes_only_after_settle(controller):
    ticket = controller.submit(orient(
        speech_id="00000000-0000-0000-0000-000000000001", target_yaw=.4,
    ))

    controller.tick_once(now=1.)

    assert ticket.accepted.result().state == "accepted"
    assert not ticket.completed.done()
    for index in range(300):
        controller.tick_once(now=1.01 + index / 100)
        if ticket.completed.done():
            break
    assert ticket.completed.result().code == "aligned"


def test_orientation_deadband_completes_immediately(controller):
    ticket = controller.submit(orient(target_yaw=.27))

    controller.tick_once(now=1.)

    assert ticket.accepted.result().state == "accepted"
    assert ticket.completed.done()
    assert ticket.completed.result().code == "aligned"


def test_orientation_acquisition_timeout_completes_ticket():
    catalog = MotionCatalog.load(RECORDINGS_DIR / "catalog.toml")
    runtime = MotionRuntime(
        primitives=catalog.library(), idle_cfg=IdleConfig(enabled=False),
        orientation_cfg=OrientationConfig(acquire_timeout=.02),
    )
    controller = MotionController(runtime, catalog)
    ticket = controller.submit(orient(target_yaw=.8))

    controller.tick_once(now=1.)
    controller.tick_once(now=1.03)

    assert ticket.completed.result().code == "timeout"


def test_duplicate_orientation_speech_id_does_not_retarget_active_request(controller):
    original = controller.submit(orient("first", speech_id="speaker", target_yaw=.4))
    duplicate = controller.submit(orient("second", speech_id="speaker", target_yaw=-.4))

    controller.tick_once(now=1.)
    controller.tick_once(now=1.01)

    assert original.accepted.result().state == "accepted"
    assert controller.runtime.orientation_snapshot().target_yaw == pytest.approx(.4)
    assert duplicate.completed.result().code == "duplicate"


def test_orientation_acquire_rejects_active_task_light(controller):
    task = controller.submit(request("task_light.place", point=[.24, 0., 0.]))
    controller.tick_once(now=1.)
    ticket = controller.submit(orient())

    controller.tick_once(now=1.01)

    assert task.accepted.result().state == "accepted"
    assert ticket.completed.result().code == "blocked_by_task_light"
    assert controller.runtime.orientation_snapshot().state == "idle"


def test_orientation_return_center_rejects_busy_primitive(controller):
    motion = controller.submit(play())
    controller.tick_once(now=1.)
    ticket = controller.submit(request("orientation.return_center", "return"))

    controller.tick_once(now=1.01)

    assert motion.accepted.result().state == "accepted"
    assert ticket.completed.result().code == "busy"


def test_orientation_return_center_rejects_busy_task_light(controller):
    task = controller.submit(request("task_light.place", point=[.24, 0., 0.]))
    controller.tick_once(now=1.)
    ticket = controller.submit(request("orientation.return_center", "return"))

    controller.tick_once(now=1.01)

    assert task.accepted.result().state == "accepted"
    assert ticket.completed.result().code == "busy"


def test_orientation_return_center_completes_after_center_settle(controller):
    acquired = controller.submit(orient(target_yaw=.4))
    for index in range(300):
        controller.tick_once(now=1. + index / 100)
        if acquired.completed.done():
            break
    returned = controller.submit(request("orientation.return_center", "return"))

    controller.tick_once(now=4.)

    assert returned.accepted.result().state == "accepted"
    assert not returned.completed.done()
    for index in range(300):
        controller.tick_once(now=4.01 + index / 100)
        if returned.completed.done():
            break
    assert returned.completed.result().code == "centered"


def test_orientation_status_and_motion_result_expose_yaw_scale(controller):
    target_yaw = controller.runtime.orientation_safe_yaw_limits()[1] - .01
    acquired = controller.submit(orient(target_yaw=target_yaw))
    controller.tick_once(now=1.)
    status = controller.submit(request("orientation.status", "status"))
    controller.tick_once(now=1.01)
    motion = controller.submit(play("motion", name="headshake"))
    controller.tick_once(now=1.02)

    assert acquired.accepted.result().state == "accepted"
    assert status.completed.result().data["state"] == "orienting"
    center_yaw = controller.runtime.orientation.cfg.center_yaw
    safe_min, safe_max = controller.runtime.orientation_safe_yaw_limits()
    assert status.completed.result().data["center_yaw"] == pytest.approx(center_yaw)
    assert status.completed.result().data["safe_yaw_min"] == pytest.approx(safe_min)
    assert status.completed.result().data["safe_yaw_max"] == pytest.approx(safe_max)
    snapshot = controller.snapshot()
    assert snapshot.orientation_state == "orienting"
    assert snapshot.orientation_speech_id == "speech-1"
    assert snapshot.orientation_target_yaw == pytest.approx(target_yaw)
    assert snapshot.orientation_current_yaw == pytest.approx(controller.runtime.traj.pos[0])
    assert snapshot.orientation_clamped is False
    assert snapshot.orientation_center_yaw == pytest.approx(center_yaw)
    assert snapshot.orientation_safe_yaw_min == pytest.approx(safe_min)
    assert snapshot.orientation_safe_yaw_max == pytest.approx(safe_max)
    assert snapshot.primitive_yaw_scale < 1.0
    for index in range(3000):
        controller.tick_once(now=1.03 + index / 100)
        if motion.completed.done():
            break
    assert motion.completed.done()
    assert snapshot.primitive_yaw_scale == pytest.approx(motion.completed.result().data["yaw_scale"])


def test_disconnect_starts_center_return_only_after_orientation_hold(controller):
    acquired = controller.submit(orient(target_yaw=.27))
    controller.tick_once(now=1.)
    assert acquired.completed.done()
    assert acquired.completed.result().code == "aligned"

    controller.remote_disconnected()
    controller.tick_once(now=2.)
    controller.tick_once(now=11.99)
    assert controller.snapshot().orientation_state == "aligned"
    controller.tick_once(now=12.)

    assert controller.snapshot().orientation_state == "returning"


def test_orientation_acquire_replaces_an_accepted_return_ticket(controller):
    initial = controller.submit(orient(target_yaw=.27))
    controller.tick_once(now=1.)
    assert initial.completed.result().code == "aligned"
    returned = controller.submit(request("orientation.return_center", "return"))
    controller.tick_once(now=2.)
    assert returned.accepted.result().state == "accepted"
    assert not returned.completed.done()

    acquired = controller.submit(orient("replacement", speech_id="new-speaker", target_yaw=.4))
    controller.tick_once(now=2.01)

    assert returned.completed.done()
    assert returned.completed.result().code == "replaced"
    assert acquired.accepted.result().state == "accepted"
    assert controller.snapshot().orientation_target_yaw == pytest.approx(.4)


def test_duplicate_acquire_keeps_an_accepted_return_ticket(controller):
    initial = controller.submit(orient(target_yaw=.27))
    controller.tick_once(now=1.)
    assert initial.completed.result().code == "aligned"
    returned = controller.submit(request("orientation.return_center", "return"))
    controller.tick_once(now=2.)
    assert returned.accepted.result().state == "accepted"

    duplicate = controller.submit(orient("duplicate", target_yaw=.4))
    controller.tick_once(now=2.01)

    assert duplicate.completed.done()
    assert duplicate.completed.result().code == "duplicate"
    assert not returned.completed.done()
    for index in range(100):
        controller.tick_once(now=2.02 + index / 100)
        if returned.completed.done():
            break
    assert returned.completed.result().code == "centered"


def test_duplicate_acquire_during_autonomous_disconnect_return_is_terminal(controller):
    initial = controller.submit(orient(target_yaw=.27))
    controller.tick_once(now=1.)
    assert initial.completed.result().code == "aligned"
    controller.remote_disconnected()
    controller.tick_once(now=2.)
    controller.tick_once(now=12.)
    assert controller.snapshot().orientation_state == "returning"
    target_before = controller.runtime.orientation_snapshot().target_yaw

    duplicate = controller.submit(orient("reconnect-returning", target_yaw=.4))
    controller.tick_once(now=12.01)

    assert duplicate.completed.done()
    assert duplicate.completed.result().code == "duplicate"
    assert controller.snapshot().orientation_state == "returning"
    assert controller.runtime.orientation_snapshot().target_yaw == target_before


def test_duplicate_acquire_after_autonomous_disconnect_center_is_terminal(controller):
    initial = controller.submit(orient(target_yaw=.27))
    controller.tick_once(now=1.)
    assert initial.completed.result().code == "aligned"
    controller.remote_disconnected()
    controller.tick_once(now=2.)
    controller.tick_once(now=12.)
    for index in range(100):
        controller.tick_once(now=12.01 + index / 100)
        if controller.snapshot().orientation_state == "centered":
            break
    assert controller.snapshot().orientation_state == "centered"
    target_before = controller.runtime.orientation_snapshot().target_yaw

    duplicate = controller.submit(orient("reconnect-centered", target_yaw=.4))
    controller.tick_once(now=13.01)

    assert duplicate.completed.done()
    assert duplicate.completed.result().code == "duplicate"
    assert controller.snapshot().orientation_state == "centered"
    assert controller.runtime.orientation_snapshot().target_yaw == target_before


@pytest.mark.parametrize("previous_anchor", [None, -.4, "returning"])
@pytest.mark.parametrize("motion_name", ["headshake", "idle"])
def test_acquiring_during_primitive_refits_yaw_before_commanding(controller, previous_anchor, motion_name):
    """A new anchor must not reuse yaw offsets fitted for a previous base pose."""
    if previous_anchor == "returning":
        controller.submit(request("orientation.return_center", "return"))
        controller.tick_once(now=0.)
    elif previous_anchor is not None:
        controller.submit(orient("old", speech_id="old", target_yaw=previous_anchor))
        controller.tick_once(now=0.)
    motion = controller.submit(play("motion", name=motion_name))
    controller.tick_once(now=1.)
    lo, hi = controller.runtime.orientation_safe_yaw_limits()
    acquired = controller.submit(orient(target_yaw=hi - .001))
    commands = []
    for index in range(1500):
        controller.tick_once(now=1.01 + index / 100)
        commands.append(controller.runtime.backend.measured()[0])
    assert acquired.accepted.result().state == "accepted"
    assert min(commands) >= lo - 1e-9
    assert max(commands) <= hi + 1e-9
    if motion.completed.done():
        assert motion.completed.result().state == "completed"
        assert motion.completed.result().data["yaw_scale"] < 1.
    else:
        assert controller.snapshot().active_motion == motion_name
        assert controller.snapshot().primitive_yaw_scale < 1.


@pytest.mark.parametrize("anchor_state", ["timeout", "returning", "centered"])
@pytest.mark.parametrize("endpoint", [0, 1])
def test_retained_anchor_fits_repeated_primitive_commands(anchor_state, endpoint):
    """Lifecycle states must not disable the margin of an active absolute layer."""
    from motion.hardware_alignment import HardwareAlignment
    limits = HardwareAlignment.load().joint_limits[0]
    target = limits[endpoint] + (1 if endpoint == 0 else -1) * (np.deg2rad(5) + .001)
    catalog = MotionCatalog.load(RECORDINGS_DIR / "catalog.toml")
    runtime = MotionRuntime(primitives=catalog.library(), idle_cfg=IdleConfig(enabled=False),
                            orientation_cfg=OrientationConfig(center_yaw=target, acquire_timeout=.02))
    controller = MotionController(runtime, catalog)
    if anchor_state == "timeout":
        acquired = controller.submit(orient(target_yaw=target))
        controller.tick_once(now=0.)
        controller.tick_once(now=.03)
        assert acquired.completed.result().code == "timeout"
    else:
        controller.submit(request("orientation.return_center"))
        controller.tick_once(now=0.)
        if anchor_state == "centered":
            for index in range(500):
                controller.tick_once(now=.01 + index / 100)
    assert controller.snapshot().orientation_state == anchor_state
    motion = controller.submit(request("motion.play", "repeat", name="headshake",
                                       replace_current=False, intensity=1., repeat=2))
    lo, hi = runtime.orientation_safe_yaw_limits()
    commands, starts = [], set()
    for index in range(3000):
        controller.tick_once(now=6. + index / 100)
        commands.append(runtime.backend.measured()[0])
        if runtime.primitive.started_at is not None:
            starts.add(runtime.primitive.started_at)
        if motion.completed.done():
            break
    assert min(commands) >= lo - 1e-9
    assert max(commands) <= hi + 1e-9
    assert len(starts) == 2
    assert motion.completed.result().data["yaw_scale"] < 1.


def test_deadband_requests_never_command_the_requested_displacement(controller):
    """Immediate aligned must hold actual yaw over later ticks and new speech IDs."""
    current = controller.runtime.traj.pos[0]
    for attempt, offset in enumerate([.07, -.07, .05]):
        ticket = controller.submit(orient(str(attempt), speech_id=str(attempt), target_yaw=current + offset))
        controller.tick_once(now=attempt * 4.)
        assert ticket.completed.result().code == "aligned"
        for index in range(300):
            controller.tick_once(now=attempt * 4. + .01 + index / 100)
            assert controller.runtime.backend.measured()[0] == pytest.approx(current, abs=1e-12)
        assert ticket.completed.result().data["target_yaw"] == pytest.approx(current)


@pytest.mark.parametrize("endpoint", [0, 1])
def test_deadband_outside_margin_moves_to_safe_anchor_before_success(endpoint):
    """Safety takes priority over no-move when starting inside the mechanical margin."""
    from motion.hardware_alignment import HardwareAlignment
    from motion.config import REST_POSE
    limits = HardwareAlignment.load().joint_limits[0]
    safe = limits[endpoint] + (1 if endpoint == 0 else -1) * np.deg2rad(5)
    initial = REST_POSE.copy()
    initial[0] = safe + (-.02 if endpoint == 0 else .02)
    catalog = MotionCatalog.load(RECORDINGS_DIR / "catalog.toml")
    runtime = MotionRuntime(initial_pose=initial, primitives=catalog.library(), idle_cfg=IdleConfig(enabled=False))
    controller = MotionController(runtime, catalog)
    ticket = controller.submit(orient(target_yaw=initial[0]))
    controller.tick_once(now=0.)
    assert not ticket.completed.done()
    commands = []
    for index in range(300):
        controller.tick_once(now=.01 + index / 100)
        commands.append(runtime.backend.measured()[0])
    assert ticket.completed.result().code == "aligned"
    assert ticket.completed.result().data["clamped"] is True
    assert commands[-1] == pytest.approx(safe, abs=1e-9)
    assert min(initial[0], safe) - 1e-9 <= min(commands)
    assert max(commands) <= max(initial[0], safe) + 1e-9


def test_deadband_retarget_during_return_waits_for_existing_velocity_to_settle(controller):
    """A nearby request cannot claim no-move success while yaw is already moving."""
    controller.submit(orient(target_yaw=.8))
    for index in range(300):
        controller.tick_once(now=index / 100)
    returned = controller.submit(request("orientation.return_center", "return"))
    for index in range(30):
        controller.tick_once(now=3. + index / 100)
    assert abs(controller.runtime.traj.vel[0]) > .08
    current = controller.runtime.traj.pos[0]
    acquired = controller.submit(orient("new", speech_id="new", target_yaw=current + .02))
    controller.tick_once(now=3.3)
    assert returned.completed.result().code == "replaced"
    assert not acquired.completed.done()
    for index in range(300):
        controller.tick_once(now=3.31 + index / 100)
    assert acquired.completed.result().code == "aligned"
    assert controller.runtime.backend.measured()[0] == pytest.approx(current, abs=1e-9)


def test_replaced_motion_result_retains_outgoing_yaw_scale(controller):
    controller.submit(orient(target_yaw=controller.runtime.orientation_safe_yaw_limits()[1] - .01))
    controller.tick_once(now=0.)
    outgoing = controller.submit(play("outgoing", name="headshake"))
    controller.tick_once(now=.01)
    outgoing_scale = controller.snapshot().primitive_yaw_scale
    assert 0. < outgoing_scale < .2
    incoming = controller.submit(request("motion.play", "incoming", name="headshake",
                                         replace_current=True, intensity=0., repeat=1))
    controller.tick_once(now=.02)
    assert incoming.accepted.result().state == "accepted"
    assert controller.snapshot().primitive_yaw_scale == 1.
    assert outgoing.completed.result().code == "replaced"
    assert outgoing.completed.result().data["yaw_scale"] == outgoing_scale


def test_replacement_load_failure_faults_without_falsely_reporting_replaced(controller, tmp_path):
    outgoing = controller.submit(play("outgoing"))
    controller.tick_once(now=0.)
    library = controller.runtime.primitive.lib
    library.recording_paths = dict(library.recording_paths)
    library.recording_paths["headshake"] = tmp_path / "missing.csv"
    incoming = controller.submit(request("motion.play", "incoming", name="headshake",
                                         replace_current=True, intensity=0., repeat=1))
    controller.tick_once(now=.01)
    assert controller.snapshot().state == "fault"
    assert outgoing.completed.result().code == "fault"
    assert incoming.completed.result().code == "fault"


@pytest.mark.parametrize("endpoint", [0, 1])
def test_outward_momentum_cannot_violate_acceleration_when_orientation_tightens(endpoint):
    """An infeasible margin change faults without sending a fabricated instant stop."""
    from motion.config import REST_POSE
    from motion.hardware_alignment import HardwareAlignment
    sign = -1 if endpoint == 0 else 1
    hard = HardwareAlignment.load().joint_limits
    initial, rest = REST_POSE.copy(), REST_POSE.copy()
    initial[0] = hard[0, endpoint] - sign * .185
    rest[0] = hard[0, endpoint] - sign * .001
    catalog = MotionCatalog.load(RECORDINGS_DIR / "catalog.toml")
    runtime = MotionRuntime(initial_pose=initial, rest_pose=rest,
                            primitives=catalog.library(), idle_cfg=IdleConfig(enabled=False))
    controller = MotionController(runtime, catalog)
    safe = runtime.orientation_safe_yaw_limits()[endpoint]
    for index in range(500):
        controller.tick_once(now=index / 100)
        assert np.all(np.abs(runtime.traj.acc) <= runtime.traj.amax + 1e-8)
        if sign * (runtime.traj.pos[0] - safe) > .012:
            break
    assert sign * runtime.traj.vel[0] > .1
    before = runtime.traj.state
    output = runtime.backend.measured().copy()
    sent_ticks = controller.snapshot().sent_ticks
    ticket = controller.submit(orient(target_yaw=runtime.traj.pos[0]), origin="local")
    controller.tick_once(now=(index + 1) / 100)
    # Check the physical invariant first so the original bug exposes its
    # measured acceleration, rather than failing only a status assertion.
    assert np.all(np.abs(runtime.traj.acc) <= runtime.traj.amax + 1e-8)
    for actual, expected in zip(runtime.traj.state, before):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(runtime.backend.measured(), output)
    np.testing.assert_array_equal(runtime.traj.position_limits, hard)
    assert controller.snapshot().sent_ticks == sent_ticks
    assert controller.snapshot().state == "fault"
    assert ticket.accepted.result().state == "accepted"
    assert ticket.completed.result().code == "fault"
    assert controller.submit(play("after-fault")).completed.result().code == "fault"


@pytest.mark.parametrize("endpoint", [0, 1])
def test_feasible_orientation_tightening_preserves_outward_motion_limits(endpoint):
    """A moving state with braking room must still acquire and settle normally."""
    from motion.config import REST_POSE
    from motion.hardware_alignment import HardwareAlignment
    sign = -1 if endpoint == 0 else 1
    hard = HardwareAlignment.load().joint_limits
    initial, rest = REST_POSE.copy(), REST_POSE.copy()
    initial[0] = hard[0, endpoint] - sign * .185
    rest[0] = hard[0, endpoint] - sign * .001
    catalog = MotionCatalog.load(RECORDINGS_DIR / "catalog.toml")
    runtime = MotionRuntime(initial_pose=initial, rest_pose=rest,
                            primitives=catalog.library(), idle_cfg=IdleConfig(enabled=False))
    controller = MotionController(runtime, catalog)
    for index in range(2):
        controller.tick_once(now=index / 100)
    assert sign * runtime.traj.vel[0] > 0.
    before = runtime.traj.pos[0]
    target = runtime.orientation_safe_yaw_limits()[endpoint]
    ticket = controller.submit(orient(target_yaw=target), origin="local")
    lo, hi = runtime.orientation_safe_yaw_limits()
    for index in range(300):
        controller.tick_once(now=.05 + index / 100)
        assert controller.snapshot().fault is None
        assert np.all(np.abs(runtime.traj.acc) <= runtime.traj.amax + 1e-8)
        assert np.all(np.abs(runtime.traj.vel) <= runtime.traj.vmax + 1e-8)
        assert lo - 1e-9 <= runtime.backend.measured()[0] <= hi + 1e-9
    assert sign * (runtime.backend.measured()[0] - before) > .05
    assert ticket.completed.result().code == "aligned"
    np.testing.assert_array_equal(runtime.traj.position_limits, hard)
