"""Controller tests use real layers/trajectory and only fake the motor clock."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from motion.catalog import MotionCatalog
from motion.config import RECORDINGS_DIR
from motion.controller import MotionController
from motion.idle import IdleConfig
from motion.protocol import Request
from motion.runtime import MotionRuntime, NullBackend


def request(kind, ident="a", expires_at=1000., **payload):
    return Request(ident, kind, payload, expires_at)


def play(ident="a", name="nod", **kw):
    return request("motion.play", ident, **dict(name=name, replace_current=False,
                   intensity=1., repeat=1, **kw))


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
