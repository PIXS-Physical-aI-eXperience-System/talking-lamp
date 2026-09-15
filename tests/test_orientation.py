import numpy as np
import pytest

from motion.blender import BlendContext
from motion.orientation import (
    BaseYawOrientationLayer,
    OrientationConfig,
    OrientationCoordinator,
    OrientationError,
)


LIMITS = np.array([[-1.0, 1.0], [-2, 2], [-2, 2], [-2, 2], [-2, 2]], float)


def make_coordinator(**config_values):
    cfg = OrientationConfig(center_yaw=0.0, yaw_margin=0.1, **config_values)
    return OrientationCoordinator(BaseYawOrientationLayer(cfg, LIMITS), cfg)


def test_orientation_layer_claims_only_base_yaw_and_clamps_with_margin():
    cfg = OrientationConfig(center_yaw=0.1, yaw_margin=np.deg2rad(5))
    layer = BaseYawOrientationLayer(cfg, LIMITS)

    clamped = layer.acquire(2.0)
    out = layer.update(BlendContext(np.zeros(5), 0.01, 0.01))

    assert clamped
    assert out.value[0] == pytest.approx(1.0 - np.deg2rad(5))
    np.testing.assert_array_equal(out.weight, [1, 0, 0, 0, 0])
    assert out.additive is False


def test_orientation_layer_release_returns_inactive_output():
    layer = BaseYawOrientationLayer(OrientationConfig(), LIMITS)
    layer.acquire(0.4)

    layer.release()
    out = layer.update(BlendContext(np.zeros(5), 0.01, 0.01))

    np.testing.assert_array_equal(out.value, np.zeros(5))
    np.testing.assert_array_equal(out.weight, np.zeros(5))
    assert out.gain == 0.0
    assert out.additive is True


@pytest.mark.parametrize(
    ("config_values", "limits"),
    [
        ({"center_yaw": float("nan")}, LIMITS),
        ({"deadband": 0.0}, LIMITS),
        ({"settle_error": 0.0}, LIMITS),
        ({"settle_velocity": 0.0}, LIMITS),
        ({"settle_duration": 0.0}, LIMITS),
        ({"acquire_timeout": 0.0}, LIMITS),
        ({"disconnect_hold": 0.0}, LIMITS),
        ({"yaw_margin": 1.0}, LIMITS),
        ({}, np.zeros((4, 2))),
        ({}, np.array([[1.0, -1.0], [-2, 2], [-2, 2], [-2, 2], [-2, 2]], float)),
        ({}, np.array([[np.nan, 1.0], [-2, 2], [-2, 2], [-2, 2], [-2, 2]], float)),
    ],
)
def test_orientation_layer_rejects_invalid_configuration_or_limits(config_values, limits):
    cfg = OrientationConfig(**config_values)

    with pytest.raises(OrientationError) as error:
        BaseYawOrientationLayer(cfg, limits)

    assert error.value.code in {"invalid_config", "invalid_joint_limits"}


@pytest.mark.parametrize("speech_id", ["", "   ", 42])
def test_coordinator_rejects_missing_or_nonstring_speech_id(speech_id):
    coordinator = make_coordinator()

    with pytest.raises(OrientationError) as error:
        coordinator.acquire(speech_id, 0.4, now=1.0, current_yaw=0.0, task_light_busy=False)

    assert error.value.code == "invalid_speech_id"


def test_coordinator_settles_only_after_continuous_position_and_velocity_window():
    coordinator = make_coordinator(settle_duration=.15)
    first = coordinator.acquire("speech-1", target_yaw=.4, now=1.0,
                                current_yaw=0.0, task_light_busy=False)
    assert first.state == "orienting"

    for index in range(14):
        state = coordinator.observe(now=1.01 + index * .01,
                                    current_yaw=.39, velocity=.01)
        assert state.state == "orienting"
    state = coordinator.observe(now=1.15, current_yaw=.39, velocity=.01)
    assert state.state == "orienting"
    state = coordinator.observe(now=1.16, current_yaw=.39, velocity=.01)
    assert state.state == "aligned"


def test_coordinator_aligns_immediately_inside_deadband():
    coordinator = make_coordinator()

    state = coordinator.acquire("speech-1", .04, now=1.0, current_yaw=0.0, task_light_busy=False)

    assert state.state == "aligned"
    assert state.code == "aligned"
    assert state.target_yaw == pytest.approx(0.)
    assert coordinator.layer.update(BlendContext(np.zeros(5), 1.01, .01)).value[0] == 0.


def test_coordinator_resets_settle_window_after_fast_tick():
    coordinator = make_coordinator(settle_duration=.15)
    coordinator.acquire("speech-1", .4, now=1.0, current_yaw=0.0, task_light_busy=False)
    assert coordinator.observe(now=1.01, current_yaw=.39, velocity=.01).state == "orienting"

    assert coordinator.observe(now=1.10, current_yaw=.39, velocity=.09).state == "orienting"
    assert coordinator.observe(now=1.11, current_yaw=.39, velocity=.01).state == "orienting"
    assert coordinator.observe(now=1.24, current_yaw=.39, velocity=.01).state == "orienting"
    assert coordinator.observe(now=1.25, current_yaw=.39, velocity=.01).state == "orienting"
    assert coordinator.observe(now=1.26, current_yaw=.39, velocity=.01).state == "aligned"


def test_coordinator_times_out_after_acquire_timeout():
    coordinator = make_coordinator(acquire_timeout=2.0)
    coordinator.acquire("speech-1", .4, now=1.0, current_yaw=0.0, task_light_busy=False)

    state = coordinator.observe(now=3.0, current_yaw=0.0, velocity=.2)

    assert state.state == "timeout"
    assert state.code == "timeout"


def test_coordinator_reports_task_light_conflict_without_acquiring_layer():
    coordinator = make_coordinator()

    state = coordinator.acquire("speech-1", .4, now=1.0, current_yaw=0.0, task_light_busy=True)

    assert state.state == "idle"
    assert state.code == "blocked_by_task_light"
    assert state.speech_id is None
    assert state.target_yaw is None


def test_coordinator_reports_task_light_conflict_after_aligned_without_retargeting():
    coordinator = make_coordinator()
    coordinator.acquire("speech-1", .4, now=1.0, current_yaw=.4, task_light_busy=False)

    blocked = coordinator.acquire("speech-2", -.4, now=2.0, current_yaw=.3, task_light_busy=True)

    assert blocked.state == "aligned"
    assert blocked.code == "blocked_by_task_light"
    assert blocked.speech_id == "speech-1"
    assert blocked.target_yaw == pytest.approx(.4)
    assert blocked.current_yaw == pytest.approx(.3)
    assert coordinator.acquire("speech-1", -.4, now=2.1, current_yaw=.3, task_light_busy=False).target_yaw == pytest.approx(.4)


def test_coordinator_reports_task_light_conflict_after_centered_without_retargeting():
    coordinator = make_coordinator(settle_duration=.1)
    coordinator.acquire("speech-1", .4, now=1.0, current_yaw=.4, task_light_busy=False)
    coordinator.return_center(now=2.0, current_yaw=.4, motion_busy=False)
    coordinator.observe(now=2.01, current_yaw=.0, velocity=.01)
    assert coordinator.observe(now=2.11, current_yaw=.0, velocity=.01).state == "centered"

    blocked = coordinator.acquire("speech-2", -.4, now=3.0, current_yaw=.1, task_light_busy=True)

    assert blocked.state == "centered"
    assert blocked.code == "blocked_by_task_light"
    assert blocked.speech_id == "speech-1"
    assert blocked.target_yaw == pytest.approx(0.0)
    assert blocked.current_yaw == pytest.approx(.1)


def test_coordinator_rejects_layer_configuration_mismatch():
    layer_cfg = OrientationConfig(center_yaw=0.0, yaw_margin=.1)
    layer = BaseYawOrientationLayer(layer_cfg, LIMITS)
    coordinator_cfg = OrientationConfig(center_yaw=.2, yaw_margin=.1)

    with pytest.raises(OrientationError) as error:
        OrientationCoordinator(layer, coordinator_cfg)

    assert error.value.code == "config_mismatch"


def test_coordinator_returns_existing_snapshot_for_duplicate_speech_id():
    coordinator = make_coordinator()
    first = coordinator.acquire("speech-1", .4, now=1.0, current_yaw=0.0, task_light_busy=False)

    duplicate = coordinator.acquire("speech-1", -.4, now=1.2, current_yaw=.1, task_light_busy=False)

    assert duplicate == first
    assert duplicate.target_yaw == pytest.approx(.4)


def test_coordinator_release_resets_lifecycle_and_allows_same_speech_id_reacquisition():
    """Leaving the speech ID latched would suppress a legitimate retry after release."""
    coordinator = make_coordinator()
    coordinator.acquire("speech-1", .4, now=1.0, current_yaw=0.0, task_light_busy=False)

    released = coordinator.release()

    assert released.state == "idle"
    assert released.speech_id is None
    assert released.target_yaw is None
    assert released.clamped is False
    assert released.code == "idle"
    np.testing.assert_array_equal(
        coordinator.layer.update(BlendContext(np.zeros(5), 1.1, .01)).weight,
        np.zeros(5),
    )

    reacquired = coordinator.acquire(
        "speech-1", -.4, now=1.2, current_yaw=0.0, task_light_busy=False,
    )

    assert reacquired.state == "orienting"
    assert reacquired.target_yaw == pytest.approx(-.4)
    assert coordinator.layer.update(BlendContext(np.zeros(5), 1.2, .01)).weight[0] == 1.0


def test_coordinator_rejects_return_while_motion_busy():
    coordinator = make_coordinator()
    coordinator.acquire("speech-1", .4, now=1.0, current_yaw=0.0, task_light_busy=False)

    with pytest.raises(OrientationError) as error:
        coordinator.return_center(now=1.1, current_yaw=0.1, motion_busy=True)

    assert error.value.code == "busy"


def test_coordinator_centers_after_return_settle_window_and_latches_speech_id():
    coordinator = make_coordinator(settle_duration=.15)
    coordinator.acquire("speech-1", .4, now=1.0, current_yaw=.4, task_light_busy=False)

    returning = coordinator.return_center(now=2.0, current_yaw=.4, motion_busy=False)
    assert returning.state == "returning"
    for index in range(14):
        state = coordinator.observe(now=2.01 + index * .01, current_yaw=.01, velocity=.01)
        assert state.state == "returning"
    centered = coordinator.observe(now=2.15, current_yaw=.01, velocity=.01)
    assert centered.state == "returning"
    centered = coordinator.observe(now=2.16, current_yaw=.01, velocity=.01)

    assert centered.state == "centered"
    assert centered.speech_id == "speech-1"
    assert centered.target_yaw == pytest.approx(0.0)


def test_coordinator_returns_after_exact_disconnected_hold():
    coordinator = make_coordinator(disconnect_hold=10.0)
    assert coordinator.acquire("speech-1", .2, now=1.0, current_yaw=.2, task_light_busy=False).state == "aligned"

    coordinator.disconnected(now=5.0)
    assert coordinator.observe(now=14.99, current_yaw=.2, velocity=.0).state == "aligned"
    state = coordinator.observe(now=15.0, current_yaw=.2, velocity=.0)

    assert state.state == "returning"
    assert state.code == "returning"


def test_coordinator_disconnect_hold_returns_an_orienting_anchor():
    coordinator = make_coordinator(disconnect_hold=.1, acquire_timeout=2.0)
    assert coordinator.acquire("speech-1", .4, now=0.0, current_yaw=0.0,
                               task_light_busy=False).state == "orienting"

    coordinator.disconnected(now=1.0)
    assert coordinator.observe(now=1.09, current_yaw=0.0, velocity=.2).state == "orienting"

    assert coordinator.observe(now=1.1, current_yaw=0.0, velocity=.2).state == "returning"


def test_coordinator_disconnect_hold_returns_a_timed_out_anchor():
    coordinator = make_coordinator(disconnect_hold=.1, acquire_timeout=.05)
    coordinator.acquire("speech-1", .4, now=0.0, current_yaw=0.0, task_light_busy=False)
    assert coordinator.observe(now=.05, current_yaw=0.0, velocity=.2).state == "timeout"

    coordinator.disconnected(now=1.0)
    assert coordinator.observe(now=1.09, current_yaw=0.0, velocity=.2).state == "timeout"

    assert coordinator.observe(now=1.1, current_yaw=0.0, velocity=.2).state == "returning"
