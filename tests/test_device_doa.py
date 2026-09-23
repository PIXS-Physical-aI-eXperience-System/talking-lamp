import math
from uuid import UUID

import pytest

from device.doa import (
    DoaCalibration,
    DoaError,
    DoaSample,
    DoaStabilizer,
    calibrate_target,
    circular_distance_deg,
    circular_mad_deg,
    circular_medoid_deg,
)


def test_circular_medoid_does_not_average_across_180():
    samples = [358.0, 359.0, 0.0, 1.0, 2.0]
    center = circular_medoid_deg(samples)

    assert circular_distance_deg(center, 0.0) <= 1.0
    assert circular_mad_deg(samples, center) == 1.0


def test_circular_medoid_breaks_ties_by_input_order():
    assert circular_medoid_deg([90.0, 270.0]) == 90.0
    assert circular_medoid_deg([270.0, 90.0]) == 270.0


@pytest.mark.parametrize("samples", [[], [math.nan], [math.inf], [True]])
def test_circular_statistics_reject_invalid_samples(samples):
    with pytest.raises(DoaError, match="samples"):
        circular_medoid_deg(samples)


def test_calibration_uses_nonzero_motion_center_and_wraps():
    result = calibrate_target(
        359.0,
        DoaCalibration(zero_deg=1.0, direction_sign=1),
        center_yaw=math.radians(15.0),
        safe_limits=(math.radians(-80.0), math.radians(80.0)),
    )

    assert result.raw_doa_deg == 359.0
    assert result.relative_rad == pytest.approx(math.radians(-2.0))
    assert result.target_yaw == pytest.approx(math.radians(13.0))
    assert result.clamped is False


def test_calibration_applies_direction_sign_and_clamps():
    result = calibrate_target(
        60.0,
        DoaCalibration(zero_deg=90.0, direction_sign=-1),
        center_yaw=math.radians(15.0),
        safe_limits=(math.radians(-10.0), math.radians(35.0)),
    )

    assert result.relative_rad == pytest.approx(math.radians(30.0))
    assert result.target_yaw == pytest.approx(math.radians(35.0))
    assert result.clamped is True


def test_calibration_rejects_rear_direction():
    with pytest.raises(DoaError) as error:
        calibrate_target(
            181.0,
            DoaCalibration(zero_deg=0.0, direction_sign=1),
            center_yaw=0.0,
            safe_limits=(-2.0, 2.0),
        )

    assert error.value.code == "rear_direction"


@pytest.mark.parametrize(
    "calibration, center, limits",
    [
        (DoaCalibration(0.0, 0), 0.0, (-1.0, 1.0)),
        (DoaCalibration(0.0, 1, 181.0), 0.0, (-1.0, 1.0)),
        (DoaCalibration(math.nan, 1), 0.0, (-1.0, 1.0)),
        (DoaCalibration(0.0, 1), math.nan, (-1.0, 1.0)),
        (DoaCalibration(0.0, 1), 0.0, (1.0, -1.0)),
    ],
)
def test_calibration_rejects_invalid_configuration(calibration, center, limits):
    with pytest.raises(DoaError):
        calibrate_target(0.0, calibration, center_yaw=center, safe_limits=limits)


def _ids(*values):
    iterator = iter(UUID(value) for value in values)
    return lambda: next(iterator)


def test_stabilizer_waits_400_ms_then_emits_one_stable_direction():
    stabilizer = DoaStabilizer(
        id_factory=_ids("00000000-0000-0000-0000-000000000001")
    )
    assert stabilizer.observe(DoaSample(0.00, 359.0, True)) is None
    for index, angle in enumerate([0, 1, 359, 2, 0, 1, 359], start=1):
        result = stabilizer.observe(DoaSample(index * 0.05, angle, True))
    assert result is None

    result = stabilizer.observe(DoaSample(0.40, 0.0, True))

    assert result.state == "ready"
    assert result.speech_id == "00000000-0000-0000-0000-000000000001"
    assert result.sample_count == 9
    assert circular_distance_deg(result.doa_deg, 0.0) <= 1.0
    assert result.dispersion_deg <= 1.0
    assert result.code == "stable"
    assert stabilizer.observe(DoaSample(0.45, 20.0, True)) is None


def test_stabilizer_extends_unstable_samples_then_rejects_at_deadline():
    stabilizer = DoaStabilizer(
        id_factory=_ids("00000000-0000-0000-0000-000000000002")
    )
    angles = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0] * 3 + [0.0, 60.0, 120.0]
    result = None
    for index, angle in enumerate(angles):
        result = stabilizer.observe(DoaSample(index * 0.05, angle, True))

    assert result.state == "rejected"
    assert result.code == "unstable"
    assert result.doa_deg is not None
    assert result.dispersion_deg > 12.0


def test_stabilizer_rejects_insufficient_samples_at_deadline():
    stabilizer = DoaStabilizer(
        id_factory=_ids("00000000-0000-0000-0000-000000000003")
    )
    assert stabilizer.observe(DoaSample(0.0, 10.0, True)) is None
    assert stabilizer.observe(DoaSample(0.1, 11.0, True)) is None
    result = stabilizer.observe(DoaSample(1.0, 12.0, False))

    assert result.state == "rejected"
    assert result.code == "insufficient_samples"
    assert result.sample_count == 2


def test_stabilizer_creates_new_id_only_after_falling_then_rising_edge():
    stabilizer = DoaStabilizer(id_factory=_ids(
        "00000000-0000-0000-0000-000000000004",
        "00000000-0000-0000-0000-000000000005",
    ))
    stabilizer.observe(DoaSample(0.0, 10.0, True))
    first_id = stabilizer.speech_id
    stabilizer.observe(DoaSample(0.1, 10.0, True))
    assert stabilizer.speech_id == first_id
    stabilizer.observe(DoaSample(0.2, 10.0, False))
    stabilizer.observe(DoaSample(0.3, 10.0, True))

    assert stabilizer.speech_id == "00000000-0000-0000-0000-000000000005"


@pytest.mark.parametrize(
    "samples",
    [
        [DoaSample(0.0, -1.0, True)],
        [DoaSample(0.0, 360.0, True)],
        [DoaSample(math.nan, 0.0, True)],
        [DoaSample(0.0, 0.0, 1)],
        [DoaSample(0.1, 0.0, False), DoaSample(0.0, 0.0, False)],
    ],
)
def test_stabilizer_rejects_invalid_or_out_of_order_samples(samples):
    stabilizer = DoaStabilizer(
        id_factory=_ids("00000000-0000-0000-0000-000000000006")
    )

    with pytest.raises(DoaError):
        for sample in samples:
            stabilizer.observe(sample)
