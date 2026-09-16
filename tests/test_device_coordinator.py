import asyncio
from dataclasses import asdict
import math

import pytest

from device.coordinator import DirectionCoordinator
from device.doa import DoaCalibration, DoaDecision


class FakeMotionClient:
    def __init__(self, *, acquire_state="aligned"):
        self.calls = []
        self.acquire_state = acquire_state

    async def request(self, kind, payload, ttl_ms=1000):
        self.calls.append((kind, payload, ttl_ms))
        if kind == "orientation.status":
            data = {
                "state": "idle", "speech_id": None, "target_yaw": None,
                "current_yaw": math.radians(15), "clamped": False, "code": "idle",
                "center_yaw": math.radians(15),
                "safe_yaw_min": math.radians(-80),
                "safe_yaw_max": math.radians(80),
            }
        elif kind == "orientation.acquire":
            data = {
                "state": self.acquire_state,
                "speech_id": payload["speech_id"],
                "target_yaw": payload["target_yaw"],
                "current_yaw": payload["target_yaw"],
                "clamped": False,
                "code": self.acquire_state,
                "center_yaw": math.radians(15),
                "safe_yaw_min": math.radians(-80),
                "safe_yaw_max": math.radians(80),
            }
        elif kind == "orientation.return_center":
            data = {
                "state": "centered", "speech_id": "speech", "target_yaw": math.radians(15),
                "current_yaw": math.radians(15), "clamped": False, "code": "centered",
                "center_yaw": math.radians(15),
                "safe_yaw_min": math.radians(-80),
                "safe_yaw_max": math.radians(80),
            }
        else:
            raise AssertionError(kind)
        return [
            {"state": "accepted", "code": "accepted"},
            {"state": "completed", "code": data["code"], "data": data},
        ]


def decision(*, state="ready", doa=100.0, code="stable"):
    return DoaDecision(
        state=state,
        speech_id="00000000-0000-0000-0000-000000000001",
        doa_deg=doa,
        dispersion_deg=2.0,
        sample_count=9,
        code=code,
        timestamp=4.2,
    )


def test_coordinator_uses_motion_owned_center_and_limits_then_acquires_once():
    async def scenario():
        motion = FakeMotionClient()
        coordinator = DirectionCoordinator(
            motion, DoaCalibration(zero_deg=90.0, direction_sign=1))

        event = await coordinator.handle_decision(decision(doa=100.0))

        assert event.state == "aligned"
        assert event.speech_id == "00000000-0000-0000-0000-000000000001"
        assert event.raw_doa_deg == 100.0
        assert event.relative_rad == pytest.approx(math.radians(10))
        assert event.target_yaw == pytest.approx(math.radians(25))
        assert event.current_yaw == pytest.approx(math.radians(25))
        assert event.clamped is False
        assert motion.calls == [
            ("orientation.status", {}, 1000),
            ("orientation.acquire", {
                "speech_id": event.speech_id,
                "target_yaw": pytest.approx(math.radians(25)),
            }, 1000),
        ]
    asyncio.run(scenario())


def test_coordinator_rejects_unstable_without_motion_call():
    async def scenario():
        motion = FakeMotionClient()
        coordinator = DirectionCoordinator(motion, DoaCalibration(90.0, 1))

        event = await coordinator.handle_decision(
            decision(state="rejected", doa=None, code="unstable"))

        assert event.state == "rejected"
        assert event.code == "unstable"
        assert event.raw_doa_deg is None
        assert motion.calls == []
    asyncio.run(scenario())


def test_coordinator_rejects_rear_direction_before_acquire():
    async def scenario():
        motion = FakeMotionClient()
        coordinator = DirectionCoordinator(motion, DoaCalibration(0.0, 1))

        event = await coordinator.handle_decision(decision(doa=181.0))

        assert event.state == "rejected"
        assert event.code == "rear_direction"
        assert [call[0] for call in motion.calls] == ["orientation.status"]
    asyncio.run(scenario())


def test_coordinator_preserves_motion_timeout_and_explicit_return_order():
    async def scenario():
        motion = FakeMotionClient(acquire_state="timeout")
        coordinator = DirectionCoordinator(motion, DoaCalibration(90.0, 1))
        oriented = await coordinator.handle_decision(decision(doa=100.0))
        assert oriented.state == "timeout"

        centered = await coordinator.return_center()

        assert centered.state == "centered"
        assert [call[0] for call in motion.calls] == [
            "orientation.status", "orientation.acquire", "orientation.return_center"]
    asyncio.run(scenario())


def test_orientation_event_is_immutable_serializable_state():
    async def scenario():
        coordinator = DirectionCoordinator(FakeMotionClient(), DoaCalibration(90.0, 1))
        event = await coordinator.handle_decision(decision(doa=100.0))
        assert asdict(event)["code"] == "aligned"
        with pytest.raises(Exception):
            event.state = "idle"
    asyncio.run(scenario())


def test_coordinator_status_reads_motion_without_changing_last_doa_context():
    async def scenario():
        motion = FakeMotionClient()
        coordinator = DirectionCoordinator(motion, DoaCalibration(90.0, 1), clock=lambda: 9.0)
        aligned = await coordinator.handle_decision(decision(doa=100.0))

        status = await coordinator.status()

        assert status.state == "idle"
        assert status.speech_id == aligned.speech_id
        assert status.raw_doa_deg == 100.0
        assert status.relative_rad == pytest.approx(math.radians(10))
        assert status.current_yaw == pytest.approx(math.radians(15))
        assert status.code == "idle"
        assert status.timestamp == 9.0
        assert [call[0] for call in motion.calls] == [
            "orientation.status", "orientation.acquire", "orientation.status"]
    asyncio.run(scenario())
