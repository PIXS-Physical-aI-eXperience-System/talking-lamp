import importlib

import numpy as np
import pytest

from motion.config import REST_POSE
from motion import trajectory


class FakeClock:
    def __init__(self, oversleeps=()):
        self.now = 0.0
        self.oversleeps = iter(oversleeps)

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds + next(self.oversleeps, 0.0)


class FakeBackend:
    def __init__(self, clock, *, cost=0.002, error=None):
        self.clock = clock
        self.cost = cost
        self.error = error
        self.times = []
        self.commands = []
        self.sent_ticks = 0
        self.clamped_ticks = 0
        self.clamped_joints = {}
        self.closed = False

    def measured(self):
        return REST_POSE + 0.2

    def send(self, q):
        if self.error:
            raise self.error
        self.times.append(self.clock())
        self.commands.append(q.copy())
        self.sent_ticks += 1
        self.clock.now += self.cost

    def close(self):
        self.closed = True


def run_fake(*, duration=0.1, cost=0.002, error=None, clock=None, **kwargs):
    runner = importlib.import_module("motion.hardware_run")
    clock = clock or FakeClock()
    backend = FakeBackend(clock, cost=cost, error=error)
    report = runner.run_hardware(
        port="fake", lamp_id="test", duration=duration,
        allow_analytic_fallback=True, backend_factory=lambda **kw: backend,
        clock=clock, sleep=clock.sleep, **kwargs,
    )
    return report, backend


def test_deadline_schedule_sends_100_hz_without_accumulating_work_time():
    report, backend = run_fake(duration=1.0)
    np.testing.assert_allclose(backend.times, np.arange(100) / 100, atol=1e-12)
    assert report.sent_ticks == 100
    assert report.deadline_misses == 0
    assert backend.closed
    assert np.max(np.abs(backend.commands[0] - (REST_POSE + 0.2))) < 0.003


def test_slow_send_skips_missed_slots_without_burst_catchup():
    report, backend = run_fake(cost=0.025)
    np.testing.assert_allclose(backend.times, [0, 0.03, 0.06, 0.09])
    assert report.deadline_misses == 6
    assert report.sent_ticks == 4
    assert backend.closed


def test_ctrl_c_parks_and_returns_interrupted_report():
    report, backend = run_fake(error=KeyboardInterrupt())
    assert report.interrupted
    assert report.sent_ticks == 0
    assert backend.closed


def test_send_exception_still_closes():
    runner = importlib.import_module("motion.hardware_run")
    clock = FakeClock()
    backend = FakeBackend(clock, error=OSError("serial failure"))
    with pytest.raises(OSError, match="serial failure"):
        runner.run_hardware(port="fake", lamp_id="test", duration=1,
                            allow_analytic_fallback=True,
                            backend_factory=lambda **kw: backend,
                            clock=clock, sleep=clock.sleep)
    assert backend.closed


def test_ruckig_required_before_hardware_connection(monkeypatch):
    runner = importlib.import_module("motion.hardware_run")
    monkeypatch.setattr(trajectory, "_HAVE_RUCKIG", False)
    opened = []
    with pytest.raises(RuntimeError, match="Ruckig"):
        runner.run_hardware(port="fake", lamp_id="test", duration=1,
                            backend_factory=lambda **kw: opened.append(kw))
    assert opened == []


@pytest.mark.parametrize("duration", [-1, 0, float("nan"), float("inf")])
def test_invalid_duration_rejected_before_connection(duration):
    runner = importlib.import_module("motion.hardware_run")
    opened = []
    with pytest.raises(ValueError):
        runner.run_hardware(port="fake", lamp_id="test", duration=duration,
                            backend_factory=lambda **kw: opened.append(kw))
    assert opened == []


def test_cli_passes_options_and_reports_counts(monkeypatch, capsys):
    runner = importlib.import_module("motion.hardware_run")
    received = {}

    def fake_run(**kwargs):
        received.update(kwargs)
        return runner.HardwareRunReport(100, 2, 3, {"elbow_pitch": 3}, False)

    monkeypatch.setattr(runner, "run_hardware", fake_run)
    assert runner.main(["--port", "fake", "--lamp-id", "lamp", "--primitive", "nod",
                        "--duration", "1", "--feedback-hz", "10",
                        "--allow-analytic-fallback"]) == 0
    assert received == dict(port="fake", lamp_id="lamp", primitive="nod", duration=1,
                            feedback_hz=10, allow_analytic_fallback=True)
    text = capsys.readouterr().out
    assert "sent_ticks=100" in text
    assert "deadline_misses=2" in text
    assert "clamped_ticks=3" in text


@pytest.mark.parametrize("overshoot, expected_times, misses", [
    (0.027, [0, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09], 3),
    (0.009, [0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09], 1),
    (0.0205, [0, 0.0305, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09], 2),
    (0.11, [0], 9),
])
def test_sleep_overshoot_discards_expired_slots_before_sending(
    overshoot, expected_times, misses,
):
    report, backend = run_fake(clock=FakeClock([overshoot]))
    np.testing.assert_allclose(backend.times, expected_times, atol=1e-12)
    assert np.all(np.diff(backend.times) >= 0.009 - 1e-12)
    assert report.deadline_misses == misses
    assert report.sent_ticks + report.deadline_misses == 10
    assert backend.closed


def test_normal_sleep_jitter_stays_on_grid_without_false_misses():
    report, backend = run_fake(clock=FakeClock([0.0005] * 9))
    np.testing.assert_allclose(backend.times, [0, 0.0105, 0.0205, 0.0305, 0.0405,
                                              0.0505, 0.0605, 0.0705, 0.0805, 0.0905])
    assert report.sent_ticks == 10
    assert report.deadline_misses == 0
