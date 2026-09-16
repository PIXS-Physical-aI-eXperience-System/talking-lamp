import asyncio
import signal
from uuid import uuid4

import pytest

from device.audio import (
    AudioConfig,
    AudioError,
    AudioSupervisor,
    capture_pipeline,
    playback_pipeline,
)


def joined(argv):
    return " ".join(argv)


def test_capture_pipeline_uses_stable_card_processed_mono_opus_and_wired_bind():
    command = joined(capture_pipeline(AudioConfig(), probe_port=5010))
    assert command.startswith("gst-launch-1.0 -q -e ")
    assert "alsasrc device=plughw:CARD=L16K6Ch,DEV=0" in command
    assert "audio/x-raw,format=S32LE,rate=16000,channels=6" in command
    assert "audio/x-raw,format=S16LE,rate=16000,channels=1" in command
    assert "opusenc frame-size=20" in command
    assert "rtpopuspay pt=96 timestamp-offset=0" in command
    assert "udpsink host=192.168.100.1 port=5004 bind-address=192.168.100.2" in command
    assert "udpsink host=127.0.0.1 port=5010 sync=false async=false" in command


def test_playback_pipeline_has_exact_opus_caps_jitter_and_stable_alsa_sink():
    command = joined(playback_pipeline(AudioConfig()))
    assert "udpsrc address=192.168.100.2 port=5006" in command
    assert "application/x-rtp,media=audio,encoding-name=OPUS,payload=96,clock-rate=48000" in command
    assert "rtpjitterbuffer latency=40 drop-on-latency=true" in command
    assert "rtpopusdepay ! opusdec" in command
    assert "audio/x-raw,format=S16LE,rate=16000,channels=2" in command
    assert "audio/x-raw,format=S32LE" not in command
    assert "alsasink device=plughw:CARD=L16K6Ch,DEV=0 sync=true" in command


class FakeProcess:
    def __init__(self, *, returncode=None):
        self.returncode = returncode
        self.signals = []
        self.waited = 0

    def send_signal(self, signum):
        self.signals.append(signum)
        self.returncode = 0

    async def wait(self):
        self.waited += 1
        return self.returncode

    def kill(self):
        self.signals.append(signal.SIGKILL)
        self.returncode = -signal.SIGKILL


class ProcessFactory:
    def __init__(self):
        self.calls = []
        self.processes = []

    async def __call__(self, argv):
        self.calls.append(tuple(argv))
        process = FakeProcess()
        self.processes.append(process)
        return process


def metadata(stream_id=None, **updates):
    value = {
        "stream_id": stream_id or str(uuid4()),
        "sample_rate": 16000,
        "channels": 1,
        "encoding": "pcm_s16le",
    }
    value.update(updates)
    return value


def test_supervisor_starts_continuous_capture_and_drains_one_playback_stream():
    async def scenario():
        factory = ProcessFactory()
        audio = AudioSupervisor(AudioConfig(), process_factory=factory)
        started = await audio.start()
        assert started.capture_running
        assert len(factory.calls) == 1

        stream = metadata()
        playing = await audio.play_start(stream)
        assert playing.playback_running
        assert playing.stream_id == stream["stream_id"]
        assert len(factory.calls) == 2

        drained = await audio.play_stop(stream["stream_id"])
        assert drained.state == "idle"
        assert drained.code == "drained"
        assert factory.processes[1].signals == [signal.SIGINT]
        assert factory.processes[1].waited == 1

        await audio.close()
        assert factory.processes[0].signals == [signal.SIGINT]
        assert factory.processes[0].waited == 1

    asyncio.run(scenario())


def test_supervisor_maps_vad_clock_from_observed_wire_rtp_after_delayed_start():
    async def scenario():
        now = [10.0]
        factory = ProcessFactory()
        audio = AudioSupervisor(
            AudioConfig(), process_factory=factory, clock=lambda: now[0])

        await audio.start()
        packet = bytearray(12)
        packet[0] = 0x80
        packet[4:8] = (1_000).to_bytes(4, "big")
        audio.observe_capture_rtp(bytes(packet), received_at=10.5)
        now[0] = 10.75

        assert audio.capture_rtp_timestamp() == 13_000
        assert audio.capture_rtp_timestamp(11.0) == 25_000
        await audio.close()

    asyncio.run(scenario())


def test_supervisor_discards_old_rtp_anchor_when_capture_restarts():
    async def scenario():
        factory = ProcessFactory()
        audio = AudioSupervisor(AudioConfig(), process_factory=factory, clock=lambda: 20.0)

        await audio.start()
        packet = bytearray(12)
        packet[0] = 0x80
        packet[4:8] = (2_000).to_bytes(4, "big")
        audio.observe_capture_rtp(bytes(packet), received_at=19.5)
        assert audio.capture_rtp_timestamp() == 26_000

        factory.processes[0].returncode = 1
        await audio.start()
        with pytest.raises(AudioError, match="capture_not_running"):
            audio.capture_rtp_timestamp()

        packet[4:8] = (7_000).to_bytes(4, "big")
        audio.observe_capture_rtp(bytes(packet), received_at=20.0)
        assert audio.capture_rtp_timestamp() == 7_000
        await audio.close()

    asyncio.run(scenario())


def test_supervisor_rejects_invalid_or_competing_streams_before_process_creation():
    async def scenario():
        factory = ProcessFactory()
        audio = AudioSupervisor(AudioConfig(), process_factory=factory)
        await audio.start()
        for updates in (
            {"stream_id": "not-uuid"}, {"sample_rate": 48000},
            {"channels": 2}, {"encoding": "opus"}, {"extra": True},
        ):
            with pytest.raises(AudioError) as error:
                await audio.play_start(metadata(**updates))
            assert error.value.code == "invalid_stream"
        assert len(factory.calls) == 1

        first = metadata()
        await audio.play_start(first)
        with pytest.raises(AudioError) as busy:
            await audio.play_start(metadata())
        assert busy.value.code == "audio_busy"
        with pytest.raises(AudioError) as unknown:
            await audio.play_stop(str(uuid4()))
        assert unknown.value.code == "unknown_stream"
        assert len(factory.calls) == 2
        await audio.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "updates",
    [
        {"alsa_card": ""}, {"pi_host": ""}, {"jetson_host": ""},
        {"capture_port": 0}, {"playback_port": 65536},
        {"jitter_ms": 0}, {"drain_timeout": float("nan")},
    ],
)
def test_audio_config_rejects_invalid_values(updates):
    with pytest.raises(AudioError, match="invalid_config"):
        AudioConfig(**updates)
