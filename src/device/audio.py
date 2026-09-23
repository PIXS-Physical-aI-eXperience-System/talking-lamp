"""GStreamer RTP audio pipelines and process lifecycle for Raspberry Pi."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import os
import re
import signal
import time
from typing import Any, Awaitable, Callable, Mapping
from uuid import UUID


class AudioError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


_CARD_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class AudioConfig:
    alsa_card: str = "L16K6Ch"
    pi_host: str = "192.168.100.2"
    jetson_host: str = "192.168.100.1"
    capture_port: int = 5004
    playback_port: int = 5006
    jitter_ms: int = 40
    drain_timeout: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.alsa_card, str) or not _CARD_ID.fullmatch(self.alsa_card):
            raise AudioError("invalid_config", "alsa_card must be a stable ALSA card ID")
        if not isinstance(self.pi_host, str) or not self.pi_host:
            raise AudioError("invalid_config", "pi_host must not be empty")
        if not isinstance(self.jetson_host, str) or not self.jetson_host:
            raise AudioError("invalid_config", "jetson_host must not be empty")
        for name in ("capture_port", "playback_port"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
                raise AudioError("invalid_config", f"{name} must be in 1..65535")
        if (
            isinstance(self.jitter_ms, bool)
            or not isinstance(self.jitter_ms, int)
            or not 1 <= self.jitter_ms <= 2000
        ):
            raise AudioError("invalid_config", "jitter_ms must be in 1..2000")
        if (
            isinstance(self.drain_timeout, bool)
            or not isinstance(self.drain_timeout, (int, float))
            or not math.isfinite(self.drain_timeout)
            or self.drain_timeout <= 0
        ):
            raise AudioError("invalid_config", "drain_timeout must be finite and positive")


@dataclass(frozen=True)
class AudioStatus:
    capture_running: bool
    playback_running: bool
    stream_id: str | None
    state: str
    code: str
    message: str


def _rtp_timestamp(packet: bytes) -> int:
    if len(packet) < 12 or packet[0] >> 6 != 2:
        raise AudioError("invalid_rtp", "capture probe received an invalid RTP header")
    return int.from_bytes(packet[4:8], "big")


class CaptureRtpClock:
    def __init__(self) -> None:
        self._anchor: tuple[float, int] | None = None

    def reset(self) -> None:
        self._anchor = None

    def observe(self, packet: bytes, received_at: float) -> None:
        if not math.isfinite(received_at):
            raise AudioError("invalid_timestamp", "RTP observation time must be finite")
        self._anchor = (received_at, _rtp_timestamp(packet))

    def at(self, now: float) -> int:
        if self._anchor is None:
            raise AudioError("capture_not_running", "capture RTP clock has no packet anchor")
        received_at, timestamp = self._anchor
        if not math.isfinite(now) or now < received_at:
            raise AudioError("invalid_timestamp", "capture timestamp precedes RTP observation")
        return (timestamp + int((now - received_at) * 48_000)) & 0xFFFFFFFF


class _CaptureProbe(asyncio.DatagramProtocol):
    def __init__(self, callback: Callable[[bytes], None]) -> None:
        self.callback = callback

    def datagram_received(self, data: bytes, _address) -> None:
        self.callback(data)


def capture_pipeline(config: AudioConfig, *, probe_port: int) -> tuple[str, ...]:
    if isinstance(probe_port, bool) or not isinstance(probe_port, int) or not 1 <= probe_port <= 65535:
        raise AudioError("invalid_config", "probe_port must be in 1..65535")
    device = f"plughw:CARD={config.alsa_card},DEV=0"
    return (
        "gst-launch-1.0", "-q", "-e",
        "alsasrc", f"device={device}", "buffer-time=40000",
        "!", "audio/x-raw,format=S32LE,rate=16000,channels=6",
        "!", "audioconvert", "!", "audioresample",
        "!", "audio/x-raw,format=S16LE,rate=16000,channels=1",
        "!", "opusenc", "frame-size=20", "bitrate=32000", "inband-fec=true",
        "!", "rtpopuspay", "pt=96", "timestamp-offset=0",
        "!", "tee", "name=capture_rtp",
        "capture_rtp.", "!", "queue", "!", "udpsink",
        f"host={config.jetson_host}", f"port={config.capture_port}",
        f"bind-address={config.pi_host}", "sync=false", "async=false",
        "capture_rtp.", "!", "queue", "!", "udpsink",
        "host=127.0.0.1", f"port={probe_port}", "sync=false", "async=false",
    )


def playback_pipeline(config: AudioConfig) -> tuple[str, ...]:
    device = f"plughw:CARD={config.alsa_card},DEV=0"
    return (
        "gst-launch-1.0", "-q", "-e",
        "udpsrc", f"address={config.pi_host}", f"port={config.playback_port}",
        "caps=application/x-rtp,media=audio,encoding-name=OPUS,payload=96,clock-rate=48000",
        "!", "rtpjitterbuffer", f"latency={config.jitter_ms}", "drop-on-latency=true",
        "!", "rtpopusdepay", "!", "opusdec", "!", "audioconvert", "!", "audioresample",
        "!", "audio/x-raw,format=S16LE,rate=16000,channels=2",
        "!", "alsasink", f"device={device}", "sync=true",
    )


async def _spawn(argv: tuple[str, ...]):
    return await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )


def _canonical_uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def _stream_metadata(metadata: object) -> str:
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "stream_id", "sample_rate", "channels", "encoding",
    }:
        raise AudioError("invalid_stream", "audio stream metadata fields are invalid")
    stream_id = metadata["stream_id"]
    if (
        not _canonical_uuid(stream_id)
        or metadata["sample_rate"] != 16000
        or isinstance(metadata["sample_rate"], bool)
        or metadata["channels"] != 1
        or isinstance(metadata["channels"], bool)
        or metadata["encoding"] != "pcm_s16le"
    ):
        raise AudioError(
            "invalid_stream", "audio requires UUID, 16 kHz mono pcm_s16le metadata")
    assert isinstance(stream_id, str)
    return stream_id


class AudioSupervisor:
    """Own one continuous capture and at most one drainable playback process."""

    def __init__(
        self,
        config: AudioConfig,
        *,
        process_factory: Callable[[tuple[str, ...]], Awaitable[Any]] = _spawn,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            not isinstance(config, AudioConfig) or not callable(process_factory)
            or not callable(clock)
        ):
            raise AudioError("invalid_config", "audio config and process factory are required")
        self.config = config
        self.process_factory = process_factory
        self.clock = clock
        self._capture = None
        self._playback = None
        self._stream_id: str | None = None
        self._closed = False
        self._rtp_clock = CaptureRtpClock()
        self._probe_transport = None
        self._last = AudioStatus(False, False, None, "idle", "idle", "")

    @property
    def status(self) -> AudioStatus:
        capture = self._capture is not None and self._capture.returncode is None
        playback = self._playback is not None and self._playback.returncode is None
        return AudioStatus(
            capture, playback, self._stream_id if playback else None,
            self._last.state, self._last.code, self._last.message,
        )

    async def start(self) -> AudioStatus:
        if self._closed:
            raise AudioError("closed", "audio supervisor is closed")
        if self._capture is not None and self._capture.returncode is None:
            return self.status
        try:
            if self._probe_transport is None:
                loop = asyncio.get_running_loop()
                self._probe_transport, _ = await loop.create_datagram_endpoint(
                    lambda: _CaptureProbe(lambda packet: self.observe_capture_rtp(packet)),
                    local_addr=("127.0.0.1", 0),
                )
            address = self._probe_transport.get_extra_info("sockname")
            if not isinstance(address, tuple) or not isinstance(address[1], int):
                raise AudioError("capture_start_failed", "capture probe has no UDP port")
            self._rtp_clock.reset()
            self._capture = await self.process_factory(capture_pipeline(
                self.config, probe_port=address[1]))
        except (OSError, RuntimeError) as exc:
            raise AudioError("capture_start_failed", str(exc)) from exc
        if self._capture.returncode is not None:
            raise AudioError("capture_start_failed", "capture pipeline exited during startup")
        self._last = AudioStatus(True, False, None, "idle", "capture_running", "")
        return self.status

    def observe_capture_rtp(
        self, packet: bytes, *, received_at: float | None = None,
    ) -> None:
        self._rtp_clock.observe(
            packet, float(self.clock()) if received_at is None else float(received_at))

    def capture_rtp_timestamp(self, now: float | None = None) -> int:
        if not self.status.capture_running:
            raise AudioError("capture_not_running", "capture RTP clock is unavailable")
        current = float(self.clock()) if now is None else float(now)
        return self._rtp_clock.at(current)

    async def play_start(self, metadata: object) -> AudioStatus:
        if self._closed:
            raise AudioError("closed", "audio supervisor is closed")
        stream_id = _stream_metadata(metadata)
        if self._playback is not None and self._playback.returncode is None:
            raise AudioError("audio_busy", "another playback stream is active")
        try:
            self._playback = await self.process_factory(playback_pipeline(self.config))
        except (OSError, RuntimeError) as exc:
            raise AudioError("playback_start_failed", str(exc)) from exc
        if self._playback.returncode is not None:
            self._playback = None
            raise AudioError("playback_start_failed", "playback pipeline exited during startup")
        self._stream_id = stream_id
        self._last = AudioStatus(
            self.status.capture_running, True, stream_id, "playing", "playing", "")
        return self.status

    async def play_stop(self, stream_id: str) -> AudioStatus:
        if not _canonical_uuid(stream_id) or self._playback is None or stream_id != self._stream_id:
            raise AudioError("unknown_stream", "playback stream is not active")
        process = self._playback
        if process.returncode is None:
            process.send_signal(signal.SIGINT)
            try:
                returncode = await asyncio.wait_for(
                    process.wait(), timeout=self.config.drain_timeout)
            except TimeoutError as exc:
                process.kill()
                await process.wait()
                self._playback = None
                self._stream_id = None
                self._last = AudioStatus(
                    self.status.capture_running, False, None,
                    "fault", "drain_timeout", "playback did not drain before timeout")
                raise AudioError("drain_timeout", self._last.message) from exc
            if returncode != 0:
                self._playback = None
                self._stream_id = None
                self._last = AudioStatus(
                    self.status.capture_running, False, None,
                    "fault", "playback_failed", f"playback exited with {returncode}")
                raise AudioError("playback_failed", self._last.message)
        self._playback = None
        self._stream_id = None
        self._last = AudioStatus(
            self.status.capture_running, False, None, "idle", "drained", "")
        return self.status

    async def disconnect(self) -> None:
        """Stop only network-owned playback; continuous capture remains live."""
        if self._playback is not None and self._stream_id is not None:
            try:
                await self.play_stop(self._stream_id)
            except AudioError:
                pass

    async def close(self) -> None:
        if self._closed:
            return
        await self.disconnect()
        if self._capture is not None and self._capture.returncode is None:
            self._capture.send_signal(signal.SIGINT)
            try:
                await asyncio.wait_for(
                    self._capture.wait(), timeout=self.config.drain_timeout)
            except TimeoutError:
                self._capture.kill()
                await self._capture.wait()
        self._capture = None
        self._rtp_clock.reset()
        if self._probe_transport is not None:
            self._probe_transport.close()
            self._probe_transport = None
        self._closed = True
        self._last = AudioStatus(False, False, None, "closed", "closed", "")
