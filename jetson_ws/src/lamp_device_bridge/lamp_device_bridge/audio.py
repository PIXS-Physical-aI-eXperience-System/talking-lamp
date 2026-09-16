"""Validated PCM frame contract and lazy GStreamer app adapters."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Mapping
from uuid import UUID


FRAME_BYTES = 640
FRAME_DURATION_NS = 20_000_000
RTP_CLOCK_RATE = 48_000
RTP_FRAME_TICKS = 960
RTP_MODULUS = 1 << 32


class AudioFrameError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class PlaybackSession:
    """Thread-safe terminal signal that retains the first frame failure."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._error: AudioFrameError | None = None

    @property
    def error(self) -> AudioFrameError | None:
        with self._lock:
            return self._error

    def reset(self) -> None:
        with self._lock:
            self._error = None
            self._event.clear()

    def fail(self, error: AudioFrameError) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
            self._event.set()

    def finish(self) -> None:
        self._event.set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


@dataclass(frozen=True)
class CaptureChunk:
    data: bytes
    pts: int
    rtp_timestamp: int
    speech_id: str


def _rtp_delta(value: int, reference: int) -> int:
    return ((value - reference + (RTP_MODULUS // 2)) % RTP_MODULUS) - (
        RTP_MODULUS // 2)


class CaptureSpeechCorrelator:
    """Delay capture briefly so TCP VAD markers can label RTP frames."""

    def __init__(self, *, pre_roll_frames: int = 10) -> None:
        if isinstance(pre_roll_frames, bool) or not isinstance(pre_roll_frames, int) or pre_roll_frames < 1:
            raise AudioFrameError("invalid_config", "pre_roll_frames must be positive")
        self.pre_roll_frames = pre_roll_frames
        self._pending: deque[tuple[bytes, int, int]] = deque()
        self._completed: deque[tuple[int, int, str]] = deque(maxlen=8)
        self._active: tuple[int, str] | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _timestamp(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < RTP_MODULUS:
            raise AudioFrameError("invalid_activity", "RTP timestamp must be uint32")
        return value

    def activity(self, active: bool, speech_id: str, *, rtp_timestamp: int) -> None:
        timestamp = self._timestamp(rtp_timestamp)
        if not isinstance(active, bool) or not _uuid(speech_id):
            raise AudioFrameError("invalid_activity", "activity requires bool and canonical speech_id")
        with self._lock:
            if active:
                self._active = (timestamp, speech_id)
                return
            if self._active is None or self._active[1] != speech_id:
                raise AudioFrameError("invalid_activity", "VAD end does not match active speech")
            start, ident = self._active
            self._completed.append((start, timestamp, ident))
            self._active = None

    def _speech_id(self, timestamp: int) -> str:
        pre_roll = self.pre_roll_frames * RTP_FRAME_TICKS
        for start, end, ident in reversed(self._completed):
            if _rtp_delta(timestamp, (start - pre_roll) % RTP_MODULUS) >= 0 and _rtp_delta(
                timestamp, end) <= 0:
                return ident
        if self._active is not None:
            start, ident = self._active
            if _rtp_delta(timestamp, (start - pre_roll) % RTP_MODULUS) >= 0:
                return ident
        return ""

    def push(self, data: bytes, *, pts: int, rtp_timestamp: int) -> list[CaptureChunk]:
        timestamp = self._timestamp(rtp_timestamp)
        if not isinstance(data, bytes) or not isinstance(pts, int) or pts < 0:
            raise AudioFrameError("invalid_capture", "capture frame metadata is invalid")
        with self._lock:
            self._pending.append((data, pts, timestamp))
            if len(self._pending) <= self.pre_roll_frames:
                return []
            payload, frame_pts, frame_timestamp = self._pending.popleft()
            return [CaptureChunk(
                payload, frame_pts, frame_timestamp, self._speech_id(frame_timestamp))]


@dataclass(frozen=True)
class PcmFrame:
    stream_id: str
    speech_id: str
    sequence: int
    data: bytes
    end_of_stream: bool


def _uuid(value: object, *, empty: bool = False) -> bool:
    if empty and value == "":
        return True
    if not isinstance(value, str):
        return False
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


class PcmFrameValidator:
    def __init__(self, stream_id: str) -> None:
        if not _uuid(stream_id):
            raise AudioFrameError("invalid_stream", "stream_id must be a canonical UUID")
        self.stream_id = stream_id
        self.next_sequence = 0
        self.ended = False

    def accept(self, value: object) -> PcmFrame:
        if self.ended:
            raise AudioFrameError("stream_ended", "audio stream already ended")
        fields = {
            "stream_id", "speech_id", "sequence", "sample_rate", "channels",
            "encoding", "data", "end_of_stream",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise AudioFrameError("invalid_frame", "audio frame fields are invalid")
        if value["stream_id"] != self.stream_id:
            raise AudioFrameError("wrong_stream", "frame belongs to another stream")
        if not _uuid(value["speech_id"], empty=True):
            raise AudioFrameError("invalid_frame", "speech_id must be empty or a canonical UUID")
        sequence = value["sequence"]
        if (
            isinstance(sequence, bool) or not isinstance(sequence, int)
            or sequence != self.next_sequence
        ):
            raise AudioFrameError("out_of_order", "audio sequence is not the next frame")
        if (
            value["sample_rate"] != 16000 or isinstance(value["sample_rate"], bool)
            or value["channels"] != 1 or isinstance(value["channels"], bool)
            or value["encoding"] != "pcm_s16le"
        ):
            raise AudioFrameError("invalid_format", "audio must be 16 kHz mono pcm_s16le")
        try:
            data = bytes(value["data"])
        except (TypeError, ValueError) as exc:
            raise AudioFrameError("invalid_frame", "audio data must contain bytes") from exc
        eos = value["end_of_stream"]
        if not isinstance(eos, bool):
            raise AudioFrameError("invalid_frame", "end_of_stream must be boolean")
        if (eos and data) or (not eos and len(data) != FRAME_BYTES):
            raise AudioFrameError(
                "invalid_frame", "PCM frames are 640 bytes and EOS has no data")
        self.next_sequence += 1
        self.ended = eos
        return PcmFrame(
            self.stream_id, value["speech_id"], sequence, data, eos)  # type: ignore[arg-type]


def capture_receiver_pipeline(
    *, bind_host: str = "192.168.100.1", port: int = 5004, jitter_ms: int = 40,
) -> tuple[str, ...]:
    return (
        "udpsrc", f"address={bind_host}", f"port={port}",
        "caps=application/x-rtp,media=audio,encoding-name=OPUS,payload=96,clock-rate=48000",
        "!", "rtpjitterbuffer", f"latency={jitter_ms}", "drop-on-latency=true",
        "!", "rtpopusdepay", "!", "opusdec", "!", "audioconvert", "!", "audioresample",
        "!", "audio/x-raw,format=S16LE,rate=16000,channels=1",
        "!", "appsink", "name=capture", "emit-signals=true", "max-buffers=10",
        "drop=true", "sync=false",
    )


def playback_sender_pipeline(
    *, bind_host: str = "192.168.100.1", pi_host: str = "192.168.100.2",
    port: int = 5006,
) -> tuple[str, ...]:
    return (
        "appsrc", "name=playback", "is-live=true", "format=time",
        "caps=audio/x-raw,format=S16LE,rate=16000,channels=1,layout=interleaved",
        "!", "queue", "max-size-buffers=10", "leaky=downstream",
        "!", "audioconvert", "!", "audioresample",
        "!", "opusenc", "frame-size=20", "bitrate=32000", "inband-fec=true",
        "!", "rtpopuspay", "pt=96",
        "!", "udpsink", f"host={pi_host}", f"port={port}",
        f"bind-address={bind_host}", "sync=false", "async=false",
    )


def _gst():
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ImportError, ValueError) as exc:
        raise AudioFrameError("dependency_missing", "GStreamer GI bindings are required") from exc
    Gst.init(None)
    return Gst


class CaptureReceiver:
    """GI appsink wrapper; callback receives PCM bytes, PTS and RTP clock."""

    def __init__(self, callback, **pipeline_options) -> None:
        if not callable(callback):
            raise AudioFrameError("invalid_config", "capture callback must be callable")
        self.callback = callback
        self.Gst = _gst()
        self.pipeline = self.Gst.parse_launch(" ".join(
            capture_receiver_pipeline(**pipeline_options)))
        self.sink = self.pipeline.get_by_name("capture")
        if self.sink is None:
            raise AudioFrameError("pipeline_failed", "capture appsink is missing")
        self.sink.connect("new-sample", self._sample)

    def _sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        ok, info = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            return self.Gst.FlowReturn.ERROR
        try:
            pts = int(buffer.pts)
            rtp_timestamp = int(pts * RTP_CLOCK_RATE / 1_000_000_000) & 0xFFFFFFFF
            self.callback(bytes(info.data), pts, rtp_timestamp)
        finally:
            buffer.unmap(info)
        return self.Gst.FlowReturn.OK

    def start(self) -> None:
        if self.pipeline.set_state(self.Gst.State.PLAYING) == self.Gst.StateChangeReturn.FAILURE:
            raise AudioFrameError("pipeline_failed", "capture pipeline failed to start")

    def close(self) -> None:
        self.pipeline.set_state(self.Gst.State.NULL)


class PlaybackSender:
    """GI appsrc wrapper driven by validated 20 ms ROS PCM frames."""

    def __init__(self, stream_id: str, **pipeline_options) -> None:
        self.validator = PcmFrameValidator(stream_id)
        self.Gst = _gst()
        self.pipeline = self.Gst.parse_launch(" ".join(
            playback_sender_pipeline(**pipeline_options)))
        self.source = self.pipeline.get_by_name("playback")
        if self.source is None:
            raise AudioFrameError("pipeline_failed", "playback appsrc is missing")

    def start(self) -> None:
        if self.pipeline.set_state(self.Gst.State.PLAYING) == self.Gst.StateChangeReturn.FAILURE:
            raise AudioFrameError("pipeline_failed", "playback pipeline failed to start")

    def push(self, value: object) -> PcmFrame:
        frame = self.validator.accept(value)
        if frame.end_of_stream:
            result = self.source.emit("end-of-stream")
        else:
            buffer = self.Gst.Buffer.new_allocate(None, len(frame.data), None)
            buffer.fill(0, frame.data)
            buffer.pts = frame.sequence * FRAME_DURATION_NS
            buffer.duration = FRAME_DURATION_NS
            result = self.source.emit("push-buffer", buffer)
        if result != self.Gst.FlowReturn.OK:
            raise AudioFrameError("pipeline_failed", f"appsrc returned {result}")
        return frame

    def close(self) -> None:
        self.pipeline.set_state(self.Gst.State.NULL)
