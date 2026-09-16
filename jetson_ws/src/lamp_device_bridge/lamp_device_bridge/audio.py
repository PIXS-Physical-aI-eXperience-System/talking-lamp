"""Validated PCM frame contract and lazy GStreamer app adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from uuid import UUID


FRAME_BYTES = 640
FRAME_DURATION_NS = 20_000_000


class AudioFrameError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


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
    """GI appsink wrapper; callback receives immutable PCM bytes and PTS."""

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
            self.callback(bytes(info.data), int(buffer.pts))
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
