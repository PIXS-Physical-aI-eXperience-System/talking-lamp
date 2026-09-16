from uuid import uuid4

import pytest
import lamp_device_bridge.audio as audio_module

from lamp_device_bridge.audio import (
    AudioFrameError,
    PcmFrameValidator,
    capture_receiver_pipeline,
    playback_sender_pipeline,
)


def frame(stream_id, sequence, *, data=None, eos=False, **updates):
    value = {
        "stream_id": stream_id,
        "speech_id": "",
        "sequence": sequence,
        "sample_rate": 16000,
        "channels": 1,
        "encoding": "pcm_s16le",
        "data": bytes(640) if data is None and not eos else (b"" if data is None else data),
        "end_of_stream": eos,
    }
    value.update(updates)
    return value


def test_playback_frame_validator_requires_ordered_20ms_pcm_and_one_eos():
    stream_id = str(uuid4())
    validator = PcmFrameValidator(stream_id)
    first = validator.accept(frame(stream_id, 0))
    second = validator.accept(frame(stream_id, 1, data=bytes([1]) * 640))
    eos = validator.accept(frame(stream_id, 2, eos=True))
    assert len(first.data) == len(second.data) == 640
    assert eos.end_of_stream and eos.data == b""
    with pytest.raises(AudioFrameError) as after_eos:
        validator.accept(frame(stream_id, 3))
    assert after_eos.value.code == "stream_ended"


@pytest.mark.parametrize("change,code", [
    ({"stream_id": str(uuid4())}, "wrong_stream"),
    ({"sequence": 2}, "out_of_order"),
    ({"sample_rate": 48000}, "invalid_format"),
    ({"channels": 2}, "invalid_format"),
    ({"encoding": "opus"}, "invalid_format"),
    ({"data": bytes(639)}, "invalid_frame"),
    ({"end_of_stream": True, "data": b"x"}, "invalid_frame"),
])
def test_playback_frame_validator_rejects_bad_frame_before_gstreamer(change, code):
    stream_id = str(uuid4())
    candidate = frame(stream_id, 0)
    candidate.update(change)
    with pytest.raises(AudioFrameError) as error:
        PcmFrameValidator(stream_id).accept(candidate)
    assert error.value.code == code


def test_jetson_gstreamer_pipelines_use_appsink_appsrc_opus_and_wired_addresses():
    capture = " ".join(capture_receiver_pipeline())
    assert "udpsrc address=192.168.100.1 port=5004" in capture
    assert "rtpjitterbuffer latency=40 drop-on-latency=true" in capture
    assert "identity name=rtp_probe" in capture
    assert "rtpopusdepay ! opusdec" in capture
    assert "audio/x-raw,format=S16LE,rate=16000,channels=1" in capture
    assert "appsink name=capture emit-signals=true max-buffers=10 drop=true" in capture

    playback = " ".join(playback_sender_pipeline())
    assert "appsrc name=playback is-live=true format=time" in playback
    assert "audio/x-raw,format=S16LE,rate=16000,channels=1,layout=interleaved" in playback
    assert "opusenc frame-size=20" in playback
    assert "rtpopuspay pt=96" in playback
    assert "udpsink host=192.168.100.2 port=5006 bind-address=192.168.100.1" in playback


def test_rtp_header_timestamp_reads_wire_clock_instead_of_local_pts():
    packet = bytearray(12)
    packet[0] = 0x80
    packet[4:8] = (0xF1020304).to_bytes(4, "big")

    assert audio_module.rtp_header_timestamp(bytes(packet)) == 0xF1020304
    with pytest.raises(AudioFrameError, match="RTP header"):
        audio_module.rtp_header_timestamp(b"short")


def test_playback_session_preserves_frame_failure_until_action_consumes_it():
    session = audio_module.PlaybackSession()
    failure = AudioFrameError("out_of_order", "missing frame")

    session.reset()
    session.fail(failure)
    session.finish()

    assert session.wait(0)
    assert session.error is failure
    session.reset()
    assert session.error is None
    assert not session.wait(0)


def test_capture_correlator_tags_delayed_preroll_and_clears_after_vad_end():
    correlator = audio_module.CaptureSpeechCorrelator(pre_roll_frames=2)
    speech_id = str(uuid4())

    assert correlator.push(b"a", pts=0, rtp_timestamp=0) == []
    assert correlator.push(b"b", pts=20_000_000, rtp_timestamp=960) == []
    correlator.activity(True, speech_id, rtp_timestamp=960)

    ready = correlator.push(b"c", pts=40_000_000, rtp_timestamp=1920)
    assert [(item.data, item.speech_id) for item in ready] == [(b"a", speech_id)]

    correlator.activity(False, speech_id, rtp_timestamp=1920)
    correlator.push(b"d", pts=60_000_000, rtp_timestamp=2880)
    after_end = correlator.push(b"e", pts=80_000_000, rtp_timestamp=3840)
    final = correlator.push(b"f", pts=100_000_000, rtp_timestamp=4800)

    assert [(item.data, item.speech_id) for item in after_end] == [(b"c", speech_id)]
    assert [(item.data, item.speech_id) for item in final] == [(b"d", "")]
