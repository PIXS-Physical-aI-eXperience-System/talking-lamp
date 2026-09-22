"""Exercise the real ROS callbacks with only ROS/GStreamer/Pi boundaries faked."""
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest
from lamp_device_bridge.audio import PlaybackSession, PcmFrameValidator


@pytest.fixture
def bridge(monkeypatch):
    for name, attributes in {
        'rclpy': [], 'rclpy.action': ['ActionServer', 'CancelResponse'],
        'rclpy.callback_groups': ['ReentrantCallbackGroup'],
        'rclpy.executors': ['ExternalShutdownException', 'MultiThreadedExecutor'],
        'rclpy.node': ['Node'], 'rclpy.qos': ['QoSProfile'],
        'sensor_msgs.msg': ['Image'], 'std_srvs.srv': ['Trigger'],
        'lamp_interfaces.action': ['PlayAudio', 'ReturnCenter'],
        'lamp_interfaces.msg': ['AudioFrame', 'AudioStatus', 'LedStatus', 'OrientationStatus'],
        'lamp_interfaces.srv': ['ListLedExpressions', 'SetLedExpression', 'SetLedSolid'],
    }.items():
        module = ModuleType(name)
        for attribute in attributes:
            setattr(module, attribute, type(attribute, (), {'Result': SimpleNamespace}))
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / 'jetson_ws/src/lamp_device_bridge/lamp_device_bridge/node.py'
    spec = importlib.util.spec_from_file_location('lamp_device_bridge._test_node', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node = module.DeviceBridgeNode.__new__(module.DeviceBridgeNode)
    node._playback_lock = threading.Lock()
    node._playback_owner = None
    node._playback_sender = node._playback_stream = None
    node._playback_session = PlaybackSession()
    node.get_parameter = lambda _: SimpleNamespace(value=5006)
    node.get_logger = lambda: SimpleNamespace(error=lambda _: None)
    senders = []
    class Sender:
        def __init__(self, stream_id, **kwargs):
            self.validator = PcmFrameValidator(stream_id)
            self.frames = []
            self.closed = False
            senders.append(self)
        def start(self):
            pass
        def push(self, frame):
            assert not self.closed
            value = self.validator.accept(frame)
            self.frames.append(value)
            return value
        def close(self):
            self.closed = True
    monkeypatch.setattr(module, 'PlaybackSender', Sender)
    return node, senders


def goal(stream):
    handle = SimpleNamespace(request=SimpleNamespace(
        stream_id=stream, sample_rate=16000, channels=1, encoding='pcm_s16le'),
        is_cancel_requested=False, status=None)
    for name in ('succeed', 'abort', 'canceled'):
        setattr(handle, name, lambda name=name: setattr(handle, 'status', name))
    return handle


@pytest.mark.parametrize('during_start', [False, True])
def test_busy_goal_preserves_first_playback_through_eos(bridge, during_start):
    node, senders = bridge
    first = goal('10000000-0000-0000-0000-000000000001')
    second = goal('10000000-0000-0000-0000-000000000002')
    entered, release, ready = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def request(kind, payload, **kwargs):
        calls.append((kind, payload['stream_id']))
        if kind == 'audio.play.start':
            if payload['stream_id'] == second.request.stream_id:
                return {'state': 'failed', 'code': 'audio_busy'}
            entered.set()
            assert release.wait(3)
        return {'state': 'completed', 'data': {'code': 'drained'}}
    node._request = request
    # Observe the callback entering its frame wait, without a timing-based sleep.
    original_wait = PlaybackSession.wait
    def wait(session, timeout):
        ready.set()
        return original_wait(session, timeout)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PlaybackSession, 'wait', wait)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(node._play_audio, first)
            try:
                assert entered.wait(3)
                if not during_start:
                    release.set()
                    assert ready.wait(3)
                result = node._play_audio(second)
                assert not result.success
                assert second.status == 'abort'
                release.set()
                assert ready.wait(3)
                # First stream must still accept PCM and EOS after rejection.
                for sequence, eos in enumerate((False, True)):
                    node._playback_frame(SimpleNamespace(
                        stream_id=first.request.stream_id, speech_id='', sequence=sequence,
                        sample_rate=16000, channels=1, encoding='pcm_s16le',
                        data=b'' if eos else bytes(640), end_of_stream=eos))
                assert future.result(timeout=3).success
                assert result.code == 'audio_busy'
                assert first.status == 'succeed'
                assert len(senders) == 1 and senders[0].closed
                assert [f.end_of_stream for f in senders[0].frames] == [False, True]
                assert calls == [('audio.play.start', first.request.stream_id),
                                 ('audio.play.stop', first.request.stream_id)]
                assert node._playback_sender is None and node._playback_stream is None
            finally:
                release.set()
                first.is_cancel_requested = True
                node._playback_session.finish()


def test_remote_start_failure_releases_slot_without_stopping_another_stream(bridge):
    node, senders = bridge
    calls = []
    def request(kind, payload, **kwargs):
        calls.append(kind)
        return {'state': 'failed', 'code': 'audio_busy'}
    node._request = request
    for stream in ('10000000-0000-0000-0000-000000000001', '10000000-0000-0000-0000-000000000002'):
        handle = goal(stream)
        result = node._play_audio(handle)
        assert not result.success and result.code == 'audio_busy'
        assert handle.status == 'abort'
    assert calls == ['audio.play.start', 'audio.play.start']
    assert not senders
    assert node._playback_owner is None


def test_cancelled_playback_releases_own_sender_and_allows_next_goal(bridge):
    node, senders = bridge
    calls = []
    def request(kind, payload, **kwargs):
        calls.append((kind, payload['stream_id']))
        return {'state': 'completed', 'data': {'code': 'drained'}}
    node._request = request
    for stream in ('10000000-0000-0000-0000-000000000001', '10000000-0000-0000-0000-000000000002'):
        handle = goal(stream)
        handle.is_cancel_requested = True
        result = node._play_audio(handle)
        assert result.code == 'cancelled' and not result.success
        assert handle.status == 'canceled'
        assert node._playback_sender is None and node._playback_stream is None
    assert len(senders) == 2 and all(sender.closed for sender in senders)
    assert calls == [(kind, stream) for stream in ('10000000-0000-0000-0000-000000000001', '10000000-0000-0000-0000-000000000002')
                     for kind in ('audio.play.start', 'audio.play.stop')]
