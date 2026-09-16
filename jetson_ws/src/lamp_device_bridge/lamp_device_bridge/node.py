"""ROS 2 adapters for Pi audio, orientation and LED device control."""

from __future__ import annotations

import os
import threading
import time
from uuid import uuid4

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from lamp_interfaces.action import PlayAudio, ReturnCenter
from lamp_interfaces.msg import AudioFrame, AudioStatus, LedStatus, OrientationStatus
from lamp_interfaces.srv import SetLedSolid
from .audio import AudioFrameError, CaptureReceiver, PlaybackSender, PlaybackSession
from .runner import AsyncRunner
from .transport import DeviceTransport, TransportError


class DeviceBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("lamp_device_bridge")
        self.declare_parameter("pi_host", "192.168.100.2")
        self.declare_parameter("device_port", 8766)
        self.declare_parameter("capture_rtp_port", 5004)
        self.declare_parameter("playback_rtp_port", 5006)
        self.declare_parameter("token_env", "TALKING_LAMP_DEVICE_TOKEN")
        self.declare_parameter("audio_frame_ms", 20)
        self.declare_parameter("jitter_buffer_ms", 40)
        token_name = self.get_parameter("token_env").value
        token = os.environ.get(token_name, "")
        if not token:
            raise RuntimeError(f"required token environment variable is empty: {token_name}")

        self.runner = AsyncRunner()
        self.command_group = ReentrantCallbackGroup()
        self.transport = DeviceTransport(
            self.get_parameter("pi_host").value,
            int(self.get_parameter("device_port").value), token)
        self.runner.submit(self.transport.start()).result(timeout=5)
        latest = QoSProfile(depth=1)
        stream = QoSProfile(depth=10)
        self.capture_pub = self.create_publisher(
            AudioFrame, "/lamp/audio/capture", stream)
        self.audio_status_pub = self.create_publisher(
            AudioStatus, "/lamp/audio_status", latest)
        self.orientation_pub = self.create_publisher(
            OrientationStatus, "/lamp/orientation_status", latest)
        self.led_status_pub = self.create_publisher(
            LedStatus, "/lamp/led/status", latest)
        self.create_subscription(
            AudioFrame, "/lamp/audio/playback_frames", self._playback_frame, stream,
            callback_group=self.command_group)
        self.create_subscription(
            Image, "/lamp/led/frame", self._led_frame, latest,
            callback_group=self.command_group)
        self.create_service(
            SetLedSolid, "/lamp/led/set_solid", self._led_solid,
            callback_group=self.command_group)
        self.create_service(
            Trigger, "/lamp/led/clear", self._led_clear,
            callback_group=self.command_group)
        self.play_action = ActionServer(
            self, PlayAudio, "/lamp/play_audio", execute_callback=self._play_audio,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self.command_group)
        self.center_action = ActionServer(
            self, ReturnCenter, "/lamp/return_center", execute_callback=self._return_center,
            cancel_callback=lambda _: CancelResponse.REJECT,
            callback_group=self.command_group)

        self._capture_stream = str(uuid4())
        self._capture_sequence = 0
        self._speech_id = ""
        self._playback_lock = threading.Lock()
        self._playback_sender = None
        self._playback_stream = None
        self._playback_session = PlaybackSession()
        self.capture = CaptureReceiver(
            self._capture_pcm,
            bind_host="192.168.100.1",
            port=int(self.get_parameter("capture_rtp_port").value),
            jitter_ms=int(self.get_parameter("jitter_buffer_ms").value),
        )
        self.capture.start()
        self._event_future = self.runner.submit(self._event_loop())

    def _request(self, kind, payload, timeout=8):
        return self.runner.submit(
            self.transport.request(kind, payload)).result(timeout=timeout)

    def _capture_pcm(self, data: bytes, _pts: int):
        if len(data) != 640:
            return
        message = AudioFrame()
        message.stamp = self.get_clock().now().to_msg()
        message.stream_id = self._capture_stream
        message.speech_id = self._speech_id
        message.sequence = self._capture_sequence
        message.sample_rate = 16000
        message.channels = 1
        message.encoding = "pcm_s16le"
        message.data = list(data)
        message.end_of_stream = False
        self._capture_sequence += 1
        self.capture_pub.publish(message)

    async def _event_loop(self):
        while True:
            event = await self.transport.next_event()
            name, data = event["event"], event["data"]
            if name == "orientation.status":
                self._publish_orientation(data)
            elif name == "audio.activity":
                self._speech_id = str(data.get("speech_id", "")) if data.get("active") else ""
            elif name == "audio.status":
                self._publish_audio_status(data)
            elif name == "led.status":
                self._publish_led_status(data)

    def _publish_orientation(self, data):
        message = OrientationStatus()
        message.stamp = self.get_clock().now().to_msg()
        message.speech_id = str(data.get("speech_id") or "")
        message.raw_doa_deg = float(data.get("raw_doa_deg") or 0.0)
        message.relative_rad = float(data.get("relative_rad") or 0.0)
        message.target_yaw = float(data.get("target_yaw") or 0.0)
        message.current_yaw = float(data.get("current_yaw") or 0.0)
        message.clamped = bool(data.get("clamped", False))
        message.state = str(data.get("state", "unknown"))
        message.code = str(data.get("code", "unknown"))
        message.message = str(data.get("message", ""))
        self.orientation_pub.publish(message)

    def _publish_audio_status(self, data):
        message = AudioStatus()
        message.stamp = self.get_clock().now().to_msg()
        message.capture_connected = bool(data.get("capture_running", False))
        message.playback_active = bool(data.get("playback_running", False))
        message.stream_id = str(data.get("stream_id") or "")
        message.state = str(data.get("state", "unknown"))
        message.code = str(data.get("code", "unknown"))
        message.message = str(data.get("message", ""))
        self.audio_status_pub.publish(message)

    def _publish_led_status(self, data):
        message = LedStatus()
        message.stamp = self.get_clock().now().to_msg()
        message.active = bool(data.get("active", False))
        message.requested_brightness = float(data.get("requested_brightness", 0.0))
        message.applied_brightness = float(data.get("applied_brightness", 0.0))
        message.clamped = bool(data.get("clamped", False))
        message.fault = str(data.get("fault") or "")
        self.led_status_pub.publish(message)

    @staticmethod
    def _frame_dict(message):
        return {
            "stream_id": message.stream_id, "speech_id": message.speech_id,
            "sequence": int(message.sequence), "sample_rate": int(message.sample_rate),
            "channels": int(message.channels), "encoding": message.encoding,
            "data": bytes(message.data), "end_of_stream": message.end_of_stream,
        }

    def _playback_frame(self, message):
        with self._playback_lock:
            sender = self._playback_sender
            if sender is None or message.stream_id != self._playback_stream:
                return
            try:
                frame = sender.push(self._frame_dict(message))
            except AudioFrameError as exc:
                self.get_logger().error(str(exc))
                self._playback_session.fail(exc)
                return
            if frame.end_of_stream:
                self._playback_session.finish()

    def _play_audio(self, goal_handle):
        goal = goal_handle.request
        result = PlayAudio.Result()
        metadata = {
            "stream_id": goal.stream_id, "sample_rate": int(goal.sample_rate),
            "channels": int(goal.channels), "encoding": goal.encoding,
        }
        sender = None
        remote_started = False
        remote_stopped = False
        cancelled = False
        try:
            terminal = self._request("audio.play.start", metadata)
            if terminal.get("state") != "completed":
                raise RuntimeError(str(terminal.get("code", "start_failed")))
            remote_started = True
            sender = PlaybackSender(
                goal.stream_id, bind_host="192.168.100.1",
                pi_host=self.get_parameter("pi_host").value,
                port=int(self.get_parameter("playback_rtp_port").value))
            sender.start()
            with self._playback_lock:
                self._playback_sender = sender
                self._playback_stream = goal.stream_id
                self._playback_session.reset()
            while not self._playback_session.wait(0.05):
                if goal_handle.is_cancel_requested:
                    cancelled = True
                    break
            terminal = self._request(
                "audio.play.stop", {"stream_id": goal.stream_id}, timeout=10)
            remote_stopped = True
            frame_error = self._playback_session.error
            if frame_error is not None:
                raise frame_error
            if cancelled:
                result.success = False
                result.code = "cancelled"
                goal_handle.canceled()
                return result
            result.success = terminal.get("state") == "completed" and (
                terminal.get("data", {}).get("code") == "drained")
            result.code = str(terminal.get("data", {}).get(
                "code", terminal.get("code", "invalid_response")))
            result.message = str(terminal.get("message", ""))
        except (TransportError, AudioFrameError, RuntimeError, TimeoutError) as exc:
            result.success = False
            result.code = getattr(exc, "code", "playback_failed")
            result.message = str(exc)
        finally:
            if remote_started and not remote_stopped:
                try:
                    self._request(
                        "audio.play.stop", {"stream_id": goal.stream_id}, timeout=10)
                except (TransportError, RuntimeError, TimeoutError):
                    pass
            with self._playback_lock:
                self._playback_sender = None
                self._playback_stream = None
            if sender is not None:
                sender.close()
        (goal_handle.succeed if result.success else goal_handle.abort)()
        return result

    def _return_center(self, goal_handle):
        terminal = self._request("orientation.return_center", {}, timeout=10)
        data = terminal.get("data", {})
        result = ReturnCenter.Result()
        result.success = terminal.get("state") == "completed" and data.get("state") == "centered"
        result.code = str(data.get("code", terminal.get("code", "invalid_response")))
        result.message = str(terminal.get("message", ""))
        result.current_yaw = float(data.get("current_yaw", 0.0))
        (goal_handle.succeed if result.success else goal_handle.abort)()
        return result

    def _led_frame(self, image):
        if image.height != 8 or image.width != 8 or image.encoding != "rgb8" or len(image.data) != 192:
            self.get_logger().error("LED image must be 8x8 rgb8")
            return
        self.runner.submit(self.transport.request("led.frame", {
            "rgb": list(image.data), "brightness": 1.0}))

    def _led_solid(self, request, response):
        terminal = self._request("led.solid", {
            "rgb": [int(request.r), int(request.g), int(request.b)],
            "brightness": float(request.brightness)})
        data = terminal.get("data", {})
        response.success = terminal.get("state") == "completed"
        response.code = str(terminal.get("code", "invalid_response"))
        response.message = str(terminal.get("message", ""))
        response.applied_brightness = float(data.get("applied_brightness", 0.0))
        response.clamped = bool(data.get("clamped", False))
        return response

    def _led_clear(self, _request, response):
        terminal = self._request("led.clear", {})
        response.success = terminal.get("state") == "completed"
        response.message = str(terminal.get("message", terminal.get("code", "")))
        return response

    def destroy_node(self):
        self.capture.close()
        self._event_future.cancel()
        self.runner.submit(self.transport.close()).result(timeout=5)
        self.runner.stop()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DeviceBridgeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
