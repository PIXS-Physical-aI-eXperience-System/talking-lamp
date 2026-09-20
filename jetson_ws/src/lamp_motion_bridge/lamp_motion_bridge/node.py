"""ROS 2 adapters for the authenticated Pi motion transport."""

from __future__ import annotations

import os
from uuid import uuid4

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PointStamped, Vector3Stamped

from lamp_interfaces.action import PlaceTaskLight, PlayMotion
from lamp_interfaces.msg import MotionStatus
from lamp_interfaces.srv import InterruptMotion, ListMotions
from lamp_device_bridge.action_wait import wait_cancelable
from lamp_device_bridge.runner import AsyncRunner
from .transport import MotionTransport, TransportError


MOTION_ACTION_TIMEOUT_SECONDS = 240


class MotionBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("lamp_motion_bridge")
        self.declare_parameter("pi_host", "192.168.100.2")
        self.declare_parameter("motion_port", 8765)
        self.declare_parameter("token_env", "TALKING_LAMP_MOTION_TOKEN")
        token_name = self.get_parameter("token_env").value
        token = os.environ.get(token_name, "")
        if not token:
            raise RuntimeError(f"required token environment variable is empty: {token_name}")
        self.runner = AsyncRunner()
        self.command_group = ReentrantCallbackGroup()
        self.transport = MotionTransport(
            self.get_parameter("pi_host").value,
            int(self.get_parameter("motion_port").value), token)
        self.runner.submit(self.transport.start()).result(timeout=5)

        latest = QoSProfile(depth=1)
        self.status_pub = self.create_publisher(
            MotionStatus, "/lamp/motion_status", latest)
        self.create_subscription(
            PointStamped, "/lamp/track_point", self._track_point, latest,
            callback_group=self.command_group)
        self.create_subscription(
            Vector3Stamped, "/lamp/track_bearing", self._track_bearing, latest,
            callback_group=self.command_group)
        self.create_service(
            ListMotions, "/lamp/list_motions", self._list,
            callback_group=self.command_group)
        self.create_service(
            InterruptMotion, "/lamp/interrupt_motion", self._interrupt,
            callback_group=self.command_group)
        self.play_action = ActionServer(
            self, PlayMotion, "/lamp/play_motion", execute_callback=self._play,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self.command_group)
        self.task_action = ActionServer(
            self, PlaceTaskLight, "/lamp/place_task_light", execute_callback=self._task,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self.command_group)
        self.create_timer(
            0.2, self._publish_status, callback_group=self.command_group)

    def _request(self, kind, payload, timeout=8):
        return self.runner.submit(
            self.transport.request(
                kind, payload, response_timeout=timeout)).result(timeout=timeout + 1)

    def _cancelable_request(
        self, goal_handle, kind, payload, cancel_kind, *, timeout,
    ):
        request_id = str(uuid4())
        future = self.runner.submit(self.transport.request(
            kind, payload, response_timeout=timeout, request_id=request_id))
        return wait_cancelable(
            future,
            cancel_requested=lambda: goal_handle.is_cancel_requested,
            cancel=lambda: self._request(
                cancel_kind, {"request_id": request_id}, timeout=3),
            timeout=timeout + 1,
        )

    def _play(self, goal_handle):
        request = goal_handle.request
        outcome = self._cancelable_request(goal_handle, "motion.play", {
            "name": request.name,
            "replace_current": request.replace_current,
            "intensity": float(request.intensity),
            "repeat": int(request.repeat),
        }, "motion.cancel", timeout=MOTION_ACTION_TIMEOUT_SECONDS)
        response = outcome.terminal
        result = PlayMotion.Result()
        result.success = response.get("state") == "completed"
        result.code = str(response.get("code", "invalid_response"))
        result.message = str(response.get("message", ""))
        if outcome.cancelled:
            goal_handle.canceled()
        else:
            (goal_handle.succeed if result.success else goal_handle.abort)()
        return result

    def _task(self, goal_handle):
        point = goal_handle.request.target.point
        outcome = self._cancelable_request(
            goal_handle, "task_light.place", {"point": [point.x, point.y, point.z]},
            "task_light.cancel", timeout=15)
        response = outcome.terminal
        result = PlaceTaskLight.Result()
        result.success = response.get("state") == "completed"
        result.code = str(response.get("code", "invalid_response"))
        result.message = str(response.get("message", ""))
        if outcome.cancelled:
            goal_handle.canceled()
        else:
            (goal_handle.succeed if result.success else goal_handle.abort)()
        return result

    def _list(self, _request, response):
        terminal = self._request("motion.list", {})
        response.motions = list(terminal.get("data", {}).get("motions", []))
        return response

    def _interrupt(self, _request, response):
        terminal = self._request("motion.interrupt", {})
        response.success = terminal.get("state") == "completed"
        response.code = str(terminal.get("code", "invalid_response"))
        response.message = str(terminal.get("message", ""))
        return response

    def _track_point(self, message):
        p = message.point
        self.runner.submit(self.transport.request(
            "track.point", {"point": [p.x, p.y, p.z]}, ttl_ms=250))

    def _track_bearing(self, message):
        v = message.vector
        self.runner.submit(self.transport.request(
            "track.bearing", {"direction": [v.x, v.y, v.z]}, ttl_ms=250))

    def _publish_status(self):
        message = MotionStatus()
        message.stamp = self.get_clock().now().to_msg()
        message.connected = self.transport.connected
        try:
            terminal = self._request("motion.status", {}, timeout=2)
            data = terminal.get("data", {})
            message.state = str(data.get("state", "unknown"))
            message.active_motion = str(data.get("active_motion", ""))
            message.busy = bool(data.get("busy", False))
            message.fault = str(data.get("fault", ""))
            message.sent_ticks = int(data.get("sent_ticks", 0))
            message.deadline_misses = int(data.get("deadline_misses", 0))
        except (TransportError, TimeoutError) as exc:
            message.connected = False
            message.state = "disconnected"
            message.fault = str(exc)
        self.status_pub.publish(message)

    def destroy_node(self):
        self.runner.submit(self.transport.close()).result(timeout=5)
        self.runner.stop()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotionBridgeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
