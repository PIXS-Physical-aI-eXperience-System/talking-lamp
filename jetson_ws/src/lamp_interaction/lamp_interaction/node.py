"""Run the S2 conversation flow, S1 task light on request, and the S4
away/return reaction.

``HeadArbiter`` decides which scenario may move the head; the current owner is
published on ``/lamp/head_owner`` so the vision node can hold back too.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger

from lamp_interfaces.action import PlayMotion, ReturnCenter
from lamp_interfaces.msg import MotionStatus, OrientationStatus
from lamp_interfaces.srv import InterruptMotion, SetLedSolid

from .arbiter import HeadArbiter
from .gesture import SPEAK_NOD, gesture_for
from .s1 import TaskLightFlow, ends_task_light, is_wake_call, task_light_command
from .s2 import S2Coordinator, VoiceEvent
from .s4 import S4Coordinator
from .behavior import motion_commands
from .light import (
    WARM_WHITE, LightCommand, LightState, device_brightness, light_command)


# The motion runtime fades the away pose out over 3 s. Greet after that.
GREET_DELAY_S = 3.5
# The voice agent sometimes opens a turn and never closes it. Without a voice
# event for this long, end the turn here so following and idle come back.
VOICE_SILENCE_S = 60.0
# S1 found no book: say so and shake the head (2026-11-01). Intensity 1.0
# turns the base about 11 deg; 0.5 was too small to read as "no".
NO_BOOK_SAY = "책이 안 보여요."
NO_BOOK_MOTION = ("headshake", 1.0)
# Called, but the next words were not understood: look left and right,
# slower and wider than the head shake (scanning, about 23 deg of base yaw).
CONFUSED_MOTION = ("scanning", 0.7)
REACT_WAIT_S = 5.0


class InteractionNode(Node):
    def __init__(self) -> None:
        super().__init__("lamp_interaction")
        self.s2 = S2Coordinator()
        self.declare_parameter("away_after_s", 20.0)
        self.s4 = S4Coordinator(
            float(self.get_parameter("away_after_s").value))
        self.arbiter = HeadArbiter()
        self.owner_pub = self.create_publisher(
            String, "/lamp/head_owner",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.owner = None
        self.epoch = 0
        self.center_handle = None
        self.center_pending = False
        self.follow_client = self.create_client(SetBool, "/lamp/vision/follow_face")
        self.interrupt_client = self.create_client(
            InterruptMotion, "/lamp/interrupt_motion")
        self.center_client = ActionClient(self, ReturnCenter, "/lamp/return_center")
        self.motion_client = ActionClient(self, PlayMotion, "/lamp/play_motion")
        self.motion_busy = False
        self.away_pub = self.create_publisher(Bool, "/lamp/away_pose", 1)
        # Renewed every tick; the Pi lets the hold lapse on its own.
        self.hold_pub = self.create_publisher(Bool, "/lamp/orientation_hold", 1)
        self.greet_timer = None
        self._voice_at = 0.0
        self.create_subscription(String, "/lamp/voice/events", self._voice, 10)
        self.create_subscription(String, "/lamp/cognition/response", self._cognition, 10)
        self.create_subscription(String, "/lamp/voice/heard", self._heard, 10)
        self.say_pub = self.create_publisher(String, "/lamp/voice/say", 10)
        self.light = LightState()
        self.led_solid_client = self.create_client(SetLedSolid, "/lamp/led/set_solid")
        self.led_clear_client = self.create_client(Trigger, "/lamp/led/clear")
        self.s1 = TaskLightFlow()
        self._vision_placing = False
        self._pending_react = None
        self._gesture_turn = None     # speech ID that already got a gesture
        self._aim_state = "off"
        self.aim_client = self.create_client(SetBool, "/lamp/vision/aim_object")
        self.create_subscription(
            OrientationStatus, "/lamp/orientation_status", self._orientation, 10)
        self.create_subscription(Bool, "/lamp/vision/presence", self._presence, 10)
        self.create_subscription(MotionStatus, "/lamp/motion_status", self._motion_status, 10)
        self.create_subscription(String, "/lamp/vision/status", self._vision_status, 10)
        self.create_timer(1.0, self._tick_s4)
        self._publish_owner()

    def _voice(self, msg: String) -> None:
        try:
            event = VoiceEvent.from_json(msg.data)
        except (TypeError, ValueError) as exc:
            self.get_logger().warn(f"잘못된 음성 이벤트: {exc}")
            return
        self._handle_voice(event)

    def _handle_voice(self, event: VoiceEvent) -> None:
        self._voice_at = self._now()
        if event.state == "wake" and event.speech_id != self.s2.active_id:
            self.epoch += 1
            self._cancel_center()
        effects = self.s2.voice(event)
        if event.state == "wake":
            self.arbiter.wake(event.speech_id)
            # While S1 lights the book the lamp listens where it is: the user
            # asked for the light to stay on the book (2026-10-31). S2's gaze
            # effects are dropped in _apply until S1 lets go.
            # Orientation turns the base toward the voice; the greeting gaze
            # must not pull against it. S2 re-enables following once aligned.
            # A wake word while away always ends the away pose.
            for effect in self.s4.conversation_started(self._now()):
                if effect.kind == "resume" or not self.s2.following:
                    self._apply_s4([effect])
        self.arbiter.conversation = self.s2.active_id is not None
        self._apply(effects)
        if event.state == "confused" and event.speech_id == self.s2.active_id:
            # The voice agent says it did not catch that; look left and right.
            self._gesture_turn = event.speech_id
            # The nod on "네?" runs 6 s, so wait for it rather than skip.
            self._react(CONFUSED_MOTION, wait_s=REACT_WAIT_S)
        elif (event.state == "speaking" and event.speech_id == self.s2.active_id
              and self._gesture_turn != event.speech_id):
            # Nothing else moved this turn: nod a little while talking.
            self._gesture_turn = event.speech_id
            self._react(SPEAK_NOD)
        self._publish_owner()

    def _orientation(self, msg: OrientationStatus) -> None:
        self.arbiter.orientation(msg.speech_id, msg.state, self._now())
        self._apply(self.s2.orientation(msg.speech_id, msg.state, msg.code))
        self._publish_owner()

    def _motion_status(self, msg: MotionStatus) -> None:
        self.arbiter.motion_status(connected=bool(msg.connected), fault=msg.fault)

    def _vision_status(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
            placing = bool(status.get("placing", False))
            self._aim_state = str(status.get("aim", "off"))
        except (TypeError, ValueError, AttributeError):
            return
        self._vision_placing = placing
        self.arbiter.task_light = placing or self.s1.holds_head

    def _publish_owner(self) -> None:
        owner = self.arbiter.owner().value
        if owner != self.owner:
            self.owner = owner
            self.owner_pub.publish(String(data=owner))
            self.get_logger().info(f"헤드 담당: {owner}")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _presence(self, msg: Bool) -> None:
        self.arbiter.presence(bool(msg.data), self._now())
        self._apply_s4(self.s4.presence(bool(msg.data), self._now()))

    def _greeted(self) -> None:
        self._apply_s4(self.s4.greeted(self._now()))

    def _tick_s4(self) -> None:
        now = self._now()
        self.arbiter.motion_busy = self.motion_busy
        for effect in self.arbiter.tick(now):
            if effect.kind == "sound_return":
                self.get_logger().info("호출어 없는 소리 회전: 원래 방향으로 복귀")
                self._center(self.epoch)
        if self.s2.active_id is not None and now - self._voice_at >= VOICE_SILENCE_S:
            self.get_logger().warn("음성 이벤트가 끊김: 대화를 끝낸 것으로 처리")
            self._handle_voice(VoiceEvent(self.s2.active_id, "done", "timeout"))
        self._tick_s1(now)
        self._tick_react(now)
        self._apply_s4(self.s4.tick(now, busy=not self.arbiter.may_leave()))
        if self.arbiter.holds_sound_turn(now):
            self.hold_pub.publish(Bool(data=True))
        self._publish_owner()

    def _apply_s4(self, effects) -> None:
        self.arbiter.away = self.s4.away
        self.arbiter.greeting = self.s4.greeting
        for effect in effects:
            if effect.kind == "center":
                self.get_logger().info("사용자 자리 비움: 대기 자세")
                self._cancel_timer("greet_timer")
                self._follow(False)
                self.away_pub.publish(Bool(data=True))
                # Nobody is at the desk, so the light goes off (2026-11-01).
                if self.light.brightness > 0.0:
                    self._set_light(LightCommand("off"))
            elif effect.kind == "greet":
                self.get_logger().info("사용자 복귀: 원래 자세 후 wake_up 인사")
                self.away_pub.publish(Bool(data=False))
                self._cancel_timer("greet_timer")
                self.greet_timer = self.create_timer(GREET_DELAY_S, self._greet)
            elif effect.kind == "resume":
                self.get_logger().info("대기 중 호출어: 인사 없이 원래 자세로")
                self._cancel_timer("greet_timer")
                self.away_pub.publish(Bool(data=False))
            elif effect.kind == "follow_on":
                if not self.s2.following:
                    self.get_logger().info("복귀 인사 뒤 얼굴 추종 잠깐 켬")
                    self._follow(True)
            elif effect.kind == "follow_off":
                # A voice turn that started meanwhile owns following now.
                if not self.s2.following:
                    self._follow(False)

    def _cancel_timer(self, name: str) -> None:
        timer = getattr(self, name)
        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)
            setattr(self, name, None)

    def _greet(self) -> None:
        self._cancel_timer("greet_timer")
        if self.motion_busy:
            self.get_logger().warn("이전 행동이 진행 중이어서 복귀 인사를 건너뜀")
            return
        self.motion_busy = True
        self._play_commands((("wake_up", 0.5),), on_done=self._greeted)

    def _heard(self, msg: String) -> None:
        request = task_light_command(msg.data)
        if request is not None:
            self._task_light_request(request.kind)
            return
        command = light_command(msg.data)
        if self.s1.holds_head and ends_task_light(msg.data, command and command.kind):
            # Asking Pixs for anything else ends the task light (2026-11-01).
            self._task_light_request("release")
        if command is not None:
            self._set_light(command)
            return
        if is_wake_call(msg.data):
            # The lamp only answers "네?" and listens. A nod there looked like
            # pretending to listen (2026-11-01), so this turn gets none.
            self._gesture_turn = self.s2.active_id
            return
        gesture = gesture_for(msg.data)
        if gesture is not None and self.s2.active_id is not None:
            self._gesture_turn = self.s2.active_id
            self._react(gesture, wait_s=REACT_WAIT_S)

    # -- S1 -------------------------------------------------------------------

    def _task_light_request(self, kind: str) -> None:
        if kind == "place":
            self.get_logger().info("작업 조명 요청: 대화가 끝나면 책상을 본다")
            self.s1.request()
        else:
            self.get_logger().info("작업 조명 해제")
            if self.s1.release():
                self._aim(False)
        self.arbiter.task_light = self._vision_placing or self.s1.holds_head
        self._publish_owner()

    def _tick_s1(self, now: float) -> None:
        busy = self.arbiter.conversation or self.motion_busy
        for action in self.s1.tick(now, busy=busy, aim=self._aim_state):
            if action == "start":
                self.get_logger().info("작업 조명: 고개를 숙여 책을 찾는다")
                self._aim_state = "off"     # until vision reports this run
                self._aim(True)
            elif action == "lit":
                self.get_logger().info("작업 조명: 책 쪽으로 팔을 뻗었다. 불을 켠다")
                if self.light.brightness == 0.0:
                    self._set_light(LightCommand("on"))
            elif action == "give_up":
                self.get_logger().warn("작업 조명: 책을 찾지 못했다")
                self._aim(False)
                self.say_pub.publish(String(data=NO_BOOK_SAY))
                # Vision still reports placing until it sees the aim end, so
                # the head may not be free yet; the reaction waits for it.
                self._react(NO_BOOK_MOTION, wait_s=REACT_WAIT_S)
        self.arbiter.task_light = self._vision_placing or self.s1.holds_head

    def _aim(self, on: bool) -> None:
        if not self.aim_client.wait_for_service(timeout_sec=0.0):
            self.get_logger().warn("/lamp/vision/aim_object 서비스가 없다")
            return
        self.aim_client.call_async(SetBool.Request(data=on))

    def _set_light(self, command: LightCommand) -> None:
        brightness = self.light.apply(command)
        self.get_logger().info(f"조명 {command.kind}: 밝기 {brightness:.2f}")
        if brightness == 0.0:
            client, request = self.led_clear_client, Trigger.Request()
        else:
            client = self.led_solid_client
            request = SetLedSolid.Request(
                r=WARM_WHITE[0], g=WARM_WHITE[1], b=WARM_WHITE[2],
                brightness=device_brightness(brightness))
        if not client.wait_for_service(timeout_sec=0.0):
            self.get_logger().warn(f"{client.srv_name} 서비스가 없다")
            return

        def finished(done):
            try:
                result = done.result()
                if not result.success:
                    self.get_logger().warn(f"조명 변경 실패: {result.message}")
            except Exception as exc:
                self.get_logger().error(f"조명 요청 실패: {exc}")

        client.call_async(request).add_done_callback(finished)

    def _react(self, command: tuple[str, float], wait_s: float = 0.0) -> None:
        """Play one reaction clip once the head is free, waiting up to wait_s."""
        if self.arbiter.may_play() and not self.motion_busy:
            self._pending_react = None
            self.get_logger().info(f"{command[0]} 반응")
            self.motion_busy = True
            self._play_commands((command,))
        elif wait_s > 0.0:
            self._pending_react = (command, self._now() + wait_s)
        else:
            self._pending_react = None
            self.get_logger().info(f"{command[0]} 반응을 건너뜀")

    def _tick_react(self, now: float) -> None:
        if self._pending_react is None:
            return
        command, deadline = self._pending_react
        self._react(command, wait_s=max(0.0, deadline - now))

    def _cognition(self, msg: String) -> None:
        try:
            commands = motion_commands(msg.data)
        except (TypeError, ValueError) as exc:
            self.get_logger().warn(f"잘못된 인지 행동: {exc}")
            return
        if not commands:
            return
        if not self.arbiter.may_play():
            self.get_logger().warn(
                f"헤드 담당이 {self.arbiter.owner().value}여서 행동을 건너뜀")
            return
        if self.motion_busy:
            self.get_logger().warn("이전 행동이 진행 중이어서 새 행동을 건너뜀")
            return
        self.motion_busy = True
        self._play_commands(commands)

    def _play_commands(self, commands: tuple[tuple[str, float], ...],
                       on_done=None) -> None:
        if not commands:
            self.motion_busy = False
            if on_done is not None:
                on_done()
            return
        if not self.motion_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn("/lamp/play_motion 액션이 없다")
            self.motion_busy = False
            return
        name, intensity = commands[0]
        goal = PlayMotion.Goal()
        goal.name = name
        goal.intensity = intensity
        goal.repeat = 1
        goal.replace_current = False
        future = self.motion_client.send_goal_async(goal)

        def accepted(done):
            try:
                handle = done.result()
                if not handle.accepted:
                    self.get_logger().warn(f"{name} 행동이 거절됨")
                    self.motion_busy = False
                    return
                handle.get_result_async().add_done_callback(finished)
            except Exception as exc:
                self.get_logger().error(f"{name} 행동 요청 실패: {exc}")
                self.motion_busy = False

        def finished(done):
            try:
                result = done.result().result
                if not result.success:
                    self.get_logger().warn(f"{name} 행동 실패: {result.code}")
                    self.motion_busy = False
                    return
            except Exception as exc:
                self.get_logger().error(f"{name} 행동 결과 확인 실패: {exc}")
                self.motion_busy = False
                return
            self._play_commands(commands[1:], on_done)

        future.add_done_callback(accepted)

    def _apply(self, effects) -> None:
        if self.s1.holds_head:
            # Following the face, centring, or interrupting motion would all
            # pull the head off the book (interrupt also drops the task light).
            effects = [e for e in effects if e.kind not in ("follow", "center", "interrupt")]
        center = any(effect.kind == "center" for effect in effects)
        follow_off = any(effect.kind == "follow" and effect.enabled is False
                         for effect in effects)
        for effect in effects:
            if effect.kind == "follow":
                self._follow(bool(effect.enabled), center_after=center and follow_off)
            elif effect.kind == "interrupt":
                self._interrupt()
            elif effect.kind == "center" and not follow_off:
                self._center(self.epoch)

    def _follow(self, enabled: bool, *, center_after: bool = False) -> None:
        epoch = self.epoch
        if not self.follow_client.wait_for_service(timeout_sec=0.0):
            self.get_logger().warn("/lamp/vision/follow_face 서비스가 없다")
            if center_after:
                self._center(epoch)
            return
        future = self.follow_client.call_async(SetBool.Request(data=enabled))

        def finished(done):
            try:
                result = done.result()
                if not result.success:
                    self.get_logger().warn(f"얼굴 추종 전환 실패: {result.message}")
            except Exception as exc:
                self.get_logger().error(f"얼굴 추종 요청 실패: {exc}")
            if epoch != self.epoch:
                # An old off request can complete after a new turn enabled gaze.
                if self.s2.following:
                    self._follow(True)
                return
            if center_after and epoch == self.epoch:
                self._center(epoch)

        future.add_done_callback(finished)

    def _interrupt(self) -> None:
        if not self.interrupt_client.wait_for_service(timeout_sec=0.0):
            self.get_logger().warn("/lamp/interrupt_motion 서비스가 없다")
            return
        self.interrupt_client.call_async(InterruptMotion.Request())

    def _center(self, epoch: int) -> None:
        if epoch != self.epoch:
            return
        if not self.center_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn("/lamp/return_center 액션이 없다")
            return
        self.center_pending = True
        future = self.center_client.send_goal_async(ReturnCenter.Goal())

        def accepted(done):
            self.center_pending = False
            try:
                handle = done.result()
            except Exception as exc:
                self.get_logger().error(f"중앙 복귀 요청 실패: {exc}")
                return
            if not handle.accepted:
                self.get_logger().warn("중앙 복귀가 거절됐다")
                return
            if epoch != self.epoch:
                handle.cancel_goal_async()
                return
            self.center_handle = handle
            handle.get_result_async().add_done_callback(
                lambda result: self._center_finished(handle, result))

        future.add_done_callback(accepted)

    def _center_finished(self, handle, future) -> None:
        if self.center_handle is handle:
            self.center_handle = None
        try:
            result = future.result().result
        except Exception as exc:
            self.get_logger().error(f"중앙 복귀 결과 확인 실패: {exc}")
            return
        if not result.success:
            self.get_logger().warn(f"중앙 복귀 실패: {result.code}")

    def _cancel_center(self) -> None:
        if self.center_handle is not None:
            self.center_handle.cancel_goal_async()
            self.center_handle = None


def main() -> None:
    rclpy.init()
    node = InteractionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
