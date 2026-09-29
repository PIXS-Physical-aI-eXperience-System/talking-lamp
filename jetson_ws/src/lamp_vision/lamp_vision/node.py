"""ROS 2 node for part D: camera -> detections -> 3D targets.

A shell around ``src/vision``. All the judgement lives in ``vision.runtime``,
which has no ROS in it and is unit-tested; this file only moves bytes.

Wiring
------

Subscribes
  ``/lamp/motion_status``       E's state. ``busy`` means the head is moving,
                                so the rest pose no longer holds.
  ``/lamp/orientation_status``  ``current_yaw``, the one joint angle E
                                publishes. Used to correct the assumed rest
                                pose for whichever way the base is turned.

Publishes
  ``/lamp/track_point``         PointStamped, S2. Where E should look.
  ``/lamp/vision/presence``     Bool, S4, on change. Is the user at the desk.
  ``/lamp/vision/labels``       String (JSON), on change. Korean names of what
                                is in view, for A's cognition prompt. Owned by
                                D under its own namespace until A and D agree a
                                message type. **[미정]**
  ``/lamp/vision/status``       String (JSON), 1 Hz. Diagnostics.

Calls
  ``/lamp/return_center``       Action. To restore the rest pose, which is the
                                only posture where D knows the camera pose.
  ``/lamp/place_task_light``    Action, S1. The desk point to light.

Services offered
  ``/lamp/vision/center``       Trigger. Return to rest and re-fix the pose.
  ``/lamp/vision/place_light``  Trigger. Run S1 now.
  ``/lamp/vision/follow_face``  SetBool. S2 on/off; B turns it on at the wake
                                word. Off by default, so the lamp does not
                                stare at the user unprompted.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger

from lamp_interfaces.action import PlaceTaskLight, ReturnCenter
from lamp_interfaces.msg import MotionStatus, OrientationStatus


def _find_src() -> Path | None:
    """The repo's ``src/`` when the package runs from a checkout.

    ``vision`` is not a ROS package and is not installed into the overlay, so
    an installed copy of this node needs ``PYTHONPATH`` or the ``src_dir``
    parameter instead. Looking upward keeps both cases working.
    """
    for parent in Path(__file__).resolve().parents:
        cand = parent / "src" / "vision"
        if cand.is_dir():
            return cand.parent
    return None


_SRC = _find_src()
if _SRC is not None and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vision import FRAME_ID, Intrinsics, VisionPipeline            # noqa: E402
from vision.detector import FaceDetector, ObjectDetector           # noqa: E402
from vision.head_camera import HeadCameraMount, HeadKinematics     # noqa: E402
from vision.runtime import RuntimeConfig, VisionRuntime            # noqa: E402


class VisionNode(Node):
    def __init__(self) -> None:
        super().__init__("lamp_vision")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("width", 1920)
        self.declare_parameter("height", 1080)
        self.declare_parameter("fps", 10.0)
        default_calib = "" if _SRC is None else str(_SRC.parent / "vision-bench" / "calib")
        self.declare_parameter("calib_dir", default_calib)
        self.declare_parameter("model_dir", os.path.expanduser("~/vision-bench/models"))
        self.declare_parameter("object_conf", 0.3)
        self.declare_parameter("face_conf", 0.6)
        self.declare_parameter("center_on_start", True)

        calib = Path(self.get_parameter("calib_dir").value)
        cam = self._load_intrinsics(calib)
        mount, kin = self._load_mount(calib)
        models = Path(self.get_parameter("model_dir").value)
        pipeline = VisionPipeline(
            cam, _placeholder_pose(),
            ObjectDetector(models / "yolox_s.onnx",
                           conf_thr=float(self.get_parameter("object_conf").value)),
            FaceDetector(models / "yunet_2023mar.onnx",
                         conf_thr=float(self.get_parameter("face_conf").value)))
        self.runtime = VisionRuntime(cam, mount, kin, pipeline, RuntimeConfig())

        self.group = ReentrantCallbackGroup()
        latest = QoSProfile(depth=1)
        self.track_pub = self.create_publisher(PointStamped, "/lamp/track_point", latest)
        self.presence_pub = self.create_publisher(Bool, "/lamp/vision/presence", latest)
        self.labels_pub = self.create_publisher(String, "/lamp/vision/labels", latest)
        self.status_pub = self.create_publisher(String, "/lamp/vision/status", latest)
        self.create_subscription(MotionStatus, "/lamp/motion_status", self._motion, latest,
                                 callback_group=self.group)
        self.create_subscription(OrientationStatus, "/lamp/orientation_status",
                                 self._orientation, latest, callback_group=self.group)
        self.center_client = ActionClient(self, ReturnCenter, "/lamp/return_center",
                                          callback_group=self.group)
        self.light_client = ActionClient(self, PlaceTaskLight, "/lamp/place_task_light",
                                         callback_group=self.group)
        self.create_service(Trigger, "/lamp/vision/center", self._srv_center,
                            callback_group=self.group)
        self.create_service(Trigger, "/lamp/vision/place_light", self._srv_place,
                            callback_group=self.group)
        self.create_service(SetBool, "/lamp/vision/follow_face", self._srv_follow,
                            callback_group=self.group)
        self.create_timer(1.0, self._publish_status, callback_group=self.group)

        self.current_yaw: float | None = None
        self.motion_busy = False
        self._placing = False           # True while a task light D sent is being held
        self._rest_at = -1e9
        self._fps = 0.0
        self._object_fps = 0.0
        self._object_frames = 0
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._capture = threading.Thread(target=self._loop, name="vision-capture", daemon=True)
        self._capture.start()
        if bool(self.get_parameter("center_on_start").value):
            threading.Thread(target=self._center_at_start, daemon=True).start()

    def _center_at_start(self) -> None:
        """Centre once at startup, and say so either way.

        Failing silently here left the node running with no rest pose and no
        hint why, which is how a whole test session got lost.
        """
        ok, why = self._center(timeout=12.0)
        if not ok:
            self.get_logger().error(f"시작 시 휴식 자세 복귀 실패: {why}")
            self.get_logger().error("  /lamp/vision/center 로 다시 시도할 수 있다")

    # -- setup ---------------------------------------------------------------

    def _load_intrinsics(self, calib: Path) -> Intrinsics:
        path = calib / "intrinsics.json"
        if path.exists():
            cam = Intrinsics.load(path)
            self.get_logger().info(f"내부 파라미터 {path} (RMS {cam.rms:.2f} px)")
            return cam
        w = int(self.get_parameter("width").value)
        h = int(self.get_parameter("height").value)
        self.get_logger().warn(f"{path} 없음 — 카탈로그 화각으로 근사한다. 거리가 10% 가까이 틀린다")
        return Intrinsics.from_fov(w, h, 120.0)

    def _load_mount(self, calib: Path):
        path = calib / "head_camera.json"
        if not path.exists():
            self.get_logger().warn(
                f"{path} 없음 — 절대 좌표(S1)를 낼 수 없다. calib.py mount 를 먼저 돌린다")
            return None, None
        mount = HeadCameraMount.load(path)
        try:
            kin = HeadKinematics()
        except Exception as exc:                                  # MuJoCo missing
            self.get_logger().warn(f"E의 기구학을 못 불러왔다 ({exc}); S1 비활성")
            return mount, None
        self.get_logger().info(f"헤드→카메라 장착값 {path} (축 오차 {mount.axis_offset_deg:.1f}°)")
        return mount, kin

    # -- status --------------------------------------------------------------

    def _motion(self, msg: MotionStatus) -> None:
        was, self.motion_busy = self.motion_busy, bool(msg.busy)
        if not (self.motion_busy and not was and self.runtime.at_rest):
            return
        # The bridge keeps reporting busy for a moment after return_center
        # finishes. Taking that as "someone moved the arm" would drop the rest
        # pose D has only just established, so ignore it just after settling.
        if time.monotonic() - self._rest_at < 2.0:
            return
        what = msg.active_motion if msg.active_motion not in ("", "None") else msg.state
        mine = self._placing
        self.runtime.leave_rest("D가 조명을 배치 중" if mine else f"E가 {what} 실행 중")
        self.get_logger().info(
            "휴식 자세 해제: " + ("조명 배치(D)" if mine else f"E가 {what} 실행 중"))
        if mine:
            return          # the lamp is holding the light D asked for; leave it there
        # Otherwise nothing else will ask for a return_center, and a single
        # stray busy would leave S1 refusing for good.
        threading.Thread(target=self._recenter_when_idle, daemon=True).start()

    def _orientation(self, msg: OrientationStatus) -> None:
        self.current_yaw = float(msg.current_yaw)

    def _publish_status(self) -> None:
        tgt, why = self.runtime.task_light_target()
        self.status_pub.publish(String(data=json.dumps({
            "at_rest": self.runtime.at_rest,
            "following": self.runtime.following,
            "fps": round(self._fps, 1),
            "object_fps": round(self._object_fps, 2),
            "tracks": len(self.runtime.averager.tracks),
            "task_light": None if tgt is None else {
                "label": tgt.label, "conf": round(tgt.conf, 2),
                "pos": [round(float(v), 3) for v in tgt.pos]},
            "task_light_blocked": why,
            "present": self.runtime.presence.present,
        }, ensure_ascii=False)))

    # -- capture loop --------------------------------------------------------

    def _loop(self) -> None:
        import cv2
        idx = int(self.get_parameter("camera_index").value)
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.get_parameter("width").value))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.get_parameter("height").value))
        if not cap.isOpened():
            self.get_logger().error(f"카메라 {idx} 를 열지 못했다")
            return
        period = 1.0 / float(self.get_parameter("fps").value)
        self.get_logger().info("카메라 시작")
        while not self._stop.is_set() and rclpy.ok():
            t0 = time.monotonic()
            ok, bgr = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            try:
                with self._lock:
                    ran_objects = self.runtime.due_for_objects(t0)
                    out = self.runtime.process(bgr, t0)
                self._object_frames += int(ran_objects)
                self._emit(out)
            except Exception:                        # a bad frame must not kill the node
                if not rclpy.ok():
                    break                            # shutting down: publishers are gone
                self.get_logger().error("프레임 처리 실패\n" + traceback.format_exc())
            dt = time.monotonic() - t0
            self._fps = 1.0 / dt if dt > 0 else 0.0
            elapsed = time.monotonic() - self._started
            self._object_fps = self._object_frames / elapsed if elapsed > 0 else 0.0
            time.sleep(max(0.0, period - dt))
        cap.release()

    def _emit(self, out) -> None:
        if out.track_point is not None:
            self.track_pub.publish(_point(out.track_point, self.get_clock().now()))
            if out.note:
                self.get_logger().debug(f"track_point: {out.note}")
        if out.presence is not None:
            self.presence_pub.publish(Bool(data=out.presence))
            self.get_logger().info("사용자 " + ("복귀" if out.presence else "자리 비움"))
        if out.labels is not None:
            self.labels_pub.publish(String(data=json.dumps(
                {"stamp": out.frame.stamp, "labels": out.labels}, ensure_ascii=False)))

    # -- services ------------------------------------------------------------

    def _center(self, timeout: float = 8.0) -> tuple[bool, str]:
        """Return to the rest pose and re-fix the camera pose from FK."""
        self._placing = False
        if self.runtime.kin is None:
            return False, "장착 캘리브레이션이나 기구학이 없다"
        if not self.center_client.wait_for_server(timeout_sec=2.0):
            return False, "/lamp/return_center 액션 서버가 없다"
        goal = self.center_client.send_goal_async(ReturnCenter.Goal())
        if not _spin_for(goal, timeout) or not goal.result().accepted:
            return False, "return_center 를 받아주지 않았다"
        res = goal.result().get_result_async()
        if not _spin_for(res, timeout):
            return False, "return_center 응답 없음"
        # An aborted or cancelled goal comes back with a default-constructed
        # result: success False and an empty message. Report the status so that
        # case is not indistinguishable from the action genuinely refusing.
        status, result = res.result().status, res.result().result
        if not result.success:
            return False, (f"return_center 실패: {result.message or _STATUS.get(status, status)}")
        yaw = float(result.current_yaw)
        # The action returns before the arm has stopped ringing, and the bridge
        # keeps publishing busy for a little longer. Wait both out, or the pose
        # is fixed while the head is still moving.
        deadline = time.monotonic() + 3.0
        while self.motion_busy and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.8)
        with self._lock:
            pose = self.runtime.enter_rest(yaw)
            self._rest_at = time.monotonic()
        self.get_logger().info(
            f"휴식 자세 고정: 카메라 높이 {pose.height * 100:.1f} cm, "
            f"아래로 {pose.tilt_deg:.1f}°, base_yaw {np.degrees(yaw):.1f}°")
        return True, "휴식 자세에서 카메라 자세를 고정했다"

    def _recenter_when_idle(self) -> None:
        """Re-establish the rest pose once E has finished whatever it was doing.

        Only when the arm is genuinely idle again. Centring while E still holds
        a pose would yank the head away from whatever it was asked to do --
        including a task light D itself placed.
        """
        deadline = time.monotonic() + 60.0
        while self.motion_busy and time.monotonic() < deadline:
            time.sleep(0.2)
        if self.motion_busy:
            self.get_logger().info("자동 복귀 보류: E 가 아직 자세를 잡고 있다")
            return
        if self.runtime.following or self.runtime.at_rest or self._placing:
            return                      # S2 owns the head, or someone beat us to it
        ok, why = self._center()
        self.get_logger().info(f"자동 복귀: {why}" if ok else f"자동 복귀 실패: {why}")

    def _srv_center(self, _req, res):
        res.success, res.message = self._center()
        return res

    def _srv_place(self, _req, res):
        with self._lock:
            target, why = self.runtime.task_light_target()
        if target is None:
            res.success, res.message = False, why
            return res
        if not self.light_client.wait_for_server(timeout_sec=2.0):
            res.success, res.message = False, "/lamp/place_task_light 액션 서버가 없다"
            return res
        goal = PlaceTaskLight.Goal()
        goal.target = _point(target.pos, self.get_clock().now())
        self._placing = True
        fut = self.light_client.send_goal_async(goal)
        if not _spin_for(fut, 5.0) or not fut.result().accepted:
            self._placing = False
            res.success, res.message = False, "조명 배치를 받아주지 않았다"
            return res
        res.success = True
        res.message = (f"{target.label} (신뢰도 {target.conf:.2f}) "
                       f"→ ({target.pos[0]:.3f}, {target.pos[1]:.3f}, {target.pos[2]:.3f}) m")
        self.get_logger().info(f"조명 배치: {res.message}")
        return res

    def _srv_follow(self, req, res):
        with self._lock:
            self._placing = False       # S2 takes the head back from the task light
            self.runtime.set_following(bool(req.data))
        res.success = True
        res.message = "얼굴 추종 " + ("시작" if req.data else "정지")
        if req.data and not self.runtime.at_rest:
            res.message += " (주의: 휴식 자세가 아니라 첫 조준을 못 한다. center 먼저)"
        return res

    def destroy_node(self) -> None:
        self._stop.set()
        self._capture.join(timeout=2.0)
        super().destroy_node()


_STATUS = {1: "ACCEPTED", 2: "EXECUTING", 3: "CANCELING", 4: "SUCCEEDED",
           5: "CANCELED", 6: "ABORTED"}


def _placeholder_pose():
    """Stand-in until ``enter_rest`` fixes the real one; nothing absolute uses it."""
    from vision.geometry import Pose
    return Pose.look_at([0.18, 0.0, 0.35], [0.6, 0.0, 0.0], source="placeholder")


def _point(p, now) -> PointStamped:
    msg = PointStamped()
    msg.header.stamp = now.to_msg()
    msg.header.frame_id = FRAME_ID
    msg.point.x, msg.point.y, msg.point.z = (float(v) for v in p)
    return msg


def _spin_for(future, timeout: float) -> bool:
    """Wait on a future from a thread that is not the executor's."""
    deadline = time.monotonic() + timeout
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    return future.done()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
