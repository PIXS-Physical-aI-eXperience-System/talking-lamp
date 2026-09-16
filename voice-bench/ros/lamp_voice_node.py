"""ROS 노드 — 오디오 토픽과 우리 음성 판단부를 잇는다.

    source /opt/ros/jazzy/setup.bash
    source ~/talking-lamp-integration/jetson_ws/install/setup.bash
    python3 ros/lamp_voice_node.py --agent 127.0.0.1:5150

이 파일은 시스템 파이썬(ROS Jazzy, 3.12)에서 돈다. 모델은 여기서 돌리지
않는다 — venvs/melo-onnx 의 직접 빌드한 onnxruntime·ctranslate2 휠이 ROS
의존성과 부딪히는 것이 가장 깨지기 쉬운 지점이라, 아예 다른 인터프리터에
두고 localhost 소켓으로 잇는다. 이 노드는 rclpy 말고는 아무것도 필요 없다.

  sub  /lamp/audio/capture          20ms AudioFrame, 16kHz 모노 pcm_s16le
  sub  /lamp/orientation_status     파이의 방향 정렬 상태
  pub  /lamp/audio/playback_frames  같은 형식으로 되돌려 보낸다
  act  /lamp/play_audio             재생 시작·EOS·드레인 완료를 소유한다

주의: 이 파일은 실기기에서 아직 돌려보지 않았다. rclpy 가 없는 곳에서는
검증할 수 없어 문법과 계약만 맞춰 둔 상태다.
"""
import argparse
import socket
import sys
import threading
import time
import uuid

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from lamp_interfaces.action import PlayAudio
from lamp_interfaces.msg import AudioFrame, OrientationStatus

FRAME_BYTES = 640          # 20ms @ 16kHz 모노 s16le. 이 크기가 아니면 거부된다
MAX_AHEAD_S = 2.0          # 재생 시각보다 이만큼 이상 앞서가지 않는다
RATE = 16000
HDR_LEN = 8

# voice/link.py 와 같은 값. 이 노드는 그 모듈을 import 하지 않는다 —
# 다른 인터프리터라 경로를 끌어오면 venv 가 섞인다.
CAP_FRAME = b"CAPF"
ORIENT = b"ORNT"
SPEAK_BEGIN = b"SPKB"
SPEAK_AUDIO = b"SPKA"
SPEAK_END = b"SPKE"
BARGE_IN = b"BRGI"
HEARD = b"HERD"
SPEECH_ID_LEN = 36


def pack(kind, payload=b""):
    return kind + len(payload).to_bytes(4, "big") + payload


def pack_id(speech_id):
    return (speech_id or "").strip().ljust(SPEECH_ID_LEN).encode("ascii")


class LampVoiceNode(Node):
    def __init__(self, host, port):
        super().__init__("lamp_voice")
        self.host, self.port = host, port
        self.sock = None
        self.send_lock = threading.Lock()

        self.stream_id = ""
        self.sequence = 0
        self.goal_handle = None
        self.play_done = threading.Event()
        # 목표가 수락되기 전에 발행한 프레임은 파이가 거부한다. 수락될 때까지
        # 들고 있다가 한꺼번에 내보낸다.
        self.accepted = threading.Event()
        self.pending = []
        self.frame_lock = threading.Lock()
        self.play_t0 = 0.0

        sensor = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(AudioFrame, "/lamp/audio/capture",
                                 self.on_capture, sensor)
        self.create_subscription(OrientationStatus, "/lamp/orientation_status",
                                 self.on_orientation, 10)
        self.playback = self.create_publisher(
            AudioFrame, "/lamp/audio/playback_frames", sensor)
        self.play_audio = ActionClient(self, PlayAudio, "/lamp/play_audio")

        self.connect()
        threading.Thread(target=self.read_loop, daemon=True).start()
        self.get_logger().info("lamp_voice 시작")

    # ── 판단부와의 연결 ─────────────────────────────────────────────────
    def connect(self):
        while rclpy.ok():
            try:
                s = socket.create_connection((self.host, self.port), timeout=5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(None)
                self.sock = s
                self.get_logger().info(f"판단부 연결 {self.host}:{self.port}")
                return
            except OSError as e:
                self.get_logger().warn(f"판단부에 못 붙었다({e}). 2초 뒤 재시도")
                time.sleep(2.0)

    def send(self, kind, payload=b""):
        with self.send_lock:
            if self.sock is None:
                return
            try:
                self.sock.sendall(pack(kind, payload))
            except OSError as e:
                self.get_logger().error(f"판단부로 못 보냈다: {e}")
                self.sock = None

    def recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            b = self.sock.recv(n - len(buf))
            if not b:
                raise ConnectionError("판단부가 연결을 끊었다")
            buf += b
        return bytes(buf)

    def read_loop(self):
        while rclpy.ok():
            if self.sock is None:
                self.connect()
                continue
            try:
                head = self.recv_exact(HDR_LEN)
                kind, n = head[:4], int.from_bytes(head[4:], "big")
                body = self.recv_exact(n) if n else b""
            except (OSError, ConnectionError) as e:
                self.get_logger().warn(f"판단부 연결 끊김: {e}")
                self.sock = None
                continue
            try:
                self.on_agent_message(kind, body)
            except Exception as e:
                self.get_logger().error(f"{kind!r} 처리 실패: {e}")

    # ── ROS → 판단부 ───────────────────────────────────────────────────
    def on_capture(self, msg: AudioFrame):
        if msg.end_of_stream:
            return
        if msg.sample_rate != RATE or msg.channels != 1 or len(msg.data) != FRAME_BYTES:
            # 계약상 이 형식만 온다. 다른 것이 오면 형식이 바뀐 것이므로
            # 조용히 넘기지 말고 남긴다.
            self.get_logger().warn(
                f"예상 밖 프레임: {msg.sample_rate}Hz {msg.channels}ch "
                f"{len(msg.data)}바이트 — 무시한다")
            return
        self.send(CAP_FRAME, pack_id(msg.speech_id) + bytes(msg.data))

    def on_orientation(self, msg: OrientationStatus):
        self.send(ORIENT, pack_id(msg.speech_id) + msg.state.encode("utf-8"))

    # ── 판단부 → ROS ───────────────────────────────────────────────────
    def on_agent_message(self, kind, body):
        # 이름을 handle 로 두면 안 된다. rclpy.Node 에 같은 이름의 속성이 있어서
        # Node.__init__ 의 with self.handle: 이 우리 메서드를 잡고 터진다.
        if kind == CAP_FRAME or kind == ORIENT:
            return
        if kind == HEARD:
            self.get_logger().info(f"들은 말: {body.decode('utf-8', 'replace')}")
        elif kind == SPEAK_BEGIN:
            self.start_playback()
        elif kind == SPEAK_AUDIO:
            self.publish_frame(bytes(body))
        elif kind == SPEAK_END:
            self.finish_playback()
        elif kind == BARGE_IN:
            self.cancel_playback()

    def start_playback(self):
        self.stream_id = str(uuid.uuid4())
        self.sequence = 0
        self.play_done.clear()
        self.accepted.clear()
        with self.frame_lock:
            self.pending = []
        self.play_t0 = time.time()
        if not self.play_audio.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("/lamp/play_audio 가 없다")
            return
        goal = PlayAudio.Goal(stream_id=self.stream_id, sample_rate=RATE,
                              channels=1, encoding="pcm_s16le")
        fut = self.play_audio.send_goal_async(goal)
        fut.add_done_callback(self._goal_accepted)

    def _goal_accepted(self, fut):
        handle = fut.result()
        if not handle.accepted:
            self.get_logger().error("PlayAudio 거부됨 — 프레임을 버린다")
            with self.frame_lock:
                self.pending = []
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self._goal_result)
        # 들고 있던 프레임을 순서대로 내보낸다.
        with self.frame_lock:
            held, self.pending = self.pending, []
        self.accepted.set()
        for data in held:
            self._publish(data)
        if held:
            self.get_logger().info(f"수락 전 프레임 {len(held)}개를 내보냈다")

    def _goal_result(self, fut):
        r = fut.result().result
        lvl = self.get_logger().info if r.success else self.get_logger().error
        lvl(f"재생 결과 success={r.success} code={r.code} {r.message}")
        self.goal_handle = None
        self.play_done.set()

    def publish_frame(self, data):
        if len(data) != FRAME_BYTES:
            self.get_logger().error(f"보낼 프레임이 {len(data)}바이트다 — 버린다")
            return
        if not self.accepted.is_set():
            with self.frame_lock:
                self.pending.append(data)
            return
        self._publish(data)

    def _publish(self, data):
        # 합성은 재생보다 훨씬 빠르다(RTF 0.17). 만드는 대로 쏟아부으면 파이의
        # 버퍼가 넘칠 수 있으므로, 재생 시각보다 너무 앞서가지 않게 늦춘다.
        ahead = self.sequence * 0.02 - (time.time() - self.play_t0)
        if ahead > MAX_AHEAD_S:
            time.sleep(ahead - MAX_AHEAD_S)
        self.playback.publish(self._frame(data, eos=False))

    def finish_playback(self):
        # 수락을 못 받은 채 끝났다면 들고 있던 것은 버린다. 다음 재생과
        # 섞이면 순번이 어긋나 통째로 거부된다.
        if not self.accepted.is_set():
            with self.frame_lock:
                n, self.pending = len(self.pending), []
            if n:
                self.get_logger().error(f"목표가 수락되지 않아 프레임 {n}개를 버렸다")
            return
        # 계약상 EOS 는 데이터가 빈 프레임 하나다. 이걸 빠뜨리면 파이가
        # 드레인을 끝내지 못해 PlayAudio 가 완료되지 않는다.
        self.playback.publish(self._frame(b"", eos=True))

    def _frame(self, data, eos):
        msg = AudioFrame()
        msg.stamp = self.get_clock().now().to_msg()
        msg.stream_id = self.stream_id
        msg.speech_id = ""
        msg.sequence = self.sequence
        msg.sample_rate = RATE
        msg.channels = 1
        msg.encoding = "pcm_s16le"
        msg.data = list(data)
        msg.end_of_stream = eos
        self.sequence += 1
        return msg

    def cancel_playback(self):
        if self.goal_handle is not None:
            self.get_logger().info("barge-in — 재생 취소")
            self.goal_handle.cancel_goal_async()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="127.0.0.1:5150",
                    help="판단부 주소 (bench/voice_agent.py 가 띄운다)")
    args, ros_args = ap.parse_known_args()
    host, _, port = args.agent.partition(":")

    rclpy.init(args=ros_args)
    node = LampVoiceNode(host, int(port or 5150))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
