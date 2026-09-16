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
import queue
import socket
import sys
import threading
import time
import uuid

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (HistoryPolicy, QoSPresetProfiles, QoSProfile,
                       ReliabilityPolicy)

from lamp_interfaces.action import PlayAudio
from lamp_interfaces.msg import AudioFrame, OrientationStatus

FRAME_BYTES = 640          # 20ms @ 16kHz 모노 s16le. 이 크기가 아니면 거부된다
# 프레임은 반드시 20 ms 간격으로 내보낸다. 몰아서 보내면 안 된다.
#
# 브리지는 ReentrantCallbackGroup 과 다중 스레드 실행기를 쓴다. 구독 콜백이
# 병렬로 돌기 때문에, 프레임이 한꺼번에 도착하면 서로 다른 스레드가 순서를
# 뒤집어 처리하고 검사기가 out_of_order 로 거부한다. 한 번 거부되면 검사기가
# 순번을 못 올려 그 스트림이 통째로 죽는다.
#
# 최소 재현 스크립트(ros/playback_probe.py)가 이것을 갈랐다. 20 ms 간격으로
# 보내면 소리가 나고, 몰아서 보내면 out_of_order 가 난다. 앞서 보내는 것은
# 이득도 없다 — 어차피 실시간으로 재생된다.
FRAME_INTERVAL_S = 0.02

# 목표가 수락된 뒤에도 브리지가 곧바로 프레임을 받을 수 있는 것은 아니다.
# 브리지는 실행 단계에서 파이에 audio.play.start 를 보내고 GStreamer 송신기를
# 만든 다음에야 _playback_sender 를 대입한다. 그 전에 도착한 프레임은
#     if sender is None or message.stream_id != self._playback_stream: return
# 으로 조용히 버려진다. 수락은 실행보다 먼저이므로, 수락 직후 쏟아부으면
# 전부 사라진다. 실제로 그래서 파이가 RTP 를 한 개도 못 받았고, 파이프라인이
# 재생 상태에 도달하지 못해 SIGINT 에 그대로 죽었다(playback exited with -2).
#
# 브리지가 준비됐다는 신호가 없다. PlayAudio 에 received_sequence 피드백이
# 정의돼 있지만 브리지는 그것을 발행하지 않는다(publish_feedback 호출이 없다).
# 그래서 시간으로 맞추는 수밖에 없다.
#
# 그리고 한 번만 놓치면 끝이다. 브리지 검사기는 통과한 프레임에서만
# next_sequence 를 올리므로, 첫 프레임(순번 0)이 버려지면 그 뒤로 오는
# 1, 2, 3 이 전부 out_of_order 로 거부되고 스트림이 통째로 죽는다.
# 그래서 넉넉하게 잡는다. 늦게 말하는 것이 아예 말 못 하는 것보다 낫다.
# 브리지 송신기가 만들어질 때까지 기다리는 시간. 준비 신호가 없어서 짐작이다.
# 이 시간이 그대로 응답 지연에 얹히므로 짧을수록 좋지만, 모자라면 첫 프레임이
# 버려져 스트림이 통째로 죽는다. TTS 합성과 겹치므로 실제 손해는 이보다 작다.
SENDER_READY_S = 0.6
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
    def __init__(self, host, port, pi_host="192.168.100.2"):
        super().__init__("lamp_voice")
        self.host, self.port = host, port
        self.pi_host = pi_host
        self.sock = None
        self.send_lock = threading.Lock()

        self.stream_id = ""
        self.sent = 0
        self.goal_handle = None
        self.play_done = threading.Event()
        # 발행은 반드시 한 곳에서만 한다. 모아둔 것을 한 스레드가 내보내는
        # 동안 새로 온 것을 다른 스레드가 내보내면 순서가 뒤집히고 순번도
        # 겹친다. 브리지가 out_of_order 로 전부 거부했다.
        self.cancelled = False
        self.play_done.set()          # 처음에는 기다릴 재생이 없다
        self.accepted = threading.Event()
        self.outq = queue.Queue()
        self.pub_thread = None
        self.play_t0 = 0.0

        sensor = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(AudioFrame, "/lamp/audio/capture",
                                 self.on_capture, sensor)
        self.create_subscription(OrientationStatus, "/lamp/orientation_status",
                                 self.on_orientation, 10)
        # 재생 토픽은 RELIABLE 이어야 한다. SENSOR_DATA(BEST_EFFORT)로 두면
        # 구독자와 QoS 가 맞지 않아 메시지가 한 개도 나가지 않는다 —
        # "requesting incompatible QoS. No messages will be sent to it".
        # 오디오 프레임은 하나만 빠져도 순번이 어긋나 뒤가 통째로 거부되므로
        # best-effort 로 흘려보낼 성질의 것이 아니다.
        #
        # 깊이는 넉넉히 둔다. 합성이 재생보다 빨라(RTF 0.17) 한꺼번에 몰릴 수 있다.
        self.playback = self.create_publisher(
            AudioFrame, "/lamp/audio/playback_frames",
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=200))
        self.play_audio = ActionClient(self, PlayAudio, "/lamp/play_audio")

        # 아무 일도 안 일어날 때 어디서 멈췄는지 알 수 없었다. 5초마다
        # 받은 프레임 수와 발화 표시 여부를 찍는다.
        self.n_frames = 0
        self.n_speech = 0
        self.n_bad = 0
        self.create_timer(5.0, self._heartbeat)

        self.connect()
        threading.Thread(target=self.read_loop, daemon=True).start()
        self.get_logger().info("lamp_voice 시작")

    def _heartbeat(self):
        if self.n_frames == 0:
            self.get_logger().warn(
                "마이크 프레임이 하나도 안 온다 — /lamp/audio/capture 를 확인할 것")
            return
        self.get_logger().info(
            f"프레임 {self.n_frames}개 (발화 표시 {self.n_speech}개"
            f"{f', 형식 불일치 {self.n_bad}개' if self.n_bad else ''})")
        self.n_frames = self.n_speech = self.n_bad = 0

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
        self.n_frames += 1
        if msg.speech_id:
            self.n_speech += 1
        if msg.sample_rate != RATE or msg.channels != 1 or len(msg.data) != FRAME_BYTES:
            self.n_bad += 1
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
        # 앞 재생이 아직 끝나지 않았으면 기다린다. 겹쳐서 시작하면 파이가
        # audio_busy 로 거부한다.
        if self.goal_handle is not None or not self.play_done.is_set():
            if not self.play_done.wait(3.0):
                self.get_logger().warn("앞 재생이 안 끝난다 — 그대로 진행한다")
        self.stream_id = str(uuid.uuid4())
        self.sent = 0
        self.play_done.clear()
        self.accepted.clear()
        self.cancelled = False
        self.outq = queue.Queue()
        self.play_t0 = time.time()
        # 큐와 스트림 id 를 스레드에 넘긴다. 인스턴스 변수를 함께 쓰면
        # 이전 스트림의 스레드가 살아 있을 때 섞인다.
        self.pub_thread = threading.Thread(
            target=self._publisher, args=(self.stream_id, self.outq), daemon=True)
        self.pub_thread.start()

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
            self.cancelled = True
            self.outq.put(None)
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self._goal_result)
        self.accepted.set()

    def _goal_result(self, fut):
        r = fut.result().result
        sent_s = self.sent * 0.02
        took = time.time() - self.play_t0
        self.get_logger().info(
            f"프레임 {self.sent}개({sent_s:.1f}초 분량)를 {took:.1f}초에 보냈다")
        lvl = self.get_logger().info if r.success else self.get_logger().error
        lvl(f"재생 결과 success={r.success} code={r.code} {r.message}")
        self.goal_handle = None
        self.play_done.set()

    def _publisher(self, stream_id, q):
        """이 스레드만 발행한다. 순번과 큐를 스트림마다 따로 둔다.

        앞서 self.sequence 를 공유하다가 이전 스트림의 발행 스레드가 아직
        살아 있을 때 둘이 같은 카운터를 증가시켰다. 브리지는 순번이 다음 것이
        아니면 거부하므로 out_of_order 로 전부 버려졌다.
        """
        if not self.accepted.wait(5.0):
            self.get_logger().error("수락되지 않았다 — 발행하지 않는다")
            return
        if stream_id != self.stream_id:
            return                      # 그사이 다음 스트림이 시작됐다
        # 수락돼도 브리지는 아직 송신기를 안 만들었을 수 있다. 실행 단계에서
        # 파이에 audio.play.start 를 보내고 나서야 대입하므로, 그 전에 보낸
        # 것은 sender is None 으로 조용히 버려진다. 준비됐다는 신호가 없어
        # 기다리는 수밖에 없다 — PlayAudio 에 received_sequence 피드백이
        # 정의돼 있으나 브리지는 발행하지 않는다.
        #
        # udpsink 소켓을 찾아보려 했으나 안 된다. udpsink 는 소켓을 connect
        # 하지 않고 sendto 로 보내므로 /proc/net/udp 에 상대 주소가 안 남는다.
        time.sleep(SENDER_READY_S)
        t_ready = time.time()

        t0 = time.time()
        seq = 0
        while True:
            data = q.get()
            if self.cancelled or stream_id != self.stream_id:
                return
            if data is None:            # 끝 신호
                self.playback.publish(self._frame(b"", True, stream_id, seq))
                self.sent = seq + 1
                return
            # 다음 프레임 시각까지 기다린다. 절대 시각으로 잡아야 오차가
            # 쌓이지 않는다.
            if seq == 0:
                self.get_logger().info(
                    f"첫 프레임 발행까지 {t_ready - self.play_t0:.2f}초 "
                    f"(대기 {SENDER_READY_S:.1f}초 포함)")
            due = t0 + seq * FRAME_INTERVAL_S
            delay = due - time.time()
            if delay > 0:
                time.sleep(delay)
            self.playback.publish(self._frame(data, False, stream_id, seq))
            seq += 1
            self.sent = seq

    def publish_frame(self, data):
        if len(data) != FRAME_BYTES:
            self.get_logger().error(f"보낼 프레임이 {len(data)}바이트다 — 버린다")
            return
        self.outq.put(data)

    def finish_playback(self):
        self.outq.put(None)

    def _frame(self, data, eos, stream_id, seq):
        msg = AudioFrame()
        msg.stamp = self.get_clock().now().to_msg()
        msg.stream_id = stream_id
        msg.speech_id = ""
        msg.sequence = seq
        msg.sample_rate = RATE
        msg.channels = 1
        msg.encoding = "pcm_s16le"
        msg.data = list(data)
        msg.end_of_stream = eos
        return msg

    def cancel_playback(self):
        self.cancelled = True
        self.outq.put(None)        # 발행 스레드를 깨워 끝낸다
        if self.goal_handle is not None:
            self.get_logger().info("barge-in — 재생 취소")
            self.goal_handle.cancel_goal_async()
        else:
            self.play_done.set()


def main() -> int:
    global SENDER_READY_S   # 이 이름을 쓰기 전에 선언해야 한다
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="127.0.0.1:5150",
                    help="판단부 주소 (bench/voice_agent.py 가 띄운다)")
    ap.add_argument("--ready-wait", type=float, default=SENDER_READY_S,
                    help="브리지 송신 소켓이 열릴 때까지 기다리는 최대 초")
    ap.add_argument("--pi-host", default="192.168.100.2",
                    help="송신 소켓을 찾을 때 쓰는 파이 주소")

    args, ros_args = ap.parse_known_args()
    host, _, port = args.agent.partition(":")

    SENDER_READY_S = args.ready_wait

    rclpy.init(args=ros_args)
    node = LampVoiceNode(host, int(port or 5150), args.pi_host)
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
