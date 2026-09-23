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
import functools
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


MIN_GAP_S = FRAME_INTERVAL_S / 2   # 연속 두 발행 사이 최소 간격


def next_due(t0, seq, now, interval=FRAME_INTERVAL_S, last=None,
             min_gap=MIN_GAP_S):
    """이 프레임을 언제 보낼지. (보낼 시각, 새 t0, 밀린 시간).

    절대 시각으로 잡아야 20ms 오차가 쌓이지 않는다. 다만 큐가 비어
    기다린 동안에는 seq 가 안 늘어나므로 일정이 통째로 밀린다. 그대로
    두면 밀린 만큼 몰아 보내게 되고, 브리지가 순서를 뒤집어
    out_of_order 로 스트림 전체가 죽는다. 한 프레임 넘게 밀렸으면
    따라잡지 않고 지금으로 다시 잡는다.

    한 프레임 안쪽으로 늦은 것은 절대 일정이 다음 간격을 줄여 따라잡는다.
    그런데 앞 프레임이 OS 스케줄링으로 거의 20ms 늦으면 다음 간격이
    거의 0 이 된다 — 두 프레임이 붙어서 나간다. 시험 60번에 한 번 EOS 가
    마지막 프레임 0.1ms 뒤에 나갔다. 브리지가 순서를 뒤집는 바로 그
    조건이라, 따라잡더라도 last 에서 min_gap 은 벌린다. 늦은 만큼은
    프레임마다 최대 (interval - min_gap) 씩 나눠서 따라잡는다.
    """
    due = t0 + seq * interval
    behind = now - due
    if behind > interval:
        due, t0 = now, now - seq * interval
    else:
        behind = 0.0
    if last is not None and due < last + min_gap:
        due = last + min_gap
    return due, t0, behind

SENDER_READY_S = 0.6
PREV_PLAY_WAIT_S = 3.0   # 앞 재생이 끝나기를 기다리는 상한
RATE = 16000
HDR_LEN = 8

# voice/link.py 와 같은 값. 이 노드는 그 모듈을 import 하지 않는다 —
# 다른 인터프리터라 경로를 끌어오면 venv 가 섞인다.
CAP_FRAME = b"CAPF"
ORIENT = b"ORNT"
SPEAK_BEGIN = b"SPKB"
SPEAK_DONE = b"SPKD"
SPEAK_AUDIO = b"SPKA"
SPEAK_END = b"SPKE"
BARGE_IN = b"BRGI"
HEARD = b"HERD"
SPEECH_ID_LEN = 36


def pack(kind, payload=b""):
    return kind + len(payload).to_bytes(4, "big") + payload


def pack_id(speech_id):
    return (speech_id or "").strip().ljust(SPEECH_ID_LEN).encode("ascii")


class Stream:
    """재생 한 번의 상태. 콜백은 자기 스트림 것만 건드린다.

    전에는 goal_handle·cancel_pending·outq·accepted·speech_id 를 노드 필드로
    두고 새 재생이 시작될 때 덮어썼다. 그러면 앞 재생의 늦은 콜백이 새
    재생의 필드를 건드린다. 실제로 난 것:

      - A 의 결과가 B 를 기다리는 3초 사이에 오면 speech_id 가 이미 B 라
        A 의 완료를 B 의 이름으로 보냈다 → B 가 시작도 전에 끝남
      - 상한이 지나 B 로 넘어가도 goal_handle 이 A 것이라, B 가 수락되기
        전에 끼어들면 A 를 또 취소하고 B 취소는 사라졌다
      - 0초 A / 1초 A 취소 / 4초 B / 5초 A 결과 → B 가 끝난 것으로 처리

    필드 하나씩 막으면 다음 수정 때 또 샌다. 스트림마다 객체를 만들고
    콜백에는 그 객체를 묶어 넘긴다.
    """

    def __init__(self, speech_id):
        self.id = str(uuid.uuid4())
        self.speech_id = speech_id
        self.q = queue.Queue()
        self.accepted = threading.Event()
        self.handle = None
        self.goal_sent = False
        self.cancel_pending = False
        self.cancelled = False
        self.reported = False          # 판단부에 결과를 알렸는가 (한 번만)
        self.sent = 0
        self.t0 = time.time()
        self.thread = None


class LampVoiceNode(Node):
    def __init__(self, host, port, pi_host="192.168.100.2"):
        super().__init__("lamp_voice")
        self.host, self.port = host, port
        self.pi_host = pi_host
        self.sock = None
        self.send_lock = threading.Lock()

        # 지금 재생. 재생마다 따로 상태를 둔다(Stream 참고).
        # 발행은 스트림마다 스레드 하나만 한다. 모아둔 것을 한 스레드가
        # 내보내는 동안 다른 스레드가 내보내면 순서가 뒤집히고 순번도
        # 겹친다. 브리지가 out_of_order 로 전부 거부했다.
        self.cur = None
        # 지금 재생이 끝났는가. 다음 재생은 이것을 기다린다(겹치면 파이가
        # audio_busy 로 거부한다). 지금 스트림의 결과만 이것을 세운다.
        self.play_done = threading.Event()
        self.play_done.set()          # 처음에는 기다릴 재생이 없다
        # 지금 재생을 바꾸는 것과 "지금 재생이 끝났다" 를 세우는 것을 묶는다.
        # 안 묶으면 앞 재생의 결과가 "지금 것인가" 를 본 뒤 세우기 전에
        # 다음 재생으로 바뀌어, 다음 재생의 완료가 켜진다.
        self.play_lock = threading.Lock()

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
            # 어느 발화인지 스트림에 묶어 두었다가 완료 신호에 되돌려 준다.
            # 노드 필드에 먼저 적으면 안 된다 — 앞 재생을 기다리는 사이에
            # 온 앞 재생의 결과가 이 이름을 달고 나간다.
            self.start_playback(
                bytes(body)[:SPEECH_ID_LEN].decode("ascii", "replace").strip())
        elif kind == SPEAK_AUDIO:
            self.publish_frame(bytes(body))
        elif kind == SPEAK_END:
            self.finish_playback()
        elif kind == BARGE_IN:
            self.cancel_playback()

    def start_playback(self, speech_id=""):
        prev = self.cur
        if prev is not None and not self.play_done.wait(PREV_PLAY_WAIT_S):
            # 무한정 기다리지 않는다 — 취소가 10초까지 걸릴 수 있고 그동안
            # 대화가 멈춘다. 앞 재생을 버리고 넘어가되, 액션 서버에 남은
            # 목표는 취소한다. 안 그러면 다음 목표가 audio_busy 로 거부된다.
            self.get_logger().warn(
                f"앞 재생이 {PREV_PLAY_WAIT_S:.0f}초 안에 안 끝난다 "
                "— 취소하고 다음으로 넘어간다")
            self._abandon(prev)

        st = Stream(speech_id)
        with self.play_lock:
            self.cur = st
            self.play_done.clear()
        st.thread = threading.Thread(target=self._publisher, args=(st,),
                                     daemon=True)
        st.thread.start()

        if not self.play_audio.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("/lamp/play_audio 가 없다")
            st.cancelled = True
            st.q.put(None)
            self._report(st, "no_server")
            return
        goal = PlayAudio.Goal(stream_id=st.id, sample_rate=RATE,
                              channels=1, encoding="pcm_s16le")
        st.goal_sent = True
        fut = self.play_audio.send_goal_async(goal)
        fut.add_done_callback(functools.partial(self._goal_accepted, st))

    def _abandon(self, st):
        """앞 재생을 버린다. 결과는 나중에 와도 그 스트림 것으로만 처리된다."""
        if st.cancelled:
            return                      # 끼어들 때 이미 취소를 보냈다
        st.cancelled = True
        st.q.put(None)
        if st.handle is not None:
            st.handle.cancel_goal_async()
        elif st.goal_sent:
            st.cancel_pending = True      # 수락되면 그때 취소한다

    def _goal_accepted(self, st, fut):
        handle = fut.result()
        if not handle.accepted:
            # 거부도 끝이다. 알리지 않으면 판단부는 상한까지 말하기 상태로
            # 남고, 다음 재생은 이 재생이 끝나기를 3초 기다린다.
            self.get_logger().error("PlayAudio 거부됨 — 프레임을 버린다")
            st.cancelled = True
            st.q.put(None)
            self._report(st, "rejected")
            return
        st.handle = handle
        handle.get_result_async().add_done_callback(
            functools.partial(self._goal_result, st))
        if st.cancel_pending or st is not self.cur:
            # 수락 응답을 기다리는 사이 취소가 들어왔거나, 그사이 다음
            # 재생으로 넘어갔다. 여기서 취소를 안 보내면 로컬 발행만 멈추고
            # 액션 서버 쪽 목표는 오디오와 EOS 를 기다리며 남는다.
            st.cancel_pending = False
            self.get_logger().info("수락된 목표를 바로 취소한다")
            handle.cancel_goal_async()
            return
        st.accepted.set()

    def _goal_result(self, st, fut):
        r = fut.result().result
        st.handle = None
        stale = st is not self.cur
        self.get_logger().info(
            f"프레임 {st.sent}개({st.sent * 0.02:.1f}초 분량)를 "
            f"{time.time() - st.t0:.1f}초에 보냈다"
            + (" — 지난 재생" if stale else ""))
        # 심각도를 골라서 한 줄에서 부르면 안 된다. rclpy 는 호출 위치로
        # 심각도를 캐싱해서, 같은 줄에서 info 를 쓰다가 error 를 쓰면
        # ValueError 를 던지고 그게 executor 를 타고 올라와 노드가 죽는다.
        # 성공하다가 한 번 실패하는 순간 죽으므로 평소에는 안 보인다.
        msg = f"재생 결과 success={r.success} code={r.code} {r.message}"
        if r.success:
            self.get_logger().info(msg)
        else:
            self.get_logger().error(msg)
        self._report(st, r.code or "")

    def _report(self, st, code):
        """이 재생이 끝났다고 판단부에 한 번만 알린다.

        지난 재생의 결과도 알린다 — 자기 발화 id 를 달고 가므로 판단부가
        알아서 거른다. 다만 play_done 은 지금 재생 것이라 지난 재생이
        세우면 안 된다. 세우면 지금 재생이 끝난 것으로 처리된다.
        """
        if st.reported:
            return
        st.reported = True
        # 액션 서버가 끝났다고 했으면 더 보내지 않는다. 보통은 EOS 뒤라
        # 발행이 이미 끝났지만, 파이가 재생 도중 실패로 끝내면 발행 스레드는
        # 다음 재생이 시작될 때까지 죽은 스트림에 계속 보낸다.
        st.cancelled = True
        st.q.put(None)
        with self.play_lock:
            if st is self.cur:
                self.play_done.set()
        self.send(SPEAK_DONE, pack_id(st.speech_id) + code.encode("utf-8"))

    def _publisher(self, st):
        """이 스트림의 발행은 이 스레드만 한다. 순번도 스트림마다 따로다.

        앞서 self.sequence 를 공유하다가 이전 스트림의 발행 스레드가 아직
        살아 있을 때 둘이 같은 카운터를 증가시켰다. 브리지는 순번이 다음 것이
        아니면 거부하므로 out_of_order 로 전부 버려졌다.
        """
        if not st.accepted.wait(5.0):
            return                      # 거부·취소·수락 안 됨
        if st is not self.cur or st.cancelled:
            return
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
        last = None                     # 마지막으로 발행한 시각
        while True:
            data = st.q.get()
            if st.cancelled or st is not self.cur:
                return
            if data is None:            # 끝 신호
                # EOS 도 자기 20ms 자리를 지킨다. 마지막 오디오 프레임 바로
                # 뒤에 붙여 보내면 두 콜백이 겹쳐 실행될 수 있고, 브리지는
                # ReentrantCallbackGroup + MultiThreadedExecutor 라 순서가
                # 뒤집힐 여지가 있다. EOS 가 먼저 처리되면 마지막 프레임이
                # out_of_order 로 거부되고 파이가 드레인을 못 끝낸다.
                #
                # 실기기에서 역전을 본 적은 없다. 다만 간격을 지키는 것이
                # 공짜라(20ms) 확인되지 않은 경합을 남겨 둘 이유가 없다.
                due, t0, _ = next_due(t0, seq, time.time(), last=last)
                delay = due - time.time()
                if delay > 0:
                    time.sleep(delay)
                self.playback.publish(self._frame(b"", True, st.id, seq))
                st.sent = seq + 1
                return
            # 다음 프레임 시각까지 기다린다. 절대 시각으로 잡아야 오차가
            # 쌓이지 않는다.
            if seq == 0:
                self.get_logger().info(
                    f"첫 프레임 발행까지 {t_ready - st.t0:.2f}초 "
                    f"(대기 {SENDER_READY_S:.1f}초 포함)")
            due, t0, behind = next_due(t0, seq, time.time(), last=last)
            if behind > 0.2:
                self.get_logger().info(
                    f"큐가 {behind:.1f}초 비었다 — 일정을 다시 잡는다"
                    " (몰아 보내면 브리지가 순서를 뒤집는다)")
            delay = due - time.time()
            if delay > 0:
                time.sleep(delay)
            self.playback.publish(self._frame(data, False, st.id, seq))
            last = time.time()
            seq += 1
            st.sent = seq

    def publish_frame(self, data):
        if len(data) != FRAME_BYTES:
            self.get_logger().error(f"보낼 프레임이 {len(data)}바이트다 — 버린다")
            return
        if self.cur is not None:
            self.cur.q.put(data)

    def finish_playback(self):
        if self.cur is not None:
            self.cur.q.put(None)

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
        st = self.cur
        if st is None or st.reported:
            return
        st.cancelled = True
        st.q.put(None)             # 발행 스레드를 깨워 끝낸다
        if st.handle is not None:
            self.get_logger().info("barge-in — 재생 취소")
            st.handle.cancel_goal_async()
            return
        # 목표를 보냈는데 아직 수락 응답이 안 온 구간이다. 여기서 그냥
        # 끝내면 로컬 발행만 멈추고 액션 서버 쪽 목표는 살아 있다. 그
        # 목표는 오디오 프레임과 EOS 를 기다리며 남아, 다음 재생이
        # audio_busy 로 거부된다. 수락되면 바로 취소하도록 남겨 둔다.
        if st.goal_sent:
            st.cancel_pending = True
            self.get_logger().info("barge-in — 수락 전이라 수락되면 취소한다")
            return
        self._report(st, "cancelled")

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
