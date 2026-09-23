"""재생 경로만 시험하는 최소 스크립트.

    source /opt/ros/jazzy/setup.bash
    source ~/talking-lamp-integration/jetson_ws/install/setup.bash
    python3 ros/playback_probe.py                # 기본: 1초 440Hz 톤
    python3 ros/playback_probe.py --wait 2.0     # 수락 후 대기를 늘려서
    python3 ros/playback_probe.py --seconds 3    # 더 길게

우리 판단부도, 마이크도, 상태 기계도 쓰지 않는다. PlayAudio 목표를 걸고
톤 프레임을 순번대로 발행하고 결과를 찍는 것이 전부다. 이것이 되면 경로는
멀쩡하고 우리 노드 쪽이 문제이며, 이것도 안 되면 브리지 문제다.

모든 발행을 한 스레드에서 하고 20ms 간격으로 내보낸다.
"""
import argparse
import math
import sys
import threading
import time
import uuid

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from lamp_interfaces.action import PlayAudio
from lamp_interfaces.msg import AudioFrame

RATE = 16000
SAMPLES = 320               # 20 ms
FRAME_BYTES = SAMPLES * 2


def tone_frames(seconds, freq=440.0, level=0.5):
    """S16LE 640바이트 프레임 목록."""
    out = []
    n = int(seconds * RATE)
    for start in range(0, n - SAMPLES + 1, SAMPLES):
        buf = bytearray()
        for i in range(start, start + SAMPLES):
            v = int(level * 32767 * math.sin(2 * math.pi * freq * i / RATE))
            buf += int(v).to_bytes(2, "little", signed=True)
        out.append(bytes(buf))
    return out


class Probe(Node):
    def __init__(self):
        super().__init__("playback_probe")
        self.pub = self.create_publisher(
            AudioFrame, "/lamp/audio/playback_frames",
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=400))
        self.client = ActionClient(self, PlayAudio, "/lamp/play_audio")
        self.accepted = threading.Event()
        self.done = threading.Event()
        self.result = None
        self.stream_id = ""

    def frame(self, data, seq, eos):
        m = AudioFrame()
        m.stamp = self.get_clock().now().to_msg()
        m.stream_id = self.stream_id
        m.speech_id = ""
        m.sequence = seq
        m.sample_rate = RATE
        m.channels = 1
        m.encoding = "pcm_s16le"
        m.data = list(data)
        m.end_of_stream = eos
        return m

    def run(self, frames, wait_s, pace):
        if not self.client.wait_for_server(timeout_sec=5.0):
            print("  ✗ /lamp/play_audio 가 없다")
            return 1
        self.stream_id = str(uuid.uuid4())
        print(f"  목표 전송 stream={self.stream_id[:8]}")
        fut = self.client.send_goal_async(
            PlayAudio.Goal(stream_id=self.stream_id, sample_rate=RATE,
                           channels=1, encoding="pcm_s16le"))
        fut.add_done_callback(self._accepted)
        if not self.accepted.wait(10.0):
            print("  ✗ 수락되지 않았다")
            return 1
        print(f"  수락됨. {wait_s:.1f}초 기다린 뒤 발행")
        time.sleep(wait_s)

        t0 = time.time()
        for seq, data in enumerate(frames):
            if seq < 3:
                print(f"    발행 seq={seq} {len(data)}바이트")
            self.pub.publish(self.frame(data, seq, False))
            if pace:
                nxt = t0 + (seq + 1) * 0.02
                d = nxt - time.time()
                if d > 0:
                    time.sleep(d)
        print(f"    발행 EOS seq={len(frames)}")
        self.pub.publish(self.frame(b"", len(frames), True))
        print(f"  {len(frames)}프레임({len(frames)*0.02:.1f}초)을 "
              f"{time.time()-t0:.1f}초에 발행")

        if not self.done.wait(20.0):
            print("  ✗ 결과가 오지 않았다")
            return 1
        r = self.result
        print(f"\n  결과 success={r.success} code={r.code} {r.message}")
        return 0 if r.success else 1

    def _accepted(self, fut):
        h = fut.result()
        if not h.accepted:
            print("  ✗ 목표 거부됨")
            self.done.set()
            return
        h.get_result_async().add_done_callback(self._result)
        self.accepted.set()

    def _result(self, fut):
        self.result = fut.result().result
        self.done.set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--freq", type=float, default=440.0)
    ap.add_argument("--wait", type=float, default=1.0,
                    help="수락 후 발행까지 기다리는 초")
    ap.add_argument("--no-pace", action="store_true",
                    help="20ms 간격을 두지 않고 한꺼번에 발행")
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = Probe()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    frames = tone_frames(args.seconds, args.freq)
    print(f"  {args.freq:.0f}Hz {args.seconds:.1f}초 → {len(frames)}프레임\n")
    try:
        rc = node.run(frames, args.wait, not args.no_pace)
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
