"""웨이크워드 "픽스야" 학습용 녹음 — 실제 마이크 경로로.

    source /opt/ros/jazzy/setup.bash
    source ~/talking-lamp-integration/jetson_ws/install/setup.bash
    python3 ros/wake_record_ros.py --name 최승원

노트북 내장 마이크로 녹음해서 학습하면 안 된다. 실제로 램프가 듣는 소리는
XVF3800 의 잡음·에코 제거를 거치고 Opus 로 압축된 뒤 16 kHz 로 풀린 것이다.
성격이 다른 소리로 학습한 모델은 실제 환경에서 잘 깨어나지 않는다.

여기서는 /lamp/audio/capture 를 그대로 받는다. 배포될 때와 같은 소리다.

전제: 파이 device 서비스와 젯슨 브리지가 떠 있어야 한다.
      (판단부와 lamp_voice_node 는 꺼도 된다)
"""
import argparse
import os
import sys
import threading
import time
import wave

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from lamp_interfaces.msg import AudioFrame

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from bench.wake_record import (NEGATIVE_SENTENCES, NEGATIVE_WORDS,  # noqa: E402
                               PHRASE, POSITIVE_STYLES)

RATE = 16000
TAKE_S = 2.0


class Recorder(Node):
    def __init__(self):
        super().__init__("wake_recorder")
        self.buf = []
        self.collecting = False
        self.lock = threading.Lock()
        self.frames_seen = 0
        self.create_subscription(
            AudioFrame, "/lamp/audio/capture", self.on_frame,
            QoSPresetProfiles.SENSOR_DATA.value)

    def on_frame(self, msg):
        if msg.end_of_stream or len(msg.data) != 640:
            return
        self.frames_seen += 1
        with self.lock:
            if self.collecting:
                self.buf.append(bytes(msg.data))

    def take(self, seconds):
        with self.lock:
            self.buf = []
            self.collecting = True
        time.sleep(seconds)
        with self.lock:
            self.collecting = False
            data = b"".join(self.buf)
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        return x


def level_db(x):
    if len(x) == 0:
        return -120.0
    return 20 * np.log10(float(np.sqrt(np.mean(np.square(x)))) + 1e-12)


def save(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def prompt_take(rec, label, path, seconds=TAKE_S, quiet_ok=False):
    while True:
        input(f"    {label}  → Enter 누르고 말하기 ")
        print("      ● 녹음 중…", end="", flush=True)
        x = rec.take(seconds)
        db = level_db(x)
        print(f" 끝  ({db:.0f} dB, {len(x)/RATE:.1f}초)")
        if len(x) < RATE * seconds * 0.5:
            print("      ! 프레임이 모자란다. 브리지가 살아 있는지 확인할 것")
            continue
        if not quiet_ok and db < -50:
            print("      ! 너무 작다. 마이크 쪽을 보고 다시.")
            continue
        if float(np.max(np.abs(x))) > 0.99:
            print("      ! 소리가 넘쳤다. 조금 작게 다시.")
            continue
        save(path, x)
        return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="wake-data")
    ap.add_argument("--positives", type=int, default=20)
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    rec = Recorder()
    threading.Thread(target=rclpy.spin, args=(rec,), daemon=True).start()

    print("마이크 연결 확인 중…")
    time.sleep(2.0)
    if rec.frames_seen == 0:
        print("  ✗ /lamp/audio/capture 에서 프레임이 안 온다.")
        print("    파이 device 서비스와 젯슨 브리지가 떠 있는지 확인할 것")
        return 1
    print(f"  ✔ 프레임 {rec.frames_seen}개 수신 중\n")

    base = os.path.join(ROOT, args.out, args.name)
    print(f"웨이크워드 녹음 — {args.name}")
    print(f"  저장 위치 {base}")
    print("  램프 마이크에서 30~50 cm, 평소 말하는 자리에서 하세요.\n")

    print("[1/3] 방 소리 10초 — 아무 말도 하지 마세요")
    input("    준비되면 Enter ")
    print("      ● 녹음 중…", end="", flush=True)
    n = rec.take(10.0)
    print(f" 끝  ({level_db(n):.0f} dB)")
    save(os.path.join(base, "noise.wav"), n)

    print(f"\n[2/3] \"{PHRASE}\" {args.positives}번 — 말투를 바꿔 가며")
    for i in range(args.positives):
        style = POSITIVE_STYLES[i % len(POSITIVE_STYLES)]
        prompt_take(rec, f"[{i+1:>2}/{args.positives}] \"{PHRASE}\" — {style}",
                    os.path.join(base, "pos", f"{i:02d}.wav"))

    print(f"\n[3/3] 헷갈리는 말 {len(NEGATIVE_WORDS)}개 + 문장 {len(NEGATIVE_SENTENCES)}개")
    print("     이게 없으면 \"픽\" 이나 \"믹스\" 에도 깨어난다.")
    for i, w in enumerate(NEGATIVE_WORDS):
        prompt_take(rec, f"[{i+1:>2}/{len(NEGATIVE_WORDS)}] \"{w}\"",
                    os.path.join(base, "neg", f"w{i:02d}.wav"))
    for i, s in enumerate(NEGATIVE_SENTENCES):
        prompt_take(rec, f"[{i+1:>2}/{len(NEGATIVE_SENTENCES)}] \"{s}\"",
                    os.path.join(base, "neg", f"s{i:02d}.wav"), seconds=3.0)

    npos = len(os.listdir(os.path.join(base, "pos")))
    nneg = len(os.listdir(os.path.join(base, "neg")))
    print(f"\n끝났습니다. 긍정 {npos}개, 부정 {nneg}개, 잡음 1개")
    print(f"  {base}")
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
