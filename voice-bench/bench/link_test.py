"""파이↔젯슨 규약을 하드웨어 없이 확인한다.

가짜 STT·TTS 로 서버를 띄우고 원격 대역으로 호출해 본다. 마이크도 GPU 도
필요 없으므로 맥에서도 돌고, 규약을 고칠 때마다 여기서 먼저 깨진다.

    venvs/melo-onnx/bin/python bench/link_test.py
"""
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from voice.remote import RemoteLink, RemoteStt, RemoteTts   # noqa: E402
from voice.server import VoiceServer                        # noqa: E402


class FakeStt:
    def transcribe(self, audio, sr=16000):
        return f"{len(audio) / sr:.1f}초짜리 소리"


class FakeTts:
    samplerate = 16000

    def synth(self, text):
        n = max(1, int(0.05 * len(text) * self.samplerate))
        return (0.1 * np.sin(2 * np.pi * 440 * np.arange(n) / self.samplerate)).astype(np.float32)


class FakePlayer:
    def __init__(self):
        self.chunks = []

    def put(self, audio, sr):
        self.chunks.append(audio)


def main() -> int:
    port = 5199
    srv = VoiceServer(FakeStt(), FakeTts(), "127.0.0.1", port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.4)

    conn = RemoteLink("127.0.0.1", port)
    stt, tts = RemoteStt(conn), RemoteTts(conn)
    fails = []

    audio = np.zeros(16000 * 3, dtype=np.float32)
    got = stt.transcribe(audio, 16000)
    print(f"  STT: {got!r}")
    if "3.0초" not in got:
        fails.append("STT 왕복")

    p, marks = FakePlayer(), []
    t0 = time.perf_counter()
    ok = tts.speak("아직 작업 중이시네요. 커피 한 잔 하시는 건 어때요? 좀 더 밝게 해드릴까요?",
                   p, None, on_first=lambda: marks.append(time.perf_counter() - t0))
    print(f"  TTS: 성공={ok} 조각 {len(p.chunks)}개 첫 조각 {marks[0]*1000:.0f}ms")
    if not ok or len(p.chunks) < 2:
        fails.append("TTS 분할 전송")

    stop = threading.Event()
    stop.set()
    if tts.speak("보내면 안 된다", FakePlayer(), stop) is not False:
        fails.append("정지 상태에서 요청을 보냈다")
    print("  정지 상태: 요청 보내지 않음 ✔")

    if "3.0초" not in stt.transcribe(audio, 16000):
        fails.append("재연결")
    print("  재연결 ✔")

    print()
    if fails:
        print("  ✗ 실패:", ", ".join(fails))
        return 1
    print("  ✔ 규약 이상 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
