"""젯슨에서 띄우는 음성 서버. 파이가 붙어서 STT·TTS 를 요청한다.

    venvs/melo-onnx/bin/python bench/voice_server.py

모델을 한 번만 올리고 계속 떠 있는다. 적재에 40초 넘게 걸리므로 요청마다
올리는 구조로는 대화가 안 된다.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from voice.server import VoiceServer   # noqa: E402
from voice.stt import Stt              # noqa: E402
from voice.tts import Tts              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5150)
    ap.add_argument("--stt-device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    args = ap.parse_args()

    print("적재 중…")
    t0 = time.time()
    tts = Tts(providers=args.providers)
    stt = Stt(device=args.stt_device)
    print(f"  TTS {tts.providers}  {tts.load_s}s")
    print(f"  STT {stt.name}")
    print(f"  합계 {time.time() - t0:.1f}s\n")

    VoiceServer(stt, tts, args.host, args.port).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
