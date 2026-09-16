"""파이프라인 골격을 실제로 돌려 본다.

    venvs/melo-onnx/bin/python bench/pipeline_demo.py
    venvs/melo-onnx/bin/python bench/pipeline_demo.py --wake-model models/wake/pixs-ya.onnx

웨이크워드 모델이 없으면 대역으로 돈다 — 아무 말에나 깨어난다. 파이프라인이
끝까지 도는지 확인하는 용도이며, 그 상태는 제품이 아니다.

'생각' 자리는 VLM(A 파트)이다. 여기서는 되받아 말하는 것으로 대신한다.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from voice import VoicePipeline          # noqa: E402
from voice.stt import Stt                # noqa: E402
from voice.tts import Tts                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wake-model", default="models/wake/pixs-ya.onnx")
    ap.add_argument("--stt-device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    ap.add_argument("--say", help="시작하자마자 이 문장을 말한다 (TTS 만 확인)")
    args = ap.parse_args()

    print("적재 중…")
    t0 = time.time()
    tts = Tts(providers=args.providers)
    stt = Stt(device=args.stt_device)
    print(f"  TTS {tts.providers}  {tts.load_s}s")
    print(f"  STT {stt.name}")
    print(f"  합계 {time.time() - t0:.1f}s\n")

    def think(text):
        print(f"  들은 말: {text}")
        return f"{text}, 라고 하셨네요."

    mark = {"대기": "·", "듣기": "◉", "생각": "…", "말하기": "▶"}
    pipe = VoicePipeline(
        stt, tts,
        wake_model=args.wake_model if os.path.exists(args.wake_model) else None,
        on_utterance=think,
        on_state=lambda s: print(f"  [{mark.get(s, ' ')}] {s}"))

    pipe.start()
    if args.say:
        pipe.say(args.say)
    print('  준비됐다. "픽스야" 라고 부른 뒤 말해 보라. Ctrl+C 로 종료.\n')
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n종료 중…")
        pipe.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
