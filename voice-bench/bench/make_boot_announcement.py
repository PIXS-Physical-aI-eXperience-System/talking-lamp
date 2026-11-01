"""부팅 안내 음성을 한 번 만들어 둔다.

    venvs/melo-onnx/bin/python bench/make_boot_announcement.py

부팅 때는 TTS 가 아직 올라오지 않았다. 모델을 다 올리는 데 3~4분이 걸리고
그동안 불러도 대답이 없으니, 켜지자마자 기다려 달라고 말해 두려면 미리
합성한 소리가 있어야 한다. 그래서 같은 목소리로 한 번 만들어 파일로 둔다.

파이가 부팅 15초 만에 이 파일을 aplay 로 직접 튼다
(deploy/pi/talking-lamp-boot-announce.service). 젯슨 ROS 노드는 부팅 후
2분 넘게 지나야 뜨므로 거기서 틀면 늦다(2026-11-01 실측). 파이 스피커
경로와 같은 16 kHz 모노 S16LE 로 쓴다.
"""
import argparse
import os
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np                        # noqa: E402

from voice import link                    # noqa: E402
from voice.audio import resample          # noqa: E402
from voice.tts import Tts, split_sentences  # noqa: E402

TEXT = "픽스가 깨어나는 중이에요. 다 켜질 때까지 3분쯤 걸려요. 조금만 기다려 주세요."
GAP_S = 0.25                              # 문장 사이 쉼


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(ROOT), "deploy/pi/boot-announce.wav"))
    args = ap.parse_args()

    # 램프가 떠 있는 채로 만들 수 있게 GPU 를 건드리지 않는다.
    tts = Tts(model_dir=os.path.join(ROOT, "models/melo-ko-onnx"),
              providers="CPUExecutionProvider")
    gap = np.zeros(int(GAP_S * link.RATE), dtype=np.float32)
    parts = []
    for part in split_sentences(args.text):
        wav = resample(np.asarray(tts.synth(part), dtype=np.float32),
                       tts.samplerate, link.RATE)
        parts += [wav, gap]
    pcm = link.pcm_from(np.concatenate(parts))

    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(link.RATE)
        w.writeframes(pcm)
    print(f"  {args.out}  {len(pcm) / 2 / link.RATE:.1f}초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
