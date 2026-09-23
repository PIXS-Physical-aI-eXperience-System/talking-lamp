"""sherpa 한국어 모델의 VITS 샘플링 파라미터를 바꿔가며 변형을 만든다.

모델은 mimic3 ko_KO/kss_low 하나뿐이고 단일 화자라 '다른 목소리'로는 못 바꾼다.
대신 VITS의 샘플링 파라미터 3개로 발화 스타일이 꽤 달라진다.

  noise_scale    표현 변동. 낮추면 안정적·단조로움, 높이면 표현력·불안정
  noise_scale_w  운율(길이) 변동. 낮추면 또박또박, 높이면 자연스럽지만 흔들림
  length_scale   말 속도. 높이면 느려지고 조음이 또렷해진다
"""
import os
import sys

import sherpa_onnx
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # voice-bench/ — 공용 모듈과 산출물이 있는 곳
sys.path.insert(0, ROOT)
from common import load_sentences, repo_paths          # noqa: E402
from ko_normalize import normalize                     # noqa: E402

D = os.path.join(ROOT, "models", "vits-mimic3-ko_KO-kss_low")

# 사용자 평가가 "약간 어눌하다"였으므로, 변동을 줄이고 속도를 늦추는 쪽을 중심에 둔다.
VARIANTS = [
    ("1-default",   0.667, 0.8, 1.00),   # 기준
    ("2-clear",     0.333, 0.4, 1.00),   # 변동 억제
    ("3-clear-slow",0.333, 0.4, 1.15),   # 변동 억제 + 느리게
    ("4-crisp",     0.200, 0.3, 1.10),   # 더 강하게 억제
    ("5-slow",      0.667, 0.8, 1.20),   # 속도만 느리게
    ("6-expressive",0.900, 1.0, 1.00),   # 반대 방향 — 표현력 강화
]


def build(noise, noise_w, length):
    onnx = [f for f in os.listdir(D) if f.endswith(".onnx")][0]
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=os.path.join(D, onnx),
                tokens=os.path.join(D, "tokens.txt"),
                data_dir=os.path.join(D, "espeak-ng-data"),
                noise_scale=noise, noise_scale_w=noise_w, length_scale=length),
            num_threads=2)))


def main() -> int:
    sents = [normalize(s) for s in load_sentences(repo_paths()["sentences"])]
    print(f"{'변형':<14}{'noise':>7}{'noise_w':>9}{'length':>8}   길이(2번 문장)")
    print("-" * 58)
    for name, n, nw, ls in VARIANTS:
        out = os.path.join(ROOT, "out", "tts", f"sherpa-{name}")
        os.makedirs(out, exist_ok=True)
        tts = build(n, nw, ls)
        dur2 = 0.0
        for i, s in enumerate(sents, 1):
            a = tts.generate(s, sid=0, speed=1.0)
            sf.write(os.path.join(out, f"{i:02d}.wav"), a.samples, a.sample_rate)
            if i == 2:
                dur2 = len(a.samples) / a.sample_rate
        print(f"{name:<14}{n:>7.3f}{nw:>9.1f}{ls:>8.2f}   {dur2:>5.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
