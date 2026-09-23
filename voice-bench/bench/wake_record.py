"""웨이크워드 "픽스야" 학습용 녹음.

    python bench/wake_record.py --name 최승원

사람마다 제각각으로 녹음해 오면 못 쓴다. 길이·표본율·조건을 여기서 고정한다.

무엇을 모으는가:

  긍정  "픽스야" — 평소 말투, 조용히, 크게, 빠르게, 천천히, 멀리서.
        한 가지 말투만 모으면 그 말투에만 반응하는 모델이 된다.

  부정  비슷하게 들리는 말과 평범한 대화.
        부정 표본이 없으면 "픽" 이나 "믹스" 에도 깨어난다. 기성 영어 모델이
        한국어 발화에 6문장 중 2회 잘못 깨어난 것이 그 예다.

  잡음  방 안의 무음 10초. 학습 증강에도 쓰고, VAD 임계값을 정하는 데도 쓴다
        (예전 VAD 측정은 녹음에 선행 무음이 없어서 아무것도 재지 못했다).

16 kHz 모노로 저장한다. 램프 마이크가 16 kHz 이고, 학습·추론이 같은 표본율
이어야 한다.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RATE = 16000
TAKE_S = 2.0

PHRASE = "픽스야"

# 말투를 흩어 놓는다. 같은 조로만 20번 말하면 표본 20개가 아니라 1개나 다름없다.
POSITIVE_STYLES = [
    "평소 말투로",
    "평소 말투로",
    "조금 빠르게",
    "조금 천천히",
    "작은 목소리로",
    "조금 크게",
    "2~3 m 떨어져서",
    "고개를 돌린 채로",
    "말끝을 올려서 (픽스야?)",
    "혼잣말하듯 툭",
]

# 비슷하게 들리는 말. 이게 부정 표본의 핵심이다 — 평범한 문장보다 훨씬 중요하다.
NEGATIVE_WORDS = [
    "픽스", "픽", "믹스", "식스", "픽셀", "피식",
    "박스", "픽스터", "빅스", "키스야", "미스야", "십시오",
]

NEGATIVE_SENTENCES = [
    "왼쪽 좀 더 밝게 비춰줘.",
    "아직 작업 중이시네요.",
    "USB 케이블이랑 노트북 좀 찾아줘.",
    "오늘 회의가 세 시였나 네 시였나.",
    "그거 어디에 뒀는지 기억나?",
    "잠깐만, 이것만 마무리하고.",
]


def record(seconds, rate=RATE):
    import sounddevice as sd
    x = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32")
    sd.wait()
    return x[:, 0]


def level_db(x):
    return 20 * np.log10(float(np.sqrt(np.mean(np.square(x)))) + 1e-12)


def save(path, x, rate=RATE):
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x, rate)


def take(prompt, path, seconds=TAKE_S, quiet_ok=False):
    """한 번 녹음하고, 너무 작거나 넘치면 다시 하게 한다."""
    while True:
        input(f"    {prompt}  → Enter 누르고 말하기 ")
        print("      ● 녹음 중…", end="", flush=True)
        x = record(seconds)
        db = level_db(x)
        peak = float(np.max(np.abs(x)))
        print(f" 끝  ({db:.0f} dB)")
        if not quiet_ok and db < -45:
            print("      ! 너무 작다. 마이크에 더 가까이서 다시.")
            continue
        if peak > 0.99:
            print("      ! 소리가 넘쳤다(클리핑). 조금 작게 다시.")
            continue
        save(path, x)
        return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="녹음하는 사람 (파일 폴더 이름)")
    ap.add_argument("--out", default="wake-data")
    ap.add_argument("--positives", type=int, default=20)
    ap.add_argument("--skip-noise", action="store_true")
    args = ap.parse_args()

    try:
        import sounddevice, soundfile  # noqa: F401
    except Exception as e:
        print(f"필요한 것이 없다: {e}\n  pip install sounddevice soundfile numpy")
        return 1

    base = os.path.join(ROOT, args.out, args.name)
    print(f"\n웨이크워드 녹음 — {args.name}")
    print(f"  저장 위치 {base}")
    print(f"  조용한 방에서, 마이크와 30~50 cm 거리에서 하세요.\n")

    # ── 1. 잡음 ────────────────────────────────────────────
    if not args.skip_noise:
        print("[1/3] 방 소리 10초 — 아무 말도 하지 마세요")
        input("    준비되면 Enter ")
        print("      ● 녹음 중…", end="", flush=True)
        n = record(10.0)
        print(f" 끝  ({level_db(n):.0f} dB)")
        save(os.path.join(base, "noise.wav"), n)
        if level_db(n) > -40:
            print("      ! 방이 시끄럽다. 더 조용한 곳을 권한다(계속 진행은 가능)")

    # ── 2. 긍정 ────────────────────────────────────────────
    print(f"\n[2/3] \"{PHRASE}\" {args.positives}번 — 말투를 바꿔 가며")
    for i in range(args.positives):
        style = POSITIVE_STYLES[i % len(POSITIVE_STYLES)]
        take(f"[{i+1:>2}/{args.positives}] \"{PHRASE}\" — {style}",
             os.path.join(base, "pos", f"{i:02d}.wav"))

    # ── 3. 부정 ────────────────────────────────────────────
    print(f"\n[3/3] 헷갈리는 말 {len(NEGATIVE_WORDS)}개 + 평범한 문장 {len(NEGATIVE_SENTENCES)}개")
    print("     이게 없으면 \"픽\" 이나 \"믹스\" 에도 깨어난다.")
    for i, w in enumerate(NEGATIVE_WORDS):
        take(f"[{i+1:>2}/{len(NEGATIVE_WORDS)}] \"{w}\"",
             os.path.join(base, "neg", f"w{i:02d}.wav"))
    for i, s in enumerate(NEGATIVE_SENTENCES):
        take(f"[{i+1:>2}/{len(NEGATIVE_SENTENCES)}] \"{s}\"",
             os.path.join(base, "neg", f"s{i:02d}.wav"), seconds=3.0)

    npos = len(os.listdir(os.path.join(base, "pos")))
    nneg = len(os.listdir(os.path.join(base, "neg")))
    print(f"\n끝났습니다. 긍정 {npos}개, 부정 {nneg}개, 잡음 1개")
    print(f"  {base} 폴더를 통째로 최승원에게 보내주세요.")
    print(f"  (폴더 이름이 곧 사람 이름입니다)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
