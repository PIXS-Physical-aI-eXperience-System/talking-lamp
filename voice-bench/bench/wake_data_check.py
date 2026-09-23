"""모인 웨이크워드 녹음이 쓸 만한지 확인한다.

    venvs/melo-onnx/bin/python bench/wake_data_check.py

사람마다 다른 기기로 녹음해 오므로 조건이 제각각이다. 학습에 넣기 전에
못 쓸 것을 걸러야 한다. 특히 무음은 조용히 섞여 들어가 모델을 망친다 —
실제로 첫 녹음에서 방 소리가 -240 dB(완전한 디지털 무음)로 나왔다.
맥이 첫 접근 때 마이크 권한을 묻느라 그 10초가 통째로 빈 것이었다.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from wake_data import people  # noqa: E402


def load(path):
    import wave
    with wave.open(path, "rb") as w:
        n, ch, width, sr = (w.getnframes(), w.getnchannels(),
                            w.getsampwidth(), w.getframerate())
        raw = w.readframes(n)
    if width != 2:
        return None, sr, ch, width
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr, ch, width


def db(x):
    return 20 * np.log10(float(np.sqrt(np.mean(np.square(x)))) + 1e-12)


def check(path):
    """(문제 목록, 요약) — 문제가 비면 쓸 수 있다."""
    x, sr, ch, width = load(path)
    bad = []
    if x is None:
        return [f"16비트가 아니다({width*8}비트)"], ""
    if sr != 16000:
        bad.append(f"{sr}Hz (16000 이어야 한다)")
    level = db(x)
    zeros = float(np.count_nonzero(x == 0.0)) / max(len(x), 1)
    if level < -70:
        bad.append(f"무음에 가깝다 ({level:.0f} dB)")
    elif level < -50:
        bad.append(f"너무 작다 ({level:.0f} dB)")
    if zeros > 0.2:
        bad.append(f"0인 표본 {zeros*100:.0f}%")
    if float(np.max(np.abs(x))) > 0.999:
        bad.append("소리가 넘쳤다(클리핑)")
    return bad, f"{len(x)/sr:.1f}s {level:.0f}dB"


def main() -> int:
    base = os.path.join(ROOT, "wake-data")
    everyone = people(base)
    if not everyone:
        print(f"{base} 에 아무것도 없다")
        return 1

    total_pos = total_neg = 0
    problems = []
    print(f"{'사람':<14}{'긍정':>6}{'부정':>6}{'잡음':>6}   문제")
    print("-" * 56)
    for name, pos, neg, noise in everyone:
        bad_here = []
        for f in pos + neg + noise:
            bad, _ = check(f)
            if bad:
                bad_here.append(f"{os.path.relpath(f, base)}: {', '.join(bad)}")
        total_pos += len(pos)
        total_neg += len(neg)
        mark = "" if not bad_here else f"{len(bad_here)}개 파일"
        if not noise:
            mark = (mark + " / 잡음 없음").strip(" /")
        print(f"{name:<14}{len(pos):>6}{len(neg):>6}{len(noise):>6}   {mark}")
        problems += bad_here

    print(f"\n합계 긍정 {total_pos}개, 부정 {total_neg}개, 사람 {len(everyone)}명")
    if problems:
        print(f"\n문제 {len(problems)}건:")
        for p in problems[:20]:
            print(f"  {p}")
        if len(problems) > 20:
            print(f"  … 외 {len(problems)-20}건")
    # 학습에 필요한 최소량은 경험칙이다. 적으면 적은 대로 만들되 알고 만든다.
    print()
    if total_pos < 100:
        print(f"  · 긍정 {total_pos}개는 적다. 사람이 늘수록 다른 목소리에 반응한다.")
    if len(everyone) < 3:
        print(f"  · {len(everyone)}명 분량이다. 이 목소리들에만 반응할 가능성이 크다.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
