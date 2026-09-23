"""학습 설정 둘을 **같은 녹음 하나하나에 대해** 짝지어 비교한다.

비율만 보면 판단이 안 된다 — 문장 30개에서 4개가 1개로 준 것은 동전
던지기로도 일어난다. 같은 파일의 점수가 내려갔는지 올라갔는지 짝지어 세면
표본이 그대로여도 훨씬 예민하다.

    venvs/melo-onnx/bin/python bench/wake_compare.py \
        --a out/wake-feats.npz --b out/wake-feats-tts.npz \
        --a-name "사람 녹음만" --b-name "합성 추가"
"""
import argparse
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from wake_data import people  # noqa: E402
from wake_eval import kind, stream  # noqa: E402
from wake_train import export_onnx, read_wav, train  # noqa: E402


def maxima(cache, data, seeds):
    """{파일경로: 최고점수}. 그 사람을 뺀 모델로 재고 씨앗 여러 개를 평균낸다.

    같은 씨앗으로 두 번 돌려도 결과가 미세하게 달랐다(0.985 vs 0.986).
    BLAS 가 여러 스레드로 더하는 순서가 매번 같지 않고, 그 차이가 학습
    300회를 거치며 벌어진다. 한 번만 재면 그 잡음을 효과로 착각한다.
    """
    from openwakeword.model import Model
    d = np.load(cache, allow_pickle=True)
    X, y, who = d["X"], d["y"], d["who"]
    tmp = tempfile.mkdtemp()
    acc = {}
    for seed in seeds:
        for name, pos, neg, _ in people(data):
            m = train(X[who != name], y[who != name], seed)
            path = export_onnx(m, os.path.join(tmp, f"{name}-{seed}.onnx"))
            oww = Model(wakeword_models=[path], inference_framework="onnx")
            for p in pos + neg:
                sc = stream(oww, read_wav(p))
                acc.setdefault(p, []).append(float(sc.max()) if len(sc) else 0.0)
        print(f"  씨앗 {seed} 끝", flush=True)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def sign_test(n_down, n_up):
    """부호 검정 p값(양측). 차이가 없다면 오르내림이 반반이어야 한다."""
    from math import comb
    n = n_down + n_up
    if n == 0:
        return 1.0
    k = min(n_down, n_up)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--data", default=os.path.join(ROOT, "wake-data"))
    ap.add_argument("--seeds", type=int, default=5, help="씨앗 몇 개를 평균낼지")
    a = ap.parse_args()

    seeds = list(range(7, 7 + a.seeds))
    print(f"{a.a_name} 재는 중… (씨앗 {a.seeds}개)", flush=True)
    A = maxima(a.a, a.data, seeds)
    print(f"{a.b_name} 재는 중… (씨앗 {a.seeds}개)", flush=True)
    B = maxima(a.b, a.data, seeds)

    groups = {"호출어": [], "헷갈리는 말": [], "문장": []}
    for name, pos, neg, _ in people(a.data):
        for p in pos:
            groups["호출어"].append(p)
        for p in neg:
            groups[kind(p)].append(p)

    print(f"\n같은 녹음을 두 설정으로 재서 짝지어 비교")
    print(f"  A = {a.a_name}   B = {a.b_name}\n")
    from scipy.stats import wilcoxon
    print(f"{'':<12}{'A 평균':>8}{'B 평균':>8}{'내려감':>7}{'올라감':>7}"
          f"{'부호 p':>9}{'크기 p':>9}")
    print("-" * 62)
    for label, paths in groups.items():
        av = np.array([A[p] for p in paths])
        bv = np.array([B[p] for p in paths])
        diff = bv - av
        down = int((diff < -0.01).sum())
        up = int((diff > 0.01).sum())
        ps = sign_test(down, up)
        # 부호만 세면 이미 0점인 파일들이 희석시킨다. 부호순위 검정은
        # 얼마나 움직였는지까지 본다.
        moved = diff[np.abs(diff) > 1e-9]
        pw = float(wilcoxon(moved).pvalue) if len(moved) >= 6 else float("nan")
        # 방향은 개수가 아니라 실제 이동량(중앙값)으로 본다. 개수로 보면
        # 크기 검정과 방향이 어긋난다 — 실제로 어긋나서 틀린 표시를 냈다.
        shift = float(np.median(moved)) if len(moved) else 0.0
        good = (shift < 0) if label != "호출어" else (shift > 0)
        mark = ""
        if min(ps, pw) < 0.05:
            mark = "  ←" + ("좋아짐" if good else "나빠짐")
        print(f"{label:<12}{av.mean():>8.3f}{bv.mean():>8.3f}"
              f"{down:>7}{up:>7}{ps:>9.3f}{pw:>9.3f}{mark}")
    print("\n낮을수록 좋은 것은 헷갈리는 말·문장, 높을수록 좋은 것은 호출어다.")
    print("부호 p — 오르내림 개수만 센다. 이미 0점인 파일이 많으면 둔해진다.")
    print("크기 p — 얼마나 움직였는지까지 센다(부호순위 검정).")


if __name__ == "__main__":
    main()
