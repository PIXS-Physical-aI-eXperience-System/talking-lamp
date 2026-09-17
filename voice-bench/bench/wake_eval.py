"""학습한 웨이크워드를 **실제 돌아가는 방식 그대로** 재본다.

wake_train.py 가 내놓는 숫자는 창(window) 하나당 비율이다. 그건 제품의
숫자가 아니다. 실제로는 openWakeWord 가 80 ms 마다 창을 밀며 점수를 내고,
한 번이라도 임계를 넘으면 램프가 깨어난다. 그러니 재야 할 것은 두 가지다.

  깨어남   — 호출어 녹음 한 개를 흘려보냈을 때 한 번이라도 넘는가
  헛깨움   — 헷갈리는 말과 평범한 문장에서 넘는가
  방 소리  — 아무도 말하지 않을 때 분당 몇 번 넘는가

여기서도 **학습에 없던 목소리**로만 잰다. 사람마다 그 사람을 뺀 모델을
만들어 그 사람 녹음으로 시험한다. 증강하지 않은 원본 녹음을 쓴다.
"""
import argparse
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from wake_augment import SR, rms_db  # noqa: E402
from wake_data import people  # noqa: E402
from wake_train import SILENT_DB, export_onnx, read_wav, train  # noqa: E402

CHUNK = 1280   # openWakeWord 가 한 걸음에 먹는 표본 수(80 ms)


def stream(model, x):
    """녹음 하나를 흘려보내고 창마다의 점수를 돌려준다."""
    model.reset()
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    out = []
    for i in range(0, len(pcm) - CHUNK + 1, CHUNK):
        s = model.predict(pcm[i:i + CHUNK])
        out.append(float(max(s.values())) if s else 0.0)
    return np.array(out)


def fired(scores, th, need):
    """임계를 need 번 연속으로 넘었는가.

    한 창만 넘어도 깨우면 스치는 소리에 깨어난다. 연속을 요구하면 헛깨움이
    크게 준다 — 호출어는 여러 창에 걸쳐 있으므로 진짜는 연속으로 넘는다.
    """
    if len(scores) < need:
        return False
    over = scores >= th
    run = 0
    for v in over:
        run = run + 1 if v else 0
        if run >= need:
            return True
    return False


def kind(path):
    """헷갈리는 말인가 평범한 문장인가. 사람마다 폴더와 이름이 다르다."""
    parent = os.path.basename(os.path.dirname(path))
    base = os.path.basename(path)
    return "문장" if parent == "sentences" or base.startswith("s") else "헷갈리는 말"


def count_over(scores, th, need):
    """방 소리용 — 몇 번 깨어났는지. 연속으로 넘는 동안은 한 번으로 센다."""
    over = scores >= th
    n, run, armed = 0, 0, True
    for v in over:
        if not v:
            run, armed = 0, True
            continue
        run += 1
        if run >= need and armed:
            n += 1
            armed = False
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "wake-data"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "out/wake-feats.npz"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    from openwakeword.model import Model

    d = np.load(a.cache, allow_pickle=True)
    X, y, who = d["X"], d["y"], d["who"]
    ths = (0.3, 0.5, 0.7, 0.9)
    needs = (1, 2, 3)

    rows, noise = [], []
    tmp = tempfile.mkdtemp()

    for name, pos_paths, neg_paths, noise_paths in people(a.data):
        tr = who != name
        print(f"  {name} 뺀 모델 학습 중…", flush=True)
        m = train(X[tr], y[tr], a.seed)
        path = export_onnx(m, os.path.join(tmp, f"{name}.onnx"))
        oww = Model(wakeword_models=[path], inference_framework="onnx")

        pos = [stream(oww, read_wav(p)) for p in pos_paths]
        hard = [stream(oww, read_wav(p)) for p in neg_paths if kind(p) == "헷갈리는 말"]
        sent = [stream(oww, read_wav(p)) for p in neg_paths if kind(p) == "문장"]
        rows.append((name, pos, hard, sent))

        for p in noise_paths:
            x = read_wav(p)
            if x is not None and rms_db(x) >= SILENT_DB:
                noise.append(stream(oww, x))

    def rate(group, th, need):
        return 100.0 * np.mean([fired(s, th, need) for s in group]) if group else 0.0

    allpos = [s for r in rows for s in r[1]]
    allhard = [s for r in rows for s in r[2]]
    allsent = [s for r in rows for s in r[3]]
    print(f"\n호출어 {len(allpos)}개 / 헷갈리는 말 {len(allhard)}개 / 평범한 문장 {len(allsent)}개")
    print("학습에 없던 목소리로만 잰다. 원본 녹음을 실제와 같이 흘려보낸다.\n")

    print(f"{'연속':>4} {'임계':>5} {'깨어남':>9} {'헷갈리는 말':>13} {'문장':>9} {'방 소리':>10}")
    print("-" * 56)
    best = None
    for need in needs:
        for th in ths:
            hit = rate(allpos, th, need)
            h = rate(allhard, th, need)
            sn = rate(allsent, th, need)
            nz = sum(count_over(s, th, need) for s in noise)
            secs = sum(len(s) for s in noise) * CHUNK / SR
            per_min = nz / (secs / 60) if secs else 0.0
            print(f"{need:>4} {th:>5.1f} {hit:>8.0f}% {h:>12.0f}% {sn:>8.0f}% {per_min:>7.1f}회/분")
            if hit >= 95 and (best is None or (sn, h) < best[0]):
                best = ((sn, h), need, th, hit, per_min)
        print()

    if best:
        (sn, h), need, th, hit, per_min = best
        print(f"깨어남 95% 이상 중 문장 헛깨움이 가장 낮은 자리: "
              f"연속 {need}창, 임계 {th}")
        print(f"  깨어남 {hit:.0f}%  문장 {sn:.0f}%  헷갈리는 말 {h:.0f}%  방 소리 {per_min:.1f}회/분")

    print("\n사람별 (위에서 고른 자리 기준)")
    if best:
        _, need, th, _, _ = best
        print(f"{'사람':<8}{'깨어남':>9}{'헷갈리는 말':>13}{'문장':>9}")
        for name, pos, hard, sent in rows:
            print(f"{name:<8}{rate(pos, th, need):>8.0f}%{rate(hard, th, need):>12.0f}%"
                  f"{rate(sent, th, need):>8.0f}%")


if __name__ == "__main__":
    main()
