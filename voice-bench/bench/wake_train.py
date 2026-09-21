"""웨이크워드 "픽스야" 학습.

    venvs/melo-onnx/bin/python bench/wake_train.py            # 검증 + 학습 + 내보내기
    venvs/melo-onnx/bin/python bench/wake_train.py --lopo-only # 검증만

구조: openWakeWord 와 같다. 구글 speech_embedding(고정) 위에 작은 분류기만
얹는다. 임베딩은 학습하지 않으므로 녹음 190개로도 분류기는 학습된다.

**한 사람 빼고 학습(LOPO)** 을 먼저 한다. 5명 중 4명으로 학습하고 나머지
한 명으로 시험한다. 이게 유일하게 정직한 숫자다 — 학습에 쓴 목소리로 재면
100%가 나오지만, 심사 때 부르는 사람은 학습에 없던 목소리다.

내보내는 ONNX 는 openWakeWord 가 그대로 읽는다(입력 [N,16,96], 출력 [N,1]).
"""
import argparse
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from wake_augment import (SR, WINDOW, noise_bed, place, resample,  # noqa: E402
                          rms_db, speech_span, to_int16)
from wake_data import people  # noqa: E402

BATCH = 256          # 임베딩 한 번에 넣을 창 개수
SILENT_DB = -70.0    # 이보다 조용한 방 소리는 녹음 실패다(마이크 권한)
TTS_TAG = "합성"      # 합성 부정에 붙이는 이름. 사람이 아니므로 LOPO 에서
                     # 시험 쪽으로 가지 않고 항상 학습에만 들어간다.


def read_wav(path):
    with wave.open(path, "rb") as w:
        n, ch, width, sr = (w.getnframes(), w.getnchannels(),
                            w.getsampwidth(), w.getframerate())
        raw = w.readframes(n)
    if width != 2:
        return None
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        x = resample(x, sr / SR)
    return x


def _noises(paths, who, quiet=False):
    out = []
    for p in paths:
        x = read_wav(p)
        if x is None:
            continue
        if rms_db(x) < SILENT_DB:
            if not quiet:
                print(f"  ! {who}/{os.path.basename(p)} 무음 — 뺀다")
            continue
        out.append(x)
    return out


def make_windows(base, rng, n_pos_aug, n_neg_aug, tts_dir=None, n_tts_aug=3):
    """(창 int16, 라벨, 사람) 를 하나씩 내놓는다. 오디오를 다 들고 있지 않는다."""
    pool = []
    for name, _, _, noise_paths in people(base):
        pool += _noises(noise_paths, name, quiet=True)

    for name, pos_paths, neg_paths, noise_paths in people(base):
        noises = _noises(noise_paths, name)

        pos = [(p, read_wav(p)) for p in pos_paths]
        pos = [(p, x) for p, x in pos if x is not None]
        for path, x in pos:
            a, b = speech_span(x)
            seg0 = x[a:b]
            for _ in range(n_pos_aug):
                seg = resample(seg0, rng.uniform(0.9, 1.12))
                # 창 안 어디에 놓든 깨어나야 한다. 말이 막 끝난 순간(1.0)부터
                # 끝난 지 0.45초 지난 순간(0.775)까지.
                yield (to_int16(place(seg, noises, rng,
                                      end_frac=rng.uniform(0.775, 1.0),
                                      snr_db=rng.uniform(3, 25),
                                      gain_db=rng.uniform(-12, 4))), 1, name)
            # 뒤가 잘린 호출어 = 부정. 이게 없으면 말 끝나기 전에 깨어난다.
            for _ in range(max(n_pos_aug // 4, 1)):
                seg = resample(seg0, rng.uniform(0.92, 1.08))
                yield (to_int16(place(seg, noises, rng,
                                      end_frac=rng.uniform(1.05, 1.45),
                                      snr_db=rng.uniform(5, 25),
                                      gain_db=rng.uniform(-10, 4))), 0, name)

        for path in neg_paths:
            x = read_wav(path)
            if x is None:
                continue
            a, b = speech_span(x)
            seg0 = x[a:b]
            for _ in range(n_neg_aug):
                seg = resample(seg0, rng.uniform(0.9, 1.12))
                yield (to_int16(place(seg, noises, rng,
                                      end_frac=rng.uniform(0.7, 1.2),
                                      snr_db=rng.uniform(3, 25),
                                      gain_db=rng.uniform(-12, 4))), 0, name)

        # 아무도 말하지 않는 창. 방 소리만으로 깨어나면 안 된다.
        for _ in range(n_neg_aug * 6):
            bed = noise_bed(noises, rng)
            g = rng.uniform(-6, 18)
            yield (to_int16(bed * (10 ** (g / 20.0))), 0, name)

    if not tts_dir:
        return
    import glob
    paths = sorted(glob.glob(os.path.join(tts_dir, "*.wav")))
    print(f"  합성 부정 {len(paths)}개 (사람 목소리가 아니다 — 학습에만 넣는다)")
    for path in paths:
        x = read_wav(path)
        if x is None:
            continue
        a, b = speech_span(x)
        seg0 = x[a:b]
        for _ in range(n_tts_aug):
            # 사람 다양성을 못 얻으므로 속도·음높이를 사람 녹음보다 넓게
            # 흔들어 가짜 화자를 만든다. 진짜 다섯 명과 같지는 않다.
            seg = resample(seg0, rng.uniform(0.82, 1.25))
            yield (to_int16(place(seg, pool, rng,
                                  end_frac=rng.uniform(0.7, 1.2),
                                  snr_db=rng.uniform(3, 25),
                                  gain_db=rng.uniform(-12, 4))), 0, TTS_TAG)


def embed_all(base, rng, n_pos_aug, n_neg_aug, tts_dir=None):
    from openwakeword.utils import AudioFeatures
    fx = AudioFeatures(inference_framework="onnx")

    X, y, who = [], [], []
    buf_a, buf_y, buf_w = [], [], []

    def flush():
        if not buf_a:
            return
        emb = fx.embed_clips(np.stack(buf_a), batch_size=BATCH)
        X.append(emb.astype(np.float32))
        y.extend(buf_y)
        who.extend(buf_w)
        buf_a.clear(); buf_y.clear(); buf_w.clear()
        print(f"\r  창 {len(y)}개", end="", flush=True)

    for clip, label, name in make_windows(base, rng, n_pos_aug, n_neg_aug, tts_dir):
        buf_a.append(clip); buf_y.append(label); buf_w.append(name)
        if len(buf_a) >= BATCH:
            flush()
    flush()
    print()
    return np.concatenate(X), np.array(y), np.array(who)


def train(Xtr, ytr, seed, hidden=(128, 64), iters=300):
    from sklearn.neural_network import MLPClassifier
    m = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                      alpha=1e-3, batch_size=128, learning_rate_init=1e-3,
                      max_iter=iters, early_stopping=True, n_iter_no_change=15,
                      validation_fraction=0.1, random_state=seed)
    m.fit(Xtr.reshape(len(Xtr), -1), ytr)
    return m


def score(m, X):
    return m.predict_proba(X.reshape(len(X), -1))[:, 1]


def rates(s_pos, s_neg, th):
    return float(np.mean(s_pos >= th)), float(np.mean(s_neg >= th))


def export_onnx(m, out_path):
    """sklearn MLP → openWakeWord 가 읽는 ONNX. 손으로 그래프를 만든다.

    변환 라이브러리(skl2onnx)를 새로 들이지 않는다. 층 3개짜리 MLP 는
    Gemm/Relu/Sigmoid 몇 개면 끝이고, 그러면 모양을 우리가 정확히 맞출 수 있다.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inp = helper.make_tensor_value_info("onnx::Flatten_0", TensorProto.FLOAT, [1, 16, 96])
    out = helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 1])

    nodes = [helper.make_node("Flatten", ["onnx::Flatten_0"], ["f0"], axis=1)]
    inits, cur = [], "f0"
    for i, (W, b) in enumerate(zip(m.coefs_, m.intercepts_)):
        wn, bn = f"W{i}", f"b{i}"
        inits.append(numpy_helper.from_array(W.astype(np.float32), wn))
        inits.append(numpy_helper.from_array(b.astype(np.float32), bn))
        z = f"z{i}"
        nodes.append(helper.make_node("Gemm", [cur, wn, bn], [z], alpha=1.0, beta=1.0))
        if i < len(m.coefs_) - 1:
            a = f"a{i}"
            nodes.append(helper.make_node("Relu", [z], [a]))
            cur = a
        else:
            nodes.append(helper.make_node("Sigmoid", [z], ["score"]))

    g = helper.make_graph(nodes, "pixs-ya", [inp], [out], inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)],
                              producer_name="talking-lamp/wake_train")
    model.ir_version = 8
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    onnx.save(model, out_path)
    return out_path


def verify_onnx(path, X, want):
    """내보낸 모델이 sklearn 과 같은 값을 내는지 확인한다. 조용히 틀리면 못 찾는다."""
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    n = s.get_inputs()[0]
    got = np.array([s.run(None, {n.name: X[i:i + 1]})[0][0][0] for i in range(len(X))])
    return float(np.max(np.abs(got - want)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "wake-data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models/wake/pixs-ya.onnx"))
    ap.add_argument("--pos-aug", type=int, default=40)
    ap.add_argument("--neg-aug", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lopo-only", action="store_true")
    ap.add_argument("--holdout", metavar="이름",
                    help="그 사람을 빼고 학습해 따로 내보낸다. 실제 마이크로 "
                         "'학습에 없던 목소리' 를 재려면 이게 필요하다 — "
                         "팀원이 전부 학습에 들어가 있어서, 남을 구하는 대신 "
                         "그 사람을 뺀 모델을 만든다")
    ap.add_argument("--cache", default=os.path.join(ROOT, "out/wake-feats.npz"))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--tts-neg", help="합성 부정 wav 디렉터리. 주면 학습에만 넣는다")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    if os.path.exists(a.cache) and not a.rebuild:
        d = np.load(a.cache, allow_pickle=True)
        X, y, who = d["X"], d["y"], d["who"]
        print(f"특징 재사용: {a.cache} ({len(y)}창)")
    else:
        print("창 만들고 임베딩 뽑는 중…")
        X, y, who = embed_all(a.data, rng, a.pos_aug, a.neg_aug, a.tts_neg)
        os.makedirs(os.path.dirname(a.cache), exist_ok=True)
        np.savez_compressed(a.cache, X=X, y=y, who=who)

    names = [n for n in sorted(set(who.tolist())) if n != TTS_TAG]
    print(f"\n사람 {len(names)}명, 창 {len(y)}개 (호출어 {int(y.sum())}, 아닌 것 {int((y == 0).sum())})")

    print("\n한 사람 빼고 학습 — 학습에 없던 목소리로 시험한다")
    print(f"{'뺀 사람':<10} {'깨어남':>8} {'헛깨움':>8}   임계 0.5 기준")
    print("-" * 46)
    lopo = []
    for name in names:
        tr, te = who != name, who == name
        m = train(X[tr], y[tr], a.seed)
        s = score(m, X[te])
        yp, yn = s[y[te] == 1], s[y[te] == 0]
        hit, fa = rates(yp, yn, 0.5)
        lopo.append((name, hit, fa, yp, yn))
        print(f"{name:<10} {hit*100:>7.1f}% {fa*100:>7.2f}%")

    hits = np.array([r[1] for r in lopo])
    fas = np.array([r[2] for r in lopo])
    print("-" * 46)
    print(f"{'평균':<10} {hits.mean()*100:>7.1f}% {fas.mean()*100:>7.2f}%")
    print(f"{'최악':<10} {hits.min()*100:>7.1f}% {fas.max()*100:>7.2f}%")

    allp = np.concatenate([r[3] for r in lopo])
    alln = np.concatenate([r[4] for r in lopo])
    print("\n임계값을 어디에 둘까 (학습에 없던 목소리 전부 모아서)")
    print(f"{'임계':>6} {'깨어남':>9} {'헛깨움':>9}")
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        hit, fa = rates(allp, alln, th)
        print(f"{th:>6.1f} {hit*100:>8.1f}% {fa*100:>8.2f}%")

    if a.lopo_only:
        return

    if a.holdout:
        if a.holdout not in names:
            raise SystemExit(f"그런 사람이 없다: {a.holdout} (있는 사람: {names})")
        keep = who != a.holdout
        print(f"\n{a.holdout} 를 빼고 학습 — 그 사람에게는 처음 듣는 목소리가 된다")
        m = train(X[keep], y[keep], a.seed)
        # 파일 이름에 사람 이름을 넣지 않는다. 한글 파일명이 젯슨 로케일에서
        # 깨지면 --wake-model 로 가리키기가 성가시다. 누구를 뺐는지는
        # 화면과 문서에 적는다.
        out = a.out.replace(".onnx", "-holdout.onnx")
        path = export_onnx(m, out)
        sc = score(m, X[keep])
        err = verify_onnx(path, X[keep][:100], sc[:100])
        print(f"  저장 {os.path.relpath(path, ROOT)}  "
              f"(sklearn 과 최대 차이 {err:.2e})")
        print("  이 모델로 실제 마이크에서 재면 그게 학습 밖 숫자다:")
        print(f"    bench/wake_field.py --wake-model {os.path.relpath(path, ROOT)}")
        return

    print("\n다섯 명 전부로 최종 학습")
    m = train(X, y, a.seed)
    s = score(m, X)
    hit, fa = rates(s[y == 1], s[y == 0], 0.5)
    print(f"  (학습 데이터 자신에 대해 깨어남 {hit*100:.1f}%, 헛깨움 {fa*100:.2f}% — 참고용일 뿐이다)")

    path = export_onnx(m, a.out)
    idx = rng.choice(len(X), size=min(200, len(X)), replace=False)
    err = verify_onnx(path, X[idx], s[idx])
    print(f"\n저장 {os.path.relpath(path, ROOT)}  ({os.path.getsize(path)/1024:.0f} KB)")
    print(f"sklearn 과 최대 차이 {err:.2e} — {'같다' if err < 1e-5 else '다르다! 내보내기가 틀렸다'}")


if __name__ == "__main__":
    main()
