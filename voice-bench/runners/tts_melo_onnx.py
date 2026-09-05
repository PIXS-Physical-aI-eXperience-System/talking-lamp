"""MeloTTS 한국어 — torch 없이 ONNX만으로 돌리는 러너.

torch 기반 melo는 피크 2 GB였고, 그중 모델 가중치는 652 MB뿐이었다.
양자화·헤드제거·BERT제거·스레드조정을 다 시도해도 피크가 안 줄었는데,
바닥을 만드는 게 모델이 아니라 torch 런타임과 그 순간 할당이었기 때문이다.

그래서 torch를 통째로 걷어낸다:
  - VITS  → ONNX (melo_export_onnx.py 로 생성)
  - BERT  → ONNX (melo_export_bert.py 로 생성)
  - 텍스트 프론트엔드 → melo_text/ (torch 안 쓰는 파일만 복사해 온 사본)

이 러너는 torch가 설치조차 되지 않은 venv(venvs/melo-onnx)에서 돌아야 의미가 있다.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from common import Timer, emit, load_sentences, repo_paths  # noqa: E402
from melo_text.cleaner import clean_text                    # noqa: E402
from melo_text import cleaned_text_to_sequence              # noqa: E402


def intersperse(lst, item):
    """melo commons.intersperse — 심볼 사이에 blank(0)를 끼운다."""
    out = [item] * (len(lst) * 2 + 1)
    out[1::2] = lst
    return out


def bert_features(sess, tok, text, word2ph):
    """melo의 get_bert_feature를 numpy로 옮긴 것.

    BERT의 뒤에서 3번째 은닉층을 꺼내, 각 토큰 특징을 그 토큰이 만드는
    음소 개수(word2ph)만큼 복제해 음소 단위로 펼친다.
    """
    enc = tok(text, return_tensors="np")
    hidden = sess.run(None, {
        "input_ids": enc["input_ids"].astype(np.int64),
        "token_type_ids": enc["token_type_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    })[0][0]                                    # (토큰수, 768)
    assert hidden.shape[0] == len(word2ph), f"{hidden.shape[0]} != {len(word2ph)}"
    return np.concatenate(
        [np.tile(hidden[i], (word2ph[i], 1)) for i in range(len(word2ph))], axis=0).T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/melo-ko-onnx")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--int8", action="store_true",
                    help="int8 양자화 모델 사용 (melo_quantize.py 로 생성)")
    ap.add_argument("--providers", default="auto",
                    help="onnxruntime 실행 공급자. auto=사용 가능한 것 중 가속기 우선, "
                         "또는 쉼표로 직접 지정 (예: CUDAExecutionProvider,CPUExecutionProvider)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="측정 전 버리는 합성 횟수. GPU는 첫 실행에 커널을 준비하느라 "
                         "수 초가 걸리므로, 가속기로 잴 때는 1 이상을 준다")
    ap.add_argument("--quiet-ort", action="store_true",
                    help="onnxruntime 경고 억제 (CUDA 경로에서 ScatterND 경고가 대량 발생)")
    ap.add_argument("--threads", type=int, default=2,
                    help="intra-op 스레드. Jetson은 코어가 6개라 다른 파트와의 경합을 고려할 것")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    d = os.path.join(HERE, args.model_dir) if not os.path.isabs(args.model_dir) else args.model_dir
    fe = json.load(open(os.path.join(d, "frontend.json"), encoding="utf-8"))
    sid = np.array([fe["spk2id"]["KR"]], dtype=np.int64)
    sr = fe["sampling_rate"]

    os.makedirs(args.out_dir, exist_ok=True)
    sentences = load_sentences(repo_paths()["sentences"])
    if args.normalize:
        from ko_normalize import normalize
        sentences = [normalize(s) for s in sentences]

    # 실행 공급자 선택. Jetson에서는 TensorRT > CUDA > CPU 순으로 빠르지만,
    # JetPack용 onnxruntime-gpu 가 설치돼 있어야 앞의 둘이 잡힌다.
    available = ort.get_available_providers()
    if args.providers == "auto":
        prefer = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in prefer if p in available]
    else:
        providers = [p.strip() for p in args.providers.split(",")]
        missing = [p for p in providers if p not in available]
        if missing:
            raise SystemExit(f"사용할 수 없는 공급자: {missing}\n설치된 것: {available}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    if args.quiet_ort:
        # 세션 로거만 낮추면 안 된다. CUDA 커널의 ScatterND 경고는 세션이 아니라
        # 환경(Default) 로거로 나가므로, 둘 다 올려야 조용해진다.
        opts.log_severity_level = 3  # 오류만
        ort.set_default_logger_severity(3)
    with Timer() as t_load:
        tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
        suffix = ".int8" if args.int8 else ""
        bert = ort.InferenceSession(os.path.join(d, f"bert-kor-base{suffix}.onnx"),
                                    opts, providers=providers)
        vits = ort.InferenceSession(os.path.join(d, f"melo-ko-vits{suffix}.onnx"),
                                    opts, providers=providers)

    def synth(text):
        """텍스트 → 오디오. 측정 루프와 워밍업이 같은 경로를 타야 의미가 있다."""
        norm_text, phone, tone, word2ph = clean_text(text, "KR")
        phone, tone, language = cleaned_text_to_sequence(
            phone, tone, "KR", fe["symbol_to_id"])
        if fe["add_blank"]:
            ph, tn, lg = (intersperse(x, 0) for x in (phone, tone, language))
            w2p = [w * 2 for w in word2ph]
            w2p[0] += 1
        else:
            ph, tn, lg, w2p = phone, tone, language, word2ph
        ja_bert = bert_features(bert, tok, norm_text, w2p).astype(np.float32)
        # 한국어 경로에서 1024차원 bert 입력은 쓰이지 않는다 (melo utils 참고)
        return vits.run(None, {
            "phones": np.array([ph], dtype=np.int64),
            "phone_lengths": np.array([len(ph)], dtype=np.int64),
            "sid": sid,
            "tones": np.array([tn], dtype=np.int64),
            "lang_ids": np.array([lg], dtype=np.int64),
            "bert": np.zeros((1024, len(ph)), dtype=np.float32)[None],
            "ja_bert": ja_bert[None],
        })[0].squeeze()

    for _ in range(args.warmup):
        synth(sentences[0])

    wavs, synth_s, audio_s = [], [], []
    for i, text in enumerate(sentences, 1):
        path = os.path.join(args.out_dir, f"{i:02d}.wav")
        t0 = time.perf_counter()
        audio = synth(text)
        sf.write(path, audio, sr)
        synth_s.append(round(time.perf_counter() - t0, 3))
        audio_s.append(round(len(audio) / sr, 3))
        wavs.append(path)

    emit({"label": args.label, "kind": "tts", "wavs": wavs,
          "load_s": round(t_load.elapsed, 2), "synth_s": synth_s, "audio_s": audio_s,
          "config": {"runtime": "onnxruntime (torch 미설치)", "sample_rate": sr,
                     "normalize": args.normalize, "int8": args.int8,
                     "providers": vits.get_providers(), "threads": args.threads,
                     "warmup": args.warmup}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
