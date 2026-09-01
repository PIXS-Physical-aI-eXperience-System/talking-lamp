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

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    with Timer() as t_load:
        tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
        suffix = ".int8" if args.int8 else ""
        bert = ort.InferenceSession(os.path.join(d, f"bert-kor-base{suffix}.onnx"),
                                    opts, providers=["CPUExecutionProvider"])
        vits = ort.InferenceSession(os.path.join(d, f"melo-ko-vits{suffix}.onnx"),
                                    opts, providers=["CPUExecutionProvider"])

    wavs, synth_s, audio_s = [], [], []
    for i, text in enumerate(sentences, 1):
        path = os.path.join(args.out_dir, f"{i:02d}.wav")
        t0 = time.perf_counter()

        norm_text, phone, tone, word2ph = clean_text(text, "KR")
        phone, tone, language = cleaned_text_to_sequence(
            phone, tone, "KR", fe["symbol_to_id"])
        if fe["add_blank"]:
            phone, tone, language = (intersperse(x, 0) for x in (phone, tone, language))
            word2ph = [w * 2 for w in word2ph]
            word2ph[0] += 1

        ja_bert = bert_features(bert, tok, norm_text, word2ph).astype(np.float32)
        # 한국어 경로에서 1024차원 bert 입력은 쓰이지 않는다 (melo utils 참고)
        zero_bert = np.zeros((1024, len(phone)), dtype=np.float32)

        audio = vits.run(None, {
            "phones": np.array([phone], dtype=np.int64),
            "phone_lengths": np.array([len(phone)], dtype=np.int64),
            "sid": sid,
            "tones": np.array([tone], dtype=np.int64),
            "lang_ids": np.array([language], dtype=np.int64),
            "bert": zero_bert[None], "ja_bert": ja_bert[None],
        })[0].squeeze()

        sf.write(path, audio, sr)
        synth_s.append(round(time.perf_counter() - t0, 3))
        audio_s.append(round(len(audio) / sr, 3))
        wavs.append(path)

    emit({"label": args.label, "kind": "tts", "wavs": wavs,
          "load_s": round(t_load.elapsed, 2), "synth_s": synth_s, "audio_s": audio_s,
          "config": {"runtime": "onnxruntime (torch 미설치)", "sample_rate": sr,
                     "normalize": args.normalize, "int8": args.int8}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
