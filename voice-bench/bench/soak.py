"""장시간 반복 후 메모리가 수렴하는지 본다 (누수 확인).

짧은 측정으로는 '피크가 계속 오르는' 것처럼 보이는데, 할당자가 해제한 메모리를
OS에 즉시 반납하지 않아서 생기는 현상일 수 있다. 실제 누수라면 몇 시간 동작 후
터지므로, 예산 숫자를 확정하기 전에 가려야 한다.

한 사이클 = STT 1문장 + TTS 1문장 (반이중 구조 그대로: 동시 실행하지 않는다).

    venvs/piper/bin/python soak.py sherpa 100
    venvs/melo/bin/python  soak.py melo   100
"""
import glob
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # voice-bench/ — 공용 모듈과 산출물이 있는 곳
sys.path.insert(0, ROOT)
from common import load_sentences, repo_paths  # noqa: E402


def rss_now_mb() -> float:
    """지금 이 순간의 상주량. ru_maxrss는 최고 수위라 수렴 여부를 못 본다."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1024


def build(engine):
    # torch 스레드 수는 피크 RSS에 크게 영향을 준다 (스레드마다 메모리 아레나를 잡는다).
    # 실측상 macOS에서는 2가 가장 낮았다. 리눅스는 할당자가 달라 Jetson에서 재확인 필요.
    threads = int(os.environ.get("TORCH_THREADS", "0"))
    if threads:
        import torch
        torch.set_num_threads(threads)

    from faster_whisper import WhisperModel
    stt = WhisperModel("small", device="cpu", compute_type="int8")

    if engine == "sherpa":
        import sherpa_onnx
        import soundfile as sf
        d = os.path.join(ROOT, "models", "vits-mimic3-ko_KO-kss_low")
        onnx = [f for f in os.listdir(d) if f.endswith(".onnx")][0]
        tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=os.path.join(d, onnx),
                    tokens=os.path.join(d, "tokens.txt"),
                    data_dir=os.path.join(d, "espeak-ng-data")),
                num_threads=2)))
        tmp = os.path.join(tempfile.mkdtemp(), "o.wav")

        def speak(text):
            a = tts.generate(text, sid=0, speed=1.0)
            sf.write(tmp, a.samples, a.sample_rate)
    elif engine in ("melo-onnx", "melo-onnx-int8"):
        # torch 없이 ONNX만 쓰는 경로. VITS·BERT 모두 onnxruntime.
        import json
        import numpy as np
        import onnxruntime as ort
        import soundfile as sf
        from transformers import AutoTokenizer
        sys.path.insert(0, ROOT)
        from melo_text.cleaner import clean_text
        from melo_text import cleaned_text_to_sequence
        from runners.tts_melo_onnx import bert_features, intersperse

        d = os.path.join(ROOT, "models", "melo-ko-onnx")
        fe = json.load(open(os.path.join(d, "frontend.json"), encoding="utf-8"))
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 2
        tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
        sfx = ".int8" if engine.endswith("int8") else ""
        bert_s = ort.InferenceSession(os.path.join(d, f"bert-kor-base{sfx}.onnx"), opts,
                                      providers=["CPUExecutionProvider"])
        vits_s = ort.InferenceSession(os.path.join(d, f"melo-ko-vits{sfx}.onnx"), opts,
                                      providers=["CPUExecutionProvider"])
        tmp = os.path.join(tempfile.mkdtemp(), "o.wav")

        def speak(text):
            nt, ph, tn, w2p = clean_text(text, "KR")
            ph, tn, lg = cleaned_text_to_sequence(ph, tn, "KR", fe["symbol_to_id"])
            if fe["add_blank"]:
                ph, tn, lg = (intersperse(x, 0) for x in (ph, tn, lg))
                w2p = [w * 2 for w in w2p]; w2p[0] += 1
            jb = bert_features(bert_s, tok, nt, w2p).astype(np.float32)
            a = vits_s.run(None, {
                "phones": np.array([ph], np.int64),
                "phone_lengths": np.array([len(ph)], np.int64),
                "sid": np.array([fe["spk2id"]["KR"]], np.int64),
                "tones": np.array([tn], np.int64), "lang_ids": np.array([lg], np.int64),
                "bert": np.zeros((1, 1024, len(ph)), np.float32), "ja_bert": jb[None]})[0].squeeze()
            sf.write(tmp, a, fe["sampling_rate"])
    else:
        from melo.api import TTS
        m = TTS(language="KR", device="cpu")
        spk = m.hps.data.spk2id["KR"]
        tmp = os.path.join(tempfile.mkdtemp(), "o.wav")

        def speak(text):
            m.tts_to_file(text, spk, tmp)

    return stt, speak


def main() -> int:
    engine = sys.argv[1] if len(sys.argv) > 1 else "sherpa"
    cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    refs = sorted(glob.glob(os.path.join(ROOT, "ref", "*.wav")))
    sents = load_sentences(repo_paths()["sentences"])
    stt, speak = build(engine)

    # MeloTTS는 합성할 때마다 stdout에 문장을 찍는다. 측정값을 stdout으로 내보내면
    # 거기 섞여 유실되므로, 샘플은 항상 별도 파일에 기록한다.
    log_path = os.path.join(ROOT, "out", f"soak-{engine}.tsv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    log.write("cycle\trss_mb\n")

    base = rss_now_mb()
    print(f"### {engine} 내구 테스트 — {cycles} 사이클 (STT 1 + TTS 1 / 사이클)")
    print(f"로드 직후 상주: {base:.0f} MB\n")
    print(f"{'사이클':>6}{'상주 RSS':>11}{'로드 대비':>11}")
    print("-" * 30)

    samples = []
    for i in range(1, cycles + 1):
        list(stt.transcribe(refs[i % len(refs)], language="ko", beam_size=1)[0])
        speak(sents[i % len(sents)])
        if i % 10 == 0 or i == 1:
            v = rss_now_mb()
            samples.append((i, v))
            log.write(f"{i}\t{v:.1f}\n")
            log.flush()

    log.close()
    for i, v in samples:
        print(f"{i:>6}{v:>9.0f} MB{v - base:>+9.0f} MB")

    # 뒤쪽 절반의 기울기로 수렴 여부를 본다
    tail = samples[len(samples) // 2:]
    if len(tail) >= 2:
        (x0, y0), (x1, y1) = tail[0], tail[-1]
        slope = (y1 - y0) / max(x1 - x0, 1)
        print(f"\n후반 기울기: {slope:+.3f} MB/사이클")
        if abs(slope) < 0.05:
            print("→ 수렴. 누수 아님")
        elif slope > 0:
            print(f"→ 계속 증가. 1000 사이클당 약 {slope * 1000:+.0f} MB — 장시간 동작 시 확인 필요")
        else:
            print("→ 감소 추세 (할당자가 반납 중)")
    print(f"\n최종 상주 {samples[-1][1]:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
