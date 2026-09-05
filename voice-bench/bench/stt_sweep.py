"""STT 속도·정확도를 모델 크기와 스레드 수로 훑는다.

Jetson 실측에서 faster-whisper small(CPU)이 RTF 1.23 이었다. 실시간보다 느려
대화에 못 쓴다. 그런데 ctranslate2 의 aarch64 휠에는 CUDA 가 없어서
(ValueError: not compiled with CUDA support) GPU 로 넘길 수가 없다.

소스 빌드는 몇 시간짜리라, 그 전에 공짜로 되는 것부터 확인한다.
  - 스레드: 기본값이 코어를 다 쓰고 있는지 아무도 안 봤다. Jetson 은 6코어다
  - 모델 크기

CER 이 같이 나와야 의미가 있다. 빨라졌는데 못 알아들으면 소용없다.

base 는 이미 탈락했다. 심사 문장 6개 중 4개가 틀렸고(CER 0.148), 6번은
"승원아, 3시에 회의 있어" 를 "뭐 나 센시也可以 있어" 로 냈다 — 한자가 섞인다.
그래서 기본값은 small 이다. 굳이 다시 보려면 --models base,small 로 준다.

    venvs/melo-onnx/bin/python bench/stt_sweep.py
    venvs/melo-onnx/bin/python bench/stt_sweep.py --models base,small --threads 2,4,6
"""
import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from common import cer, load_sentences, repo_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="small")
    ap.add_argument("--threads", default="2,4,6")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    args = ap.parse_args()

    import soundfile as sf
    from faster_whisper import WhisperModel

    wavs = sorted(glob.glob(os.path.join(ROOT, "ref", "*.wav")))
    if not wavs:
        print("ref/*.wav 가 없다.", file=sys.stderr)
        return 2
    audio_s = sum(sf.info(w).duration for w in wavs)
    # ref/NN.wav 는 sentences.txt 의 N번째 문장을 읽은 것이다. 순서가 어긋나면
    # CER 이 통째로 무의미해지므로 개수를 먼저 확인한다.
    refs = load_sentences(repo_paths()["sentences"])
    if len(refs) != len(wavs):
        print(f"문장 {len(refs)}개 vs 녹음 {len(wavs)}개 — 짝이 안 맞는다.", file=sys.stderr)
        return 2

    print(f"참조 오디오 {audio_s:.1f}s ({len(wavs)}개)\n")
    print(f"{'모델':<8}{'스레드':>6}{'적재':>8}{'추론':>9}{'RTF':>8}{'CER':>8}")
    print("-" * 47)

    rows = []
    for m in [x.strip() for x in args.models.split(",")]:
        for th in [int(x) for x in args.threads.split(",")]:
            t0 = time.perf_counter()
            model = WhisperModel(m, device=args.device,
                                 compute_type=args.compute_type, cpu_threads=th)
            load = time.perf_counter() - t0

            t0 = time.perf_counter()
            hyps = []
            for w in wavs:
                segs, _ = model.transcribe(w, language="ko", beam_size=1)
                hyps.append("".join(s.text for s in segs).strip())
            infer = time.perf_counter() - t0

            c = sum(cer(r, h) for r, h in zip(refs, hyps)) / len(refs)
            rtf = infer / audio_s
            rows.append((m, th, rtf, c, hyps))
            print(f"{m:<8}{th:>6}{load:>7.1f}s{infer:>8.1f}s{rtf:>8.2f}{c:>8.3f}")
            del model

    best = min(rows, key=lambda r: r[2])
    print(f"\n가장 빠른 조합: {best[0]} / 스레드 {best[1]} → RTF {best[2]:.2f}, CER {best[3]:.3f}")
    if best[2] > 1.0:
        print("※ RTF 가 1.0 을 넘는다. CPU 로는 실시간이 안 된다 —")
        print("  ctranslate2 CUDA 빌드나 다른 STT 로 가야 한다.")
    print("\n가장 빠른 조합의 인식 결과:")
    for r, h in zip(refs, best[4]):
        mark = "  " if cer(r, h) < 0.05 else "! "
        print(f"  {mark}{h}")
        if mark == "! ":
            print(f"     (정답: {r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
