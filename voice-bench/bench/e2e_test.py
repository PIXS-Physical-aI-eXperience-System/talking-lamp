"""최종 구성으로 대화를 반복해서 돌린다 — 통합 검증.

지금까지는 부분별로만 쟀다. 한 번씩만 돌려서 나온 숫자라, 실제 운영에서
드러나는 두 가지를 못 본다.

  1) 메모리가 조금씩 새는가. 한 번 재서는 절대 안 보인다. 램프는 몇 시간
     켜져 있을 물건이라, 턴당 10 MB 만 새도 하루면 죽는다.
  2) 지연이 일정한가. 평균이 좋아도 가끔 크게 튀면 대화가 끊긴 느낌이 난다.
     평균이 아니라 최악값(p95)으로 봐야 한다.

한 턴 = STT(사용자 발화 인식) + TTS(램프 응답 합성). VLM 은 우리 파트가
아니라 빠져 있으므로, 여기 숫자에 VLM 시간을 더해야 실제 응답 시간이 된다.

최종 구성:
  STT  faster-whisper small / CUDA / int8_float16
  TTS  melo ONNX  VITS fp32 + BERT int8 / CUDA

    venvs/melo-onnx/bin/python bench/e2e_test.py --turns 30
"""
import argparse
import glob
import os
import statistics
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "runners"))

from common import cer, load_sentences, repo_paths  # noqa: E402


class Sampler(threading.Thread):
    """20 Hz 로 메모리를 기록한다. 턴이 끝난 뒤에만 찍으면 턴 도중의 최고점을
    놓친다 — 실제로 메모리가 모자라 죽는 건 그 순간이다."""

    def __init__(self, hz=20):
        super().__init__(daemon=True)
        self.dt = 1.0 / hz
        self.rows = []          # (t, sys_used)
        self._done = threading.Event()   # _stop 은 Thread 내부 이름이라 못 쓴다

    def run(self):
        while not self._done.is_set():
            self.rows.append((time.perf_counter(), sys_used_mb()))
            time.sleep(self.dt)

    def stop(self):
        self._done.set()
        self.join(timeout=2)

    def peak(self, t0, t1):
        w = [r[1] for r in self.rows if t0 <= r[0] <= t1]
        return max(w) if w else None


def sys_used_mb():
    info = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        info[k] = float(v.strip().split()[0]) / 1024
    return info["MemTotal"] - info["MemAvailable"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--stt-model", default="small")
    ap.add_argument("--bargein-every", type=int, default=5,
                    help="N 턴마다 한 번은 겹쳐서 돌린다 (0이면 안 함)")
    args = ap.parse_args()

    if not os.path.exists("/proc/meminfo"):
        print("Jetson 에서 돌릴 것.", file=sys.stderr)
        return 2

    import soundfile as sf
    from faster_whisper import WhisperModel
    from tts_melo_onnx import build_synth
    from ko_normalize import normalize

    wavs = sorted(glob.glob(os.path.join(ROOT, "ref", "*.wav")))
    refs = load_sentences(repo_paths()["sentences"])
    if not wavs or len(wavs) != len(refs):
        print("ref/*.wav 와 sentences.txt 가 짝이 안 맞는다.", file=sys.stderr)
        return 2
    tmp = os.path.join(tempfile.mkdtemp(), "o.wav")

    print("최종 구성으로 적재 중…")
    stt = WhisperModel(args.stt_model, device="cuda", compute_type="int8_float16")
    synth, sr, provs, load_s, _ = build_synth(
        "models/melo-ko-onnx", providers="CUDAExecutionProvider,CPUExecutionProvider",
        threads=2, quiet=True, bert_int8=True)   # VITS fp32 + BERT int8
    print(f"  TTS 공급자 {provs}  적재 {load_s}s")

    def transcribe(w):
        segs, _ = stt.transcribe(w, language="ko", beam_size=1)
        return "".join(s.text for s in segs).strip()

    # 워밍업. 첫 턴은 커널 준비 비용이 실려서 대표값이 아니다.
    transcribe(wavs[0]); synth(normalize(refs[0]))
    time.sleep(0.5)
    base = sys_used_mb()

    sam = Sampler()
    sam.start()
    rows = []
    print(f"\n{args.turns} 턴 시작 (바닥 {base:.0f} MB)\n")
    print(f"{'턴':>4}{'STT':>8}{'TTS':>8}{'합계':>8}{'끝난뒤':>9}{'최고점':>9}  비고")
    print("-" * 61)

    for i in range(args.turns):
        w = wavs[i % len(wavs)]
        ref = refs[i % len(refs)]
        # 램프의 응답은 다음 문장으로 대신한다. 실제로는 VLM 이 만들 자리다.
        reply = refs[(i + 1) % len(refs)]
        overlap = args.bargein_every and (i + 1) % args.bargein_every == 0

        t_start = time.perf_counter()
        t0 = t_start
        if overlap:
            # barge-in: 램프가 말하는 도중 사용자가 끊고 들어온다.
            got = []
            th = threading.Thread(target=lambda: got.append(transcribe(w)))
            th.start()
            sf.write(tmp, synth(normalize(reply)), sr)
            th.join()
            hyp, stt_s, tts_s = got[0], float("nan"), float("nan")
        else:
            t1 = time.perf_counter()
            hyp = transcribe(w)
            stt_s = time.perf_counter() - t1
            t2 = time.perf_counter()
            sf.write(tmp, synth(normalize(reply)), sr)
            tts_s = time.perf_counter() - t2
        total = time.perf_counter() - t0

        used = sys_used_mb() - base
        pk = sam.peak(t_start, time.perf_counter())
        pk = (pk - base) if pk is not None else used
        c = cer(ref, hyp)
        note = "barge-in" if overlap else ("" if c < 0.05 else f"CER {c:.2f}")
        rows.append((total, used, c, overlap, pk))
        s_txt = "     —" if overlap else f"{stt_s:>7.2f}s"
        t_txt = "     —" if overlap else f"{tts_s:>7.2f}s"
        print(f"{i+1:>4}{s_txt}{t_txt}{total:>7.2f}s{used:>6.0f} MB{pk:>6.0f} MB  {note}")

    sam.stop()

    # ── 판정 ──────────────────────────────────────────────
    lat = sorted(r[0] for r in rows if not r[3])
    mem = [r[1] for r in rows]
    cers = [r[2] for r in rows]
    p95 = lat[int(len(lat) * 0.95) - 1] if lat else float("nan")

    # 앞쪽 턴은 초기 할당이 끝나지 않은 구간이라 빼야 한다. 이걸 포함해서
    # 비교하면 "준비 중" 과 "안정 상태" 를 비교하는 꼴이라, 평평한 결과도
    # 누수로 잡힌다 (실제로 그렇게 오판했다: 1~6턴 15->296 MB 상승은
    # 램프업이고 7턴부터 30턴까지는 300 MB 에서 평평했다).
    warm = max(1, len(mem) // 4)
    stable = mem[warm:]
    h = max(1, len(stable) // 2)
    first, last = statistics.mean(stable[:h]), statistics.mean(stable[-h:])
    drift = last - first

    print(f"\n### {args.turns} 턴 결과")
    print(f"  지연  중앙값 {statistics.median(lat):.2f}s   p95 {p95:.2f}s   최대 {max(lat):.2f}s")
    print(f"        (VLM 시간은 빠져 있다. 실제 응답 시간은 여기에 더해야 한다)")
    print(f"  정확도 평균 CER {statistics.mean(cers):.3f}   최악 {max(cers):.3f}")
    peaks = [r[4] for r in rows]
    print(f"  메모리 최고점 {max(peaks):.0f} MB   (턴이 끝난 뒤 기준으로는 {max(mem):.0f} MB)")
    print(f"        처음 {n}턴 평균 {first:.0f} → 마지막 {n}턴 평균 {last:.0f} MB")
    bi = [r[4] for r in rows if r[3]]
    if bi:
        print(f"        barge-in 턴 최고 {max(bi):.0f} MB / 일반 턴 최고 "
              f"{max(r[4] for r in rows if not r[3]):.0f} MB")

    print()
    if drift > 50:
        print(f"  ! 메모리가 {drift:.0f} MB 늘었다. 턴당 약 {drift/len(stable):.1f} MB —")
        print("    몇 시간 켜두면 문제가 된다. 누수를 찾아야 한다.")
    elif drift > 20:
        print(f"  · 메모리가 {drift:.0f} MB 늘었다. 누수인지 단순 변동인지 더 긴 시험이 필요하다.")
    else:
        print(f"  ✔ 메모리 안정 (변동 {drift:+.0f} MB)")
    if p95 > 3.0:
        print(f"  ! p95 가 {p95:.2f}s 다. 가끔 크게 튄다 — 대화가 끊긴 느낌이 난다.")
    else:
        print(f"  ✔ 지연 안정 (p95 {p95:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
