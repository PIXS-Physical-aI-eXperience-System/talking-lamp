"""실제 운영 조건에서 음성 파트가 차지하는 메모리를 잰다.

기존 measure_combined.py 의 한계 두 가지를 고친다.

1) 피크 RSS 는 한 번 올라가면 안 내려가는 최고 수위라, 이미 반납된 STT 작업
   메모리가 TTS 몫과 더해진 것처럼 보였다. 여기서는 백그라운드로 계속 표본을
   뜨고 구간별 최댓값을 쓴다.
2) STT 와 TTS 는 반이중 설계라 동시에 추론하지 않는다. 유일한 예외가
   barge-in — 램프가 말하는 도중 사용자가 끊고 들어오는 순간이다. 그게 최악의
   경우인데 아무도 재본 적이 없어서, 여기서 실제로 겹쳐 돌려 잰다.

    venvs/melo-onnx/bin/python bench/mem_profile.py
    venvs/melo-onnx/bin/python bench/mem_profile.py --bert-int8 --no-arena
"""
import argparse
import glob
import os
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "runners"))

from common import load_sentences, repo_paths  # noqa: E402


def sys_used_mb():
    """시스템 전체 사용량(MB). 통합 메모리라 GPU 몫도 여기 잡힌다."""
    info = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        info[k] = float(v.strip().split()[0]) / 1024
    return info["MemTotal"] - info["MemAvailable"]


def rss_mb():
    """현재 RSS(MB). 피크가 아니라 지금 값이라야 구간별 비교가 된다."""
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024
    return 0.0


class Sampler(threading.Thread):
    """20 Hz 로 메모리를 기록한다. 구간 최댓값은 나중에 시각으로 잘라 낸다."""

    def __init__(self, hz=20):
        super().__init__(daemon=True)
        self.dt = 1.0 / hz
        self.rows = []          # (t, sys, rss)
        self._done = threading.Event()

    def run(self):
        while not self._done.is_set():
            self.rows.append((time.perf_counter(), sys_used_mb(), rss_mb()))
            time.sleep(self.dt)

    def stop(self):
        # 이름을 _stop 으로 두면 안 된다. Thread 내부에 같은 이름의 메서드가
        # 있어서 join() 이 self._stop() 을 부를 때 Event 를 호출하려다 터진다.
        self._done.set()
        self.join(timeout=2)

    def peak(self, t0, t1):
        w = [r for r in self.rows if t0 <= r[0] <= t1]
        if not w:
            return None, None
        return max(r[1] for r in w), max(r[2] for r in w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt-model", default="small")
    ap.add_argument("--stt-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    ap.add_argument("--int8", action="store_true", help="TTS 전체 int8 (GPU 에서는 금물)")
    ap.add_argument("--bert-int8", action="store_true")
    ap.add_argument("--no-arena", action="store_true")
    args = ap.parse_args()

    if not os.path.exists("/proc/meminfo"):
        print("이 스크립트는 리눅스(Jetson)에서만 의미가 있다.", file=sys.stderr)
        return 2

    refs = sorted(glob.glob(os.path.join(ROOT, "ref", "*.wav")))
    if not refs:
        print("ref/*.wav 가 없다. STT 를 잴 수 없다.", file=sys.stderr)
        return 2
    sentences = load_sentences(repo_paths()["sentences"])
    tmp = os.path.join(tempfile.mkdtemp(), "o.wav")

    sam = Sampler()
    sam.start()
    time.sleep(0.5)
    base = min(r[1] for r in sam.rows)   # 다른 프로세스 변동을 피해 최솟값을 바닥으로
    phases = []

    def phase(name, fn):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        s, r = sam.peak(t0, t1)
        phases.append((name, s, r, t1 - t0))
        return out

    # ── 적재 ──────────────────────────────────────────────
    from faster_whisper import WhisperModel
    import soundfile as sf
    from tts_melo_onnx import build_synth
    from ko_normalize import normalize

    ct = "int8_float16" if args.stt_device == "cuda" else "int8"
    stt = phase("STT 적재", lambda: WhisperModel(
        args.stt_model, device=args.stt_device, compute_type=ct))

    built = phase("TTS 적재", lambda: build_synth(
        "models/melo-ko-onnx", int8=args.int8, providers=args.providers,
        threads=2, quiet=True,
        bert_int8=True if args.bert_int8 else None, arena=not args.no_arena))
    synth, sr, provs, load_s, _ = built

    def transcribe_all():
        for w in refs:
            list(stt.transcribe(w, language="ko", beam_size=1)[0])

    def synth_all():
        for s in sentences:
            sf.write(tmp, synth(normalize(s)), sr)

    phase("TTS 워밍업", lambda: synth(normalize(sentences[0])))
    phase("상주만 (유휴)", lambda: time.sleep(1.0))

    # ── 단독 추론 ─────────────────────────────────────────
    phase("STT 단독 추론", transcribe_all)
    phase("TTS 단독 추론", synth_all)

    # ── barge-in: 실제로 겹쳐서 돌린다 ────────────────────
    # 램프가 말하는 도중 사용자가 끊고 들어오는 순간. 반이중 설계에서
    # 유일하게 둘이 같이 도는 구간이고, 최악의 경우다.
    def overlap():
        err = []

        def bg():
            try:
                transcribe_all()
            except Exception as e:      # 겹침 자체가 실패하면 그것도 결과다
                err.append(e)

        t = threading.Thread(target=bg)
        t.start()
        synth_all()
        t.join()
        return err

    errs = phase("barge-in (STT+TTS 동시)", overlap)

    try:
        sam.stop()
    except Exception as e:
        # 몇 분 걸린 측정이다. 표본 수집을 접다가 터졌다고 결과까지 버리면 안 된다.
        print(f"표본 수집 정리 중 오류(결과는 그대로 낸다): {e!r}", file=sys.stderr)

    print(f"\n### 음성 파트 메모리 — STT {args.stt_model}/{args.stt_device}, "
          f"TTS {'int8' if args.int8 else 'fp32'}"
          f"{', BERT int8' if args.bert_int8 else ''}"
          f"{', 아레나 끔' if args.no_arena else ''}")
    print(f"TTS 공급자 {provs}  적재 {load_s}s   바닥 {base:.0f} MB\n")
    print(f"{'구간':<24}{'시스템 증가':>12}{'RSS':>10}{'소요':>9}")
    print("-" * 56)
    for name, s, r, dt in phases:
        s_txt = f"{s - base:>9.0f} MB" if s is not None else "        —"
        r_txt = f"{r:>7.0f} MB" if r is not None else "      —"
        print(f"{name:<22}{s_txt}{r_txt}{dt:>8.1f}s")

    worst = max((p[1] for p in phases if p[1] is not None), default=base)
    print(f"\n최악의 경우 {worst - base:.0f} MB — 이 값을 예산 근거로 쓴다.")
    if errs:
        print(f"※ 겹침 구간에서 예외 발생: {errs[0]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
