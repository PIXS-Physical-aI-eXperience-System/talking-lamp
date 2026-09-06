"""긴 발화에서 메모리가 얼마나 더 늘어나는지 잰다.

30턴 통합 시험에서 메모리가 6턴에 걸쳐 300 MB 까지 오르다 멈췄다. 참조
파일이 6개이므로, 서로 다른 입력을 전부 한 번씩 본 시점이다. 추론 버퍼 풀이
"지금까지 본 것 중 가장 긴 입력"에 맞춰 커지고 반납하지 않기 때문이다.

그렇다면 지금까지의 최악값(barge-in 1549 MB)은 참조 파일 중 가장 긴
3.62초에 맞춰진 값일 뿐이다. 사용자가 램프한테 15초씩 말하면 풀이 더
커진다. 얼마나 커지는지 모르는 채로 예산을 내면 안 된다.

새로 녹음하지 않고 기존 참조 파일을 이어붙여 긴 발화를 만든다. 내용은
어색해지지만 버퍼 크기를 재는 데는 상관없다.

    venvs/melo-onnx/bin/python bench/long_utterance.py
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
    info = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        info[k] = float(v.strip().split()[0]) / 1024
    return info["MemTotal"] - info["MemAvailable"]


class Sampler(threading.Thread):
    def __init__(self, hz=20):
        super().__init__(daemon=True)
        self.dt = 1.0 / hz
        self.rows = []
        self._done = threading.Event()

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="4,10,20,30,60",
                    help="만들 발화 길이(초). 쉼표로 구분")
    args = ap.parse_args()

    if not os.path.exists("/proc/meminfo"):
        print("Jetson 에서 돌릴 것.", file=sys.stderr)
        return 2

    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    from tts_melo_onnx import build_synth
    from ko_normalize import normalize

    wavs = sorted(glob.glob(os.path.join(ROOT, "ref", "*.wav")))
    if not wavs:
        print("ref/*.wav 가 없다.", file=sys.stderr)
        return 2
    sentences = load_sentences(repo_paths()["sentences"])

    # ── 긴 발화 만들기 ────────────────────────────────────
    outdir = os.path.join(ROOT, "out", "long")
    os.makedirs(outdir, exist_ok=True)
    clips = [sf.read(w) for w in wavs]
    sr_in = clips[0][1]
    if any(c[1] != sr_in for c in clips):
        print("참조 파일의 표본율이 서로 다르다.", file=sys.stderr)
        return 2
    gap = np.zeros(int(sr_in * 0.3), dtype=clips[0][0].dtype)

    targets = [float(x) for x in args.lengths.split(",")]
    made = []
    for want in targets:
        buf, i = [], 0
        while sum(len(b) for b in buf) / sr_in < want:
            buf.append(clips[i % len(clips)][0]); buf.append(gap); i += 1
        audio = np.concatenate(buf)[:int(sr_in * want)]
        path = os.path.join(outdir, f"utt-{int(want):02d}s.wav")
        sf.write(path, audio, sr_in)
        made.append((want, path, len(audio) / sr_in))
    print("만든 발화:", ", ".join(f"{d:.1f}s" for _, _, d in made))

    sam = Sampler(); sam.start(); time.sleep(0.5)
    floor = min(r[1] for r in sam.rows)   # 모델 적재 전 바닥

    print("\n최종 구성으로 적재 중…")
    stt = WhisperModel("small", device="cuda", compute_type="int8_float16")
    synth, sr, provs, load_s, _ = build_synth(
        "models/melo-ko-onnx", providers="CUDAExecutionProvider,CPUExecutionProvider",
        threads=2, quiet=True, bert_int8=True)

    def transcribe(w):
        segs, _ = stt.transcribe(w, language="ko", beam_size=1)
        return "".join(s.text for s in segs).strip()

    transcribe(wavs[0]); synth(normalize(sentences[0]))   # 워밍업
    time.sleep(0.5)
    idle = sys_used_mb() - floor
    print(f"적재+워밍업 후 유휴: {idle:.0f} MB\n")

    # ── STT: 발화 길이별 ──────────────────────────────────
    print("### STT — 발화가 길어질 때")
    print(f"{'길이':>8}{'추론':>9}{'RTF':>7}{'최고 메모리':>13}{'유휴 대비':>11}")
    print("-" * 50)
    stt_rows = []
    for want, path, dur in made:
        t0 = time.perf_counter()
        transcribe(path)
        dt = time.perf_counter() - t0
        pk = sam.peak(t0, time.perf_counter()) - floor
        stt_rows.append((dur, dt, pk))
        print(f"{dur:>7.1f}s{dt:>8.2f}s{dt/dur:>7.2f}{pk:>11.0f} MB{pk-idle:>9.0f} MB")

    # ── TTS: 문장이 길어질 때 ─────────────────────────────
    print("\n### TTS — 합성할 문장이 길어질 때")
    print(f"{'글자수':>8}{'합성':>9}{'오디오':>9}{'RTF':>7}{'최고 메모리':>13}")
    print("-" * 50)
    tts_rows = []
    for mult in (1, 2, 4, 6, 8):
        text = " ".join(sentences[i % len(sentences)] for i in range(mult))
        t0 = time.perf_counter()
        audio = synth(normalize(text))
        dt = time.perf_counter() - t0
        pk = sam.peak(t0, time.perf_counter()) - floor
        adur = len(audio) / sr
        tts_rows.append((len(text), dt, adur, pk))
        print(f"{len(text):>8}{dt:>8.2f}s{adur:>8.2f}s{dt/adur:>7.2f}{pk:>11.0f} MB")

    # ── 쪼개서 합성하면 달라지는가 ────────────────────────
    # VITS 는 파형 전체를 한 번에 만든다. 긴 답변이면 중간 텐서가 그만큼
    # 커진다. 문장 단위로 잘라 따로 합성하면 각 조각이 짧으니 풀이 안 커야
    # 하고, 덤으로 첫 문장부터 바로 재생할 수 있다 (체감 지연이 줄어든다).
    print("\n### 같은 길이를 문장 단위로 쪼개면")
    long_text = " ".join(sentences[i % len(sentences)] for i in range(8))
    parts = [p.strip() for p in long_text.replace("?", "?|").replace(".", ".|").split("|") if p.strip()]
    t0 = time.perf_counter()
    first_at, total_audio = None, 0.0
    for part in parts:
        a = synth(normalize(part))
        if first_at is None:
            first_at = time.perf_counter() - t0
        total_audio += len(a) / sr
    dt = time.perf_counter() - t0
    pk_split = sam.peak(t0, time.perf_counter()) - floor
    whole = tts_rows[-1]
    print(f"  통째로 {whole[0]}자   합성 {whole[1]:.2f}s   최고 {whole[3]:.0f} MB")
    print(f"  {len(parts)}조각으로   합성 {dt:.2f}s   최고 {pk_split:.0f} MB"
          f"   첫 조각까지 {first_at:.2f}s")
    print(f"  → 메모리 {pk_split - whole[3]:+.0f} MB, 말을 시작하기까지 "
          f"{whole[1]:.2f}s → {first_at:.2f}s")

    sam.stop()

    print("\n### 판정")
    base_pk = stt_rows[0][2]
    worst = max(max(r[2] for r in stt_rows), max(r[3] for r in tts_rows))
    split_helps = pk_split < whole[3] - 100
    print(f"  가장 짧은 발화 {stt_rows[0][0]:.1f}s → {base_pk:.0f} MB")
    print(f"  가장 긴 발화  {stt_rows[-1][0]:.1f}s → {stt_rows[-1][2]:.0f} MB "
          f"({stt_rows[-1][2]-base_pk:+.0f} MB)")
    print(f"  전체 최고점 {worst:.0f} MB")
    print()
    if stt_rows[-1][2] - base_pk > 200:
        print("  ! 긴 발화에서 크게 늘어난다. 예산을 이 값 기준으로 잡거나,")
        print("    입력 길이에 상한을 걸어야 한다 (예: 15초에서 자르기).")
    elif stt_rows[-1][2] - base_pk > 50:
        print("  · 늘어나긴 하지만 감당할 수준이다. 예산에 반영할 것.")
    else:
        print("  ✔ 발화 길이에 거의 영향받지 않는다. 기존 예산이 유효하다.")
    if split_helps:
        print(f"  ✔ 문장 단위로 쪼개면 최고점이 {whole[3]:.0f} → {pk_split:.0f} MB 로 내려간다.")
        print("    긴 답변은 쪼개서 합성할 것. 체감 지연도 같이 줄어든다.")
    else:
        print(f"  · 쪼개도 최고점이 {pk_split:.0f} MB 다. 메모리로는 이득이 없다")
        print("    (체감 지연에는 여전히 이득이 있다).")
    slow = [r for r in stt_rows if r[1] / r[0] > 1.0]
    if slow:
        print(f"  ! {slow[0][0]:.0f}s 부터 RTF 가 1.0 을 넘는다 — 긴 발화는 실시간이 안 된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
