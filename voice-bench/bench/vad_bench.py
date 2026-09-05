"""VAD 후보 비교 — 발화 시작·종료 감지 지연과 오탐을 잰다.

왜 이 숫자가 필요한가:
  · 발화 시작 지연 = barge-in 감지 지연. E 에게 "150~200ms" 로 잠정 회신했고
    실측해서 확정해야 한다. TTS 발화 중에도 VAD 만은 계속 돌아야 하므로
    (반이중 구조) 이 값이 곧 끼어들기 반응 속도가 된다.
  · 발화 종료 지연 = STT 를 언제 끊을지. 너무 짧으면 말이 잘리고
    너무 길면 응답이 굼떠 보인다.
  · 오탐 = 조용할 때 말했다고 하는 빈도. 웨이크워드 오작동으로 이어진다.

기준시각(정답)은 에너지 포락선으로 잡는다. 완벽한 정답은 아니지만
세 후보에 같은 기준을 쓰므로 비교에는 충분하다.

    venvs/vad/bin/python bench/vad_bench.py
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from common import load_sentences, repo_paths  # noqa: E402

SR = 16000


def rss_mb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1024


def energy_bounds(x, frame=160, rel_db=-35.0):
    """에너지 포락선으로 발화 구간의 시작·끝을 추정한다 (기준시각).

    프레임 10ms 단위 RMS 를 구하고, 최대값 대비 rel_db 이상인 첫/마지막
    프레임을 발화 경계로 본다.
    """
    n = len(x) // frame
    rms = np.array([np.sqrt(np.mean(x[i*frame:(i+1)*frame] ** 2) + 1e-12) for i in range(n)])
    thr = rms.max() * (10 ** (rel_db / 20))
    idx = np.where(rms > thr)[0]
    if len(idx) == 0:
        return None, None
    return idx[0] * frame / SR, (idx[-1] + 1) * frame / SR


# ── 후보별 스트리밍 래퍼 ────────────────────────────────────────────────
# 각자 요구하는 청크 크기가 다르므로 그대로 존중하고, 결과는 초 단위로 환산한다.

def run_silero(x, threshold=0.5):
    from silero_vad import load_silero_vad
    import torch
    model = load_silero_vad(onnx=True)
    hop = 512
    flags = []
    for i in range(0, len(x) - hop + 1, hop):
        p = model(torch.from_numpy(x[i:i+hop]).unsqueeze(0), SR).item()
        flags.append(p >= threshold)
    return flags, hop


def run_ten(x, threshold=0.5):
    from ten_vad import TenVad
    hop = 256
    v = TenVad(hop_size=hop, threshold=threshold)
    xi = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    flags = []
    for i in range(0, len(xi) - hop + 1, hop):
        prob, flag = v.process(xi[i:i+hop])
        flags.append(bool(flag))
    return flags, hop


def run_webrtc(x, aggressiveness=2):
    import webrtcvad
    v = webrtcvad.Vad(aggressiveness)
    hop = 320  # 20 ms
    xi = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    flags = []
    for i in range(0, len(xi) - hop + 1, hop):
        flags.append(v.is_speech(xi[i:i+hop].tobytes(), SR))
    return flags, hop


CANDIDATES = {"silero": run_silero, "ten": run_ten, "webrtc": run_webrtc}


def first_run(flags, need=2):
    """연속 need 프레임이 speech 인 첫 지점 (단발 잡음에 안 흔들리게)."""
    for i in range(len(flags) - need + 1):
        if all(flags[i:i+need]):
            return i
    return None


def last_run(flags, need=2):
    for i in range(len(flags) - need, -1, -1):
        if all(flags[i:i+need]):
            return i + need - 1
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="후보 하나만 (silero/ten/webrtc)")
    args = ap.parse_args()

    p = repo_paths()
    sents = load_sentences(p["sentences"])
    wavs = sorted(f for f in os.listdir(p["ref"]) if f.endswith(".wav"))
    if not wavs:
        print("ref/*.wav 가 없다. bench/record.py 로 먼저 녹음할 것")
        return 1

    names = [args.only] if args.only else list(CANDIDATES)
    print(f"{'후보':<9}{'시작지연':>10}{'종료지연':>10}{'오탐':>8}{'RTF':>9}{'RSS':>9}")
    print("-" * 56)

    for name in names:
        fn = CANDIDATES[name]
        on_d, off_d, fp, rtf_all = [], [], 0, []
        base = rss_mb()
        for w in wavs:
            x, sr = sf.read(os.path.join(p["ref"], w), dtype="float32")
            assert sr == SR, f"{w}: {sr}Hz — 16k 로 녹음돼야 한다"
            t_on, t_off = energy_bounds(x)
            t0 = time.perf_counter()
            flags, hop = fn(x)
            rtf_all.append((time.perf_counter() - t0) / (len(x) / SR))

            i0, i1 = first_run(flags), last_run(flags)
            if i0 is not None and t_on is not None:
                on_d.append((i0 + 1) * hop / SR - t_on)      # 프레임 끝에서 판정된다
                off_d.append((i1 + 1) * hop / SR - t_off)
            # 발화 시작 전 구간에서 speech 로 판정된 프레임 = 오탐
            pre = int(t_on * SR / hop) if t_on else 0
            fp += sum(flags[:max(pre - 1, 0)])
        peak = rss_mb() - base
        print(f"{name:<9}{np.mean(on_d)*1000:>9.0f}ms{np.mean(off_d)*1000:>9.0f}ms"
              f"{fp:>8}{np.mean(rtf_all):>9.4f}{peak:>8.0f}M")

    print("\n시작지연 = barge-in 감지 지연 (작을수록 좋음, E 에게 넘길 값)")
    print("종료지연 = 발화 끝 판정 지연 (STT 를 언제 끊을지)")
    print("오탐     = 발화 시작 전 구간에서 speech 로 오판정한 프레임 수")
    return 0


if __name__ == "__main__":
    sys.exit(main())
