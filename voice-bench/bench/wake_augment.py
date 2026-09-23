"""녹음 190개로 웨이크워드를 학습하기 위한 데이터 부풀리기.

호출어 100개(5명 × 20회)는 그대로 쓰기엔 턱없이 적다. 원본을 그대로 20번
보여주면 그 20개 녹음의 잡음과 마이크 특성까지 외운다. 그래서 하나에서
여러 개를 만들어 낸다 — 위치, 크기, 방 소리, 말 빠르기를 바꿔서.

창(window)은 2.0초다. openWakeWord 임베딩이 2.0초 오디오에서 정확히
(16, 96) 을 내놓고, 이게 분류기가 받는 모양이다. 그래서 길이를 맞출 필요가
없다 — 녹음 한 개가 창 한 개다.

가장 중요한 것은 **잘린 호출어를 부정으로 넣는 것**이다. 실제로는 창이
80 ms 씩 밀려가므로, "픽스"까지만 들어온 창이 반드시 생긴다. 이것을
가르치지 않으면 램프가 말이 끝나기 전에 깨어난다.
"""
import numpy as np

SR = 16000
WINDOW_S = 2.0
WINDOW = int(SR * WINDOW_S)   # 32000 표본 = 임베딩 16개


def rms_db(x):
    return 20 * np.log10(float(np.sqrt(np.mean(np.square(x)))) + 1e-12)


def speech_span(x, sr=SR, frame_ms=20, rise_db=12.0, pad_s=0.08):
    """말이 있는 구간 (시작, 끝) 표본 번호. 못 찾으면 전체."""
    n = int(sr * frame_ms / 1000)
    frames = x[:len(x) // n * n].reshape(-1, n)
    if len(frames) < 5:
        return 0, len(x)
    lv = 20 * np.log10(np.sqrt(np.mean(np.square(frames), axis=1)) + 1e-12)
    floor = np.percentile(lv, 20)
    loud = np.where(lv > floor + rise_db)[0]
    if len(loud) == 0:
        return 0, len(x)
    pad = int(sr * pad_s)
    a = max(0, loud[0] * n - pad)
    b = min(len(x), (loud[-1] + 1) * n + pad)
    return a, b


def resample(x, rate):
    """말 빠르기 바꾸기. 선형 보간이면 충분하다 — 학습 증강이지 재생이 아니다."""
    if abs(rate - 1.0) < 1e-3:
        return x
    n = int(round(len(x) / rate))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def noise_bed(noises, rng, n=WINDOW):
    """방 소리에서 n 표본을 잘라 온다. 방 소리가 없으면 아주 작은 잡음."""
    if not len(noises):
        return rng.normal(0, 1e-4, n).astype(np.float32)
    src = noises[rng.integers(len(noises))]
    if len(src) <= n:
        return np.tile(src, n // max(len(src), 1) + 1)[:n].astype(np.float32)
    i = rng.integers(0, len(src) - n)
    return src[i:i + n].astype(np.float32)


def place(seg, noises, rng, end_frac, snr_db, gain_db):
    """seg 를 2초 창 안에 놓는다. end_frac 은 seg 의 끝이 놓일 창 안 위치(0~1).

    1을 넘으면 뒤가 잘린다 — 잘린 호출어를 만들 때 쓴다.
    """
    bed = noise_bed(noises, rng)
    bed_lv = rms_db(bed)
    seg = seg * (10 ** (gain_db / 20.0))
    seg_lv = rms_db(seg)
    # 원하는 SNR 이 되도록 방 소리 쪽을 맞춘다(말소리는 건드리지 않는다)
    bed = bed * (10 ** ((seg_lv - snr_db - bed_lv) / 20.0))

    out = bed.copy()
    end = int(WINDOW * end_frac)
    start = end - len(seg)
    s0, s1 = max(start, 0), min(end, WINDOW)
    if s1 > s0:
        out[s0:s1] += seg[s0 - start:s1 - start]
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out


def to_int16(x):
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
