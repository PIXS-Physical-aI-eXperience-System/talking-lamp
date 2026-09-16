"""AEC 성능과 barge-in 감지 지연 실측.

E 의 안건 3-1("내장 AEC 로 충분한지")과 3-3("barge-in 감지 지연 목표치")에
답하기 위한 측정. 마이크 어레이가 있어야 돌아간다.

    python bench/aec_check.py echo       # 에코 잔향 — AEC 가 얼마나 지우나
    python bench/aec_check.py bargein    # 끼어들기 감지 지연

전제: 스피커가 XVF3800 의 JST 출력에 연결돼 있고, TTS 를 그 장치로 재생해야
한다. Jetson 에 직결하면 보드가 참조신호를 못 받아 하드웨어 AEC 가 동작하지
않는다(그 경우 이 스크립트는 '지워지지 않는다'만 확인해 줄 뿐이다).
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out", "doa")
SR = 16000


def read_wav(path):
    """wav 를 float32 모노로 읽는다. soundfile 없이 표준 라이브러리만 쓴다.

    이 도구는 라즈베리파이에서 돌려야 하는데, 파이에는 인터넷이 없어서
    패키지를 하나 더 얹는 것이 곧 휠을 손으로 옮기는 일이 된다.
    """
    import wave
    with wave.open(path, "rb") as w:
        n, ch, width, sr = (w.getnframes(), w.getnchannels(),
                            w.getsampwidth(), w.getframerate())
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"16비트 wav 만 읽는다 (이 파일은 {width*8}비트)")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path, x, sr):
    import wave
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def to_sr(x, src, dst=SR):
    """표본율을 맞춘다. 보드 출력이 16 kHz 라 44.1 kHz TTS 를 그대로 못 넣는다.

    scipy 없이 선형 보간으로 한다 — 파이에 패키지를 더 얹지 않기 위해서다.
    측정 대상은 '자기 목소리가 얼마나 지워지는가' 이므로 이 정도로 충분하다.
    """
    if src == dst:
        return x
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n),
                     np.arange(len(x)), x).astype(np.float32)


def _finite(x, what):
    """값이 성한지 본다. 계산 전에 막는다.

    채워지지 않은 녹음 버퍼는 초기화되지 않은 메모리를 그대로 들고 있어서
    dBFS 가 +710 같은 불가능한 값으로 나온다. 실제로 그렇게 나왔고 판정까지
    그대로 통과했다. 물리적으로 불가능한 값은 결과가 아니라 고장이다.
    """
    x = np.asarray(x, dtype=np.float32)
    if not np.all(np.isfinite(x)):
        raise RuntimeError(f"{what}: NaN/무한대가 섞였다 — 장치를 확인할 것")
    peak = float(np.max(np.abs(x)))
    if peak > 4.0:
        raise RuntimeError(f"{what}: 표본 최대값이 {peak:.3g} 다. 정상 범위(-1~1)를 "
                           "크게 벗어났다 — 녹음 버퍼가 채워지지 않았다")
    return x


def db(x):
    """RMS 를 dBFS 로."""
    r = float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2) + 1e-12))
    return 20 * np.log10(max(r, 1e-12))


def pick_devices(sd):
    """XVF3800 을 입력·출력 양쪽에서 찾는다."""
    ins = outs = None
    for i, d in enumerate(sd.query_devices()):
        n = d["name"].lower()
        if any(k in n for k in ("xvf", "respeaker", "xmos")):
            if d["max_input_channels"] > 0 and ins is None:
                ins = i
            if d["max_output_channels"] > 0 and outs is None:
                outs = i
    return ins, outs


def cmd_echo(args):
    """TTS 를 재생하면서 마이크를 녹음해, 램프 자기 목소리가 얼마나 남는지 본다.

    남은 잔향이 사용자 목소리보다 크면 웨이크워드가 자기 소리에 반응하고
    barge-in 이 자기 말에 오작동한다. 그 여유가 몇 dB 인지가 핵심이다.
    """
    import sounddevice as sd

    ins, outs = pick_devices(sd)
    if ins is None:
        print("XVF3800 입력 장치를 못 찾았다. bench/mic_check.py 먼저 실행")
        return 1
    if outs is None:
        print("⚠ XVF3800 이 출력 장치로 안 잡힌다 — 스피커가 보드에 연결됐는지 확인.")
        print("  Jetson 직결이면 하드웨어 AEC 가 동작하지 않는다.")

    tts = args.tts or os.path.join(ROOT, "out", "tts", "melo-onnx-int8", "04.wav")
    if not os.path.exists(tts):
        print(f"재생할 TTS 파일이 없다: {tts}")
        return 1
    audio, sr = read_wav(tts)
    if sr != SR:
        print(f"  ({sr} Hz → {SR} Hz 로 맞춘다. 보드 출력이 {SR} Hz 다)")
        audio, sr = to_sr(audio, sr), SR
    dur = len(audio) / sr

    res = {}
    print(f"재생 파일 {os.path.basename(tts)} ({dur:.1f}초)\n")

    print("① 무음 기준 — 아무 소리도 내지 말고 기다리세요")
    input("   Enter → 3초 녹음 ")
    quiet = sd.rec(int(3 * SR), samplerate=SR, channels=1, device=ins, dtype="float32")
    sd.wait()
    quiet = _finite(quiet, "조용할 때 녹음")
    res["quiet_db"] = db(quiet)
    print(f"   배경소음 {res['quiet_db']:.1f} dBFS")

    print("\n② 램프만 말하는 중 — 사용자는 조용히")
    input("   Enter → 재생하며 녹음 ")
    # sd.rec 로 녹음을 걸어둔 뒤 sd.play 를 부르면 안 된다. 둘이 같은 전역
    # 스트림을 쓰기 때문에 녹음 스트림이 교체되고 버퍼가 채워지지 않은 채
    # 남는다. 그 쓰레기 값으로 +710 dBFS 가 나왔다. 동시 입출력은 playrec 다.
    # playrec 은 input_device/output_device 가 아니라 device=(입력, 출력) 를 받는다.
    # None 이면 그쪽은 기본 장치를 쓴다.
    rec = sd.playrec(audio, samplerate=SR, channels=1, dtype="float32",
                     device=(ins, outs))
    sd.wait()
    rec = rec[:, 0] if getattr(rec, "ndim", 1) > 1 else rec
    rec = _finite(rec, "램프 발화 중 녹음")
    res["echo_db"] = db(rec)
    print(f"   AEC 통과 후 남은 잔향 {res['echo_db']:.1f} dBFS"
          f"   (조용할 때 {res['quiet_db']:.1f})")

    print("\n③ 사용자 발화 기준 — 평소 위치(1 m)에서 3초간 말하세요")
    input("   Enter → 3초 녹음 ")
    speech = sd.rec(int(3 * SR), samplerate=SR, channels=1, device=ins, dtype="float32")
    sd.wait()
    speech = _finite(speech, "사용자 발화 녹음")
    res["speech_db"] = db(speech)
    print(f"   사용자 발화 {res['speech_db']:.1f} dBFS")

    margin = res["speech_db"] - res["echo_db"]
    res["margin_db"] = margin
    print(f"\n여유 {margin:+.1f} dB  (사용자 발화 − 잔향)")
    if margin >= 15:
        print("  ✔ 충분. 내장 AEC 만으로 간다")
    elif margin >= 6:
        print("  △ 빠듯하다. 스피커 음량을 낮추거나 이격을 늘릴 것")
    else:
        print("  ✗ 부족. 잔향이 사용자 목소리에 묻힌다.")
        print("    → 이격 확대 / 음량 축소 / 소프트웨어 AEC 보강 검토")

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "aec_echo.json"), "w"), indent=2)
    for name, a in (("quiet", quiet), ("echo", rec), ("speech", speech)):
        write_wav(os.path.join(OUT, f"aec_{name}.wav"), a, SR)
    print(f"저장: {OUT}/aec_echo.json + wav 3개")
    return 0


def cmd_bargein(args):
    """끼어들기 감지 지연.

    사람 반응시간이 섞이므로, TTS 를 끄고 잰 값을 기준선으로 삼아 뺀다.
    그 차이가 '램프가 말하는 중이라서 늦어진 몫' 이고, 그게 우리가 알고 싶은 값이다.
    """
    import sounddevice as sd
    from ten_vad import TenVad

    ins, outs = pick_devices(sd)
    if ins is None:
        print("XVF3800 입력 장치를 못 찾았다")
        return 1
    tts = args.tts or os.path.join(ROOT, "out", "tts", "melo-onnx-int8", "04.wav")
    audio, sr = read_wav(tts) if os.path.exists(tts) else (None, SR)
    if audio is not None and sr != SR:
        audio, sr = to_sr(audio, sr), SR

    HOP = 256
    def trial(play):
        v = TenVad(hop_size=HOP, threshold=0.5)
        print("     3..2..1..  신호 후 곧바로 '잠깐만' 이라고 말하세요")
        time.sleep(1.5)
        buf = []
        stream = sd.InputStream(samplerate=SR, channels=1, device=ins, dtype="float32",
                                blocksize=HOP, callback=lambda ind, *_: buf.append(ind.copy()))
        with stream:
            if play and audio is not None:
                sd.play(audio, sr, device=outs) if outs is not None else sd.play(audio, sr)
            print("     ▶ 지금!")
            t0 = time.perf_counter()
            hit = None
            while time.perf_counter() - t0 < 4.0:
                if buf:
                    blk = (np.clip(buf.pop(0)[:, 0], -1, 1) * 32767).astype(np.int16)
                    if len(blk) == HOP:
                        _, flag = v.process(blk)
                        if flag and hit is None:
                            hit = time.perf_counter() - t0
                            break
                else:
                    time.sleep(0.005)
            sd.stop()
        return hit

    n = args.trials
    base, with_tts = [], []
    print(f"기준선 측정 (TTS 없이) — {n}회")
    for i in range(n):
        print(f"  {i+1}/{n}")
        t = trial(False)
        if t: base.append(t)
    print(f"\nTTS 재생 중 측정 — {n}회")
    for i in range(n):
        print(f"  {i+1}/{n}")
        t = trial(True)
        if t: with_tts.append(t)

    if not base or not with_tts:
        print("측정 실패 — 발화가 감지되지 않았다")
        return 1
    b, w = statistics.median(base), statistics.median(with_tts)
    print(f"\n기준선(반응시간 포함)  {b*1000:>6.0f} ms")
    print(f"TTS 재생 중            {w*1000:>6.0f} ms")
    print(f"램프 발화로 인한 지연   {(w-b)*1000:>+6.0f} ms  ← E 에게 넘길 값")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"baseline_ms": b*1000, "with_tts_ms": w*1000,
               "penalty_ms": (w-b)*1000, "trials": n},
              open(os.path.join(OUT, "bargein.json"), "w"), indent=2)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("echo"); e.add_argument("--tts")
    b = sub.add_parser("bargein"); b.add_argument("--tts"); b.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()
    return {"echo": cmd_echo, "bargein": cmd_bargein}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
