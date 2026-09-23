"""STT + TTS를 한 프로세스에 같이 올렸을 때의 피크 RSS를 단계별로 잰다.

따로 재면 파이썬 런타임·numpy 같은 고정 비용이 후보마다 중복 계상된다.
실제 배포에서 한 프로세스에 넣을 거라면 그 몫은 한 번만 든다.
'프로세스를 합칠지'는 B가 런타임 골격에서 정할 문제인데, 그 판단에 필요한 숫자다.

    venvs/piper/bin/python     bench/measure_combined.py sherpa
    venvs/melo/bin/python      bench/measure_combined.py melo
    venvs/melo-onnx/bin/python bench/measure_combined.py melo-onnx --stt-device cuda

Jetson 은 CPU 와 GPU 가 메모리를 공유한다. 그래서 RSS 만 보면 GPU 할당분이
얼마나 잡히는지 알 수 없어, 시스템 전체 가용 메모리 감소분을 같이 잰다.
예산은 둘 중 큰 쪽으로 잡아야 안전하다.
"""
import os
import resource
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # voice-bench/ — 공용 모듈과 산출물이 있는 곳
sys.path.insert(0, ROOT)

from common import load_sentences, repo_paths  # noqa: E402


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def sys_used_mb():
    """시스템 전체가 쓰고 있는 메모리(MB). /proc 이 없으면 None.

    통합 메모리에서는 GPU 가 잡아간 몫이 프로세스 RSS 에 안 잡힐 수 있다.
    MemAvailable 의 감소로 보면 그게 드러난다.
    """
    try:
        info = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            info[k] = float(v.strip().split()[0]) / 1024
        return round(info["MemTotal"] - info["MemAvailable"], 1)
    except Exception:
        return None


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("which", nargs="?", default="sherpa",
                    choices=["sherpa", "melo", "melo-onnx"])
    ap.add_argument("--stt-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--int8", action="store_true",
                    help="melo-onnx 를 int8 로. GPU 에서는 붙이지 말 것 (10배 느리다)")
    ap.add_argument("--providers", default="auto")
    args = ap.parse_args()
    which = args.which

    tmp = os.path.join(tempfile.mkdtemp(), "o.wav")
    refs = sorted(__import__("glob").glob(os.path.join(ROOT, "ref", "*.wav")))
    base_sys = sys_used_mb()
    marks = [("파이썬만", rss_mb(), sys_used_mb())]

    def mark(name):
        marks.append((name, rss_mb(), sys_used_mb()))

    # ── STT ──────────────────────────────────────────────
    from faster_whisper import WhisperModel
    mark("+ faster-whisper import")
    ct = "int8_float16" if args.stt_device == "cuda" else "int8"
    stt = WhisperModel("small", device=args.stt_device, compute_type=ct)
    mark("+ STT 모델 로드")
    # 피크 RSS는 최고 수위 기록이라 추론을 반복할수록 올라간다.
    # 1문장만 돌리면 실사용보다 낮게 잡히므로 심사 문장 전체를 돌린다.
    for w in refs:
        list(stt.transcribe(w, language="ko", beam_size=1)[0])
    mark(f"+ STT 추론 {len(refs)}문장")

    # ── TTS ──────────────────────────────────────────────
    if which == "sherpa":
        import sherpa_onnx
        import soundfile as sf
        mark("+ sherpa import")
        d = os.path.join(ROOT, "models", "vits-mimic3-ko_KO-kss_low")
        onnx = [f for f in os.listdir(d) if f.endswith(".onnx")][0]
        tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=os.path.join(d, onnx),
                    tokens=os.path.join(d, "tokens.txt"),
                    data_dir=os.path.join(d, "espeak-ng-data")),
                num_threads=2)))
        mark("+ TTS 모델 로드")
        for s in load_sentences(repo_paths()["sentences"]):
            a = tts.generate(s, sid=0, speed=1.0)
            sf.write(tmp, a.samples, a.sample_rate)
    elif which == "melo-onnx":
        # 러너와 같은 build_synth 를 쓴다. 여기서만 따로 모델을 올리면
        # 두 곳의 숫자가 조용히 어긋난다.
        sys.path.insert(0, os.path.join(ROOT, "runners"))
        from tts_melo_onnx import build_synth
        from ko_normalize import normalize
        mark("+ onnxruntime import")
        synth, sr, provs, load_s, _ = build_synth(
            "models/melo-ko-onnx", int8=args.int8,
            providers=args.providers, threads=2, quiet=True)
        mark("+ TTS 모델 로드")
        synth(normalize(load_sentences(repo_paths()["sentences"])[0]))  # 워밍업
        mark("+ TTS 워밍업")
        import soundfile as sf
        for s in load_sentences(repo_paths()["sentences"]):
            sf.write(tmp, synth(normalize(s)), sr)
        print(f"\n  TTS 공급자 {provs}  로드 {load_s}s", file=sys.stderr)
    else:
        from melo.api import TTS
        mark("+ melo import")
        tts = TTS(language="KR", device="cpu")
        mark("+ TTS 모델 로드")
        for s in load_sentences(repo_paths()["sentences"]):
            tts.tts_to_file(s, tts.hps.data.spk2id["KR"], tmp)
    mark("+ TTS 합성 6문장")

    has_sys = base_sys is not None
    print(f"\n### STT(faster-whisper small, {args.stt_device}) + TTS({which}) 동일 프로세스")
    hdr = f"{'단계':<26}{'피크 RSS':>11}{'증가분':>10}"
    if has_sys:
        hdr += f"{'시스템 사용':>13}"
    print(hdr)
    print("-" * (len(hdr) + 8))
    prev = 0.0
    for row in marks:
        name, v, sv = row
        line = f"{name:<24}{v:>9.0f} MB{v - prev:>8.0f} MB"
        if has_sys and sv is not None:
            line += f"{sv - base_sys:>11.0f} MB"
        print(line)
        prev = v
    print(f"\n피크 RSS 합계 {marks[-1][1]:.0f} MB")
    if has_sys and marks[-1][2] is not None:
        grew = marks[-1][2] - base_sys
        print(f"시스템 사용 증가 {grew:.0f} MB")
        if grew > marks[-1][1] * 1.15:
            print("※ 시스템 증가분이 RSS 보다 크다 — GPU 가 잡아간 몫이 RSS 에 안 잡히고 있다.")
            print("  예산은 시스템 증가분 기준으로 잡을 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
