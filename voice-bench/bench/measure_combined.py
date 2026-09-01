"""STT + TTS를 한 프로세스에 같이 올렸을 때의 피크 RSS를 단계별로 잰다.

따로 재면 파이썬 런타임·numpy 같은 고정 비용이 후보마다 중복 계상된다.
실제 배포에서 한 프로세스에 넣을 거라면 그 몫은 한 번만 든다.
'프로세스를 합칠지'는 B가 런타임 골격에서 정할 문제인데, 그 판단에 필요한 숫자다.

    venvs/piper/bin/python measure_combined.py sherpa
    venvs/melo/bin/python  measure_combined.py melo
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


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "sherpa"
    tmp = os.path.join(tempfile.mkdtemp(), "o.wav")
    refs = sorted(__import__("glob").glob(os.path.join(ROOT, "ref", "*.wav")))
    marks = [("파이썬만", rss_mb())]

    # ── STT ──────────────────────────────────────────────
    from faster_whisper import WhisperModel
    marks.append(("+ faster-whisper import", rss_mb()))
    stt = WhisperModel("small", device="cpu", compute_type="int8")
    marks.append(("+ STT 모델 로드", rss_mb()))
    # 피크 RSS는 최고 수위 기록이라 추론을 반복할수록 올라간다.
    # 1문장만 돌리면 실사용보다 낮게 잡히므로 심사 문장 전체를 돌린다.
    for w in refs:
        list(stt.transcribe(w, language="ko", beam_size=1)[0])
    marks.append((f"+ STT 추론 {len(refs)}문장", rss_mb()))

    # ── TTS ──────────────────────────────────────────────
    if which == "sherpa":
        import sherpa_onnx
        import soundfile as sf
        marks.append(("+ sherpa import", rss_mb()))
        d = os.path.join(ROOT, "models", "vits-mimic3-ko_KO-kss_low")
        onnx = [f for f in os.listdir(d) if f.endswith(".onnx")][0]
        tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=os.path.join(d, onnx),
                    tokens=os.path.join(d, "tokens.txt"),
                    data_dir=os.path.join(d, "espeak-ng-data")),
                num_threads=2)))
        marks.append(("+ TTS 모델 로드", rss_mb()))
        for s in load_sentences(repo_paths()["sentences"]):
            a = tts.generate(s, sid=0, speed=1.0)
            sf.write(tmp, a.samples, a.sample_rate)
    else:
        from melo.api import TTS
        marks.append(("+ melo import", rss_mb()))
        tts = TTS(language="KR", device="cpu")
        marks.append(("+ TTS 모델 로드", rss_mb()))
        for s in load_sentences(repo_paths()["sentences"]):
            tts.tts_to_file(s, tts.hps.data.spk2id["KR"], tmp)
    marks.append(("+ TTS 합성 6문장", rss_mb()))

    print(f"\n### STT(faster-whisper small) + TTS({which}) 동일 프로세스")
    print(f"{'단계':<28}{'피크 RSS':>11}{'증가분':>10}")
    print("-" * 50)
    prev = 0.0
    for name, v in marks:
        print(f"{name:<26}{v:>9.0f} MB{v - prev:>8.0f} MB")
        prev = v
    print(f"\n합계 {marks[-1][1]:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
