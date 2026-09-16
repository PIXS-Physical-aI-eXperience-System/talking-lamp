"""STT — faster-whisper small 을 CUDA 로 돌린다.

Jetson 실측: RTF 0.35, 심사 문장 6개 CER 0.000 (30턴 반복해도 유지).
CPU 로는 6코어를 다 써도 RTF 1.07 이라 대화가 성립하지 않는다.
base 로 내리는 선택지는 없다 — 6문장 중 4개가 틀렸고 한자가 섞여 나왔다.

CUDA 로 돌리려면 ctranslate2 를 직접 빌드한 휠이 필요하다. PyPI 의 aarch64
휠은 CUDA 없이 빌드돼 있다(build-ctranslate2/ 참조).
"""
import numpy as np


class Stt:
    def __init__(self, model="small", device="cuda", compute_type=None, cpu_threads=0):
        from faster_whisper import WhisperModel
        if compute_type is None:
            compute_type = "int8_float16" if device == "cuda" else "int8"
        self.m = WhisperModel(model, device=device, compute_type=compute_type,
                              cpu_threads=cpu_threads)
        self.device = device
        self.name = f"faster-whisper {model}/{device}/{compute_type}"

    def transcribe(self, audio, samplerate=16000):
        """float32 모노 버퍼를 받아 문자열로. beam_size=1 은 실시간 대화용 설정이다."""
        x = np.asarray(audio, dtype=np.float32)
        if samplerate != 16000:
            from .audio import resample
            x = resample(x, samplerate, 16000)
        segs, _ = self.m.transcribe(x, language="ko", beam_size=1)
        return "".join(s.text for s in segs).strip()
