"""STT — faster-whisper small 을 CUDA 로 돌린다.

Jetson 실측: RTF 0.35, 심사 문장 6개 CER 0.000 (30턴 반복해도 유지).
CPU 로는 6코어를 다 써도 RTF 1.07 이라 대화가 성립하지 않는다.
base 로 내리는 선택지는 없다 — 6문장 중 4개가 틀렸고 한자가 섞여 나왔다.

CUDA 로 돌리려면 ctranslate2 를 직접 빌드한 휠이 필요하다. PyPI 의 aarch64
휠은 CUDA 없이 빌드돼 있다(build-ctranslate2/ 참조).
"""
import numpy as np

# 짧은 조명 명령이 "끝내줘", "불 펴져"로 잘못 들려서 단어를 미리 알려 준다.
# "책 비춰줘"는 "체스 비춰줘", "책 빗쳐줘"로 들렸다.
HOTWORDS = "픽스야 불 켜줘 불 꺼줘 밝게 어둡게 조명 책 비춰줘"
WAKE_HOTWORDS = "픽스야"
# 웅얼거림을 힌트를 베껴 "불 켜줘"로 적었다(2026-11-01). 그때 확신도는
# logprob -0.78, no_speech 0.81 이었고, 또렷한 말("불 켜줘", "픽스야" 10번)은
# logprob -0.22~-0.29, no_speech 0.11~0.20 이었다. 진짜 "불 꺼줘"가 -0.40,
# 0.47 까지 나왔으므로 여유를 두고, 둘 다 나쁠 때만 버린다.
UNCLEAR_NO_SPEECH = 0.65
UNCLEAR_LOGPROB = -0.6
_HOT_SET = set(HOTWORDS.split())
_HOT_ACTIONS = {"켜줘", "꺼줘", "밝게", "어둡게", "비춰줘"}


def echoes_hotwords(text):
    """힌트 단어만으로 된 문장에 명령이 두 번 이상 나오면 Whisper가 힌트를 베낀 것이다.
    barge-in 직후 "불 켜줘 불 켜줘", "불 켜줘 불 켜줘 밝게 어둡게 조명"이 실제로 나왔다."""
    words = "".join(c if c.isalnum() or c.isspace() else " " for c in text).split()
    return (bool(words) and all(w in _HOT_SET for w in words)
            and sum(w in _HOT_ACTIONS for w in words) >= 2)


class Stt:
    def __init__(self, model="small", device="cuda", compute_type=None, cpu_threads=0):
        from faster_whisper import WhisperModel
        if compute_type is None:
            compute_type = "int8_float16" if device == "cuda" else "int8"
        self.m = WhisperModel(model, device=device, compute_type=compute_type,
                              cpu_threads=cpu_threads)
        self.device = device
        self.name = f"faster-whisper {model}/{device}/{compute_type}"

    def warmup(self):
        """무음 1초를 받아쓰고 버린다. 첫 호출이 느린 값을 미리 치른다
        (실측 첫 턴 2.61s, 둘째 턴 0.70s)."""
        import time

        import numpy as np
        t = time.time()
        try:
            self.transcribe(np.zeros(16000, dtype=np.float32), 16000)
        except Exception as e:
            print(f"  ! STT 예열 실패({type(e).__name__}) — 첫 응답이 느릴 수 있다")
            return 0.0
        return time.time() - t

    def transcribe(self, audio, samplerate=16000, hotwords=HOTWORDS, vad_filter=True,
                   drop_unclear=True):
        """float32 모노 버퍼를 받아 문자열로. beam_size=1 은 실시간 대화용 설정이다."""
        x = np.asarray(audio, dtype=np.float32)
        if samplerate != 16000:
            from .audio import resample
            x = resample(x, samplerate, 16000)
        # vad_filter: 노래나 무음 구간에서 "고맙습니다." 같은 문장을 지어내던
        # 것을 막는다. 깨끗한 발화 5문장에서는 결과가 같았다.
        segs, _ = self.m.transcribe(x, language="ko", beam_size=1,
                                  hotwords=hotwords, vad_filter=vad_filter)
        segs = list(segs)
        text = "".join(s.text for s in segs).strip()
        if segs:
            # 웅얼거림이 힌트를 베껴 "불 켜줘"로 나왔다(2026-11-01). 진짜 말과
            # 가를 기준을 찾으려고 확신도를 남긴다.
            print(f"  받아쓰기 확신도 logprob "
                  f"{min(s.avg_logprob for s in segs):.2f} no_speech "
                  f"{max(s.no_speech_prob for s in segs):.2f} "
                  f"압축 {max(s.compression_ratio for s in segs):.2f}: {text!r}",
                  flush=True)
        if (drop_unclear and segs
                and max(s.no_speech_prob for s in segs) > UNCLEAR_NO_SPEECH
                and min(s.avg_logprob for s in segs) < UNCLEAR_LOGPROB):
            print(f"  ! 말이 아닌 것 같아 버린다: {text}", flush=True)
            return ""
        if echoes_hotwords(text):
            print(f"  ! 힌트 단어를 베낀 받아쓰기를 버린다: {text}", flush=True)
            return ""
        return text

    def transcribe_wake(self, audio, samplerate=16000):
        """호출어 모델이 깬 구간을 확인용으로 받아쓴다.

        명령 힌트를 그대로 주면 진짜 "픽스야"가 "뷔셜", "비트야"로 나왔다
        (2026-10-31 녹음, 호출 4번 중 2번). 힌트를 "픽스야" 하나로 줄이고
        vad_filter 를 끄자 4번 모두 픽스야/피스야로 나왔고, 노래에 깬 12번은
        모두 빈 문자열이었다(Jetson ~/backups/calls-20261031-2254.wav, music-20261031-2216.wav).
        """
        return self.transcribe(audio, samplerate, hotwords=WAKE_HOTWORDS, vad_filter=False,
                               drop_unclear=False)
