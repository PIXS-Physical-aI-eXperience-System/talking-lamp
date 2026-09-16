"""발화 감지 — barge-in 과 발화 끝 판정에 쓴다.

silero 는 쓰지 않는다. torch 를 끌고 오는데, 이 파이프라인의 전제가 torch 없음이다
(melo 를 ONNX 로 옮긴 이유가 그것이다). 대신 가벼운 것부터 찾아 쓴다.

주의: 이 파일의 임계값은 아직 실측으로 정해진 값이 아니다. 예전 VAD 벤치마크는
녹음에 선행 무음이 없어서(6개 중 5개가 0.00초에 발화 시작) 아무것도 재지
못했다. 마이크가 생겼으니 방 무음을 녹음해 다시 재야 한다.
"""
import numpy as np


class EnergyVad:
    """마지막 수단. 에너지만 본다 — 잡음이 있으면 쉽게 속는다."""

    def __init__(self, samplerate=16000, threshold_db=-45.0):
        self.threshold_db = threshold_db
        self.name = "energy"

    def is_speech(self, frame):
        rms = float(np.sqrt(np.mean(np.square(frame))) + 1e-12)
        return 20 * np.log10(rms) > self.threshold_db


class TenVad:
    def __init__(self, samplerate=16000, hop=256):
        from ten_vad import TenVad as _T
        self.v = _T(hop_size=hop)
        self.hop = hop
        self.name = "ten-vad"

    def is_speech(self, frame):
        x = (np.clip(frame, -1, 1) * 32767).astype(np.int16)
        out = False
        for i in range(0, len(x) - self.hop + 1, self.hop):
            _, flag = self.v.process(x[i:i + self.hop])
            out = out or bool(flag)
        return out


class WebrtcVad:
    def __init__(self, samplerate=16000, aggressiveness=2):
        import webrtcvad
        self.v = webrtcvad.Vad(aggressiveness)
        self.sr = samplerate
        self.frame = int(samplerate * 0.02)     # webrtc 는 10/20/30 ms 만 받는다
        self.name = "webrtc"

    def is_speech(self, frame):
        x = (np.clip(frame, -1, 1) * 32767).astype(np.int16)
        for i in range(0, len(x) - self.frame + 1, self.frame):
            if self.v.is_speech(x[i:i + self.frame].tobytes(), self.sr):
                return True
        return False


def load_vad(samplerate=16000, prefer=None):
    """쓸 수 있는 것 중 가장 나은 것을 고른다. 고른 결과를 알려준다."""
    order = [prefer] if prefer else []
    order += ["ten-vad", "webrtc", "energy"]
    for name in order:
        try:
            if name == "ten-vad":
                return TenVad(samplerate)
            if name == "webrtc":
                return WebrtcVad(samplerate)
            if name == "energy":
                return EnergyVad(samplerate)
        except Exception:
            continue
    return EnergyVad(samplerate)


class SpeechGate:
    """프레임 단위 판정을 '발화가 시작됐다 / 끝났다' 로 바꾼다.

    한 프레임만 보고 판단하면 기침 한 번에 깨어나고, 말 사이 숨 쉴 때 끊긴다.
    시작은 연속 몇 프레임, 끝은 무음이 얼마나 이어졌는지로 본다.
    """

    def __init__(self, vad, start_frames=3, end_silence_s=0.8, frame_s=0.032):
        self.vad = vad
        self.start_frames = start_frames
        self.end_frames = max(1, int(end_silence_s / frame_s))
        self.hits = 0
        self.quiet = 0
        self.speaking = False

    def update(self, frame):
        """(발화 시작됨, 발화 끝남) 을 돌려준다."""
        if self.vad.is_speech(frame):
            self.hits += 1
            self.quiet = 0
        else:
            self.hits = 0
            self.quiet += 1
        started = ended = False
        if not self.speaking and self.hits >= self.start_frames:
            self.speaking, started = True, True
        elif self.speaking and self.quiet >= self.end_frames:
            self.speaking, ended = False, True
        return started, ended

    def reset(self):
        self.hits = self.quiet = 0
        self.speaking = False
