"""웨이크워드 — "픽스야" 를 기다린다.

왜 필요한가: STT 를 계속 돌리면 GPU 와 전력을 계속 쓰고, 지나가는 말도 전부
명령으로 해석된다. 웨이크워드는 그 앞을 막는 아주 작은 모델이다.

기성 영어 모델은 쓸 수 없다. 한국어 발화에 오작동한다 — 심사 문장 6개를
들려줬을 때 2회 잘못 깨어났다. 한국어 모델을 새로 학습해야 하고, 학습에는
팀원들 목소리 녹음이 필요하다.

그래서 지금은 모델 자리를 비워 둔다. StubWake 는 파이프라인을 끝까지 돌려볼
수 있게 하는 대역이며, 실제 호출 판정을 하지 않는다.
"""
import os

import numpy as np

PHRASE = "픽스야"


class StubWake:
    """모델이 없을 때 쓰는 대역.

    아무 말이나 하면 깨어난다(에너지 기준). 파이프라인 골격을 확인하는
    용도일 뿐이며, 이 상태로는 제품이 아니다.
    """

    name = "stub(모델 없음 — 아무 말에나 깨어난다)"
    ready = False

    def __init__(self, samplerate=16000, threshold_db=-40.0):
        self.threshold_db = threshold_db

    def detect(self, frame):
        rms = float(np.sqrt(np.mean(np.square(frame))) + 1e-12)
        return 20 * np.log10(rms) > self.threshold_db

    def reset(self):
        pass


class OpenWakeWord:
    """openWakeWord (Apache-2.0, ONNX, torch 불필요).

    사전학습된 음성 임베딩 위에 작은 분류기를 얹는 구조라 한국어 호출어를
    직접 학습시킬 수 있다. 실측 RTF 0.021.
    """

    ready = True

    def __init__(self, model_path, threshold=0.5, samplerate=16000):
        from openwakeword.model import Model
        self.m = Model(wakeword_models=[model_path], inference_framework="onnx")
        self.threshold = threshold
        self.key = os.path.splitext(os.path.basename(model_path))[0]
        self.name = f"openWakeWord({self.key}, 임계 {threshold})"

    def detect(self, frame):
        x = (np.clip(frame, -1, 1) * 32767).astype(np.int16)
        scores = self.m.predict(x)
        return max(scores.values()) >= self.threshold if scores else False

    def reset(self):
        self.m.reset()


def load_wake(model_path=None, threshold=0.5, samplerate=16000):
    """모델이 있으면 쓰고, 없으면 대역을 돌려준다. 어느 쪽인지 알려준다."""
    if model_path and os.path.exists(model_path):
        try:
            return OpenWakeWord(model_path, threshold, samplerate)
        except Exception as e:
            print(f"  ! 웨이크워드 모델을 못 열었다({type(e).__name__}). 대역으로 진행한다")
    return StubWake(samplerate)
