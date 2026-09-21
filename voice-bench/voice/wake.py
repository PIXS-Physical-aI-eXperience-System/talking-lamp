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


CHUNK = 1280        # openWakeWord 가 한 걸음에 먹는 표본 수(80 ms)


def _feature_models(model_path):
    """특징 추출 모델(멜 스펙트로그램·임베딩)을 모델 옆에서 찾는다.

    openWakeWord 는 보통 자기 패키지 안의 resources/models 에서 찾는데,
    젯슨에는 의존성 충돌을 피하려고 `--no-deps` 로 깔아서 그 파일들이
    없다. 없으면 AudioFeatures 가 뜨는 순간 FileNotFoundError 다.

    둘이 합쳐 2.4 MB뿐이라 저장소에 같이 넣고 여기서 직접 가리킨다.
    옆에 없으면 인자를 비워 패키지 기본 경로에 맡긴다.
    """
    d = os.path.dirname(os.path.abspath(model_path))
    out = {}
    for key, name in (("melspec_model_path", "melspectrogram.onnx"),
                      ("embedding_model_path", "embedding_model.onnx")):
        path = os.path.join(d, name)
        if os.path.exists(path):
            out[key] = path
    return out if len(out) == 2 else {}

THRESHOLD = 0.7     # bench/wake_eval.py 실측에서 고른 자리
NEED_FRAMES = 2     # 연속 2창


class OpenWakeWord:
    """openWakeWord (Apache-2.0, ONNX, torch 불필요).

    사전학습된 음성 임베딩 위에 작은 분류기를 얹는 구조라 한국어 호출어를
    직접 학습시킬 수 있다. 실측 RTF 0.021.

    **연속 두 창을 요구한다.** 한 창만 넘어도 깨우면 스치는 소리에 깨어난다.
    학습에 없던 목소리로 잰 실측(bench/wake_eval.py):

        연속 1창, 임계 0.5 — 깨어남 99%, 평범한 문장에 헛깨움 23%
        연속 2창, 임계 0.7 — 깨어남 96%, 평범한 문장에 헛깨움 13%

    **80 ms 단위로 모은 뒤에 넣는다.** openWakeWord 는 1280 표본이 차기
    전에는 새로 계산하지 않고 직전 점수를 그대로 돌려준다. 20 ms 프레임을
    그냥 넣으면 같은 점수가 네 번 나와서 "연속 2창"이 저절로 충족된다 —
    연속을 요구한 의미가 사라진다.
    """

    ready = True

    def __init__(self, model_path, threshold=THRESHOLD,
                 need_frames=NEED_FRAMES, samplerate=16000, on_score=None):
        from openwakeword.model import Model
        self.m = Model(wakeword_models=[model_path], inference_framework="onnx",
                       **_feature_models(model_path))
        self.threshold = threshold
        self.need = need_frames
        self.key = os.path.splitext(os.path.basename(model_path))[0]
        self.name = (f"openWakeWord({self.key}, 임계 {threshold}, "
                     f"연속 {need_frames}창)")
        self._buf = np.empty(0, dtype=np.int16)
        self._run = 0
        # 80 ms 마다 나오는 점수를 그대로 흘려보낸다. 실제 마이크로 잴 때
        # 점수를 전부 남겨 두면 임계·연속 창을 바꿔 가며 다시 부르지 않아도
        # 된다(bench/wake_field.py).
        self.on_score = on_score

    def detect(self, frame):
        x = (np.clip(frame, -1, 1) * 32767).astype(np.int16)
        self._buf = np.concatenate([self._buf, x])
        hit = False
        while len(self._buf) >= CHUNK:
            chunk, self._buf = self._buf[:CHUNK], self._buf[CHUNK:]
            scores = self.m.predict(chunk)
            s = max(scores.values()) if scores else 0.0
            if self.on_score:
                self.on_score(s)
            self._run = self._run + 1 if s >= self.threshold else 0
            if self._run >= self.need:
                hit = True
        return hit

    def reset(self):
        self.m.reset()
        self._buf = np.empty(0, dtype=np.int16)
        self._run = 0


def load_wake(model_path=None, threshold=THRESHOLD,
              need_frames=NEED_FRAMES, samplerate=16000, on_score=None):
    """모델이 있으면 쓰고, 없으면 대역을 돌려준다. 어느 쪽인지 알려준다."""
    if model_path and os.path.exists(model_path):
        try:
            return OpenWakeWord(model_path, threshold, need_frames,
                                samplerate, on_score)
        except Exception as e:
            print(f"  ! 웨이크워드 모델을 못 열었다({type(e).__name__}). 대역으로 진행한다")
    return StubWake(samplerate)
