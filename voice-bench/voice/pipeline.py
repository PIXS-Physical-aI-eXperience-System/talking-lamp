"""음성 파이프라인 — 부르면 듣고, 답을 받아 말하고, 끊으면 멈춘다.

    대기 ──웨이크워드──▶ 듣기 ──발화 끝──▶ 생각 ──▶ 말하기 ──▶ 대기
                         ▲                              │
                         └────────── barge-in ──────────┘

반이중이다. 램프가 말하는 동안에는 STT 를 돌리지 않고 VAD 만 돌린다. 둘을
동시에 돌릴 이유가 없고(사용자가 끼어들었는지만 알면 된다), 그동안 GPU 를
STT 에 쓰지 않아 메모리 최고점이 낮아진다.

유일하게 겹치는 구간이 barge-in 이다. Jetson 실측에서 STT(CUDA)와 TTS(CUDA)를
동시에 돌려도 충돌하지 않았고, 그때가 메모리 최고점(1.55 GB)이다.

'생각' 은 우리 파트가 아니다. on_utterance 콜백으로 밖에 맡긴다 — VLM 은 A 다.
"""
import threading
import time

import numpy as np

from .audio import Mic, Player
from .vad import SpeechGate, load_vad
from .wake import PHRASE, load_wake

IDLE, LISTENING, THINKING, SPEAKING = "대기", "듣기", "생각", "말하기"


class VoicePipeline:
    """B 의 런타임에서 쓰는 진입점.

    on_utterance(text) -> 답할 문장. 느리면 그만큼 응답이 늦어진다.
    on_state(state)    -> 상태가 바뀔 때. 램프 표정·불빛을 붙일 자리다.
    """

    def __init__(self, stt, tts, wake_model=None, on_utterance=None,
                 on_state=None, max_utterance_s=15.0,
                 bargein_guard_s=0.4, wake_threshold=0.5):
        self.stt = stt
        self.tts = tts
        self.on_utterance = on_utterance or (lambda t: f"{t}, 라고 하셨네요.")
        self.on_state = on_state or (lambda s: None)
        self.max_utterance_s = max_utterance_s
        # 재생 직후 잠깐은 barge-in 을 보지 않는다. 하드웨어 AEC 가 자기 목소리를
        # 지우기까지 시간이 걸려서, 이 유예가 없으면 램프가 제 말에 깨어난다.
        self.bargein_guard_s = bargein_guard_s
        self._wake_model = wake_model
        self._wake_threshold = wake_threshold

        self.state = IDLE
        self.player = Player()
        self.stop_speaking = threading.Event()
        self._running = threading.Event()
        self._thread = None
        self._speak_thread = None

    # ── 상태 ────────────────────────────────────────────────────────────
    def _set(self, s):
        if s != self.state:
            self.state = s
            self.on_state(s)

    # ── 외부에서 부르는 것 ──────────────────────────────────────────────
    def say(self, text):
        """밖에서 말을 시킨다. 말하는 중이면 끊고 새로 말한다."""
        self.interrupt()
        self._start_speaking(text)

    def interrupt(self):
        """즉시 정지. 재생 큐를 비우고 남은 조각 합성도 멈춘다."""
        self.stop_speaking.set()
        self.player.stop()
        if self._speak_thread and self._speak_thread.is_alive():
            self._speak_thread.join(timeout=1.0)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        self.interrupt()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ── 내부 ────────────────────────────────────────────────────────────
    def _start_speaking(self, text):
        self.stop_speaking.clear()
        self._speak_started = None

        def run():
            self.tts.speak(text, self.player, self.stop_speaking,
                           on_first=lambda: setattr(self, "_speak_started", time.time()))

        self._speak_thread = threading.Thread(target=run, daemon=True)
        self._speak_thread.start()
        self._set(SPEAKING)

    def _loop(self):
        self.player.open()
        wake = load_wake(self._wake_model, self._wake_threshold)
        vad = load_vad()
        print(f"  웨이크워드: {wake.name}")
        print(f"  VAD: {vad.name}")
        if not getattr(wake, "ready", False):
            print(f'  ! "{PHRASE}" 모델이 없다. 지금은 아무 말에나 깨어난다 — 제품이 아니다.')

        with Mic() as mic:
            gate = SpeechGate(vad, frame_s=mic.blocksize / mic.samplerate)
            buf, t_start = [], 0.0
            for frame in mic.frames():
                if not self._running.is_set():
                    break

                if self.state == IDLE:
                    if wake.detect(frame):
                        wake.reset()
                        gate.reset()
                        mic.drain()
                        buf, t_start = [], time.time()
                        self._set(LISTENING)

                elif self.state == LISTENING:
                    buf.append(frame)
                    _, ended = gate.update(frame)
                    too_long = time.time() - t_start > self.max_utterance_s
                    if ended or too_long:
                        audio = np.concatenate(buf) if buf else np.zeros(1, np.float32)
                        self._set(THINKING)
                        text = self.stt.transcribe(audio, mic.samplerate)
                        if not text:
                            self._set(IDLE)
                            continue
                        reply = self.on_utterance(text)
                        if reply:
                            self._start_speaking(reply)
                        else:
                            self._set(IDLE)

                elif self.state == SPEAKING:
                    started = getattr(self, "_speak_started", None)
                    speaking_done = (not self.player.is_playing()
                                     and not (self._speak_thread and self._speak_thread.is_alive()))
                    if speaking_done:
                        gate.reset()
                        mic.drain()
                        self._set(IDLE)
                        continue
                    # barge-in: 유예가 지난 뒤부터 본다
                    if started and time.time() - started > self.bargein_guard_s:
                        begun, _ = gate.update(frame)
                        if begun:
                            self.interrupt()
                            gate.reset()
                            buf, t_start = [frame], time.time()
                            self._set(LISTENING)
