"""음성 판단부 — 마이크 프레임을 받아 언제 듣고 언제 말할지 정한다.

ROS 노드가 20ms 프레임을 밀어 넣고, 이쪽은 말할 내용을 오디오로 돌려준다.
장치를 직접 열지 않으므로 마이크 없이도 시험할 수 있다(bench/agent_test.py).

    대기 ──"픽스야"──▶ 듣기 ──발화 끝──▶ 생각 ──▶ 말하기 ──▶ 대기
                       ▲                            │
                       └────────── barge-in ────────┘

발화 구간을 누가 자르는가:

  대기·듣기 중  파이가 자른다. 마이크 프레임에 speech_id 가 붙어 오는데,
                그것이 XVF3800 VAD 가 "말하는 중" 이라고 판정한 구간이다.
  말하기 중     우리가 자른다. 파이는 재생 중 자기 목소리에 반응하지 않으려고
                VAD 를 꺼두기 때문이다(self-playback guard, 드레인 후 0.3초).
                그래서 barge-in 은 우리가 직접 찾아야 한다.

barge-in 감지가 가능하다는 것은 실측으로 확인했다. 램프가 말하는 동안 사람이
끼어들면 마이크 레벨이 최악 28 dB 튄다(results/aec-2026-09-16.md).
다만 절대 dB 임계값은 쓰지 않는다 — 회차마다 크게 움직였다. 재생 중 관측된
바닥 대비 얼마나 올랐는지로 판정한다.
"""
import threading
import time
import uuid

import numpy as np

from . import link
from .tts import split_sentences

IDLE, LISTENING, THINKING, SPEAKING = "대기", "듣기", "생각", "말하기"

FRAME_SAMPLES = 320          # 20 ms @ 16 kHz
MAX_UTTERANCE_S = 15.0
END_SILENCE_FRAMES = 25      # 0.5초. 끼어든 직후 우리가 발화 끝을 볼 때 쓴다
BARGE_GRACE_S = 1.5          # 그 사이에는 파이 VAD 를 믿지 않는다


def _db(x):
    return 20 * np.log10(float(np.sqrt(np.mean(np.square(x)))) + 1e-12)


class BargeInDetector:
    """재생 중 마이크 레벨이 바닥 대비 얼마나 뛰었는지로 끼어듦을 잡는다.

    고정 dB 임계값을 쓰지 않는 이유: 같은 조건에서 세 번 쟀을 때 조용할 때가
    -20 dB 였다가 -83 dB 로 나온 적이 있다. 절대값을 기준으로 삼으면 그런
    회차에 통째로 틀린다. 재생 중 바닥은 매번 새로 재면 된다.
    """

    def __init__(self, rise_db=12.0, need_frames=3, floor_frames=10):
        self.rise_db = rise_db
        self.need_frames = need_frames
        self.floor_frames = floor_frames
        self.reset()

    def reset(self):
        self._floor = []
        self._hits = 0

    def update(self, frame):
        """(끼어들었나, 현재 dB, 바닥 dB)."""
        v = _db(frame)
        if len(self._floor) < self.floor_frames:
            # 재생 시작 직후 몇 프레임으로 바닥을 잡는다. 이 구간은 판정하지
            # 않는다 — 하드웨어 AEC 가 자리를 잡기 전이다.
            self._floor.append(v)
            return False, v, None
        floor = float(np.median(self._floor))
        if v - floor >= self.rise_db:
            self._hits += 1
        else:
            self._hits = 0
        return self._hits >= self.need_frames, v, floor


class VoiceAgent:
    def __init__(self, stt, tts, wake, on_utterance, send, on_state=None,
                 rise_db=12.0):
        self.stt = stt
        self.tts = tts
        self.wake = wake
        self.on_utterance = on_utterance
        self.send = send                      # send(kind, payload)
        self.on_state = on_state or (lambda s: None)
        self.barge = BargeInDetector(rise_db=rise_db)

        self.state = IDLE
        self.speech_id = ""
        self._buf = []
        self._t0 = 0.0
        # 끼어든 직후에는 파이 VAD 를 믿을 수 없다. self-playback guard 로
        # 재생 중과 드레인 후 0.3초 동안 꺼져 있어서, 사용자가 아직 말하고
        # 있는데도 speech_id 가 비어 온다. 그것을 발화 끝으로 읽으면 끼어든
        # 사람의 말이 첫 프레임에서 잘린다. 그 사이에는 우리가 직접 본다.
        self._grace_until = 0.0
        self._grace_floor = None
        self._quiet_run = 0
        self._stop_speaking = threading.Event()
        self._speak_thread = None
        self._lock = threading.Lock()

    # ── 상태 ────────────────────────────────────────────────────────────
    def _set(self, s):
        if s != self.state:
            self.state = s
            self.on_state(s)

    # ── ROS 노드가 부르는 것 ────────────────────────────────────────────
    def on_capture(self, speech_id, pcm):
        """마이크 프레임 하나(20 ms). pcm 은 float32."""
        with self._lock:
            if self.state == SPEAKING:
                self._while_speaking(pcm)
            elif self.state in (IDLE, LISTENING):
                self._while_listening(speech_id, pcm)

    def on_orientation(self, speech_id, state):
        """파이의 방향 정렬 상태. 지금은 기록만 한다 —
        정렬 완료를 기다리는 것은 TurnOrchestrator 가 한다."""
        pass

    # ── 내부 ────────────────────────────────────────────────────────────
    def _while_listening(self, speech_id, pcm):
        active = bool(speech_id)
        if self.state == IDLE:
            # 파이가 말하는 중이라고 표시한 프레임만 웨이크워드에 넣는다.
            # 조용한 프레임까지 넣으면 헛일이고 오작동만 늘어난다.
            if active and self.wake.detect(pcm):
                self.wake.reset()
                self.speech_id = speech_id
                self._buf = [pcm]
                self._t0 = time.time()
                self._set(LISTENING)
            return

        # LISTENING
        self._buf.append(pcm)
        too_long = time.time() - self._t0 > MAX_UTTERANCE_S

        if time.time() < self._grace_until:
            loud = (self._grace_floor is None
                    or _db(pcm) - self._grace_floor >= self.barge.rise_db / 2)
            self._quiet_run = 0 if loud else self._quiet_run + 1
            ended = self._quiet_run >= END_SILENCE_FRAMES
        else:
            # speech_id 가 비면 파이 VAD 가 발화 끝으로 본 것이다.
            ended = not active

        if ended or too_long:
            audio = np.concatenate(self._buf) if self._buf else np.zeros(1, np.float32)
            self._buf = []
            threading.Thread(target=self._think_and_speak,
                             args=(audio, self.speech_id), daemon=True).start()
            self._set(THINKING)

    def _while_speaking(self, pcm):
        hit, cur, floor = self.barge.update(pcm)
        if hit:
            print(f"  barge-in: {cur:.1f} dB (바닥 {floor:.1f})")
            self.send(link.BARGE_IN, b"")
            self._stop_speaking.set()
            self.barge.reset()
            # 끼어든 말부터 다시 듣는다. 파이 VAD 는 재생 직후 0.3초 더
            # 꺼져 있으므로 speech_id 를 기다리지 않고 바로 모은다.
            self.speech_id = str(uuid.uuid4())
            self._buf = [pcm]
            self._t0 = time.time()
            self._grace_until = time.time() + BARGE_GRACE_S
            self._grace_floor = floor
            self._quiet_run = 0
            self._set(LISTENING)

    def _think_and_speak(self, audio, speech_id):
        text = ""
        try:
            text = self.stt.transcribe(audio, link.RATE)
        except Exception as e:
            print(f"  ! STT 실패: {type(e).__name__}: {e}")
        if not text:
            with self._lock:
                self._set(IDLE)
            return
        self.send(link.HEARD, text.encode("utf-8"))

        try:
            reply = self.on_utterance(text)
        except Exception as e:
            print(f"  ! 응답 생성 실패: {type(e).__name__}: {e}")
            reply = None
        if not reply:
            with self._lock:
                self._set(IDLE)
            return

        self._stop_speaking.clear()
        self.barge.reset()
        with self._lock:
            self._set(SPEAKING)
        self.send(link.SPEAK_BEGIN, link.pack_id(speech_id))
        try:
            # 문장 단위로 만들어 만드는 대로 보낸다. 통째로 만들면 첫 소리까지
            # 4.56초, 최고 메모리 1750 MB 다. 쪼개면 0.59초, 1245 MB.
            for part in split_sentences(reply):
                if self._stop_speaking.is_set():
                    break
                wav = self.tts.synth(part)
                if self._stop_speaking.is_set():
                    break
                self._send_audio(wav, self.tts.samplerate)
        except Exception as e:
            print(f"  ! TTS 실패: {type(e).__name__}: {e}")
        finally:
            self.send(link.SPEAK_END, b"")
            with self._lock:
                if self.state == SPEAKING:
                    self._set(IDLE)

    def _send_audio(self, wav, src_rate):
        """20 ms 프레임으로 잘라 보낸다. ROS AudioFrame 이 640바이트 고정이다."""
        if src_rate != link.RATE:
            from .audio import resample
            wav = resample(wav, src_rate, link.RATE)
        pcm = link.pcm_from(wav)
        step = FRAME_SAMPLES * 2
        for i in range(0, len(pcm), step):
            if self._stop_speaking.is_set():
                return
            chunk = pcm[i:i + step]
            if len(chunk) < step:
                chunk = chunk + b"\x00" * (step - len(chunk))   # 마지막 프레임 채우기
            self.send(link.SPEAK_AUDIO, chunk)
