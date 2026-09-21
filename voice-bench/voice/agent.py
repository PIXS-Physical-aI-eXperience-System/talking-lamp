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
import collections
import threading
import time
import uuid

import numpy as np

from . import link
from .tts import split_sentences

IDLE, LISTENING, THINKING, SPEAKING = "대기", "듣기", "생각", "말하기"

FRAME_SAMPLES = 320          # 20 ms @ 16 kHz
# 한 번에 받아쓸 최대 길이. 15초로 뒀더니 조용해지지 않는 방에서 발화가
# 끝나지 않고 계속 쌓였다. 파이 VAD 표시가 250프레임 중 150~250개에 붙는
# 환경이라 "표시가 끊기면 끝" 이라는 기준이 성립하지 않는다. 그 결과 15초를
# 채운 뒤에야 잘렸고, 그 덩어리를 STT 에 넣느라 다시 5초가 걸렸다.
#
# 램프에게 하는 말은 짧다. 상한을 낮춰 최악 지연을 묶는다.
# 근본 해결은 웨이크워드다 — "픽스야" 다음부터만 들으면 이 문제가 사라진다.
MAX_UTTERANCE_S = 6.0
END_SILENCE_FRAMES = 25      # 0.5초. 끼어든 직후 우리가 발화 끝을 볼 때 쓴다
# 파이 VAD 표시가 한 번 끊겼다고 바로 자르면 안 된다. 사람은 말하다 숨을 쉬고,
# 방에 사람이 있으면 표시가 들쭉날쭉하다. 실기기에서 250프레임 중 150~250개에
# 표시가 붙었고, 그 사이 끊김마다 잘려 문장 조각이 STT 로 갔다.
PI_END_FRAMES = 30           # 0.6초 연속으로 표시가 없어야 발화 끝으로 본다
MIN_UTTERANCE_FRAMES = 20    # 0.4초보다 짧으면 발화로 치지 않는다
PREROLL_FRAMES = 15          # 0.3초. 깨어나기 직전 소리도 함께 넘긴다
BARGE_GRACE_S = 1.5          # 그 사이에는 파이 VAD 를 믿지 않는다


def _db(x):
    return 20 * np.log10(float(np.sqrt(np.mean(np.square(x)))) + 1e-12)


class BargeInDetector:
    """재생 중 마이크 레벨이 바닥 대비 얼마나 뛰었는지로 끼어듦을 잡는다.

    고정 dB 임계값을 쓰지 않는 이유: 같은 조건에서 세 번 쟀을 때 조용할 때가
    -20 dB 였다가 -83 dB 로 나온 적이 있다. 절대값을 기준으로 삼으면 그런
    회차에 통째로 틀린다. 재생 중 바닥은 매번 새로 재면 된다.
    """

    def __init__(self, rise_db=12.0, need_frames=3, floor_frames=15,
                 hold_s=1.0):
        self.rise_db = rise_db
        self.need_frames = need_frames
        self.floor_frames = floor_frames
        # 말하기 시작 직후에는 바닥을 잡지 않는다. 소리가 아직 스피커에
        # 닿지 않았기 때문이다 — RTP 전송과 지터 버퍼, 드레인이 끼어 있다.
        # 그 무음을 바닥으로 삼으면, 램프가 실제로 말하기 시작할 때 레벨이
        # 올라가 자기 목소리를 끼어듦으로 오인한다. 실제로 그렇게 됐다.
        self.hold_s = hold_s
        self.reset()

    def reset(self):
        self._floor = []
        self._hits = 0
        self._hold_until = time.time() + self.hold_s

    def update(self, frame):
        """(끼어들었나, 현재 dB, 바닥 dB)."""
        v = _db(frame)
        if time.time() < self._hold_until:
            return False, v, None       # 소리가 닿기를 기다리는 중
        if len(self._floor) < self.floor_frames:
            # 램프가 실제로 말하고 있는 동안의 레벨로 바닥을 잡는다.
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
        # 깨어난 프레임부터 모으면 말 앞부분이 잘린다. 항상 최근 몇 프레임을
        # 들고 있다가 깨어날 때 앞에 붙인다.
        self._preroll = collections.deque(maxlen=PREROLL_FRAMES)
        self._pi_quiet = 0
        self._voiced = 0          # 실제로 소리가 난 프레임 수
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
        self._mic = []          # 말하는 동안의 (현재 dB, 바닥 dB)
        self._speak_thread = None
        self._lock = threading.Lock()

    # ── 상태 ────────────────────────────────────────────────────────────
    def _set(self, s):
        if s != self.state:
            if self.state == SPEAKING:
                self._report_speak_mic()
            if s == SPEAKING:
                self._mic = []
            self.state = s
            self.on_state(s)

    def _report_speak_mic(self):
        """말하는 동안 마이크가 어땠는지 한 줄로.

        barge-in 이 실제 장비에서 되는지 보려면 이게 있어야 한다. 안 될 때
        원인이 세 가지인데 로그가 없으면 구분이 안 된다.

          프레임 0개  — 재생 중에는 파이가 캡처를 아예 안 보낸다. 그러면
                        barge-in 은 구조상 불가능하다(파이 쪽을 고쳐야 한다).
          대비 작음   — 에코 제거가 약해서 램프 목소리가 바닥을 올린다.
                        rise_db 를 낮춰도 자기 목소리에 걸린다.
          대비 충분   — 임계값만 조정하면 된다.
        """
        mic = getattr(self, "_mic", None)
        if not mic:
            print("  말하는 동안 마이크 프레임 0개 "
                  "— 재생 중 캡처가 안 온다. barge-in 불가")
            return
        lv = [v for v, _ in mic]
        floors = [f for _, f in mic if f is not None]
        base = f"{min(floors):.1f}" if floors else "미정"
        print(f"  말하는 동안 마이크 {len(mic)}프레임  "
              f"바닥 {base} dB  최고 {max(lv):.1f} dB  "
              f"대비 {max(lv) - (min(floors) if floors else max(lv)):.1f} dB")

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
            self._preroll.append(pcm)
            # 파이가 말하는 중이라고 표시한 프레임만 웨이크워드에 넣는다.
            # 조용한 프레임까지 넣으면 헛일이고 오작동만 늘어난다.
            if active and self.wake.detect(pcm):
                self.wake.reset()
                self.speech_id = speech_id
                self._buf = list(self._preroll)
                self._preroll.clear()
                self._pi_quiet = 0
                self._voiced = 1
                self._t0 = time.time()
                self._set(LISTENING)
            return

        # LISTENING
        self._buf.append(pcm)
        if active:
            self._voiced += 1
        too_long = time.time() - self._t0 > MAX_UTTERANCE_S

        if time.time() < self._grace_until:
            loud = (self._grace_floor is None
                    or _db(pcm) - self._grace_floor >= self.barge.rise_db / 2)
            self._quiet_run = 0 if loud else self._quiet_run + 1
            ended = self._quiet_run >= END_SILENCE_FRAMES
        else:
            # 표시가 연속으로 없어야 발화 끝으로 본다. 한 프레임 끊겼다고
            # 자르면 숨 쉬는 자리마다 문장이 토막난다.
            self._pi_quiet = 0 if active else self._pi_quiet + 1
            ended = self._pi_quiet >= PI_END_FRAMES

        # 너무 짧은 것은 발화가 아니다. 기침이나 문 닫는 소리로 STT 를 돌리면
        # 빈 문자열이 나오고 그때마다 한 턴이 헛돈다.
        # 버퍼 길이로 세면 안 된다 — 뒤에 붙는 무음까지 세어 통과해 버린다.
        if ended and self._voiced < MIN_UTTERANCE_FRAMES:
            self._buf = []
            self._pi_quiet = 0
            self._voiced = 0
            self._set(IDLE)
            return

        if ended or too_long:
            audio = np.concatenate(self._buf) if self._buf else np.zeros(1, np.float32)
            self._buf = []
            dur = len(self._buf) * FRAME_SAMPLES / link.RATE
            why = "길이 상한" if too_long else "무음"
            print(f"  발화 {dur:.1f}초 모음 ({why}로 끊음)")
            # 발화가 끝났다고 판단한 시각. 여기부터 첫 소리까지가 체감 지연이다.
            threading.Thread(target=self._think_and_speak,
                             args=(audio, self.speech_id, time.time()), daemon=True).start()
            self._set(THINKING)

    def _while_speaking(self, pcm):
        hit, cur, floor = self.barge.update(pcm)
        self._mic.append((cur, floor))
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

    def _think_and_speak(self, audio, speech_id, t_end=None):
        t_end = t_end or time.time()
        text = ""
        t0 = time.time()
        try:
            text = self.stt.transcribe(audio, link.RATE)
        except Exception as e:
            print(f"  ! STT 실패: {type(e).__name__}: {e}")
        t_stt = time.time() - t0
        if not text:
            with self._lock:
                self._set(IDLE)
            return
        self.send(link.HEARD, text.encode("utf-8"))

        # on_utterance 는 문자열 하나를 돌려줘도 되고, 문장이 완성될 때마다
        # 하나씩 내보내도 된다(LLM 스트리밍). 뒤쪽이면 첫 문장이 나오는 즉시
        # 합성이 시작되므로 말을 훨씬 빨리 시작한다.
        t0 = time.time()
        try:
            reply = self.on_utterance(text)
        except Exception as e:
            print(f"  ! 응답 생성 실패: {type(e).__name__}: {e}")
            reply = None
        t_think = time.time() - t0
        if reply is None:
            with self._lock:
                self._set(IDLE)
            return
        chunks = [reply] if isinstance(reply, str) else reply

        self._stop_speaking.clear()
        self.barge.reset()
        with self._lock:
            self._set(SPEAKING)
        self.send(link.SPEAK_BEGIN, link.pack_id(speech_id))
        t0 = first = None
        # 스트리밍이면 여기서 한참 기다릴 수 있다. SPEAK_BEGIN 을 먼저 보내는
        # 것은 브리지가 송신기를 준비할 시간을 벌기 위해서다.
        spoke = False
        try:
            t0 = time.time()
            # 문장 단위로 만들어 만드는 대로 보낸다. 통째로 만들면 첫 소리까지
            # 4.56초, 최고 메모리 1750 MB 다. 쪼개면 0.59초, 1245 MB.
            for chunk in chunks:
                if self._stop_speaking.is_set():
                    break
                for part in split_sentences(chunk):
                    if self._stop_speaking.is_set():
                        break
                    wav = self.tts.synth(part)
                    if first is None:
                        first = time.time() - t0
                    if self._stop_speaking.is_set():
                        break
                    self._send_audio(wav, self.tts.samplerate)
                    spoke = True
        except Exception as e:
            print(f"  ! TTS 실패: {type(e).__name__}: {e}")
        finally:
            print(f"  지연  STT {t_stt:.2f}s + 응답생성 {t_think:.2f}s + "
                  f"TTS 첫문장 {first:.2f}s = {time.time()-t_end:.2f}s"
                  if first is not None else
                  f"  지연  STT {t_stt:.2f}s + 응답생성 {t_think:.2f}s")
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
