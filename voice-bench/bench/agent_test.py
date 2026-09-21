"""마이크도 ROS 도 없이 판단부를 시험한다.

    venvs/melo-onnx/bin/python bench/agent_test.py

가짜 프레임을 밀어 넣어 상태가 제대로 옮겨 가는지, barge-in 이 잡히는지 본다.
실기기에서만 확인할 수 있으면 고칠 때마다 파이·젯슨을 거쳐야 하고, 그러면
한 번 고치는 데 몇 분씩 든다. 여기서 먼저 깨지게 한다.
"""
import os
import sys
import time
import uuid

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from voice import link                       # noqa: E402
from voice.agent import (FRAME_SAMPLES, IDLE, LISTENING, SPEAKING,  # noqa: E402
                         THINKING, VoiceAgent)

SID = str(uuid.uuid4())


def frame(db=-20.0):
    """주어진 레벨의 잡음 한 프레임."""
    amp = 10 ** (db / 20)
    return (amp * np.random.randn(FRAME_SAMPLES)).astype(np.float32)


class FakeStt:
    def transcribe(self, audio, sr):
        return "왼쪽 좀 더 밝게 비춰줘"


class FakeTts:
    samplerate = 16000

    def synth(self, text):
        return np.zeros(int(0.5 * self.samplerate), dtype=np.float32)


class AlwaysWake:
    def detect(self, pcm):
        return True

    def reset(self):
        pass


class NeverWake(AlwaysWake):
    def detect(self, pcm):
        return False


def run(name, wake, script, rise_db=12.0):
    """script: (speech_id, 레벨dB, 반복) 목록. 보낸 메시지와 상태를 돌려준다."""
    sent, states = [], []
    agent = VoiceAgent(FakeStt(), FakeTts(), wake,
                       on_utterance=lambda t: "네, 밝게 할게요.",
                       send=lambda k, p: sent.append((k, p)),
                       on_state=states.append, rise_db=rise_db)
    for sid, db, n in script:
        for _ in range(n):
            agent.on_capture(sid, frame(db))
            time.sleep(0.001)
    return agent, sent, states


def main() -> int:
    fails = []

    # ① 조용하면 깨지 않는다 ─────────────────────────────────────────
    a, sent, states = run("조용", AlwaysWake(), [("", -60, 20)])
    print(f"  ① 발화 표시 없음     상태 {a.state}   보낸 것 {len(sent)}개")
    if a.state != IDLE:
        fails.append("speech_id 없이 깨어났다")

    # ② 웨이크워드가 아니면 깨지 않는다 ───────────────────────────────
    a, sent, states = run("비호출", NeverWake(), [(SID, -20, 20)])
    print(f"  ② 웨이크워드 불일치   상태 {a.state}")
    if a.state != IDLE:
        fails.append("웨이크워드 없이 깨어났다")

    # ③ 부름 → 듣기 → 발화 끝 → 말하기 ───────────────────────────────
    #    발화 1초(50프레임), 그다음 무음 0.7초(35프레임). 표시가 연속으로
    #    30프레임 없어야 끝으로 본다.
    a, sent, states = run("정상", AlwaysWake(),
                          [(SID, -20, 50), ("", -60, 35)])
    for _ in range(60):          # 생각·합성은 다른 스레드에서 돈다
        if any(k == link.SPEAK_END for k, _ in sent):
            break
        time.sleep(0.05)
    kinds = [k for k, _ in sent]
    heard = [p.decode() for k, p in sent if k == link.HEARD]
    audio = [p for k, p in sent if k == link.SPEAK_AUDIO]
    print(f"  ③ 정상 한 턴         상태 흐름 {' → '.join(states)}")
    print(f"     받아쓴 말 {heard}   오디오 프레임 {len(audio)}개")
    if link.SPEAK_BEGIN not in kinds or link.SPEAK_END not in kinds:
        fails.append("말하기 시작/끝이 안 나왔다")
    if states[:3] != [LISTENING, THINKING, SPEAKING]:
        fails.append(f"상태 흐름이 다르다: {states}")
    if any(len(p) != FRAME_SAMPLES * 2 for p in audio):
        fails.append("오디오 프레임 크기가 640바이트가 아니다")

    # ③-b 숨 쉬는 자리에서 자르지 않는다 ─────────────────────────────
    #    파이 표시는 들쭉날쭉하다. 한 프레임 끊겼다고 자르면 문장이 토막난다.
    a, sent, states = run("숨", AlwaysWake(),
                          [(SID, -20, 30), ("", -30, 10),      # 숨 쉬는 자리
                           (SID, -20, 30), ("", -60, 35)])
    for _ in range(60):
        if any(k == link.SPEAK_END for k, _ in sent):
            break
        time.sleep(0.05)
    n_turns = sum(1 for k, _ in sent if k == link.SPEAK_BEGIN)
    print(f"  ③-b 중간에 끊긴 발화   턴 {n_turns}개 (1개여야 한다)")
    if n_turns != 1:
        fails.append(f"한 문장을 {n_turns}턴으로 쪼갰다")

    # ③-c 짧은 소음은 턴을 소모하지 않는다 ───────────────────────────
    a, sent, states = run("소음", AlwaysWake(),
                          [(SID, -20, 5), ("", -60, 35)])
    time.sleep(0.3)
    print(f"  ③-c 짧은 소음         상태 {a.state}   보낸 것 {len(sent)}개")
    if sent:
        fails.append("0.1초짜리 소음으로 한 턴을 돌렸다")

    # ③-d 스트리밍 응답도 받는다 ─────────────────────────────────────
    #    LLM 이 문장을 하나씩 내보내면 그때그때 합성해야 한다. 다 모아서
    #    합성하면 스트리밍의 이점이 사라진다.
    sent3, times = [], []

    def streamed(text):
        for part in ("네, 알겠습니다.", "왼쪽을 밝게 할게요."):
            times.append(time.time())
            yield part

    agent = VoiceAgent(FakeStt(), FakeTts(), AlwaysWake(),
                       on_utterance=streamed,
                       send=lambda k, p: sent3.append((k, p, time.time())))
    agent.state = LISTENING
    agent._voiced = 100
    agent._buf = [frame(-20)] * 100
    agent._while_listening("", frame(-60))          # 발화 끝 -> 생각 -> 말하기
    for _ in range(80):
        if any(k == link.SPEAK_END for k, _, _ in sent3):
            break
        time.sleep(0.05)
    audio_t = [t for k, _, t in sent3 if k == link.SPEAK_AUDIO]
    print(f"  ③-d 스트리밍 응답     조각 {len(times)}개, 오디오 프레임 {len(audio_t)}개")
    if len(times) == 2 and audio_t:
        # 두 번째 문장을 내놓기 전에 첫 문장 오디오가 나갔어야 한다
        before = sum(1 for t in audio_t if t < times[1])
        print(f"     두 번째 문장 전에 나간 오디오 {before}개")
        if before == 0:
            fails.append("첫 문장을 합성하지 않고 다음 문장을 기다렸다")
    else:
        fails.append("스트리밍 응답을 처리하지 못했다")

    # ④ 말하는 중에 끼어들기 ─────────────────────────────────────────
    sent2, states2 = [], []
    agent = VoiceAgent(FakeStt(), FakeTts(), AlwaysWake(),
                       on_utterance=lambda t: "네",
                       send=lambda k, p: sent2.append((k, p)),
                       on_state=states2.append)
    agent.state = SPEAKING

    # ④-0 말하기 시작 직후에는 끼어듦으로 보지 않는다 ─────────────────
    #    소리가 아직 스피커에 닿지 않았다(RTP·지터 버퍼·드레인). 그 무음을
    #    바닥으로 삼으면 램프가 말하기 시작할 때 자기 목소리에 끊는다.
    #    실기기에서 실제로 0.66초 만에 스스로 끊었다.
    agent.barge.reset()                      # hold_s 만큼 판정하지 않는다
    for _ in range(30):                      # 소리가 닿기 전 구간 — 조용하다
        agent.on_capture("", frame(-55))
    for _ in range(10):                      # 램프가 말하기 시작 — 레벨이 오른다
        agent.on_capture("", frame(-30))
    early = [k for k, _ in sent2]
    print(f"  ④-0 시작 직후        보낸 것 {[k.decode() for k in early] or '없음'}")
    if link.BARGE_IN in early:
        fails.append("말하기 시작 직후 자기 목소리를 끼어듦으로 오인했다")

    # ④-0b 램프 자기 목소리(바닥 대비 15 dB)에는 끊지 않는다 ───────────
    #    실기기에서 에코 제거를 통과하고 남은 자기 목소리가 바닥 대비
    #    15.1 dB 까지 올라갔다. 임계가 12 였을 때 램프가 자기 말을 끊었다
    #    (파이가 code=cancelled 를 돌려줬다). 여기서 막는다.
    sent_self = []
    agent_self = VoiceAgent(FakeStt(), FakeTts(), AlwaysWake(),
                            on_utterance=lambda t: "네",
                            send=lambda k, p: sent_self.append((k, p)),
                            on_state=lambda s: None)
    agent_self.state = SPEAKING
    agent_self.barge.hold_s = 0.0
    agent_self.barge.reset()
    for _ in range(20):                      # 바닥 — 램프가 조용히 말하는 중
        agent_self.on_capture("", frame(-55))
    for _ in range(40):                      # 자기 목소리가 15 dB 올라간다
        agent_self.on_capture("", frame(-40))
    self_fired = [k.decode() for k, _ in sent_self]
    print(f"  ④-0b 자기 목소리 15dB  보낸 것 {self_fired or '없음'}   "
          f"(임계 {agent_self.barge.rise_db:.0f}dB)")
    if link.BARGE_IN in [k for k, _ in sent_self]:
        fails.append("램프 자기 목소리(15dB)를 끼어듦으로 오인했다 — "
                     "rise_db 가 너무 낮다")

    # 이제 대기가 끝난 상태로 만들어 바닥을 잡게 한다
    agent.barge.hold_s = 0.0
    agent.barge.reset()
    for _ in range(20):                      # 바닥을 잡는 구간
        agent.on_capture("", frame(-55))
    before = [k for k, _ in sent2]
    for _ in range(5):                       # 갑자기 크게 — 끼어듦
        # 실측에서 사람이 끼어들면 바닥 대비 28 dB 튀었다. 바닥이 -55 이므로
        # -27 이 그 값이다. 시험을 실제 값에 맞춰 둬야 임계를 올렸을 때
        # 여기서 잡힌다.
        agent.on_capture("", frame(-27))
    after = [k for k, _ in sent2]
    print(f"  ④ barge-in           바닥 구간 {len(before)}건 → 이후 "
          f"{[k.decode() for k in after]}   상태 {agent.state}")
    if link.BARGE_IN not in after:
        fails.append("끼어들었는데 BARGE_IN 이 안 나갔다")
    if LISTENING not in states2:
        fails.append(f"끼어든 뒤 듣기로 넘어가지 않았다: {states2}")
    if link.BARGE_IN in before:
        fails.append("바닥을 잡는 중에 끼어듦으로 오판했다")

    # ⑤ 끼어든 말을 첫 프레임에서 자르지 않는다 ───────────────────────
    #    파이 VAD 는 재생 직후 0.3초 더 꺼져 있어 speech_id 가 비어 온다.
    #    그걸 발화 끝으로 읽으면 끼어든 사람의 말이 통째로 날아간다.
    n_before = len(agent._buf)
    for _ in range(20):                      # speech_id 는 비었지만 계속 말하는 중
        agent.on_capture("", frame(-20))
    print(f"  ⑤ 끼어든 뒤 계속 말함  모은 프레임 {n_before} → {len(agent._buf)}   "
          f"상태 {agent.state}")
    if agent.state != LISTENING or len(agent._buf) <= n_before:
        fails.append("끼어든 직후 speech_id 가 비었다고 발화를 잘라버렸다")

    # ⑥ 깨우면 "네" 하고 대답한다 ────────────────────────────────────
    #    파이는 재생 중과 그 뒤 0.3초 동안 VAD 를 끈다. 그 구간의 speech_id
    #    는 비어서 오는데 그것을 "말이 끝났다" 로 읽으면 대답하자마자 턴이
    #    끝난다. 이 시험이 그것만 본다.
    sent6, states6 = [], []
    ack = np.zeros(int(16000 * 0.4), dtype=np.float32)
    agent6 = VoiceAgent(FakeStt(), FakeTts(), AlwaysWake(),
                        on_utterance=lambda t: "네",
                        send=lambda k, p: sent6.append((k, p)),
                        on_state=states6.append, ack_wav=ack)
    agent6.on_capture("sid-6", frame(-20))           # 깨어남
    time.sleep(0.3)                                   # 대답 스레드가 나가도록
    kinds = [k for k, _ in sent6]
    print(f"  ⑥ 깨우면 대답        보낸 것 "
          f"{sorted(set(k.decode() for k in kinds))}   상태 {agent6.state}")
    if link.SPEAK_AUDIO not in kinds:
        fails.append("깨웠는데 대답 오디오가 안 나갔다")

    for _ in range(40):                               # 0.8초 — 기본 0.6초보다 길다
        agent6.on_capture("", frame(-50))
    print(f"  ⑥-b 대답 중 표시 없음  상태 {agent6.state} (듣기여야 한다)")
    if agent6.state != LISTENING:
        fails.append(f"대답하는 동안 턴이 끝났다: {agent6.state}")

    before6 = len(agent6._buf)
    for _ in range(20):
        agent6.on_capture("sid-6", frame(-20))
    print(f"  ⑥-c 대답 뒤 사람 말함  모은 프레임 {before6} → {len(agent6._buf)}")
    if len(agent6._buf) <= before6:
        fails.append("대답 뒤에 들어온 말을 안 모았다")

    print()
    if fails:
        print("  ✗ 실패:", ", ".join(fails))
        return 1
    print("  ✔ 판단부 이상 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
