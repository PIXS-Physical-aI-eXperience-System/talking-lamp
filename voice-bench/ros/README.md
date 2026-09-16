# 음성 어댑터 — 젯슨에서 띄우기

파트 C(음성) / 최승원

저쪽(E·B)이 만든 ROS 인터페이스에 우리 STT·TTS·웨이크워드를 붙이는 부분이다.
계약은 `docs/jetson-integration.md` 를 따른다.

```
[ROS 노드 — 시스템 파이썬 3.12, rclpy]     [판단부 — venvs/melo-onnx]
 sub /lamp/audio/capture ───────────────▶  웨이크워드 · STT · TTS
 pub /lamp/audio/playback_frames ◀──────
 act /lamp/play_audio                      localhost:5150
```

## 왜 프로세스를 나누는가

젯슨의 onnxruntime 과 ctranslate2 는 **직접 빌드한 휠**이다. PyPI 에 sm_87
커널이 있는 onnxruntime 도, CUDA 가 들어간 ctranslate2 도 없어서 각각
`build-onnxruntime/`, `build-ctranslate2/` 에서 만들었다. 그게 이 환경에서
가장 깨지기 쉬운 부분이다.

ROS 노드를 같은 인터프리터에 넣으면 rosdep 이나 ROS 패키지가 의존성을
건드리다 그 휠을 덮을 수 있다. 실제로 openwakeword 를 그냥 깔았다가 같은
일이 날 뻔했다(`--no-deps` 로 막았다). 그래서 아예 다른 파이썬에 둔다.

노드는 rclpy 말고 아무것도 필요 없다.

## 띄우는 순서

**① 판단부** (venv, 모델을 올린다. 40초쯤 걸린다)

```bash
cd ~/talking-lamp/voice-bench
venvs/melo-onnx/bin/python bench/voice_agent.py
```

**② ROS 노드** (시스템 파이썬)

```bash
source /opt/ros/jazzy/setup.bash
source ~/talking-lamp-integration/jetson_ws/install/setup.bash
cd ~/talking-lamp/voice-bench
python3 ros/lamp_voice_node.py --agent 127.0.0.1:5150
```

순서가 반대여도 된다. 노드는 판단부에 못 붙으면 2초마다 다시 시도한다.

## 오디오 형식

계약상 이것만 받는다. 어기면 GStreamer 에 닿기 전에 거부된다.

| | |
| --- | --- |
| 표본율 | 16000 |
| 채널 | 1 |
| 인코딩 | `pcm_s16le` |
| 프레임 | **640바이트 고정** (20 ms) |
| 순번 | 0 에서 시작, 1씩 증가 |
| 끝 | 데이터가 빈 프레임 하나 (EOS) |

EOS 를 빠뜨리면 파이가 드레인을 끝내지 못해 `PlayAudio` 가 완료되지 않는다.

## 발화 구간을 누가 자르는가

상태마다 다르다.

- **대기·듣기 중** — 파이가 자른다. 프레임에 붙어 오는 `speech_id` 가
  XVF3800 VAD 의 판정이다.
- **말하기 중** — 우리가 자른다. 파이는 자기 목소리에 반응하지 않으려고
  재생 중과 드레인 후 0.3초 동안 VAD 를 꺼둔다(self-playback guard).
  그래서 barge-in 은 우리가 직접 찾아야 한다.

끼어든 **직후 1.5초** 도 우리가 본다. 그 사이에는 파이 VAD 가 아직 꺼져 있어
`speech_id` 가 비어 오는데, 그것을 발화 끝으로 읽으면 끼어든 사람의 말이 첫
프레임에서 잘린다. 실제로 그렇게 만들었다가 시험에서 잡았다(`bench/agent_test.py` ⑤).

## barge-in 임계값

고정 dB 값을 쓰지 않는다. 같은 조건에서 세 번 쟀을 때 조용할 때가 −20 dB
였다가 −83 dB 로 나온 적이 있어, 절대값 기준은 그 회차에 통째로 틀린다.

재생이 시작되면 바닥을 새로 잡고 **거기서 12 dB 오르면** 끼어든 것으로 본다.
실측에서 끼어들면 최악 **28 dB** 튀므로 여유가 있다
([results/aec-2026-09-16.md](../results/aec-2026-09-16.md)).

`--rise-db` 로 조절한다.

## 아직 안 된 것

- **웨이크워드 모델이 없다.** "픽스야" 한국어 모델을 학습해야 하고 팀원 녹음을
  기다리는 중이다. 지금은 아무 말에나 깨어나며 그 상태는 제품이 아니다.
  띄울 때 그렇다고 찍는다.
- **`lamp_voice_node.py` 는 실기기에서 돌려보지 않았다.** rclpy 가 없는 곳에서는
  검증할 수 없어 문법과 메시지 계약만 맞춰 둔 상태다. 판단부 쪽은
  `bench/agent_test.py` 로 장치 없이 시험한다.
- **`TurnOrchestrator` 를 아직 쓰지 않는다.** 지금은 오디오만 주고받는다.
  정렬 → 오디오·모션 동시 → 중앙 복귀 → idle 순서를 지키려면 `TurnResponse`
  에 `motion_name` 을 실어 넘겨야 하고, 그 표정 선택은 A 와 정해야 한다.
- barge-in 감지 지연은 이 경로에서 다시 재야 한다. 파이에서 직접 잰 값에는
  Opus 코덱과 지터 버퍼가 빠져 있다.
