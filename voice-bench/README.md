# 음성 (파트 C)

최승원 · 마지막 갱신 2026-09-21
관련 문서: [진행-순서.md](../docs/진행-순서.md)

램프를 부르면 듣고, 받아쓰고, 답을 만들어 말한다. 말하는 도중에 끼어들면
멈춘다.

```
  라즈베리파이                     젯슨 Orin Nano
  ─────────────                   ──────────────
  XVF3800 마이크 ──┐               ┌─ 웨이크워드  "픽스야"
  스피커        ──┤   랜 (ROS 2)   ├─ STT        faster-whisper small
  방향·LED·모션 ──┘               ├─ LLM        (붙일 자리만 있음)
                                  └─ TTS        MeloTTS ONNX
```

젯슨 쪽은 프로세스 둘로 나뉜다. **ROS 노드**는 rclpy 말고 아무것도 안 쓰고,
**판단부**는 모델을 들고 있다. 직접 빌드한 onnxruntime·ctranslate2 휠이 ROS
의존성과 부딪히는 것이 이 환경에서 가장 깨지기 쉬운 지점이라 인터프리터를
아예 나눴다. 둘은 localhost 소켓으로만 잇는다.

---

## 빠른 시작

전제부터 확인한다. **부팅 시 자동으로 안 켜진다.**

```bash
ssh pixs@192.168.100.2 sudo systemctl start talking-lamp-device.service
sudo systemctl start talking-lamp-bridges.service
ros2 topic hz /lamp/audio/capture      # 50 Hz 근처여야 한다
```

판단부 (venv, 모델을 올린다. 예열 포함 40초쯤):

```bash
cd ~/talking-lamp/voice-bench
venvs/melo-onnx/bin/python bench/voice_agent.py
```

ROS 노드 (시스템 파이썬):

```bash
source /opt/ros/jazzy/setup.bash
source ~/talking-lamp-integration/jetson_ws/install/setup.bash
cd ~/talking-lamp/voice-bench
python3 ros/lamp_voice_node.py --agent 127.0.0.1:5150
```

순서는 상관없다. 노드는 판단부에 못 붙으면 2초마다 다시 시도한다.
자세한 절차와 문제 해결은 [ros/README.md](ros/README.md).

**"픽스야"** 하고 이어서 말하면 된다. 화면에 상태가 찍힌다 —
`대기 → 듣기 → 생각 → 말하기`.

---

## 설정

기본값은 전부 실측으로 정했다. 근거는 [results/](results/).

### 판단부 (`bench/voice_agent.py`)

| 플래그 | 기본 | |
| --- | --- | --- |
| `--wake-model` | `models/wake/pixs-ya.onnx` | 학습한 웨이크워드 |
| `--wake-threshold` | 0.7 | 낮추면 잘 깨어나고 헛깨움도 는다 |
| `--wake-frames` | 3 | 연속 몇 창이 넘어야 깨울지 (1창 = 80 ms) |
| `--end-silence` | 0.6 | 이만큼 조용하면 말이 끝난 것으로 본다 |
| `--rise-db` | 20 | 재생 중 바닥 대비 이만큼 오르면 끼어든 것 |
| `--llm` | 없음 | OpenAI 호환 엔드포인트. 없으면 되받아 말한다 |
| `--llm-model` | `local` | |
| `--llm-timeout` | 20 | |
| `--stt-device` | `cuda` | |
| `--providers` | `CUDAExecutionProvider,CPUExecutionProvider` | |
| `--tone` | 끔 | TTS 대신 440 Hz 순음. 소리가 안 날 때 내용/경로를 가른다 |

### ROS 노드 (`ros/lamp_voice_node.py`)

| 플래그 | 기본 | |
| --- | --- | --- |
| `--agent` | `127.0.0.1:5150` | 판단부 주소 |
| `--ready-wait` | 0.6 | 브리지 송신기가 설 때까지 기다리는 시간 |
| `--pi-host` | `192.168.100.2` | |

### 잘 안 될 때 먼저 돌릴 것

| 증상 | |
| --- | --- |
| 소리가 안 남 | `--tone` 으로 내용 문제인지 경로 문제인지 가른다 |
| 마이크 프레임 0개 | 브리지·device 서비스 확인 ([ros/README.md](ros/README.md)) |
| 안 깨어남 | `--wake-frames 2` 로 느슨하게 |
| 혼자 말을 끊음 | `--rise-db 24` |
| 끼어들어도 안 멈춤 | `--rise-db 16` |

---

## 코드 구조

제품 코드와 측정 도구를 섞지 않는다.

```
voice/                    제품 — 장치 없이 도는 판단부
  agent.py                  상태 기계. 대기·듣기·생각·말하기와 barge-in
  wake.py                   웨이크워드 (openWakeWord)
  stt.py                    faster-whisper
  tts.py                    MeloTTS ONNX + 문장 단위 분할
  llm.py                    OpenAI 호환 스트리밍 (없으면 되받아 말함)
  link.py                   ROS 노드 ↔ 판단부 규약 (길이 접두 프레임)
  audio.py  vad.py          리샘플·VAD

ros/                      제품 — ROS 쪽
  lamp_voice_node.py        토픽·액션 ↔ 판단부. rclpy 말고 안 쓴다
  playback_probe.py         재생 경로만 시험하는 최소 스크립트
  wake_record_ros.py        실제 마이크 경로로 웨이크워드 녹음

runners/tts_melo_onnx.py  제품 — ONNX 전용 추론
melo_text/                제품 — torch 없는 한글 프론트엔드
ko_normalize.py           제품 — 숫자·영문 → 한글 읽기

bench/                    측정 도구. 램프 런타임에 안 들어간다
  voice_agent.py            판단부를 젯슨에서 띄우는 진입점
  agent_test.py             판단부 회귀 시험 (장치 불필요)
  pacing_test.py            재생 일정 시험 (ROS 불필요)
  node_test.py              노드 취소 처리·EOS 순서 (ROS 불필요)
  wake_*.py                 웨이크워드 학습·평가·측정 (아래)
  stt_sweep.py              STT 모델·스레드별 RTF 와 CER
  mem_profile.py            구간별 메모리. barge-in 겹침 포함
  e2e_test.py               N턴 반복 — 누수와 지연 흔들림
  aec_check.py              에코 제거·barge-in 여유
  doa_measure.py            음원 방향 오차
  run.py  soak.py  blind.py 후보 비교·장시간·블라인드 청취

export/                   모델 변환 — 한 번만 실행하면 된다
```

### 시험

장치도 ROS도 없이 돈다. 고치고 나면 이것부터.

```bash
venvs/melo-onnx/bin/python bench/agent_test.py    # 판단부 상태·barge-in·재생 수명
venvs/melo-onnx/bin/python bench/pacing_test.py   # 재생 일정 계산
venvs/melo-onnx/bin/python bench/node_test.py     # 노드 취소 처리·EOS 순서
```

`node_test.py` 는 rclpy 를 가짜로 채우고 `Node.__init__` 없이 객체만 만들어
메서드를 직접 부른다. 하드웨어도 ROS 도 없이 실제 코드를 시험하기 위해서다.

시험을 넣을 때는 **일부러 고장 내서 실제로 실패하는지** 확인한다. 안
실패하는 시험은 아무것도 안 지킨다.

---

## 웨이크워드

"픽스야". 5명 녹음 190개 + 합성 부정 207개로 학습한다.

```bash
venvs/melo-onnx/bin/python bench/wake_tts_neg.py                  # 합성 부정 만들기
venvs/melo-onnx/bin/python bench/wake_train.py --rebuild     --tts-neg out/wake-tts-neg                                    # 학습 + 내보내기
venvs/melo-onnx/bin/python bench/wake_eval.py                     # 부른 횟수로 평가
venvs/melo-onnx/bin/python bench/wake_field.py                    # 실제 마이크로 측정
```

| 도구 | |
| --- | --- |
| `wake_record.py` | 노트북으로 녹음 (팀원 배포용) |
| `ros/wake_record_ros.py` | 실제 마이크 경로로 녹음 |
| `wake_data.py` `wake_data_check.py` | 사람마다 다른 폴더 구조를 읽고 검사 |
| `wake_augment.py` | 녹음 190개 → 창 8,000개 |
| `wake_train.py` | 학습·한 사람 빼고 검증·ONNX 내보내기 |
| `wake_eval.py` | 부른 횟수 기준 평가 |
| `wake_compare.py` | 설정 둘을 같은 녹음으로 짝지어 비교 |
| `wake_field.py` | **실제 마이크로 측정** |

녹음은 개인 목소리라 저장소에 없다(`wake-data/`, gitignore). 모델
(`models/wake/`)은 커밋한다 — 녹음 없이 다시 만들 수 없다.

**현재 상태**: 팀원 목소리는 잘 깨어나지만 학습에 없던 목소리는 42% 다.
대화 중 헛깨움 0.40회/분. 자세한 것과 다음 할 일은
[results/wake-2026-09-17.md](results/wake-2026-09-17.md).

---

## LLM 붙이기

자리는 준비돼 있다. 모델이 정해지면 플래그 하나다.

```bash
venvs/melo-onnx/bin/python bench/voice_agent.py     --llm http://127.0.0.1:8080/v1/chat/completions --llm-model <이름>
```

OpenAI 호환 엔드포인트를 가정한다(llama.cpp 서버·vLLM·Ollama 전부 이 형식).
의존성을 늘리지 않으려고 urllib 로 직접 읽는다.

**문장이 완성될 때마다 하나씩 내보낸다.** 판단부가 받는 즉시 합성하므로
답을 다 만들 때까지 기다리지 않는다. 답이 길어져도 체감 지연은 그대로다.

시스템 프롬프트에서 두 문장을 넘지 않게 하고 200자에서 자른다 — 길면
지연도 메모리도 같이 커진다.

---

## TTS 구성과 근거

**MeloTTS 한국어를 ONNX로 변환해, VITS는 fp32 · BERT는 int8로 쓴다.**
Jetson에서는 CUDA로 돌린다.

> 맥 CPU 기준으로는 int8 전체가 주력이었으나, **Jetson GPU에서 뒤집혔다.**
> int8+CUDA 는 fp32+CUDA 보다 10.4배 느리다 (RTF 2.61 vs 0.250).
> 실측 근거는 [JETSON-측정.md](JETSON-측정.md) 를 볼 것.

| 구성 | RTF | 비고 |
| --- | --- | --- |
| **VITS fp32 + BERT int8** (주력, Jetson/CUDA) | **0.251** | BERT int8은 속도 손해 +0.3%, 메모리 −772 MB |
| VITS fp32 + BERT fp32 (Jetson/CUDA) | 0.250 | 메모리만 더 씀 |
| 전체 int8 (Jetson/CUDA) | 2.61 | **쓰면 안 된다** |
| 전체 int8 (맥 CPU) | 0.86 | CPU 전용 예비 |

실행은 `--bert-int8` 을 준다. `--int8` 은 VITS까지 내려가므로 GPU에서 금물이다.

**긴 답변은 문장 단위로 쪼개서 합성할 것.** 138자를 통째로 합성하면 메모리가
655 MB 튀고 첫 소리까지 4.56초가 걸린다. 쪼개면 각각 +95 MB, 0.59초다.

### 왜 ONNX인가

원본 melo는 피크 2 GB인데 모델 가중치는 652 MB뿐이었다. 나머지를 줄이려고
int8 양자화·MLM 헤드 제거·BERT 완전 제거·스레드 조정·캐시 반환을 모두 시도했으나
피크가 거의 변하지 않았다. **BERT(1.24 GB)를 통째로 빼도 280 MB밖에 안 줄었다.**

바닥을 만드는 것이 모델이 아니라 **torch 런타임과 그 순간 할당**이었기 때문이다.
그래서 torch를 통째로 걷어냈다.

### 합성 경로

```
텍스트
  → melo_text/          한글 → 음소·성조·심볼 ID (순수 파이썬, torch 없음)
  → bert-kor-base.onnx  운율 특징 (hidden_states[-3])
  → melo-ko-vits.onnx   음성 파형
  → 오디오 (44.1 kHz)
```

`venvs/melo-onnx`에는 **torch가 설치되어 있지 않다.** 우회가 아니라 실제 제거다.

### 모델 산출물 재생성

가중치는 용량 때문에 커밋하지 않는다. 아래 순서로 `models/melo-ko-onnx/`를 만든다.

```bash
python export/melo_export_onnx.py   # VITS  → melo-ko-vits.onnx
python export/melo_export_bert.py   # BERT  → bert-kor-base.onnx + tokenizer/ + frontend.json
python export/melo_quantize.py      # 둘 다 → *.int8.onnx
```

내보내기에는 원본 MeloTTS(torch 포함) 환경이 필요하다. 실행은 torch 없이 된다.

```bash
# Jetson (CUDA) — 최종 구성
python runners/tts_melo_onnx.py --out-dir out/tts/melo --label melo \
    --normalize --bert-int8 --warmup 2 --quiet-ort \
    --providers CUDAExecutionProvider,CPUExecutionProvider

# 맥 (CPU) — 여기서는 전체 int8 이 가장 빠르다
python runners/tts_melo_onnx.py --out-dir out/tts/melo-int8 --label melo-int8 \
    --normalize --int8
```

### Jetson 으로 옮길 때

저장소를 클론한 뒤, **git에 없는 두 가지를 따로 옮긴다.**

```bash
rsync -av ~/talking-lamp/voice-bench/models/melo-ko-onnx/ jetson:~/talking-lamp/voice-bench/models/melo-ko-onnx/
rsync -av ~/talking-lamp/voice-bench/ref/ jetson:~/talking-lamp/voice-bench/ref/
```

`models/`는 용량(743 MB) 때문에, `ref/`(녹음)는 개인 음성이라 커밋하지 않는다.
`ref/`는 맥에서 녹음한 그 파일이어야 CER을 직접 비교할 수 있다.

이어서 Jetson에서 런타임을 갖춘다. **PyPI 휠 두 개가 모두 못 쓴다.**

| 패키지 | 문제 | 해결 |
| --- | --- | --- |
| onnxruntime-gpu | sm_87 커널 없음 → `cudaErrorNoKernelImageForDevice` | `build-onnxruntime/` |
| ctranslate2 | CUDA 없이 빌드됨 → `not compiled with CUDA support` | `build-ctranslate2/` |

둘 다 aarch64 크로스 빌드다. 각 디렉터리의 `build.sh` 를 메모리 넉넉한
리눅스 장비에서 돌리고, 나온 휠을 Jetson 으로 옮겨 설치한다.
**설치 순서에 주의** — `faster-whisper` 가 의존성으로 CPU판 onnxruntime 을
끌고 오므로, 직접 빌드한 휠은 반드시 마지막에 덮어쓴다.

```bash
./bench/jetson_check.sh         # 설치 전 점검 — 아무것도 바꾸지 않는다
./bench/jetson_test.sh setup    # venv + 의존성 (직접 빌드한 휠은 건드리지 않는다)

# 아래는 venv 의 파이썬으로 돌린다.
# TOKENIZERS_PARALLELISM 은 transformers 의 포크 경고를 막는 것뿐이다.
export TOKENIZERS_PARALLELISM=false
P=venvs/melo-onnx/bin/python
$P bench/stt_sweep.py --device cuda --compute-type int8_float16 --threads 6
$P bench/mem_profile.py --stt-device cuda --bert-int8
$P bench/e2e_test.py --turns 30
$P bench/long_utterance.py
```

### 실측 결과 (2026-09-06)

**맥 CPU 기준으로 내렸던 결정 두 개가 여기서 뒤집혔다.**

| | 맥 CPU에서의 판단 | Jetson 실측 |
| --- | --- | --- |
| TTS | int8 주력 | **int8+CUDA 는 fp32 보다 10.4배 느리다** (RTF 2.61 vs 0.250) |
| STT | CPU 로도 될 것 | **6코어를 다 써도 RTF 1.07** — CUDA 필수 (0.35) |

- **BERT 만 int8** 로 내리면 속도 손해 +0.3% 에 메모리 −772 MB. `--bert-int8` 을 쓴다.
- **긴 답변은 문장 단위로 쪼개서** 합성한다. 메모리 −505 MB, 첫 소리 7.7배 빠름.
- 메모리: 유휴 ~1.1 GB / 대화 중 1.25~1.4 GB / barge-in 최악 ~1.55 GB.
- 30턴 반복에서 CER 0.000 유지, 메모리 변동 −1 MB, 지연 p95 2.19s.

전체 근거는 [JETSON-측정.md](JETSON-측정.md), 콘솔 원문은
[results/jetson-2026-09-06.md](results/jetson-2026-09-06.md).

**메모리는 피크 RSS 가 아니라 `/proc/meminfo` 기준 시스템 사용량으로 잰다.**
피크 RSS 는 한 번 올라가면 안 내려가서 이미 반납된 몫을 다음 구간에 더한 것처럼
보인다 — 그래서 초기에 3263 MB 라는 과대값이 나왔다. 통합 메모리라 GPU 가
잡아간 몫도 시스템 사용량에는 잡힌다.

## 성능

말이 끝난 것으로 판정된 순간부터 첫 소리까지. 젯슨 실측, LLM 없음.

| | |
| --- | --- |
| 발화 끝 판정 | 0.60s (고정) |
| STT | 0.70~1.19s (버퍼 길이 × RTF 0.35) |
| 응답 생성 | 0.00s — LLM 을 붙이면 여기가 는다 |
| TTS 첫 문장 | 0.91s |
| **합계** | **2.3s** |

적재는 예열 포함 40초쯤이다. **예열을 빼면 첫 턴이 8초** 가 된다 — CUDA
커널 선택과 그래프 최적화가 첫 호출에 몰리기 때문이라, 그 값을 사람이
처음 말을 건 순간에 치르지 않게 적재할 때 미리 치른다.

메모리는 유휴 ~1.1 GB, 대화 중 1.25~1.4 GB, barge-in 겹칠 때 최악 ~1.55 GB.
팀에는 **2.5 GB** 를 요청해 뒀다.

자세한 것은 [results/e2e-2026-09-21.md](results/e2e-2026-09-21.md).

---

## 문서

| | |
| --- | --- |
| [후보-선정.md](후보-선정.md) | TTS 9종·STT 4종을 무엇을 왜 떨어뜨렸나. 웨이크워드·VAD·마이크 포함 |
| [JETSON-측정.md](JETSON-측정.md) | 보드 실측과 최종 구성 |
| [ros/README.md](ros/README.md) | 젯슨에서 띄우는 절차와 문제 해결 |
| [bench/WAKE-RECORDING.md](bench/WAKE-RECORDING.md) | 웨이크워드 녹음 요청 (팀원 배포용) |
| [bench/MIC-ARRIVAL.md](bench/MIC-ARRIVAL.md) | 마이크 도착 후 할 것 |
| [results/](results/) | 측정 기록 원문 |

측정 기록은 날짜순이다.

| | |
| --- | --- |
| `jetson-2026-09-06.md` | 보드에서의 TTS·STT 실측, 최종 구성 결정 |
| `aec-2026-09-16.md` | 에코 제거 여유 — barge-in 가능 판정 |
| `playback-2026-09-16.md` | 젯슨 → 파이 재생이 안 되던 원인 |
| `wake-2026-09-17.md` | 웨이크워드 학습과 실제 마이크 실측 |
| `e2e-2026-09-21.md` | 전 구간 실측 — 지연·barge-in·버그 넷 |

---

## 남은 작업

**웨이크워드가 가장 급하다.** 자세한 것은
[results/wake-2026-09-17.md](results/wake-2026-09-17.md) 의 "다음" 절.

- 학습에 없던 목소리 42% — 그 사람들 목소리 없이는 안 된다
- 대화 중 헛깨움 0.40회/분 — 실제 채널을 거친 부정 데이터로 내릴 수 있다
- 모델이 "픽" 을 거의 안 듣는다. 고치는 방향은 확인했으나 아직 반영 안 함

그 밖에

- **사람이 실제로 끼어드는 것을 재기** — 재생 중 캡처가 온다는 것은
  확인했다. 임계 20 dB 가 맞는지는 안 재봤다
- **LLM 을 붙이고 첫 문장까지의 시간 재기** — 자리는 준비돼 있다
- **VLM 과 GPU 를 동시에 쓸 때 메모리 재측정** — 통합 메모리라 서로
  밀어낼 수 있다. 지금 숫자는 전부 음성만 돌린 상태다. A·D 와 같이 돌릴 것
- **긴 시간 누수 확인** — 30턴에서는 안정이었으나 턴당 1~2 MB 는 이
  표본에서 잡음에 묻힌다. 200턴 이상을 밤새 돌릴 것
