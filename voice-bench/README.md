# 한국어 TTS — MeloTTS ONNX 파이프라인

파트 C(음성) / 최승원 · 2026-09-01
관련 문서: [진행-순서.md](../docs/진행-순서.md) C-2 "TTS 엔진 후보 한국어 음질 실측 → 확정"

---

## 결정

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

## 왜 ONNX인가

원본 melo는 피크 2 GB인데 모델 가중치는 652 MB뿐이었다. 나머지를 줄이려고
int8 양자화·MLM 헤드 제거·BERT 완전 제거·스레드 조정·캐시 반환을 모두 시도했으나
피크가 거의 변하지 않았다. **BERT(1.24 GB)를 통째로 빼도 280 MB밖에 안 줄었다.**

바닥을 만드는 것이 모델이 아니라 **torch 런타임과 그 순간 할당**이었기 때문이다.
그래서 torch를 통째로 걷어냈다.

## 구성

```
텍스트
  → melo_text/          한글 → 음소·성조·심볼 ID (순수 파이썬, torch 없음)
  → bert-kor-base.onnx  운율 특징 (hidden_states[-3])
  → melo-ko-vits.onnx   음성 파형
  → 오디오 (44.1 kHz)
```

`venvs/melo-onnx`에는 **torch가 설치되어 있지 않다.** 우회가 아니라 실제 제거다.

## 디렉터리 구성

제품 코드와 측정 도구를 섞지 않는다.

```
melo_text/                torch 없는 텍스트 프론트엔드 (제품)
ko_normalize.py           숫자·영문 → 한글 읽기 (제품)
runners/tts_melo_onnx.py  ONNX 전용 추론 (제품)
common.py                 CER·피크 RSS·결과 프로토콜 (공용)
sentences.txt             심사 문장 6개 (공용)

export/                   모델 변환 — 한 번만 실행하면 된다
  melo_export_onnx.py       VITS → ONNX
  melo_export_bert.py       한국어 BERT → ONNX + 토크나이저 + frontend.json
  melo_quantize.py          위 둘을 int8로

bench/                    측정 도구 — 램프 런타임에는 들어가지 않는다
  setup.sh                  후보별 venv 생성
  run.py                    후보 비교 (CER·RSS·TTFB 표)
  soak.py                   장시간 반복 후 메모리 수렴 확인
  blind.py                  TTS 블라인드 청취
  record.py                 심사 문장 녹음
  candidates.json           후보 정의
  runners/                  탈락·대조 후보 실행기

  stt_sweep.py              STT 모델·스레드별 RTF 와 CER
  mem_profile.py            구간별 메모리. barge-in 겹침 포함
  e2e_test.py               N턴 반복 — 누수와 지연 흔들림
  long_utterance.py         긴 발화·긴 답변에서의 메모리
  mic_check.py              마이크 도착 시 전제 확인
  doa_measure.py            음원 방향 오차 측정
  aec_check.py              에코 제거·barge-in 지연
```

`bench/` 아래 것들은 **Jetson 재검증에 그대로 쓴다.** 지우지 말 것.

## 모델 산출물 재생성

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

## Jetson으로 옮길 때

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

# 아래는 venv 의 파이썬으로 돌린다. TOKENIZERS_PARALLELISM=false 는
# transformers 가 포크 경고를 뿜는 것을 막는다.
P="TOKENIZERS_PARALLELISM=false venvs/melo-onnx/bin/python"
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

## 남은 작업

- **문장 단위 분할 합성을 러너에 넣기** — 지금은 측정으로만 확인했다.
  첫 소리까지 4.56s → 0.59s 이고 메모리도 505 MB 줄어든다 (C-6)
- barge-in 시 즉시 정지 — C-7. 겹쳐 도는 것 자체는 확인됐다(30턴 중 6회 정상)
- **VLM 과 GPU 를 동시에 쓸 때 재측정** — 통합 메모리라 서로 밀어낼 수 있다.
  지금 숫자는 전부 음성만 돌린 상태다. A·D 파트와 같이 돌려야 한다
- **긴 시간 누수 확인** — 30턴에서는 안정이었으나 턴당 1~2 MB 씩 새는 것은
  이 표본에서 잡음에 묻힌다. 200턴 이상을 밤새 돌릴 것
- 마이크·웨이크워드·VAD 통합 — 마이크 미도착, 웨이크워드 학습 전.
  현재 숫자는 "음성 파트 전체" 가 아니라 **"STT·TTS"** 다
- B의 런타임 스텁이 나오면 이 추론 로직을 그 인터페이스에 맞춰 모듈로 감싼다.
  현재 러너는 문장을 파일로 뽑는 벤치마크용 구조다.
