# 한국어 TTS — MeloTTS ONNX 파이프라인

파트 C(음성) / 최승원 · 2026-09-01
관련 문서: [진행-순서.md](../docs/진행-순서.md) C-2 "TTS 엔진 후보 한국어 음질 실측 → 확정"

---

## 결정

**MeloTTS 한국어를 ONNX로 변환해 int8 양자화한 것**을 쓴다.
fp32 변환본을 예비로 함께 둔다.

| 구성 | STT 포함 피크 RSS | 가중치 | RTF (맥 CPU) |
| --- | --- | --- | --- |
| **melo ONNX int8** (주력) | **1407 MB** | 158 MB | 0.86 |
| melo ONNX fp32 (예비) | 1550 MB | 585 MB | 0.36 |
| melo 원본 (torch) | 2013 MB | 652 MB | 0.27 |

측정 조건: Apple M3 / CPU / 심사 문장 6개를 100 사이클 반복.
각 사이클은 STT 1문장 + TTS 1문장이며, 동시 실행하지 않는다
(반이중 구조 — TTS 발화 중에는 VAD만 돌고 STT는 돌지 않는다).

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
python runners/tts_melo_onnx.py --out-dir out/tts/melo-onnx-int8 \
    --label melo-onnx-int8 --normalize --int8
```

## Jetson으로 옮길 때

저장소를 클론한 뒤, **git에 없는 두 가지를 따로 옮긴다.**

```bash
rsync -av ~/talking-lamp/voice-bench/models/melo-ko-onnx/ jetson:~/talking-lamp/voice-bench/models/melo-ko-onnx/
rsync -av ~/talking-lamp/voice-bench/ref/ jetson:~/talking-lamp/voice-bench/ref/
```

`models/`는 용량(743 MB) 때문에, `ref/`(녹음)는 개인 음성이라 커밋하지 않는다.
`ref/`는 맥에서 녹음한 그 파일이어야 CER을 직접 비교할 수 있다.

이어서 Jetson에서:

```bash
./bench/jetson_check.sh            # 설치 전 점검 — 아무것도 바꾸지 않는다
./bench/jetson_test.sh setup       # venv + 의존성
./bench/jetson_test.sh stt         # STT 정확도·메모리 (CUDA 실패 시 CPU로 자동 전환)
./bench/jetson_test.sh tts         # int8 / fp32 비교, 실행 공급자 표시
./bench/jetson_test.sh soak        # 100 사이클 x 3회 — 예산 근거
```

`jetson_check.sh` 를 먼저 돌려 **aarch64 휠 가용성**을 확인한다.
`ctranslate2`(STT)와 `onnxruntime-gpu`(GPU 가속)가 가장 막히기 쉬운 지점이며,
휠이 없으면 소스 빌드로 넘어가 시간이 크게 든다.

- 내보내기를 다시 할 필요는 없다. `models/` 폴더가 곧 산출물이다.
- onnxruntime provider를 CUDA/TensorRT로 바꾼다.
- **RTF를 반드시 다시 잰다.** 맥 CPU에서 int8이 fp32보다 느렸고(0.86~0.99 vs 0.36~0.46),
  긴 문장에서는 1.0을 넘기도 했다.
  Jetson GPU는 int8 가속이 있어 반대로 나올 가능성이 높지만 확인 전까지는 미지수다.
  **1.0을 넘으면 실시간보다 느려 대화에 못 쓰므로 fp32로 전환한다.**
- 측정은 3회 이상 반복한다. 이 워크로드는 편차가 ±200~300 MB로 크다.

## 남은 작업

- 스트리밍 처리 (첫 음절까지의 시간 단축) — 진행 순서 C-6
- barge-in 시 즉시 정지 — C-7
- B의 런타임 스텁이 나오면 이 추론 로직을 그 인터페이스에 맞춰 모듈로 감싼다.
  현재 러너는 문장을 파일로 뽑는 벤치마크용 구조다.
