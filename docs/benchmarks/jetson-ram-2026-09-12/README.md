# Jetson VLM RAM·품질 최종 루프 (2026-09-12)

## 합격 결과

Jetson Orin Nano 8GB 실기에서 최종 경량 경로를 새 프로세스로 3회 반복했다.
합격 조건은 실행 중 `MemAvailable >= 800MB`(10진), swap I/O 0, 정상 종료,
VLM 필수 개념 검출, 한국어 렌더링 성공, TTS WAV 생성, WAV를 STT로 다시 읽은
문장의 완전 일치다.

최종 구성:

- InternVL3.5-1B 사전 NF4 체크포인트
- 이미지 1타일
- 영어 물체 목록 최대 8단어·16토큰
- 프로젝트의 닫힌 물체 집합을 `render_scene_ko.py`로 한국어 문장화
- STT, VLM, TTS를 별도 프로세스로 직렬 실행
- MeloTTS ONNX, BERT int8, CUDA Execution Provider

| 회차 | 시스템 피크 | 최소 여유 RAM | VLM 추론 | 전체 벤치 | 음성 왕복 |
|---|---:|---:|---:|---:|---|
| 1 | 4.146GB | 3.703GB | 3.86초 | 45.40초 | 완전 일치 |
| 2 | 4.108GB | 3.742GB | 3.55초 | 51.21초 | 완전 일치 |
| 3 | 4.105GB | 3.744GB | 3.62초 | 42.22초 | 완전 일치 |

최악 회차도 목표보다 2.903GB 더 남았다. 기존 VLM·STT·TTS 강제 동시 실행의
최저 여유 762.6MB와 비교하면 여유가 2.941GB 증가했고 시스템 피크는
7.087GB에서 4.146GB로 41.5% 감소했다. 여유 RAM의 상한은 두지 않았다. 더 많이
남는 실행이 더 안전하기 때문이다.

세 회차의 VLM 출력은 모두 `USB charger, power adapter.`였고 한국어 출력은
`여기에는 USB 충전기와 전원 어댑터가 보여요.`였다. 4.47초 WAV를 다시 STT에
입력했을 때 세 회차 모두 이 한국어 문장을 정확히 복원했다.

종합 판정 원본은 `final-1b-rendered/acceptance.json`, 시스템 표본은
`final-1b-rendered/attempt-*.json`, VLM·한국어 출력은 `final-vlm/`, TTS 메타데이터와
WAV는 `final-tts/`, 음성 왕복 결과는 `final-roundtrip/`에 있다.

## 실험 루프에서 기각한 구성

| 구성 | 최소 여유 RAM | 판정 이유 |
|---|---:|---|
| 2B + STT/TTS 강제 동시 실행 | 762.6MB | 800MB 게이트 실패 |
| 2B 직렬 실행 | 2.452GB | 통과했지만 1B보다 약 1.25GB 더 사용 |
| 1B 영어 완전 문장 | 3.629GB | RAM·영어 인식 통과, 한국어 변환 필요 |
| 1B + NLLB-600M fp16 | 3.191GB | `전력 스트립`, `래프` 등 번역 품질 실패 |
| 1B + 닫힌 집합 한국어 렌더러 | 3.703GB | 최종 채택 |

NLLB를 제거하면서 번역 단계의 CUDA 피크 약 1.25GB와 2.4GB 모델 파일 의존성을
없앴다. 알 수 없는 VLM 라벨은 영어로 흘려보내지 않고 렌더러가 실패 코드 3으로
중단하므로 품질 저하를 조용히 숨기지 않는다.

## 교차 장면 품질

같은 모델 설정으로 다음 네 실제 사진을 검사했다.

| 장면 | 1B 핵심 출력 | 한국어 렌더링 | 판정 |
|---|---|---|---|
| 전원 어댑터·멀티탭 | USB charger, power adapter | USB 충전기, 전원 어댑터 | 통과 |
| 손상된 사무용 의자 | Office chair, desk with items | 사무용 의자, 물건이 놓인 책상 | 통과 |
| 로봇 램프 작업대 | Workstation, robot arm, laptop, notebook | 작업대, 로봇 팔, 노트북, 공책 | 통과 |
| 손·회로기판 조립 | Blue box with Arduino, laptop, keyboard, notebook | 아두이노가 든 파란 상자, 노트북, 키보드, 공책 | 통과 |

`cross-image/`, `concise-1b/`, `rendered-1b/`에 입력과 판정 결과를 보관했다. 이
게이트는 프로젝트의 장면 인식용 닫힌 물체 집합을 검증한다. 자유 주제 한국어
대화 품질 전체를 보증하는 시험은 아니다.

## 독립 재검증 (2026-09-12, 세션 2)

`evaluate_pipeline_run.py`에 WAV 경로 폴백 수정(로컬 `tools/`에만 있고 원격에
반영 안 됐던 차이, `sha256sum` 대조로 발견)을 원격에 동기화한 뒤, 완전히 새
출력 디렉터리(`verify2*`)에 3회를 다시 돌려 첫 통과가 우연이 아닌지 확인했다.

| 회차 | 시스템 피크 | 최소 여유 RAM | 전체 벤치 |
|---|---:|---:|---:|
| 1 | 4.062GB | 3.787GB | 55.33초 |
| 2 | 4.142GB | 3.707GB | 52.97초 |
| 3 | 4.108GB | 3.741GB | 45.01초 |

세 회차 모두 `acceptance.json` `passed: true`, swap 0, VLM 출력
`USB charger, power adapter.`, 한국어 `여기에는 USB 충전기와 전원 어댑터가
보여요.`, 왕복 STT 완전 일치. 최초 실행과 오차 범위 내로 일치해 재현성을
확인했다. 원본 산출물은 `final-*`, 이 독립 재검증 산출물은
`verify2-independent/{ram,vlm,tts,roundtrip}/`에 있다(WAV는 용량상 미보관,
JSON 메타데이터만 보관).

## 재현 명령

```bash
python tools/repeat_ram_experiment.py \
  --out-dir /mnt/ssd/ram-results/final-1b-rendered \
  --attempts 3 --timeout 180 \
  --min-available-mb 800 --max-swap-pages 0 -- \
  bash tools/run_dialogue_memory.sh

python tools/evaluate_pipeline_run.py \
  --ram-summary /mnt/ssd/ram-results/final-1b-rendered/summary.json \
  --vlm-dir /mnt/ssd/ram-results/final-vlm \
  --tts-dir /mnt/ssd/ram-results/final-tts \
  --roundtrip-dir /mnt/ssd/ram-results/final-roundtrip \
  --image-sha256 24efd9a8c01727ea4780ff9bfa6047a816e7c6ac6690c7e58de63a75963fd17f \
  --require-any usb,charger --require-any power,adapter \
  --out /mnt/ssd/ram-results/final-1b-rendered/acceptance.json
```
