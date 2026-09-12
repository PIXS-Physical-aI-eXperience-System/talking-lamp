# RAM 경량화 작업 출처

Jetson VLM RAM 경량화 작업(`feat/ram-lightweight`)에서 어디까지 새로 만들고
어디까지 팀원 작업(`feat/voice-bench`)을 가져다 썼는지 기록함.

## 그대로 가져다 쓴 것

| 항목 | 위치 | 비고 |
|---|---|---|
| TTS 합성 함수 `build_synth` | `voice-bench/runners/tts_melo_onnx.py` | torch 제거, ONNX 전용 러너. 팀원 작업 |
| 한국어 정규화 `normalize` | `voice-bench/ko_normalize.py` | 팀원 작업 |
| BERT ONNX, VITS ONNX 모델 | `voice-bench/models/melo-ko-onnx/` | 팀원이 변환한 모델 파일 |

`tools/tts_single.py`는 위 함수 import해서 호출만 함. TTS 로직 재구현 안 함.

## 설정값만 따라간 것

| 항목 | 내용 |
|---|---|
| STT 모델 구성 | `faster-whisper small`, `cuda`, `int8_float16` — voice-bench 최종 확정값과 동일 |

`tools/stt_quick.py`는 코드 import 안 하고 `faster_whisper` 라이브러리 직접 호출.
모델/디바이스/양자화 설정만 voice-bench 결정 따라감.

## 새로 만든 것

| 항목 | 파일 |
|---|---|
| VLM 모델 선정·양자화(1B NF4), 1타일, 토큰 제한 | `tools/wake_latency.py` |
| 한국어 닫힌 라벨 렌더러 | `tools/render_scene_ko.py` |
| RAM 측정 하네스 | `tools/ram_probe.py`, `tools/repeat_ram_experiment.py` |
| VLM 품질·파이프라인 통합 판정 | `tools/evaluate_vlm_quality.py`, `tools/evaluate_pipeline_run.py` |
| STT→VLM→렌더러→TTS 직렬 오케스트레이션 | `tools/run_dialogue_memory.sh` |

## 결론

RAM 절감분 중 "STT/TTS 동시 실행 대신 직렬 실행"이 제일 큰 비중 차지함.
근데 이건 TTS/STT 엔진 자체를 바꾼 게 아니라 언제 띄우고 내리는지를 바꾼 거임.
엔진은 팀원 작업 그대로, 오케스트레이션과 VLM 쪽만 새로 함.
