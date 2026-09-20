# VLM 경량화 벤치와 팀 기능 연결

2026-09-20 `git fetch origin` 기준:

| 위치 | 확인한 구현 |
| --- | --- |
| `origin/main` (`118e16f`) | `src/motion`: IK, DOA bearing 입력, 100Hz 궤적, 서보 실행, 안전 종료 |
| `origin/feat/voice-bench` (`3cb97eb`) | `voice-bench/voice`: STT, TTS, wake, VAD, 음성 에이전트 및 `reply(text)` 응답 인터페이스 |
| `origin/codex/jetson-pi-middleware` (`f893c5a`) | Pi 오디오·XVF3800·DOA, LED, Jetson–Pi 모션 전송 |

메인에 있는 DOA **입력 API**와 실제 마이크 방향 **측정 구현**은 다르다.
음성 및 미들웨어 브랜치는 아직 메인에 병합되지 않았다.
현재 `feat/ram-lightweight`에는 위 메인의 모션 변경을 병합했다.
기존 미추적 벤치 결과와 음성 데이터는 유지했다.

## 공통 출력

`src/cognition/contract.py`의 `SYSTEM_PROMPT`, `CognitionResult`를 벤치와
실행 어댑터가 공유한다. 출력은 정확히 `observation`, `speech_ko`, `motion`인
JSON이며, 한국어 발화는 200자 이하, 행동은 기존 CSV 목록으로 제한한다.
`idle`은 새 모션을 보내지 않는 의미이며 기존 동작 취소 명령이 아니다.
JSON 규약 통과는 시각적 사실의 정확성을 보증하지 않는다.

벤치 결과에는 `checks.runtime_contract`와 `handoff`가 추가된다.
`handoff.tts`는 `tools/tts_single.py --input`의 `{"text": ...}` 형식이다.
`handoff.motion`은 미들웨어 `MotionClient.request(type, payload)` 인자이며,
인증 토큰·UUID·TTL을 담은 전송 패킷 자체가 아니다.

## 음성 에이전트 연결

VLM 환경은 기존 `tools/requirements-vlm-ram.txt`를 사용한다.
`PYTHONPATH=src`로 아래 모듈을 불러올 수 있다.

```python
from queue import Queue
from cognition.internvl import InternVLCognition
from cognition.voice_adapter import VlmReply

pending = Queue()
backend = InternVLCognition()  # 기존 벤치와 같은 1B / NF4 / 1 tile / 96 tokens
adapter = VlmReply(backend, snapshot=camera_snapshot, submit_motion=pending.put)
# voice-bench의 VoiceAgent 생성 시 on_utterance=adapter.reply 전달
# camera_snapshot은 현재 프레임의 PIL.Image 복사본을 반환하는 비전 팀 콜백
```

모션을 소유한 통합 루프에서 `pending.get_nowait()`로 결과를 소비하여
`runtime.play_primitive(result.motion)`을 호출한다. 별도 보드에서는
`result.handoff()["motion"]`을 기존 MotionClient에 전달한다.
VLM 추론과 카메라 캡처를 Pi 100Hz 루프 안에서 실행하지 않는다.
실서비스에서는 통합 담당이 턴 ID/취소 및 오래된 큐 항목 폐기를 관리해야 한다.
음성 끼어들기는 기존 `runtime.barge_in()`/미들웨어 interrupt에 연결한다.
어댑터는 JSON 전체 검증 후 한 문장을 반환하며 토큰 스트리밍은 하지 않는다.
이 상주 모델 방식의 STT/TTS 동시 메모리는 Jetson에서 별도 측정해야 한다.

## GPU 없이 벤치 → 모션 → TTS 입력 검사

```bash
python tools/replay_vlm_scenarios.py \
  --input docs/benchmarks/vlm-project-2026-09-15/2b.json \
  --out-dir /tmp/vlm-replay
```

통과한 케이스만 NullBackend 모션 런타임으로 100 tick 진행하고 TTS 입력을
저장한다. 실제 모터나 스피커는 구동하지 않는다. 실패한 케이스가 있으면
보고서를 남기고 종료 코드 1을 반환한다. 오디오 합성은 음성 브랜치의 모델과
런타임을 준비한 뒤 생성된 `NNN-tts.json`을 `tools/tts_single.py`에 전달한다.

기존 9월 15일 결과는 1B 0/7, 2B 1/7 통과다. 영어 관찰 키워드 검사에
한국어 관찰이 실패하는 평가 언어 문제도 있으므로 전체 실패를 그대로
시각 인식 실패율로 해석하면 안 된다. 모션 선택 실패 등은 별도로 남아 있다.
이번 변경은 연결 규약 및 실행 경로를 검증하며 모델 정확도 개선 실측은 아니다.
