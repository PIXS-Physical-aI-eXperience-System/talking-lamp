# 베이스 yaw 방향 정렬 운영 가이드

이 문서는 **모션 계층의 베이스 yaw 방향 정렬** 인터페이스를 설명한다. Pi 장치 서비스가 보호된 로컬 경로로 `base_yaw` 기준점을 설정·조회하고 중앙 복귀를 요청하는 방법을 다룬다. XVF3800 접근·DOA 수집과 안정화·오디오 전송과 재생·WS2812B·장치 서비스·Jetson ROS 브리지·대화 실행 순서는 [Pi 장치 검증](pi-device-commissioning.md), [Jetson 통합](jetson-integration.md), [배포·인수인계](deployment-and-handoff.md) 문서를 참고한다.

## 전제 조건과 담당 범위

`talking-lamp-motion.service`로 Pi 모션 데몬을 실행하면 보호된 Unix 소켓 `/run/talking-lamp/motion-control.sock`이 생성된다.

소켓 권한은 `0660`으로 서비스 사용자·그룹만 사용할 수 있다. 이 Pi 로컬 제어 경로는 8765 포트의 인증된 Jetson TCP 제어 주체를 대체하지 않는다. 원격 TCP에서는 `orientation.acquire`, `orientation.return_center`, `orientation.status`를 거부한다. Unix 전송 계층은 제어기 메일박스에 요청 티켓만 제출하며 런타임 스텝을 직접 실행하거나 별도 하드웨어 제어 주체를 만들지 않는다.

각 요청은 UTF-8 NDJSON 객체 하나에 줄바꿈 문자 하나를 붙인다.

```json
{"version":1,"id":"canonical-request-uuid","type":"...","ttl_ms":1000,"payload":{}}
```

`id`는 정규 형식의 소문자 UUID 문자열, `ttl_ms`는 1~10000의 정수여야 한다. 로컬 메시지는 위 다섯 필드만 포함하며 `token` 필드는 없다. 논리 요청마다 새 요청 UUID를 생성한다.

## 로컬 요청 형식

먼저 소켓 변수를 설정한다.

```bash
SOCKET=/run/talking-lamp/motion-control.sock
```

`socat` 예제는 서버가 응답할 수 있도록 표준 입력을 잠시 열어 둔다. 실제로 방향이 움직이는 요청은 최종 이벤트를 받을 만큼 대기 시간을 늘리거나 Python 예제를 사용한다.

### 방향 기준점 설정

`target_yaw`는 보정된 베이스 좌표계의 유한한 절대 각도이며 단위는 **라디안**, 범위는 `[-pi, pi]`다. `speech_id`는 장치 서비스가 해당 발화에 부여한 정규 형식 UUID다.

```bash
{ printf '%s\n' \
  '{"version":1,"id":"10000000-0000-0000-0000-000000000001","type":"orientation.acquire","ttl_ms":1000,"payload":{"speech_id":"20000000-0000-0000-0000-000000000001","target_yaw":0.40}}'; sleep 3; } \
  | socat - UNIX-CONNECT:"$SOCKET"
```

```python
import json
import socket

request = {
    "version": 1,
    "id": "10000000-0000-0000-0000-000000000001",
    "type": "orientation.acquire",
    "ttl_ms": 1000,
    "payload": {
        "speech_id": "20000000-0000-0000-0000-000000000001",
        "target_yaw": 0.40,
    },
}
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect("/run/talking-lamp/motion-control.sock")
    client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
    for line in client.makefile("rb"):
        event = json.loads(line)
        print(event)
        if event.get("state") != "accepted":
            break
```

이동이 수락되면 첫 이벤트는 `{"id": ..., "state":"accepted", "code":"accepted", ...}`다. 최종 이벤트는 `state:"completed"`이며 안정화 조건을 만족하면 `code:"aligned"`, 2초의 방향 획득 제한을 넘기면 `code:"timeout"`이다. 목표가 5° 불감대 안이면 현재 실제 yaw를 기준점으로 유지한다. yaw가 정지해 있고 안전 구간 안이면 즉시 `aligned`로 완료하며 가까운 요청 각도로 추가 회전하지 않는다. 이미 움직이고 있다면 유지한 기준점에서 먼저 안정화한다. 현재 자세가 기계적 여유 구간에 있으면 안전을 우선해 기준점을 안전 구간으로 제한하고 `clamped:true`를 보고한 뒤 안정화를 기다린다. `aligned` 후에도 목표를 유지하며 같은 `speech_id`의 다른 표본으로 목표를 바꾸지 않는다. 현재 고정된 `speech_id`로 다시 요청하면 최종 `code:"duplicate"`를 반환한다.

작업 조명이 활성 상태이면 `state:"failed", code:"blocked_by_task_light"` 최종 이벤트 하나로 거부하고 기존 기준점은 바꾸지 않는다. 작업 조명을 해제한 뒤 정책에 따라 새 발화의 방향을 획득한다.

### 명시적 중앙 복귀 요청

```bash
{ printf '%s\n' \
  '{"version":1,"id":"10000000-0000-0000-0000-000000000002","type":"orientation.return_center","ttl_ms":1000,"payload":{}}'; sleep 3; } \
  | socat - UNIX-CONNECT:"$SOCKET"
```

```python
import json
import socket

request = {
    "version": 1,
    "id": "10000000-0000-0000-0000-000000000002",
    "type": "orientation.return_center",
    "ttl_ms": 1000,
    "payload": {},
}
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect("/run/talking-lamp/motion-control.sock")
    client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
    for line in client.makefile("rb"):
        event = json.loads(line)
        print(event)
        if event.get("state") != "accepted":
            break
```

예상 순서는 `accepted/accepted` 다음, 중앙 목표가 안정화 조건을 만족한 뒤에만 최종 `completed/centered`가 오는 것이다. 프리미티브 모션이나 작업 조명이 동작 중이면 최종 `failed/busy` 이벤트 하나로 거부한다. 호출자는 작업이 끝나기를 기다리거나 정책에 따라 명시적으로 취소해야 한다. 모션 계층은 잘못된 실행 순서를 감추기 위해 진행 중인 작업을 임의로 끊지 않는다.

### 방향 상태 조회

```bash
{ printf '%s\n' \
  '{"version":1,"id":"10000000-0000-0000-0000-000000000003","type":"orientation.status","ttl_ms":1000,"payload":{}}'; sleep 1; } \
  | socat - UNIX-CONNECT:"$SOCKET"
```

```python
import json
import socket

request = {
    "version": 1,
    "id": "10000000-0000-0000-0000-000000000003",
    "type": "orientation.status",
    "ttl_ms": 1000,
    "payload": {},
}
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect("/run/talking-lamp/motion-control.sock")
    client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
    for line in client.makefile("rb"):
        event = json.loads(line)
        print(event)
        if event.get("state") != "accepted":
            break
```

제어기가 유효 시간 안에 인수한 상태 요청은 `accepted/accepted` 이후 `completed/completed`로 응답한다. 유효한 요청이라도 인수 전에 만료되면 최종 `failed/expired`만 반환하며 런타임 명령 처리나 하드웨어 제어에 전달하지 않는다. 완료 이벤트의 `data`에는 다음 방향 상태가 담긴다.

| 필드 | 의미 |
| --- | --- |
| `state` | `idle`, `orienting`, `aligned`, `timeout`, `returning`, `centered` |
| `speech_id` | 고정된 발화 UUID, 대기 중에는 `null` |
| `target_yaw` | 라디안 단위 기준점·중앙 목표, 대기 중에는 `null` |
| `current_yaw` | 가장 최근 명령의 베이스 yaw, 라디안 단위 |
| `clamped` | 모션 계층이 요청 목표를 안전 yaw 구간으로 제한했는지 여부 |
| `code` | 기계 판독 가능한 상태·결과 코드 |

일반 제어기 상태는 `busy`, `orientation_state`, `orientation_speech_id`, `orientation_target_yaw`, `orientation_current_yaw`, `orientation_clamped`, `primitive_yaw_scale`도 보고한다. 방향 상태가 `orienting` 또는 `returning`이거나 프리미티브·작업 조명 모션이 활성 상태이면 `busy`가 참이다. `aligned` 기준점을 유지하는 것만으로는 작업 중 상태가 되지 않는다.

### 로컬 하트비트

`system.heartbeat`는 방향 상태를 바꾸지 않는 네 번째 허용 로컬 메시지로, 통신 진단에 사용할 수 있다.

```bash
{ printf '%s\n' \
  '{"version":1,"id":"10000000-0000-0000-0000-000000000004","type":"system.heartbeat","ttl_ms":1000,"payload":{}}'; sleep 1; } \
  | socat - UNIX-CONNECT:"$SOCKET"
```

```python
import json
import socket

request = {
    "version": 1,
    "id": "10000000-0000-0000-0000-000000000004",
    "type": "system.heartbeat",
    "ttl_ms": 1000,
    "payload": {},
}
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect("/run/talking-lamp/motion-control.sock")
    client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
    for line in client.makefile("rb"):
        event = json.loads(line)
        print(event)
        if event.get("state") != "accepted":
            break
```

제어기가 유효 시간 안에 인수한 하트비트는 `accepted/accepted` 이후 `completed/completed`를 반환한다. 인수 전에 만료되면 런타임 명령 처리·하드웨어 제어 없이 최종 `failed/expired` 이벤트 하나를 반환한다.

## 안정화·프리미티브·연결 끊김 동작

방향 모션은 우선순위 15로 `base_yaw`만 제어한다. 공통 궤적 생성기가 궤적 제한을 적용하며, 완료하려면 yaw 오차 3° 이하와 yaw 속도 0.08 rad/s 이하를 동시에 150 ms 연속 유지해야 한다. 안전 yaw 구간은 보정된 관절 범위 양끝에서 설정된 여유 5°를 뺀 범위다. `clamped`는 모션 계층이 계산하므로 호출자는 이를 미리 계산하지 않고 반환값을 보고해야 한다.

방향 계층이 yaw를 제어하는 동안 두 궤적 백엔드 모두 경로 전체에 안전 구간을 적용한다. 작업 조명은 더 높은 우선순위와 보정된 절대 한계를 유지한다. 시작 자세가 여유 구간 밖이면 안전 구간 안으로 연속적으로 복귀하며 위치를 순간적으로 뛰게 만들지 않는다.

상대 프리미티브 재생 중에도 기준점은 유지한다. 재생 전에 `anchor_yaw + primitive_offset`이 안전 범위 안에 있도록 yaw에만 `[0, 1]` 배율을 계산한다. 다른 관절 오프셋과 요청한 전체 강도는 유지한다. 이는 `timeout`, `returning`, `centered`를 포함해 절대 기준점이 활성화된 모든 상태와 반복 클립마다 적용한다. 프리미티브 재생 중 방향 획득도 허용하므로 장치 서비스가 방향을 획득하는 동안 대기·몸체 모션을 이어갈 수 있다. 기준점이 바뀌면 중단 후 감쇠 구간까지 포함해 현재 클립의 원본 재표본화 오프셋에서 yaw를 다시 맞춘 뒤 다음 합성을 수행한다. 클립 시간·엔벌로프·나머지 관절 오프셋은 유지하며 반복 조정으로 배율이 누적되지 않는다. 제어기 상태의 `primitive_yaw_scale`은 현재 배율을, 프리미티브 최종 결과의 `data.yaw_scale`은 마지막 적용 배율을 보고한다. 다른 모션으로 교체된 모션도 자신의 최종 결과에 자기 배율을 유지한다.

**인증된 원격 TCP 제어 주체의 연결이 끊기면** 제어기 작업을 안전하게 중단하되 기존 방향 기준점은 정확히 10초간 유지한다. 이후 같은 저수준 중앙 복귀 경로를 시작해 최종적으로 `centered`가 된다. 재연결 시 이전 요청을 재실행하지 않는다. 로컬 Unix 클라이언트 EOF에는 이 규칙을 적용하지 않는다. EOF는 해당 클라이언트의 아직 시작하지 않은 로컬 티켓만 취소할 수 있으며 제어기가 이미 인수한 방향 요청이나 다른 클라이언트의 요청은 취소할 수 없다.

## Pi 장치 서비스가 제공할 입력

마이크 변환은 모션 계층의 담당이 아니다. 장치 서비스는 다음을 담당한다.

- 발화마다 정규 형식 UUID `speech_id` 하나를 생성하고 유지한다.
- DOA를 안정화·검증한 뒤 해당 발화에 방향 요청을 정확히 한 번 보낸다.
- `orientation.acquire` 전송 전에 보정된 DOA를 `[-pi, pi]` 범위의 유한한 절대 라디안 각도 `target_yaw`로 변환한다.
- `doa_zero_deg`와 `doa_direction_sign`은 모션 설정이 아닌 **장치 설정**에 둔다.
- 자체 제한 판단 대신 모션 제어 주체가 반환한 `clamped` 필드를 사용한다.

DOA 도 단위 값, `doa_zero_deg`, `doa_direction_sign`은 로컬 모션 요청 필드가 아니다. 장치 서비스는 이 소켓 호출 전에 DOA 표본 수집·안정화·장치 좌표계 보정·전방 반구 밖 입력 거부·도→라디안 변환을 수행한다.

## 실물 마이크 벤치 인수인계 (마이크 시험 전 필수)

실물 마이크 시험 전에 `origin/feat/voice-bench`를 갱신·확인하고 해당 참조의 `voice-bench/bench/MIC-ARRIVAL.md`를 읽는다. 최초 모션 단계 검증 당시 커밋은 `91ce0a14f6f21bfe4aee5e2cd6183bb02c6ec187`이었으나 초기 설치 시 이 과거 버전에 의존하지 않는다. 모션 작업 공간에 파일을 복사하거나 브랜치를 전환하지 않고 다음 명령을 사용한다.

```bash
git fetch origin feat/voice-bench
git rev-parse origin/feat/voice-bench
git show origin/feat/voice-bench:voice-bench/bench/MIC-ARRIVAL.md
```

납품된 어레이는 **Linear-4**다. 전방 반구 약 180°만 지원하며 Linear 16 kHz/6채널 펌웨어·장치 설정인 `L16K6Ch`를 사용해야 한다. 원형 어레이용 `C16K6Ch`를 사용하지 않는다. 램프에 장착한 뒤 해당 방향으로 다시 보정한다. 왼쪽 90/60/30°, 정면, 오른쪽 30/60/90°의 7점 표를 새로 만들고 별도의 검증 측정을 수행한다. 기존 `out/doa/calibration.json`은 장착 상태에 종속되므로 과거 보정 파일을 복사하지 않는다.

조용한 환경·팬 소음 조건뿐 아니라 서보 소음 조건에서도 벤치를 수행한다. 서보 동작은 추종에 중요한 조건이다. XVF 출력 장치가 있는지 확인하고 보드의 재생·AEC 경로를 사용한다. 마이크 결과를 사용 가능한 것으로 판정하기 전에 XVF 출력·AEC 검증도 수행한다. 소음·오차·AEC 결과는 장치 서비스 검증의 입력이며 모션 계층 시험만으로 오디오·DOA 구현을 검증할 수는 없다.

## 모션 계층 검증 근거

설계에서 요구한 모션 보장을 다음 자동 테스트로 확인한다.

| 검증 항목 | 구체적인 테스트 |
| --- | --- |
| 방향 계층은 절대 베이스 yaw만 제어하며 안전 여유 준수 | `tests/test_orientation.py::test_orientation_layer_claims_only_base_yaw_and_clamps_with_margin` |
| 위치·속도 연속 안정화 구간 이후 성공 | `tests/test_orientation.py::test_coordinator_settles_only_after_continuous_position_and_velocity_window`; `tests/test_motion_controller.py::test_orientation_ticket_completes_only_after_settle` |
| 상대 프리미티브는 기준점을 유지하고 yaw를 안전하게 축소 | `tests/integration/test_runtime.py::test_orientation_anchor_overrides_tracking_yaw_but_not_other_tracking_joints`; `tests/integration/test_runtime.py::test_anchored_motion_scales_only_yaw_to_fit_safe_range` |
| 기준점 변경·감쇠 구간에서도 안전 명령 유지 | `tests/test_motion_controller.py::test_acquiring_during_primitive_refits_yaw_before_commanding`; `tests/integration/test_runtime.py::test_anchor_refits_release_tail_without_restarting_or_scaling_body`; `tests/integration/test_runtime.py::test_autonomous_return_refits_primitive_release_before_next_blend` |
| timeout·returning·centered 상태의 양끝 반복 재생 제한 | `tests/test_motion_controller.py::test_retained_anchor_fits_repeated_primitive_commands` |
| 불감대에서 실제 yaw 유지, 불안전·이동 중 시작은 먼저 안정화 | `tests/test_motion_controller.py::test_deadband_requests_never_command_the_requested_displacement`; `tests/test_motion_controller.py::test_deadband_outside_margin_moves_to_safe_anchor_before_success`; `tests/test_motion_controller.py::test_deadband_retarget_during_return_waits_for_existing_velocity_to_settle` |
| 교체된 모션 결과에 이전 재생 메타데이터 유지 | `tests/test_motion_controller.py::test_replaced_motion_result_retains_outgoing_yaw_scale` |
| 일회성 진단은 성공·실패 최종 응답에서 종료 | `tests/test_orientation_diagnostics.py::test_one_shot_diagnostic_exits_at_first_terminal_event` |
| 작업 조명 충돌 시 방향 상태 유지 | `tests/test_orientation.py::test_coordinator_reports_task_light_conflict_without_acquiring_layer`; `tests/test_motion_controller.py::test_orientation_acquire_rejects_active_task_light` |
| 명시적 복귀는 중앙 도달 후 완료 | `tests/test_orientation.py::test_coordinator_centers_after_return_settle_window_and_latches_speech_id`; `tests/test_motion_controller.py::test_orientation_return_center_completes_after_center_settle` |
| 원격 연결 끊김 시 유지 후 중앙 복귀 | `tests/test_orientation.py::test_coordinator_returns_after_exact_disconnected_hold`; `tests/test_motion_controller.py::test_disconnect_starts_center_return_only_after_orientation_hold` |
| 로컬 경로는 두 번째 하드웨어 제어 주체를 만들지 않음 | `tests/test_motion_middleware.py::test_daemon_binds_both_listeners_before_starting_motion_owner`; `tests/test_motion_middleware.py::test_unix_disconnect_invalidates_unstarted_orientation_request` |
| 로컬 전용 프로토콜과 인증 TCP 분리 | `tests/test_motion_protocol.py::test_remote_decoder_rejects_local_orientation_acquire_with_stable_code`; `tests/test_motion_protocol.py::test_local_decoder_accepts_exact_orientation_payload_without_token` |
| 제어기 인수 전 만료된 로컬 요청은 실행하지 않고 종료 | `tests/test_motion_middleware.py::test_unix_server_forwards_expired_request_to_controller`; `tests/test_motion_controller.py::test_expired_request_never_reaches_runtime` |

저장소 루트에서 전체 소프트웨어 검증을 실행한다.

```bash
git diff --check
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/python -m compileall -q src
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest -q
```

이 검증은 시뮬레이션 모션·프로토콜의 보장을 확인한다. 승인된 설계에서 요구하는 Pi 실기·마이크·오디오·LED·Pi–Jetson 전체 경로의 검증을 대체하지 않는다.
