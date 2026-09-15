# Raspberry Pi 오디오·DOA·LED와 Jetson ROS 통합 설계

작성일: 2026-09-15  
대상 저장소: Talking Lamp  
상태: 사용자 승인 완료

## 1. 목적

Talking Lamp의 변경된 하드웨어를 기존 모션 시스템과 통합한다.

- 기존 원형 LED 대신 모델 세대와 배선 순서가 아직 확인되지 않은 `WS2812B-64` 8×8 사각 매트릭스를 사용한다.
- 기존 ReSpeaker Pi HAT 대신 `reSpeaker Flex XVF3800 Linear-4`를 사용한다.
- XVF3800과 스피커는 Raspberry Pi 5가 USB로 소유한다.
- Raspberry Pi는 마이크 음성을 유선랜으로 Jetson에 보내고 Jetson의 TTS 음성을 받아 XVF3800의 스피커 출력으로 재생한다.
- Raspberry Pi는 XVF3800의 DOA만 로컬에서 사용해 `base_yaw`를 먼저 사용자 방향으로 정렬한다.
- Jetson은 정렬 완료 뒤 ROS 2로 응답 음성과 표현 모션을 실행한다.
- Jetson은 응답과 모션이 모두 끝난 뒤 중앙 복귀를 요청하고, 중앙 복귀 완료 뒤 대기 모션을 요청한다.

이 설계는 Raspberry Pi에 ROS 2를 설치하지 않는다. Jetson 내부에는 표준 ROS 2 인터페이스를 제공하고, Jetson과 Pi 사이는 오디오 전용 RTP/UDP와 인증된 제어 TCP로 나눈다.

## 2. 현재 상태

기준 구현은 `codex/jetson-pi-middleware` 브랜치다.

- 5축 `MotionRuntime`, 계층형 블렌더, 100 Hz 궤적 생성기와 Feetech 백엔드가 구현되어 있다.
- Pi의 인증 TCP 모션 서버와 명령 수명, 하트비트, 연결 단절 안전 대기가 구현되어 있다.
- 이름 기반 모션 카탈로그와 ROS 2 모션 인터페이스 정의가 구현되어 있다.
- Jetson의 `lamp_motion_bridge` 구현은 아직 남아 있다.
- XVF3800 입력·출력·DOA, 네트워크 오디오, WS2812B-64, 방향 정렬 계층은 구현되어 있지 않다.
- 설계 작성 직전 기존 자동 테스트 전체가 통과했다.

현재 최상위 체크아웃 `feat/motion-stack-e2e-sim`은 원격보다 4개 커밋 앞서고 3개 커밋 뒤에 있다. 실제 후속 구현은 위의 깨끗한 미들웨어 작업 트리를 기준으로 진행하고, 병합 전에 원격 변경과 정리한다.

## 3. 확정된 하드웨어 토폴로지

### 3.1 Raspberry Pi 5

Pi가 다음 장치를 소유한다.

- Feetech 5축 서보 드라이버
- reSpeaker Flex XVF3800 Linear-4 USB 오디오 캡처·재생·USB 제어
- XVF3800의 JST 또는 3.5 mm 출력에 연결된 스피커
- WS2812B-64의 데이터와 전원. 데이터는 programmable 3.3 V GPIO, 전원은 40-pin 헤더의 5 V·GND 핀을 사용한다.
- 유선랜을 통한 Jetson 연결

XVF3800의 Linear-4 마이크 배열은 회전하는 헤드가 아니라 고정 베이스에 장착한다. 따라서 DOA는 `base_yaw`가 움직여도 변하지 않는 베이스 고정 좌표다.

### 3.2 Jetson

Jetson이 다음 기능을 소유한다.

- STT, 대화 판단, TTS
- 응답에 맞는 이름 기반 표현 모션 선택
- TTS와 모션의 동시 실행 및 완료 결합
- 중앙 복귀와 대기 모션 호출
- 다른 팀 모듈에 제공하는 ROS 2 API

### 3.3 확인된 XVF3800 특성

공식 자료 기준으로 Flex는 USB UAC 2.0 캡처·재생, AEC, beamforming, VAD, DOA와 USB 제어를 지원한다. Linear-4는 전방 약 180°를 대상으로 하고 후방 소리를 억제한다. USB 제어의 `DOA_VALUE`는 0–359° 방향과 speech-detected 값을 반환한다.

사용할 펌웨어는 Linear-4 형상과 일치하는 USB 이미지여야 한다. 펌웨어는 런타임이 자동 변경하지 않으며 설치·검증 절차에서 운영자가 명시적으로 확인한다.

## 4. 시스템 구조

```text
Raspberry Pi 5                                      Jetson Orin Nano

XVF3800 USB capture ─┐                         ┌─> /lamp/audio/capture -> STT
XVF3800 USB control ─┼─ lamp_device_server ────┤
WS2812B-64 GPIO ─────┘  :8766 control/status   ├─< /lamp/audio/playback_frames <- TTS
                           RTP/UDP audio  ──────┤
                                  │            │   lamp_device_bridge (ROS 2)
                                  │            │
                  local Unix RPC  │            │   lamp_motion_bridge (ROS 2)
                                  v            │
motion_server :8765 -> MotionController <─────┘
                        -> 100 Hz MotionRuntime
                        -> Feetech 5축
```

Pi에는 두 개의 독립된 systemd 서비스를 둔다.

1. `talking-lamp-motion.service`
   - 기존 100 Hz 모션 소유자다.
   - Jetson 모션 브리지의 인증 TCP 연결 하나를 소유한다.
   - 로컬 방향 정렬 명령은 네트워크 TCP와 분리된 Unix 소켓으로 받는다.
2. `talking-lamp-device.service`
   - XVF3800, GStreamer 오디오 파이프라인과 WS2812B를 소유한다.
   - DOA 안정화 뒤 Unix 소켓으로 motion service에 방향 정렬을 요청하고 정렬 상태를 돌려받는다.
   - Jetson device bridge와 별도의 인증 TCP 연결을 유지한다.

방향 정렬의 ROS 소유자는 `lamp_device_bridge`다. 이 브리지가 Pi device service의 발화·방향 이벤트를 `/lamp/orientation_status`로 내보내고, `/lamp/return_center` 요청을 device TCP와 로컬 Unix RPC를 거쳐 motion service에 전달한다. 기존 `lamp_motion_bridge`는 이름 기반 표현 모션과 작업 조명만 소유한다. 두 ROS 노드가 같은 Pi TCP 연결을 경쟁하지 않는다.

두 프로세스를 나누는 이유는 ALSA, libusb, GStreamer 또는 LED 드라이버가 지연되거나 실패해도 100 Hz 모션 스레드를 막지 않기 위해서다. Unix 소켓의 요청도 모션 런타임을 직접 변경하지 않고 기존 `MotionController` 메일박스를 거친다.

## 5. DOA 안정화와 좌표 변환

### 5.1 입력 수집

- XVF3800의 `DOA_VALUE`를 기본 20 Hz로 읽는다.
- `speech_detected=1`의 상승 에지에서 새 `speech_id`를 UUID로 만든다.
- 처음 400 ms 동안 최소 6개의 유효 방향 표본을 모은다.
- 표본이 부족하거나 흩어져 있으면 발화 시작 뒤 최대 1초까지 수집을 연장한다.
- 1초 안에 조건을 만족하지 못하면 `rejected`로 끝내고 회전하지 않는다.

방향 표본은 일반 산술 평균을 사용하지 않는다. 359°와 1°가 180°로 잘못 평균되는 일을 막기 위해 원형 거리를 사용한다. 후보 표본 중 다른 표본까지의 원형 거리 합이 가장 작은 값을 원형 중앙값으로 택한다. 중앙 원형 절대편차가 기본 12° 이하일 때 안정된 값으로 인정한다. 임계값은 설정 파일에서 바꿀 수 있다.

### 5.2 좌표 변환

설치 보정 파라미터는 다음과 같다.

- `doa_zero_deg`: 마이크 정면이 램프 정면과 이루는 설치 오프셋
- `doa_direction_sign`: DOA 증가 방향과 `base_yaw` 양의 방향의 관계, `-1` 또는 `1`
- `front_half_angle_deg`: Linear-4의 허용 전방 반평면, 기본 90°
- `yaw_margin_deg`: 기계 한계에서 확보할 여유, 기본 5°

변환식은 다음과 같다.

```text
relative = wrap_to_pi(doa_direction_sign * radians(doa - doa_zero_deg))
target   = clamp(center_yaw + relative,
                 yaw_min + margin,
                 yaw_max - margin)
```

`abs(relative)`가 허용 전방 반평면을 넘으면 후방 입력으로 거부한다. 제한으로 목표가 바뀌면 정렬은 수행하되 `clamped=true`를 상태에 포함한다.

현재 yaw와 목표 차이가 5° 이하면 모터 이동 없이 정렬 완료로 처리한다.

### 5.3 정렬 완료

- 목표 오차 3° 이하
- `base_yaw` 속도 0.08 rad/s 이하
- 두 조건이 연속 150 ms 유지

위 조건을 만족하면 `aligned`다. 정렬 시작 뒤 2초 안에 만족하지 못하면 `timeout`으로 처리한다. timeout 시 현재 위치를 강제로 성공으로 보고하지 않는다.

## 6. 방향 기준 모션

새 `BaseYawOrientationLayer`를 추가한다.

| 계층 | 우선순위 | 역할 |
| --- | ---: | --- |
| L0 idle | 0 | 호흡·미세 움직임 |
| L1 track | 10 | 얼굴·점 추적 |
| orientation | 15 | 안정화된 사용자 방향의 절대 base_yaw 기준점 |
| L2 primitive | 20 | 기준점 위에서 재생하는 상대 표현 모션 |
| L3 task light | 30 | 작업 조명 절대 자세 |

orientation은 `base_yaw` 하나만 소유한다. 다른 4개 관절은 기존 계층이 제어한다. 정렬 완료 뒤에도 Jetson이 중앙 복귀를 요청할 때까지 목표를 유지한다.

표현 모션 CSV는 상대 오프셋이므로 새 기준 yaw 위에 그대로 합성한다. 단, `anchor_yaw + primitive_yaw_offset`이 안전 범위를 넘을 수 있다. 모션 시작 전에 해당 클립의 양·음 yaw 최대 오프셋을 검사하고, 안전 범위 안에 들어오도록 yaw 성분만 0–1 범위로 축소한다. 다른 관절과 요청된 전체 intensity는 유지한다. 실제 적용된 yaw 배율은 Action feedback과 결과에 포함한다.

TaskLightLayer가 활성 상태면 DOA 정렬을 시작하지 않고 `blocked_by_task_light`를 반환한다. Jetson이 작업 조명을 해제한 뒤 새 발화에 대해 다시 정렬하도록 한다.

## 7. 상태머신과 상호작용 순서

### 7.1 Pi 방향 상태

```text
idle
  -> collecting
  -> rejected | orienting
  -> timeout | aligned
  -> returning
  -> centered
  -> idle
```

각 상태에는 다음 정보가 있다.

- `speech_id`
- 원본 DOA와 보정된 상대 방향
- 목표 yaw와 현재 yaw
- `clamped`
- 상태 코드와 설명
- 상태 발생 시각

같은 `speech_id`에서는 방향을 한 번만 확정한다. 정렬 중 추가 DOA 표본이 목표를 바꾸지 않는다.

### 7.2 정상 대화 턴

1. Pi는 마이크 오디오를 계속 Jetson으로 보낸다.
2. Pi가 발화를 감지하고 DOA를 안정화한다.
3. Pi가 base_yaw를 정렬하고 목표를 고정한다.
4. Jetson은 오디오를 STT와 대화 처리에 사용한다.
5. Jetson은 응답 준비와 `aligned`를 모두 기다린다.
6. Jetson은 TTS 재생과 이름 기반 표현 모션을 동시에 시작한다.
7. Jetson은 두 Action이 모두 terminal 상태가 될 때까지 기다린다.
8. 둘 다 성공하면 `/lamp/return_center`를 호출한다.
9. `centered` 결과를 받은 뒤 Jetson이 이름 기반 대기 모션을 호출한다.

중앙 복귀 요청은 표현 모션이 busy인 동안 `busy`로 거부한다. Jetson이 두 완료를 잘못 결합해 조기 복귀시키는 오류를 숨기지 않기 위해 Pi가 임의로 모션을 끊지 않는다.

### 7.3 barge-in

TTS 도중 사용자가 말하면 Jetson의 기존 barge-in 정책이 우선한다.

1. Jetson이 재생 Action과 표현 모션을 취소한다.
2. Jetson이 모션 interrupt를 보낸다.
3. 현재 방향 기준점은 유지한다.
4. 다음 발화의 DOA를 새로 안정화한다.
5. 새 방향이 확정되면 기준점을 교체한다.

## 8. 네트워크 오디오

### 8.1 전송 분리

오디오는 제어 TCP와 분리한다. 대용량 오디오 때문에 정지·취소·상태 명령이 지연되는 head-of-line blocking을 피하기 위해서다.

- Pi → Jetson: 처리된 마이크 채널, 16 kHz mono Opus over RTP/UDP
- Jetson → Pi: TTS Opus over RTP/UDP, Pi에서 XVF3800의 재생률로 변환
- 기본 포트: capture 5004/UDP, playback 5006/UDP
- RTP jitter buffer 기본값: 40 ms
- 늦은 패킷은 버리고 재전송하지 않는다.
- 스트림마다 UUID `stream_id`, RTP sequence와 timestamp를 사용한다.

GStreamer가 ALSA 연결, 변환, Opus, RTP, jitter buffer를 담당한다. 애플리케이션 코드는 오디오 코덱과 패킷 재조립을 직접 구현하지 않는다.

XVF3800의 캡처와 재생은 동일 USB UAC 장치로 열어 디지털 far-end 재생 신호가 보드의 AEC 경로에 들어가게 한다. ALSA 카드 번호처럼 재부팅 시 변할 수 있는 값은 고정하지 않고 udev 속성이나 안정된 장치 이름으로 찾는다.

### 8.2 ROS 스트리밍 계약

`AudioFrame.msg`:

```text
builtin_interfaces/Time stamp
string stream_id
string speech_id
uint64 sequence
uint32 sample_rate
uint8 channels
string encoding
uint8[] data
bool end_of_stream
```

초기 지원 encoding은 `pcm_s16le` 하나다. 프레임은 20 ms를 기본으로 하고 크기, 채널, sample rate와 sequence를 브리지 경계에서 검증한다. capture frame의 `speech_id`는 VAD 밖에서는 빈 문자열이고, 발화 구간에서는 Pi가 생성한 UUID다. playback frame은 `speech_id`를 비워 둔다.

Pi device service는 VAD 상승·하강 시 device TCP로 `audio.activity` 이벤트를 보내며 `speech_id`, RTP timestamp와 단조 증가 event sequence를 포함한다. RTP의 SSRC는 `stream_id`에 대응한다. Jetson device bridge는 이 이벤트와 RTP timestamp를 결합해 capture `AudioFrame`에 같은 `speech_id`를 넣는다. 따라서 STT 결과와 `/lamp/orientation_status`가 동일한 발화를 가리킬 수 있다. TCP 이벤트가 뒤늦게 도착할 수 있으므로 bridge는 기본 200 ms의 작은 capture pre-roll을 보관한다.

`/lamp/audio/capture`는 Pi에서 온 음성을 발행한다. `/lamp/audio/playback_frames`는 TTS 프레임을 받는다. `/lamp/play_audio` Action은 스트림 시작과 종료를 소유한다. 별도 Topic을 쓰는 이유는 TTS가 전체 문장을 생성할 때까지 기다리지 않고 첫 프레임부터 재생하기 위해서다.

`PlayAudio.action` goal은 stream ID, sample rate, channels, encoding을 가진다. producer는 goal accepted 뒤 같은 stream ID의 프레임을 보내고 마지막 프레임에 `end_of_stream=true`를 설정한다. Pi의 ALSA 버퍼가 drain된 뒤에만 Action result가 성공한다. feedback은 수신·재생 sequence와 buffered milliseconds를 제공한다.

## 9. ROS 2 인터페이스

### 9.1 유지할 기존 인터페이스

- `/lamp/play_motion` Action
- `/lamp/place_task_light` Action
- `/lamp/interrupt_motion` Service
- `/lamp/list_motions` Service
- `/lamp/track_point` Topic
- `/lamp/track_bearing` Topic
- `/lamp/motion_status` Topic

### 9.2 추가할 인터페이스

- `/lamp/audio/capture` Topic: `AudioFrame`
- `/lamp/audio/playback_frames` Topic: `AudioFrame`
- `/lamp/play_audio` Action: `PlayAudio`
- `/lamp/audio_status` Topic: `AudioStatus`
- `/lamp/orientation_status` Topic: `OrientationStatus`
- `/lamp/return_center` Action: `ReturnCenter`
- `/lamp/led/frame` Topic: `sensor_msgs/Image`, `rgb8`, 8×8
- `/lamp/led/set_solid` Service: `SetLedSolid`
- `/lamp/led/clear` Service: `std_srvs/Trigger`
- `/lamp/led/status` Topic: `LedStatus`

연속 최신값인 상태와 LED frame Topic은 `KEEP_LAST`를 사용한다. LED frame과 capture audio는 오래된 값보다 최신값이 중요하므로 bounded queue를 사용한다. Action과 Service는 ROS 기본 reliable 정책을 사용한다.

## 10. WS2812B-64

### 10.1 논리 좌표

Jetson은 항상 좌상단 원점, 행 우선 8×8 `rgb8` 이미지를 보낸다. Pi는 다음 설치 파라미터로 물리 인덱스를 계산한다.

- `layout`: `row_major` 또는 `serpentine`
- `origin`: `top_left`, `top_right`, `bottom_left`, `bottom_right`
- `rotation_deg`: `0`, `90`, `180`, `270`
- `color_order`: 기본 `GRB`, 필요 시 변경
- `gpio_pin`: 기본 12
- `max_brightness`: 실기 시험으로 확정

정확한 모델을 알 수 없으므로 매핑을 코드에 고정하지 않는다. 최초 연결 때 한 픽셀, 행, 열, 체커보드 테스트를 순서대로 표시해 설정을 확정한다.

### 10.2 전원 정책

사용 가능한 별도 5 V 전원이 없으므로 Raspberry Pi 5의 정품 전원과 40-pin 헤더의 5 V 핀을 대상으로 실측한다. 이는 자동으로 안전하다고 간주하는 결정이 아니다. LED 전원을 programmable GPIO나 3.3 V 핀에서 공급해서는 안 된다.

- 기본 미검증 밝기는 10%다.
- `max_brightness`를 넘는 모든 요청은 clamp한다.
- 전체 백색 시험은 10%, 25%, 50%, 75%, 100% 순서로만 진행한다.
- 각 단계는 LED만이 아니라 XVF3800 full-duplex 오디오와 Pi 부하를 함께 켠 상태로 검사한다.
- 새 undervoltage/throttling bit, USB 재연결, 재부팅, 5 V 전압 저하 또는 비정상 발열이 있으면 즉시 clear하고 시험을 중단한다.
- 실패 단계의 직전 통과값을 임시 상한으로 기록한다. 최종 상한은 반복 시험 뒤 운영자가 명시적으로 승인한다.
- 시작, 정상 종료, 프로세스 fault와 Jetson 연결 단절 시 LED를 clear한다.

Pi GPIO의 3.3 V 데이터가 실제 미상 모듈에서 불안정하면 74AHCT 계열 레벨 시프터가 필요하다. 소프트웨어가 이 전기적 문제를 보정하려고 재전송 루프를 만들지 않는다.

## 11. Pi 제어 프로토콜

기존 motion TCP `:8765`는 모션 브리지 한 연결만 허용하는 정책을 유지한다. 장치 제어용 `:8766`을 별도로 둔다.

추가 로컬 motion Unix RPC:

- `orientation.acquire` — 목표 yaw와 speech ID
- `orientation.return_center`
- `orientation.status`

device 명령:

- `audio.play.start`, `audio.play.stop`, `audio.status`
- `orientation.return_center`, `orientation.status`
- `led.frame`, `led.solid`, `led.clear`, `led.status`
- `device.status`, `system.heartbeat`

인증 뒤 Pi가 Jetson으로 밀어 보내는 device event는 `audio.activity`, `orientation.status`, `audio.status`, `led.status`다. event에는 요청 ID 대신 단조 증가 event sequence가 있고, 재연결 뒤 새 session ID와 함께 sequence가 다시 시작한다. Jetson은 이전 session의 이벤트를 폐기한다.

두 TCP 서버는 같은 token 정책, canonical UUID, receive-time TTL, 정확한 payload 필드 검사와 비재생성 원칙을 사용한다. 오디오·LED 클라이언트 연결 단절이 motion controller 연결 단절로 해석되지 않도록 포트를 분리한다.

Pi 로컬 Unix 소켓은 `/run/talking-lamp/motion-control.sock`을 사용하고 `talking-lamp` 전용 그룹만 접근할 수 있게 한다. 파일 모드는 `0660`이다.

## 12. Jetson 구현 지침

### 12.1 패키지

```text
jetson_ws/src/
├── lamp_interfaces/       ROS msg/srv/action 정의
├── lamp_motion_bridge/    기존 motion TCP ↔ ROS 2
├── lamp_device_bridge/    audio RTP, device TCP, LED ↔ ROS 2
└── lamp_interaction/      팀 B가 사용할 턴 오케스트레이션 예제와 helper
```

`lamp_motion_bridge`와 `lamp_device_bridge`의 transport/adapter 모듈은 ROS import를 하지 않는다. 소켓, framing, 재연결, request correlation, server-pushed event ordering과 데이터 검증을 일반 Python 테스트에서 검증한다. ROS node는 타입 변환과 executor 연동만 담당한다.

### 12.2 launch 파라미터

- `pi_host`
- `motion_port=8765`
- `device_port=8766`
- `capture_rtp_port=5004`
- `playback_rtp_port=5006`
- `token_env=TALKING_LAMP_TOKEN`
- `audio_frame_ms=20`
- `jitter_buffer_ms=40`
- `orientation_wait_timeout_sec`
- `return_center_timeout_sec`

IP와 token은 소스 코드에 저장하지 않는다.

### 12.3 오케스트레이션 예제

Jetson 문서와 helper는 다음 순서를 제공한다.

```python
response = await conversation.consume(audio_capture)
orientation = await wait_for_orientation(response.speech_id)
if orientation.state != "aligned":
    handle_orientation_failure(orientation)

audio_task = play_tts_stream(response.tts)
motion_task = play_motion(response.motion_name)
audio_result, motion_result = await gather(audio_task, motion_task)

if audio_result.success and motion_result.success:
    await return_center()
    await play_motion("idle")
```

실제 helper는 취소, timeout과 예외를 처리하며 위 순서를 건너뛰지 않는다. TTS와 모션 중 하나가 실패하면 자동으로 중앙 복귀·idle을 실행하지 않고 정책 계층에 실패를 알린다. 운영 정책은 필요하면 interrupt 뒤 복귀를 명시적으로 선택한다.

## 13. 장애 처리

### 13.1 유선랜 단절

- Pi는 새 명령과 새 재생 오디오를 받지 않는다.
- 진행 중인 네트워크 TTS를 중단하고 표현 모션을 interrupt한다.
- 방향 기준점은 10초 유지한 뒤 저속 중앙 복귀한다.
- 재연결 뒤 이전 요청, TTS 또는 모션을 자동 재실행하지 않는다.

### 13.2 XVF3800 분리

- audio와 DOA를 fault로 표시한다.
- 모션과 LED 서비스는 계속 유지한다.
- 지수 백오프로 USB 장치를 다시 찾는다.
- 펌웨어는 자동으로 쓰지 않는다.
- 재연결 뒤 새로운 stream ID만 허용한다.

### 13.3 오디오 손실

- 짧은 손실은 jitter buffer/Opus concealment 또는 무음으로 처리한다.
- 늦은 packet은 버린다.
- 장시간 수신이 없으면 해당 playback Action을 실패시키고 ALSA를 drain/close한다.

### 13.4 입력 검증

- non-finite 각도, 잘못된 RGB 크기·encoding, 과도한 오디오 frame, 알 수 없는 stream ID는 하드웨어 접근 전에 거부한다.
- invalid request가 모션 owner thread에 도달하지 않게 한다.
- fault와 reject는 machine-readable code와 사람이 읽을 message를 함께 반환한다.

## 14. 네트워크와 보안

- Pi와 Jetson은 전용 유선 서브넷과 고정 IP를 사용한다.
- TCP token은 `/etc/talking-lamp/*.env`의 mode `0600` 파일에서 읽는다.
- Pi의 TCP allowlist에는 Jetson 고정 IP만 둔다.
- RTP UDP 포트는 유선 인터페이스에만 bind하고 방화벽에서 Jetson IP만 허용한다.
- 오디오 RTP 자체 암호화는 초기 로컬 전용 범위에서 제외한다. 다른 네트워크와 연결되면 SRTP 또는 터널을 별도 설계한다.

## 15. 테스트 전략

### 15.1 단위 테스트

- 359°/1° 원형 통계
- 표본 부족, 큰 분산, 후방 입력 거부
- zero offset, direction sign, clamp와 deadband
- collecting → orienting → aligned → returning → centered
- 정렬 timeout과 task-light 충돌
- 방향 기준 primitive yaw 축소
- 8×8 row-major, serpentine, origin과 회전 조합
- RGB frame과 brightness clamp
- audio sequence, duplicate, stale frame와 EOS
- 새 motion/device protocol의 strict schema

### 15.2 소프트웨어 통합 테스트

- fake XVF, ALSA, PixelStrip로 device service 실행
- Unix RPC를 포함한 device-to-motion loopback
- Pi TCP 서버와 Jetson 순수 Python transport 왕복
- RTP loopback과 packet loss/reordering
- TCP 재연결, USB 분리와 오래된 요청 비재생
- ROS interface build와 node contract
- 기존 전체 motion 회귀 테스트

### 15.3 Pi 실기 검증

- USB VID/PID, Linear-4 USB 펌웨어와 버전 기록
- `arecord`/`aplay` 및 full-duplex AEC 확인
- 알려진 각도의 음원으로 DOA zero/sign/분산 측정
- base_yaw 정렬 오차 3° 이내
- LED 픽셀·행·열·체커보드 매핑
- LED 10→25→50→75→100% 전체 백색 단계 시험
- 각 단계의 `vcgencmd get_throttled`, USB kernel log, 5 V 전압, 재부팅과 발열 기록
- 오디오·LED 부하 중 zero out-of-range command와 bounded deadline miss 확인

### 15.4 Pi–Jetson 종단간 검증

- 실제 유선랜 capture audio 수신
- aligned 상태와 STT 응답 결합
- TTS와 표현 모션 동시 실행
- 두 Action이 모두 끝나기 전 중앙 복귀 금지
- centered 뒤에만 idle 시작
- LAN 분리·재연결 뒤 stale audio/motion 비재생

## 16. 완료 기준

- Jetson 개발자는 Pi 내부 구현을 import하지 않고 ROS 인터페이스만으로 마이크, 스피커, 방향 상태, LED와 모션을 사용할 수 있다.
- Pi는 발화 초기 400 ms 안정화 뒤 한 번만 방향을 확정한다.
- 정렬 성공 시 base_yaw 오차가 3° 이내고, Jetson의 명시적 복귀까지 방향 기준점을 유지한다.
- 표현 모션은 정렬 기준점에서 재생되고 모든 관절 명령이 안전 범위 안에 있다.
- Pi의 ALSA playback drain 이후에만 TTS Action이 완료된다.
- TTS와 표현 모션 완료 뒤 중앙 복귀, centered 뒤 idle이라는 순서가 자동 테스트와 실기에서 확인된다.
- LED는 실측 승인된 밝기 상한을 넘지 않는다.
- 오디오·LED·네트워크 장애가 100 Hz 모션 owner를 블록하지 않는다.
- 기존 모션 테스트와 새 단위·통합 테스트가 모두 통과한다.

## 17. 범위에서 제외

- Raspberry Pi에 ROS 2 설치
- XVF3800 펌웨어 자동 업데이트
- 마이크 원시 4/6채널을 Jetson으로 전송하는 기능
- 미확인 WS2812B 기판의 회로 역설계
- 인터넷 구간 오디오 전송
- 새 PCB 또는 별도 LED 전원 설계
- STT, TTS, 대화 모델 자체 구현

## 18. 구현 단계

하나의 구현 계획 안에서 다음 순서로 나눈다.

1. 방향 정렬 코어, BaseYawOrientationLayer와 motion protocol
2. Pi device service의 하드웨어 추상화, XVF control과 LED
3. RTP 오디오 transport와 재생 완료 의미
4. ROS interface 확장과 Jetson motion/device bridge
5. Jetson 상호작용 helper와 배포 문서
6. Pi 실기 검증 뒤 Pi–Jetson E2E 검증

## 19. 참고 자료

- Seeed Studio, reSpeaker Flex 소개와 Linear-4/USB/스피커 사양: https://wiki.seeedstudio.com/respeaker_flex_introduction/
- 공식 reSpeaker Flex 저장소: https://github.com/respeaker/reSpeaker_Flex
- 공식 Python USB 제어의 `DOA_VALUE`: https://github.com/respeaker/reSpeaker_Flex/blob/main/python_control/xvf_host.py
- Worldsemi WS2812B 제품군: https://www.world-semi.com/ws2812-family/page1/
- ROS 2 Topic, Service, Action 선택 지침: https://docs.ros.org/en/jazzy/How-To-Guides/Topics-Services-Actions.html
