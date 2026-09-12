# Jetson–Raspberry Pi 모션 미들웨어 설계

## 목적

Jetson Orin Nano 8GB에서 실행되는 인지·음성·비전·시스템 통합 노드가 유선 네트워크를 통해 Raspberry Pi의 모션 런타임을 안전하게 호출하게 한다. Jetson은 행동 이름과 공간 목표만 전달하고, Raspberry Pi는 100 Hz 궤적 생성, 모션 합성, 모터 제한과 안전 종료를 소유한다.

새 모션을 추가할 때 통신 코드나 ROS 인터페이스를 수정하지 않는 것이 핵심 확장성 요구사항이다. 새 CSV와 카탈로그 항목을 추가한 뒤 검증을 통과하면 기존 범용 모션 호출 인터페이스에서 즉시 사용할 수 있어야 한다.

## 결정

Jetson 내부 통합에는 ROS 2를 사용하고 Jetson과 Raspberry Pi 사이에는 버전이 명시된 TCP JSON 프로토콜을 사용한다. Raspberry Pi에는 ROS 2를 설치하지 않는다.

이 혼합 구조는 다음 조건을 반영한다.

- 프로젝트에는 인지, 음성, 비전, 시스템 통합, 모션처럼 서로 다른 주기와 담당자를 가진 여러 프로세스가 있다.
- 모션 실행은 완료 상태와 취소가 필요하며 ROS 2 Action의 목적에 맞는다.
- 얼굴 위치와 소리 방향은 연속 데이터이고 최신 값이 중요하므로 ROS 2 Topic의 목적에 맞는다.
- Raspberry Pi는 Debian 13과 Python 3.13을 사용하고 있으며 ROS 2가 설치되어 있지 않다.
- Raspberry Pi OS는 ROS 2 Tier 3 환경이므로 공식 바이너리 대신 컨테이너나 소스 빌드가 필요하다.
- 모터의 100 Hz 루프는 네트워크, ROS 콜백, Jetson 추론 부하와 격리되어야 한다.

검토한 대안은 다음과 같다.

1. 양쪽 장치에 ROS 2를 설치하면 표준 도구를 전 구간에서 사용할 수 있지만 Pi 배포와 USB 시리얼 접근, Python 환경 관리가 복잡해진다.
2. 모든 통신을 직접 만든 TCP 이벤트 버스로 처리하면 초기 구성은 단순하지만 Jetson의 여러 팀이 취소, 상태 구독, 타입 계약과 관찰 도구를 다시 구현해야 한다.
3. Jetson ROS 2와 Pi TCP 서버를 결합하면 Jetson 팀은 표준 ROS 인터페이스를 사용하고 Pi의 검증된 모터 환경은 유지할 수 있다. 이 설계를 채택한다.

## Jetson 버전 호환성

Jetson Orin Nano 8GB는 실제 JetPack/L4T 버전을 배포 전에 확인한다. 설치 스크립트는 `/etc/nv_tegra_release`, `/etc/os-release`, `dpkg-query -W nvidia-l4t-core` 결과를 기록하고 아래 조합만 허용한다.

| JetPack 계열 | L4T 계열 | 기반 OS | ROS 2 | 정책 |
|---|---|---|---|---|
| 6.x | 36.x | Ubuntu 22.04 | Humble | 지원 |
| 7.x | 39.x | Ubuntu 24.04 | Jazzy | 지원 |

JetPack 7에서 사용할 ROS 2는 장기 지원 기간과 생태계 호환성을 위해 Jazzy로 고정한다. 브리지 코드는 Humble과 Jazzy에 공통으로 존재하는 `rclpy`, Action, Topic, Service API만 사용한다. 배포판별 분기 로직을 애플리케이션 코드에 두지 않는다.

Jetson에는 `ros-base`와 프로젝트의 두 ROS 패키지만 설치한다. RViz, Gazebo, 데스크톱 도구와 상시 rosbag 기록은 기본 배포에서 제외한다. 이는 8GB 메모리를 VLM, STT, TTS와 비전 처리에 우선 배정하기 위한 선택이다.

## 전체 구조

```text
Jetson Orin Nano 8GB
  cognition / voice / vision nodes
              │ ROS 2
              ▼
  lamp_interfaces + lamp_motion_bridge
              │ persistent TCP, NDJSON
              ▼
Raspberry Pi 192.168.100.2
  motion_server → command arbiter → MotionRuntime 100 Hz → FeetechBackend
                      │
                MotionCatalog
```

ROS 2는 Jetson 내부에서만 사용한다. `lamp_motion_bridge`는 ROS 콜백을 Pi 명령으로 변환하고 Pi 상태를 ROS 상태로 변환한다. 브리지는 모터 각도, 시리얼 포트나 PID 레지스터를 노출하지 않는다.

## 구성 요소

### `lamp_interfaces`

ROS 인터페이스 정의만 포함하는 `ament_cmake` 패키지다. 다른 Jetson 팀은 이 패키지에만 의존해 모션을 호출할 수 있다.

- `PlayMotion.action`: 표현 모션 실행, 진행 상태, 결과와 취소
- `PlaceTaskLight.action`: 작업 조명 목표 실행과 취소
- `ListMotions.srv`: 실행 가능한 모션 카탈로그 조회
- `InterruptMotion.srv`: 표현 모션과 작업 모션 즉시 중단
- `MotionStatus.msg`: 연결, 현재 모션, busy, fault, 제어 상태

추적 입력에는 표준 `geometry_msgs/msg/PointStamped`와 `geometry_msgs/msg/Vector3Stamped`를 사용한다.

### `lamp_motion_bridge`

Jetson에서 실행되는 `ament_python` 패키지다. 하나의 ROS 노드가 다음 인터페이스를 제공한다.

| 이름 | 형식 | 용도 |
|---|---|---|
| `/lamp/play_motion` | Action | 이름 기반 표현 모션 실행 |
| `/lamp/place_task_light` | Action | 3D 목표로 조명 배치 |
| `/lamp/interrupt` | Service | 활성 모션 중단 |
| `/lamp/list_motions` | Service | Pi의 검증된 모션 목록 조회 |
| `/lamp/track_point` | Topic | 얼굴 등 3D 위치 추적 |
| `/lamp/track_bearing` | Topic | 소리 방향 추적 |
| `/lamp/motion_status` | Topic | Pi 상태 공개 |

브리지는 Pi와 단일 persistent TCP 연결을 유지한다. 지수 백오프로 재연결하며 연결 상태를 `/lamp/motion_status`에 반영한다. Action 취소는 동일한 요청 ID를 포함한 `motion.cancel` 명령으로 전달한다.

### `motion_server`

Raspberry Pi에서 systemd 서비스로 실행되는 Python 프로세스다. 네트워크 작업과 모터 작업을 별도 실행 문맥으로 분리한다.

- 네트워크 루프: 메시지 해석, 인증, 스키마와 TTL 검증, 응답 전송
- 명령 중재기: 우선순위, 중복 요청, 활성 Action과 최신 추적 목표 관리
- 제어 루프: monotonic deadline에 맞춰 `MotionRuntime.step()`을 100 Hz로 호출
- 안전 종료: sleep 자세로 이동한 뒤 토크 해제

네트워크 루프는 `MotionRuntime`이나 `FeetechBackend`를 직접 호출하지 않는다. 제어 루프만 런타임 상태를 변경하고 모터 명령을 보낼 수 있다.

### `MotionCatalog`

`lelamp_runtime/lelamp/recordings/*.csv`와 `catalog.toml`을 읽어 실행 가능한 모션의 화이트리스트를 만든다. 카탈로그에 활성화된 모션만 로드하며 서버가 모터 버스를 열기 전에 모든 모션을 검증한다.

검증 항목은 다음과 같다.

- 파일 존재, 헤더, 프레임 수와 시간 단조 증가
- 모든 값의 유한성
- 캘리브레이션 변환 후 관절 범위
- 허용 속도, 가속도와 급격한 단일 프레임 변화
- 이름 형식과 중복

검증 실패가 하나라도 있으면 해당 모션을 제외하고 fault에 사유를 기록한다. 기본 sleep 자세와 캘리브레이션 검증이 실패하면 서버 전체가 모터 버스를 열지 않는다.

## 모션 확장 방식

모션 종류마다 ROS Action이나 TCP 메시지를 새로 만들지 않는다. `PlayMotion`은 이름을 받는 범용 Action으로 유지한다.

```text
# Goal
string name
bool replace_current
float32 intensity
uint32 repeat
---
# Result
bool success
string code
string message
---
# Feedback
string state
float32 progress
```

새 모션 추가 절차는 다음으로 고정한다.

1. 녹화 CSV를 recordings 디렉터리에 추가한다.
2. `catalog.toml`에 이름, 파일, 활성화 여부와 선택적인 의미 태그를 추가한다.
3. `python -m motion.validate_catalog`로 안전 검사를 실행한다.
4. 저장소 배포 후 Pi 서비스를 재시작한다.
5. `/lamp/list_motions`에서 새 이름을 확인한다.
6. 기존 `/lamp/play_motion` Action으로 호출한다.

이 절차에서는 ROS 인터페이스, Jetson 브리지와 Pi 서버 코드를 수정하지 않는다. `catalog.toml`을 화이트리스트로 사용해 임시 파일이나 검증되지 않은 녹화가 자동 실행되는 것을 막는다.

## TCP 프로토콜

전용 유선 서브넷에서 TCP 한 연결을 사용한다. 각 메시지는 최대 16 KiB의 UTF-8 JSON 한 줄로 표현한다. 모든 메시지는 다음 공통 필드를 가진다.

```json
{
  "version": 1,
  "id": "0199f3c0-0b5f-7b55-a020-7c8b4012d8ce",
  "type": "motion.play",
  "sent_at_ms": 1789142400000,
  "ttl_ms": 1000,
  "token": "environment-provided-secret",
  "payload": {"name": "nod", "replace_current": true, "intensity": 1.0, "repeat": 1}
}
```

지원 명령은 다음으로 제한한다.

- `motion.play`, `motion.cancel`, `motion.interrupt`, `motion.status`
- `motion.list`
- `track.point`, `track.bearing`, `track.clear`
- `task_light.place`, `task_light.cancel`, `task_light.clear`
- `system.heartbeat`

응답과 비동기 이벤트는 동일한 `id`를 포함한다. 서버는 최근 완료 요청 ID를 제한된 LRU 캐시에 보관해 재연결 후 같은 요청이 중복 실행되는 것을 막는다.

## 데이터 흐름과 중재

표현 모션은 한 번에 하나만 실행한다. 활성 모션이 있을 때 `replace_current=false`인 새 요청은 `busy`로 거절한다. `replace_current=true`이면 기존 모션을 부드럽게 중단하고 새 모션을 시작한다.

추적 입력은 큐에 누적하지 않는다. `track.point`와 `track.bearing` 각각의 최신 유효 값만 보관한다. ROS Topic QoS도 `keep_last=1`로 설정한다. 연결 또는 데이터가 끊기면 TTL 만료 후 TrackLayer를 해제한다.

작업 조명은 표현 모션보다 높은 기존 레이어 우선순위를 유지한다. 말 끊기 이벤트는 표현 모션과 작업 모션을 중단하고 중립 청취 자세로 이동한다.

모션 완료는 CSV 재생 종료만 의미하지 않는다. 블렌더가 primitive 권한을 반환하고 궤적이 허용 오차 내에서 안정된 시점에 Action 결과를 완료로 보낸다.

## 장애와 안전 처리

- 인증 실패, 알 수 없는 타입, 만료된 TTL, 범위를 벗어난 좌표와 유한하지 않은 숫자는 런타임에 전달하지 않는다.
- 메시지 크기 초과 또는 연속 파싱 오류가 발생하면 연결을 닫는다.
- 브리지 연결이 끊기면 Pi는 최신 추적 목표를 즉시 폐기하고 활성 원격 모션을 중단해 안전 대기 자세로 이동한다.
- Pi 제어 루프가 deadline을 놓치면 만료된 슬롯을 건너뛰며 몰아서 전송하지 않는다.
- 하드웨어 fault가 발생하면 새 명령을 거부하고 sleep 진입을 시도한 뒤 토크를 해제한다.
- SIGTERM, Ctrl-C와 예외 종료는 모두 같은 안전 종료 경로를 사용한다.
- Pi는 `192.168.100.2`에 고정하고 서버는 전용 유선 인터페이스 주소에만 bind한다. Jetson 주소 허용 목록과 환경 변수의 공유 토큰을 함께 검사한다.

## 테스트 전략

### 단위 테스트

- 프로토콜 스키마, TTL, 인증과 메시지 크기
- 요청 ID 중복 제거
- 모션 카탈로그 검색과 안전 검증
- Action 상태 변환과 취소
- tracking 최신 값 병합
- 연결 끊김과 fault 상태 전이

### 통합 테스트

- `NullBackend` 기반 Pi 서버와 순수 Python Jetson 클라이언트 왕복
- ROS 2 Action/Topic/Service와 브리지 왕복
- 서버 재시작, 네트워크 단절과 재연결
- 장시간 100 Hz 루프에서 네트워크 부하에 따른 deadline miss 측정
- MuJoCo에서 카탈로그 전체 모션 호출 및 sleep 복귀

### 실기기 검증

1. 토크를 켜지 않은 상태에서 버전, 인증, 목록과 상태 조회를 확인한다.
2. sleep 자세에서 `nod` 한 개를 실행하고 취소와 복귀를 확인한다.
3. 추적 입력 TTL과 연결 단절 복귀를 확인한다.
4. 카탈로그 전체 모션을 한 번씩 실행한다.
5. 테스트 종료 후 sleep 자세와 토크 해제를 확인한다.

## 배포

Pi에는 `talking-lamp-motion.service`를 설치한다. 서비스는 네트워크 준비 후 시작하고 실패 시 제한적으로 재시작하며, 종료 시 충분한 park 시간을 허용한다.

Jetson에는 다음 두 패키지를 colcon workspace에 설치한다.

```text
src/
  lamp_interfaces/
  lamp_motion_bridge/
```

launch 파일은 Pi 주소, TCP 포트, 토큰 환경 변수 이름, tracking TTL과 재연결 한계를 인자로 받는다. JetPack 판별 스크립트가 Humble 또는 Jazzy 환경을 선택하고 잘못된 조합에서는 설치를 중단해 구체적인 수정 명령을 표시한다.

## 완료 기준

- Jetson에서 ROS 2 Action으로 이름 기반 모션을 실행하고 완료·취소 결과를 받을 수 있다.
- 새 CSV와 카탈로그 항목 추가만으로 새 모션이 목록과 실행 경로에 나타난다.
- 비전과 음성 추적 Topic이 Pi의 TrackLayer에 최신 값 방식으로 전달된다.
- Jetson 재시작과 유선 단절이 Pi의 100 Hz 루프를 막지 않는다.
- 알 수 없는 모션, 만료 메시지와 중복 요청이 실행되지 않는다.
- 모든 정상·오류 종료 경로에서 sleep 복귀와 토크 해제가 유지된다.
- 같은 소스가 JetPack 6/Humble과 JetPack 7/Jazzy에서 빌드된다.

## 근거 자료

- NVIDIA JetPack 6.2.1: Ubuntu 22.04 기반 Jetson Linux 36.4.4 — https://developer.nvidia.com/embedded/jetpack-sdk-621
- NVIDIA JetPack 다운로드: JetPack 7.2.1, Orin 계열 지원, Ubuntu 24.04 기반 — https://developer.nvidia.com/embedded/jetpack/downloads
- ROS 2 Topic, Service, Action 선택 지침 — https://docs.ros.org/en/jazzy/How-To-Guides/Topics-Services-Actions.html
- ROS 2 Actions 개념 — https://docs.ros.org/en/rolling/Concepts/Basic/About-Actions.html
- Raspberry Pi의 ROS 2 설치와 지원 등급 — https://docs.ros.org/en/ros2_documentation/kilted/How-To-Guides/Installing-on-Raspberry-Pi.html
