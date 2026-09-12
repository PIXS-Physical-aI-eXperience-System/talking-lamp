# Jetson–Raspberry Pi 모션 연동 개발 계획

## 결정 사항

Jetson Orin Nano 8GB 내부 구성 요소는 ROS 2로 연결하고, Jetson과 Raspberry Pi 사이는 전용 유선 TCP로 연결한다.

- Jetson: 인지·음성·비전·전역 상태 관리와 ROS 2 인터페이스 담당
- Raspberry Pi: 100 Hz 모션 루프·IK·블렌더·궤적·서보·안전 정지 담당
- 전송 데이터: 모션 이름, 추적할 3D 위치·방향, 작업 조명 목표, 중단 요청
- 전송하지 않는 데이터: 관절별 원시 각도, 시리얼 포트, 캘리브레이션 및 PID 레지스터

Raspberry Pi OS에 ROS 2를 추가하면 컨테이너 또는 소스 빌드와 모터 USB 접근 관리가 필요하다. 따라서 Pi의 검증된 Python 모션 환경은 유지하고 Jetson에만 ROS 2를 둔다.

## 버전 기준

설치 전에 Jetson의 L4T와 Ubuntu 버전을 확인한다.

| JetPack/L4T | Ubuntu | 적용할 ROS 2 |
|---|---|---|
| JetPack 6.x / L4T 36.x | 22.04 | Humble |
| JetPack 7.x / L4T 39.x | 24.04 | Jazzy |

브리지 코드는 Humble과 Jazzy에 공통으로 있는 `rclpy`, Action, Topic, Service API만 사용한다. Jetson Orin Nano 8GB의 메모리를 AI 추론에 우선 배정하기 위해 `ros-base`만 설치한다.

## Raspberry Pi에서 개발할 항목

### 1. 모션 카탈로그

`recordings/*.csv`와 `catalog.toml`을 읽어 실행 가능한 모션 목록을 만든다. 카탈로그에 활성화된 모션만 실행한다.

- CSV 헤더, 시간 순서, 유한값, 프레임 수 검사
- 캘리브레이션 변환 후 관절 범위·속도·가속도 검사
- 검증 실패 시 모터 연결 전 서버 시작 중단
- `/motion.list` 요청에 현재 허용된 이름 반환

새 모션 추가는 CSV 추가, `catalog.toml` 등록, 검증 명령 실행으로 끝나게 한다. 통신 코드와 ROS 메시지는 변경하지 않는다.

### 2. 모션 명령 중재기

네트워크 명령을 바로 모터에 전달하지 않고 제한된 명령 큐를 거쳐 100 Hz 제어 루프에서 처리한다.

- 표현 모션은 한 번에 하나만 실행
- 새 모션이 기존 모션을 대체할지 요청 값으로 결정
- 얼굴·소리 추적 값은 큐에 쌓지 않고 종류별 최신 값 하나만 유지
- 요청 TTL이 지나면 실행하지 않음
- 요청 ID를 기억해 재연결 후 같은 모션이 중복 실행되는 것을 방지
- 네트워크 처리와 모터 제어를 별도 실행 문맥으로 분리

### 3. TCP 모션 서버

Pi의 전용 유선 주소 `192.168.100.2:8765`에서 UTF-8 NDJSON 메시지를 받는다.

- 프로토콜 버전, 요청 ID, 명령 타입, TTL, 공유 토큰, payload 검증
- 메시지 최대 크기 16 KiB
- 한 개의 Jetson 제어 연결만 허용
- 500 ms 하트비트, 2.5초 연결 만료
- 요청 접수와 최종 완료·취소·실패를 분리해서 응답
- 연결이 끊기면 추적 목표를 지우고 활성 원격 모션을 중단한 뒤 안전 대기

### 4. 장치 서비스와 안전 종료

Pi 모션 서버를 systemd 서비스로 상시 실행한다.

- 시작 순서: 설정·토큰 검사 → 모션 카탈로그 검사 → Ruckig 확인 → 모터 연결
- 종료 순서: 새 요청 차단 → 활성 모션 중단 → sleep 자세 → 토크 해제
- 정상 종료, 예외, SIGINT, SIGTERM에서 같은 종료 경로 사용
- 연속 실패 시 무한 재시작하지 않도록 systemd 시작 횟수 제한
- 모터를 열지 않는 mock 모드로 네트워크만 먼저 검증

## Jetson에서 구현할 항목

### 1. ROS 2 인터페이스 패키지

`lamp_interfaces` 패키지는 다른 담당자가 모션 내부 구현을 몰라도 사용할 수 있는 계약을 제공한다.

| ROS 이름 | 형식 | 호출 주체 | 목적 |
|---|---|---|---|
| `/lamp/play_motion` | Action | 인지·시스템 통합 | 이름으로 표현 모션 실행·취소 |
| `/lamp/place_task_light` | Action | 비전·시스템 통합 | 3D 목표로 작업 조명 배치 |
| `/lamp/interrupt` | Service | 음성·시스템 통합 | 말 끊기 시 활성 모션 중단 |
| `/lamp/list_motions` | Service | 인지·운영 도구 | Pi가 허용한 모션 목록 조회 |
| `/lamp/track_point` | Topic | 비전 | 얼굴 등 3D 위치 전달 |
| `/lamp/track_bearing` | Topic | 음성 | 소리 방향 전달 |
| `/lamp/motion_status` | Topic | 시스템 통합 | 연결·busy·fault·현재 모션 구독 |

`PlayMotion.action`은 모션마다 새 타입을 만들지 않고 다음 범용 목표를 사용한다.

```text
string name
bool replace_current
float32 intensity
uint32 repeat
```

### 2. ROS–TCP 브리지

`lamp_motion_bridge` 노드가 ROS 요청과 Pi TCP 프로토콜 사이를 변환한다.

- TCP 연결과 하트비트 유지
- Action goal ID와 Pi 요청 ID 연결
- Action 취소를 `motion.cancel`로 전달
- Pi의 완료·취소·실패 이벤트를 ROS Action 결과로 변환
- Pi 상태를 `/lamp/motion_status`로 발행
- 연결이 끊기면 지수 백오프로 재연결
- 끊기기 전에 보낸 모션을 자동 재실행하지 않음

추적 Topic은 `KEEP_LAST=1`, `BEST_EFFORT`, `VOLATILE`을 사용한다. 상태 Topic은 깊이 5의 `RELIABLE`, `VOLATILE`을 사용한다.

### 3. 시스템 통합 노드 연결

Jetson의 시스템 통합 담당은 음성·인지·비전 결과를 다음 규칙으로 중재한다.

- VLM 행동 태그 → `/lamp/play_motion`
- 사용자의 말 끊기 → TTS 취소와 `/lamp/interrupt` 동시 실행
- 얼굴 검출 결과 → `/lamp/track_point`
- DOA 결과 → `/lamp/track_bearing`
- 책·키보드 목표 좌표 → `/lamp/place_task_light`
- `/lamp/motion_status`의 fault를 전역 상태머신에 반영

인지·음성·비전 노드는 Pi 주소나 TCP 메시지 형식을 알 필요가 없다.

## 고정 통신 계약

Jetson–Pi 명령 타입은 다음으로 제한한다.

- `motion.play`, `motion.cancel`, `motion.interrupt`, `motion.status`, `motion.list`
- `track.point`, `track.bearing`, `track.clear`
- `task_light.place`, `task_light.cancel`, `task_light.clear`
- `system.heartbeat`

TCP TTL은 두 장치의 시스템 시간이 아니라 Pi가 메시지를 완전히 받은 시점의 monotonic clock부터 계산한다.

## 개발 순서와 인수인계

| 단계 | Raspberry Pi 담당 | Jetson 담당 | 통과 기준 |
|---|---|---|---|
| 1 | 모션 카탈로그·검증기 | ROS 인터페이스 정의 | 이름·필드·오류 코드 문서 확정 |
| 2 | 명령 중재기와 mock 서버 | 순수 Python TCP 클라이언트 | 모터 없이 list/status/play 왕복 |
| 3 | 100 Hz 루프 연결 | ROS–TCP 브리지 | Action 완료·취소와 Topic 전달 |
| 4 | systemd 서비스 | JetPack 감지·ROS 설치 스크립트 | 재부팅 후 자동 연결 |
| 5 | MuJoCo·mock 장애 시험 | 상태머신 연결 | TTL·중복·재연결·fault 통과 |
| 6 | 실기기 안전 시험 | 실제 음성·비전 이벤트 호출 | sleep 복귀·토크 해제·전체 모션 통과 |

각 단계가 끝날 때 Pi 담당은 프로토콜 골든 벡터와 mock 서버를 제공하고, Jetson 담당은 같은 벡터를 사용하는 클라이언트 테스트 결과를 제공한다.

## 검증 순서

1. 단위 테스트: 카탈로그, 프로토콜, 큐, TTL, 중복 요청
2. Pi mock 서버와 Python 클라이언트 loopback
3. ROS 2 Jazzy 빌드 및 인터페이스 확인
4. ROS 2 Humble 빌드 및 인터페이스 확인
5. MuJoCo 전체 모션과 sleep 복귀
6. 유선 연결 단절·재연결 시험
7. Pi 실기기 단일 모션, 취소, 추적 TTL 시험
8. 전체 모션 실행 후 sleep 자세와 토크 해제 확인

Pi 전원이 없는 동안 1~5단계까지 진행할 수 있다. 6~8단계는 Pi와 Jetson을 실제 유선으로 연결한 뒤 진행한다.

## 완료 기준

- 새 모션은 CSV와 카탈로그 항목만 추가하면 기존 Action으로 호출된다.
- Jetson의 인지·음성·비전 코드는 TCP와 모터 구현에 의존하지 않는다.
- Jetson 재시작이나 추론 지연이 Pi의 100 Hz 제어 주기를 막지 않는다.
- 알 수 없는 모션, 만료된 목표와 중복 요청은 실행되지 않는다.
- 연결 단절과 모든 종료 경로에서 안전 대기 또는 sleep 자세로 복귀한다.
- JetPack 6/Humble과 JetPack 7/Jazzy에서 같은 ROS 계약을 사용한다.

## 상세 문서

- [미들웨어 설계서](superpowers/specs/2026-09-12-jetson-pi-motion-middleware-design.md)
- [단계별 구현 계획](superpowers/plans/2026-09-12-jetson-pi-motion-middleware.md)
