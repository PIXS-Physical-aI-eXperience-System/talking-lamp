# Jetson 통합 가이드

## 보드별 담당 범위

Jetson은 STT·대화·TTS·응답 정책과 ROS 2 API를 담당한다. Raspberry Pi는 XVF3800 USB 녹음·재생·제어, DOA 안정화, WS2812B 출력, Feetech 모션과 100 Hz 단일 제어 루프 등 모든 물리 장치와 안전 제어를 담당한다. Pi에는 ROS를 설치하지 않는다. Jetson 애플리케이션에서 원시 관절 각도, 서보 레지스터, GPIO 파형, Python으로 직접 조립한 Opus 패킷이나 펌웨어 조작 명령을 보내지 않는다.

직결 유선 네트워크는 다음 주소를 사용한다.

| 연결 대상 | 주소/포트 | 용도 |
| --- | --- | --- |
| Jetson | `192.168.100.1` | ROS 및 GStreamer 브리지 호스트 |
| Pi 모션 | `192.168.100.2:8765/TCP` | 이름 기반 모션·추종·작업 조명 |
| Pi 장치 | `192.168.100.2:8766/TCP` | 방향·오디오 수명주기·LED |
| Jetson 녹음 수신 | `192.168.100.1:5004/UDP` | Pi 마이크 Opus/RTP |
| Pi 재생 수신 | `192.168.100.2:5006/UDP` | Jetson TTS Opus/RTP |

TCP 세션은 서로 독립적이며 각각 인증된 Jetson 제어 주체 하나만 허용한다. 장치 브리지를 모션 포트에 연결하거나 그 반대로 연결하면 안 된다.

마이크 오디오와 방향 정보는 동일한 XVF3800에서 나오지만 소프트웨어 경로는 분리된다. Pi의 USB 제어 경로는 VAD/DOA를 읽어 로컬 `base_yaw`를 정렬하고 방향 메타데이터만 발행한다. 별도로 ALSA/GStreamer가 마이크 오디오를 Jetson으로 보내 `/lamp/audio/capture` 토픽으로 제공한다. STT·애플리케이션 개발자는 이 오디오 토픽을 구독하며 Pi의 DOA·모터 코드를 호출할 필요가 없다. 두 경로는 발화와 방향을 연결하는 정규 형식의 `speech_id`만 공유한다. 녹음이 재시작되면 ROS 오디오 `stream_id`를 교체하고 순번을 0으로 초기화하며 대기 중인 `speech_id` 연결 상태를 모두 비운다.

## Jetson 실행 환경

초기 설치 검증 당시 Jetson 환경은 Ubuntu 24.04.4, L4T 39.2, aarch64, Python 3.12였으며 ROS 2 Jazzy를 사용했다. [공식 ROS apt 설치 절차](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)에 따라 데스크톱 패키지 대신 `ros-jazzy-ros-base`를 설치한다.

배포된 통합 저장소 경로는 `/home/asdf/talking-lamp-integration`, ROS 작업 공간은 `jetson_ws`다. 기존 `/home/asdf/talking-lamp` 저장소는 그대로 유지한다. 설치 후 다음 명령을 실행한다.

```bash
source /opt/ros/jazzy/setup.bash
cd /home/asdf/talking-lamp-integration/jetson_ws
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y
colcon build --symlink-install
source install/setup.bash
```

## 인증 정보

Jetson에 root 또는 운영자가 준비한 권한 `0600` 파일 두 개를 사용한다.

```text
/etc/talking-lamp/motion-bridge.env  TALKING_LAMP_MOTION_TOKEN=...
/etc/talking-lamp/device-bridge.env  TALKING_LAMP_DEVICE_TOKEN=...
```

토큰 값은 Pi의 대응 파일과 같아야 한다. 토큰을 커밋하거나 출력하거나 ROS 실행 파일에 넣지 않는다. 재연결은 새 TCP 세션을 만들며 이전 요청·모션·TTS 스트림·LED 프레임을 재실행하지 않는다.

시스템 서비스는 두 파일을 읽고 브리지 노드 두 개를 시작한다.

```bash
sudo systemctl start talking-lamp-bridges.service
systemctl status talking-lamp-bridges.service
```

초기 설치 검증 중에는 부팅 자동 시작을 비활성화한다. 실기 오디오·모션·중앙 복귀·대기 모션 순서 검증을 통과한 뒤에만 활성화한다.

## ROS API

모션 브리지:

- `/lamp/play_motion` — `lamp_interfaces/action/PlayMotion`
- `/lamp/place_task_light` — `lamp_interfaces/action/PlaceTaskLight`
- `/lamp/interrupt_motion` — `lamp_interfaces/srv/InterruptMotion`
- `/lamp/list_motions` — `lamp_interfaces/srv/ListMotions`
- `/lamp/track_point`, `/lamp/track_bearing` — 최신 값 기반 추종 토픽
- `/lamp/motion_status` — 제어기·통신 상태

장치 브리지:

- `/lamp/audio/capture` — 20 ms `AudioFrame`, 16 kHz 모노 `pcm_s16le`
- `/lamp/audio/playback_frames` — 수락된 스트림 하나에 대한 TTS 생성 프레임
- `/lamp/play_audio` — 재생 시작·EOS·Pi 재생 완료 확인 담당
- `/lamp/audio_status` — Pi 녹음 파이프라인 장애와 관리된 재시작을 이벤트로 전달한다. 정상 시작 시 상태를 저장해 재전송하지 않으며 재생 상태 전이도 보고하지 않는다. 순번·버퍼 필드는 이번 버전에서 예약된 필드다.
- `/lamp/orientation_status` — `speech_id`를 포함한 Pi DOA·정렬 상태
- `/lamp/return_center` — Pi가 `centered` 상태에 도달해야 완료
- `/lamp/led/frame` — `sensor_msgs/Image`, 정확히 8×8 `rgb8`
- `/lamp/led/set_expression` — 이름 기반 8×8 표정 및 요청 밝기
- `/lamp/led/list_expressions` — 고정된 표정 키 10개
- `/lamp/led/set_solid`, `/lamp/led/clear`, `/lamp/led/status`

표정 키는 `neutral`, `happy`, `excited`, `sad`, `angry`, `surprised`, `curious`, `thinking`, `shy`, `love`다. `lamp_device_bridge`에 행 우선 RGB 픽셀 아트로 직접 제작해 저장했으며 외부 이모지 자산을 복사하지 않았다. 브리지는 이름을 기존 `led.frame` 장치 명령으로 변환한다. 검증된 180° 물리 배치 변환과 밝기 상한 0.08은 Pi가 적용한다.

현재 허용하는 오디오 형식은 `sample_rate=16000`, `channels=1`, `encoding=pcm_s16le`이며 EOS가 아닌 모든 프레임의 데이터 길이는 640바이트다. 순번은 0부터 1씩 증가하고 마지막에는 데이터가 빈 EOS 프레임 하나를 보낸다. 잘못된 프레임, 오래된 프레임, 중복 프레임, 다른 스트림의 프레임은 GStreamer에 전달하기 전에 거부한다.

## 대화 실행 순서

애플리케이션은 `TurnOrchestrator`에 `TurnResponse(speech_id, motion_name, audio_stream)`을 전달한다. 이 도우미는 다음 순서를 보장한다.

```text
해당 발화의 방향 상태 == aligned
        ↓
PlayAudio와 PlayMotion 동시 시작
        ↓ 둘 다 최종 성공할 때까지 대기
ReturnCenter
        ↓ centered까지 대기
PlayMotion("idle")
```

TTS나 표현 모션이 실패하면 자동 중앙 복귀·대기 모션은 실행하지 않는다. 정책 계층은 기계 판독 가능한 결과를 확인하고 중단·재시도·중앙 복귀를 명시적으로 선택해야 한다. 실패했거나 아직 실행 중인 표현 모션을 이른 복귀 요청으로 가리는 것을 막기 위한 동작이다.

사용자가 말을 끼어들면 `TurnOrchestrator.barge_in()`을 호출한다. 오디오와 모션을 취소한 뒤 모션 중단을 호출한다. 현재 방향 기준점은 유지하며, 다음 VAD 상승 에지에 새 `speech_id`가 부여되면 이를 교체할 수 있다.

## 팀 애플리케이션 어댑터

STT·대화·TTS 구현은 이 저장소의 통합 코드 범위 밖이다. 해당 어댑터는 `lamp_interaction.turn`의 주입형 `TurnPorts` 메서드를 구현해야 하며 Pi 모듈을 가져오면 안 된다. 녹음 STT 결과와 방향 이벤트는 정확히 일치하는 정규 형식 `speech_id`로 연결한다. 실제 시각이나 ‘가장 최근 발화’라는 추정으로 연결하지 않는다.

## 장애 시 동작

- 유선 TCP 연결 끊김: 대기 중인 모든 액션이 실패하며 재연결 후에도 재실행하지 않는다.
- 녹음 RTP 손실: 짧은 공백은 지터 버퍼와 Opus 손실 보정으로 처리한다. Pi 녹음 파이프라인이 종료되면 장애를 알리고 관리된 재시작을 수행하며 모션은 계속 응답한다. Jetson 브리지는 아직 장시간 무음·RTP 공백 경보를 발행하지 않는다.
- XVF 제거: Pi는 장치 TCP·LED를 유지하고 XVF 장애를 표시하며 지수 백오프로 재탐색한다. 재연결 시 새 녹음 스트림을 시작한다.
- 재생 제어·프레임 검증 실패: `PlayAudio`가 실패하므로 자동 중앙 복귀·대기는 실행하지 않는다. 성공 결과 `drained`는 Pi 재생 파이프라인이 EOS를 수락하고 종료됐음을 뜻한다. 이번 버전은 UDP 패킷 도착이나 실제 스피커 발성을 확인하지 않는다.
- 장치 세션 변경: 이벤트 순번을 1부터 다시 시작하며 브리지는 이전 세션 ID의 이벤트를 버린다.
- 알 수 없는 표정 이름: Jetson에서 거부하며 프레임을 전송하지 않는다. 물리 배치·하드웨어 장애·밝기 제한의 최종 판단은 Pi가 담당한다.

## 초기 설치 검증 기록 (2026-09-16)

- ROS 2 Jazzy 작업 공간: 패키지 4개 빌드 성공, `rosdep check`에서 시스템 의존성 충족 확인.
- Pi 마이크 → Jetson: 16 kHz 모노 `pcm_s16le`, 20 ms 프레임, 49.989~50.003 Hz.
- Jetson → Pi 재생: 제한된 길이의 440 Hz 시험음을 청취했으며 `PlayAudio`가 `success=true`, `code=drained` 반환.
- 방향 정렬: 실기 XVF3800 이벤트가 정규 형식 `speech_id`와 최종 `aligned` 상태를 생성했으며 베이스가 보고된 목표 yaw에 도달.
- WS2812B-64: Pi 5 GPIO 12의 RP1 PIO 출력 패턴 시험 통과. 공용 5 V 시험은 25·50·75% 및 3초로 제한한 100% 표본에서 `throttled=0x0`으로 통과. 이후 180° 회전을 육안 확인했으며 운영 밝기는 8%로 제한.
- 연속 방향 추종: 운영자가 방향을 바꿔 반복 발화하며 `base_yaw`가 소리를 계속 따라가는 것을 확인.
- 응답 순서: 오디오와 저강도 `nod`가 함께 수락되어 둘 다 완료(`drained` / `completed`)된 뒤 `ReturnCenter`가 yaw `0.261799` rad에서 완료. 이후에만 `idle` 수락.
- 반복 대기 모션의 응답성: `active_motion=idle` 중 상태·목록 서비스가 응답하고 중단과 재시작 성공. 240초 액션 제한 안에서 60초 대기 주기를 마친 뒤 `success=true`, `code=completed` 반환.
- 자체 재생 차단: Pi는 재생 중 및 재생 완료 후 0.3초 동안 XVF VAD/DOA를 무시해 스피커 응답이 새 방향 요청으로 처리되지 않도록 함.
- 물리 LED 배치·전원 검증은 완료했다. 서비스의 부팅 자동 시작은 전체 재부팅·연결 끊김 인수 검증 전까지 비활성 상태를 유지한다. 상세 기록은 [배포·인수인계 문서](deployment-and-handoff.md)를 참고한다.
