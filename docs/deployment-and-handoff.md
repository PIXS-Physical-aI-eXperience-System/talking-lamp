# Talking Lamp 구축·사용·인수인계 가이드

이 문서는 현재 실물 Talking Lamp를 같은 상태로 다시 구축하고, Jetson
애플리케이션에서 사용하며, 다음 개발자가 작업을 이어가기 위한 단일 진입점이다.
세부 프로토콜은 [Jetson 통합 가이드](jetson-integration.md), 마이크·LED 실물
시험은 [Pi 장치 커미셔닝](pi-device-commissioning.md), 방향 정렬 내부 동작은
[base-yaw 방향 정렬 가이드](base-yaw-orientation.md)를 따른다.

## 1. 현재 확정된 구성

### 역할 분리

| 장치 | 소유하는 기능 |
| --- | --- |
| Raspberry Pi 5 | Feetech 5축 서보, `base_yaw` 방향 정렬, XVF3800 마이크 입력·DOA, 스피커 출력, WS2812B-64 제어 |
| Jetson Orin Nano | ROS 2, STT·대화·TTS, 응답 정책, 모션·오디오·LED 고수준 호출 |
| PC | 개발·배포·SSH 진입점. Jetson에는 Tailscale로 접속하고 Pi에는 Jetson을 경유해 접속 |

Pi만 실물 장치를 연다. Jetson은 서보 레지스터, USB 제어 전송 또는 GPIO를
직접 다루지 않고 ROS API만 사용한다. Pi에는 ROS를 설치하지 않는다.

### 실물과 연결

- Raspberry Pi 5 + 정품 전원 어댑터
- Feetech STS3215 5축과 USB 시리얼 드라이버: Pi `/dev/ttyACM0`
- reSpeaker Flex XVF3800 Linear-4: Pi USB, VID:PID `2886:0022`, 펌웨어
  `L16K6Ch 1.0.3`
- 스피커: Pi에 연결된 XVF3800 재생 장치를 통해 출력
- WS2812B-64 8×8: Pi 5 V, GND, GPIO 12 데이터 사용 예정. 현재는 물리적으로
  분리되어 있으며 소프트웨어도 `NullPixelSink`를 사용한다.
- Jetson↔Pi 전용 유선 LAN. 인터넷·원격 관리는 Jetson의 Wi-Fi/Tailscale을
  별도로 사용한다.

### 네트워크와 포트

| 경로 | 주소 | 용도 |
| --- | --- | --- |
| Jetson 유선 | `192.168.100.1/24` | Pi 전용 링크 |
| Pi 유선 | `192.168.100.2/24` | Jetson 전용 링크 |
| Pi motion TCP | `192.168.100.2:8765` | 인증된 모션 명령·상태 |
| Pi device TCP | `192.168.100.2:8766` | 인증된 방향·오디오·LED 제어 |
| Pi → Jetson UDP | `192.168.100.1:5004` | Opus/RTP 마이크 오디오 |
| Jetson → Pi UDP | `192.168.100.2:5006` | Opus/RTP TTS 재생 |
| Jetson Tailscale | 현 장비 `100.79.117.124` | PC에서 Jetson SSH |

실험실 기본 SSH 경로는 다음과 같다. Tailscale 주소는 장비 재등록 시 달라질 수
있으므로 관리 콘솔에서 다시 확인한다.

```bash
ssh asdf@100.79.117.124
ssh -J asdf@100.79.117.124 pixs@192.168.100.2
```

### 현재 배포 위치와 상태

| 호스트 | 코드 위치 | systemd 서비스 | 현재 부팅 자동 시작 |
| --- | --- | --- | --- |
| Jetson | `/home/asdf/talking-lamp-integration` | `talking-lamp-bridges.service` | `disabled` |
| Pi | `/home/pixs/talking-lamp` | `talking-lamp-motion.service`, `talking-lamp-device.service` | `disabled` |

2026-09-16 인수 시점에는 세 서비스가 실행 중이고 경고 로그가 없었다. Pi의
`vcgencmd get_throttled`는 `0x0`이었다. 자동 시작은 LED와 재부팅 복구 시험이
끝나기 전까지 의도적으로 비활성이다.

같은 날 서비스 실행 중 5~8초 표본에서 Pi 전체 CPU는 약 1~4%, 두 서비스의
합산 RSS는 약 413 MiB(7.9 GiB RAM의 약 5.1%), 온도는 59.0~59.3°C였다.
모션 프로세스가 약 369 MiB, device와 GStreamer가 약 44 MiB를 사용했다. 이
값은 정상 기준선이며 장시간 최대치가 아니다. device 서비스의 systemd 누적
재시작 횟수는 설치·조정 과정의 2회, motion은 0회였다.

## 2. 처리 흐름

```text
사용자 음성
  → Pi XVF3800: VAD/DOA + 16 kHz mono 캡처
  → Pi가 DOA를 안정화하고 base_yaw를 사용자 방향으로 정렬
  → Jetson: 마이크 ROS 프레임으로 STT → 대화 → TTS/모션 선택
  → Jetson: 동일 speech_id의 aligned 상태 확인
  → PlayAudio와 PlayMotion을 동시에 시작
  → 둘 다 성공 종료할 때까지 대기
  → ReturnCenter 성공 대기
  → idle 모션 1회 시작
```

스피커가 재생 중일 때와 재생 종료 후 0.3초 동안 Pi는 XVF VAD/DOA를 무시한다.
램프가 자기 응답음을 새 사용자 발화로 인식해 다시 회전하는 것을 막기 위해서다.

## 3. 새 장비에서 같은 상태 만들기

### 3.1 안전 전제

1. 처음 설치할 때는 WS2812B의 5 V·GND·DIN을 모두 분리한다.
2. 서보 전원은 모션 서비스 설치와 설정 확인이 끝난 뒤 켠다.
3. 램프를 받쳐 둬서 토크 해제 시 헤드가 떨어지지 않게 한다.
4. 기존 장비의 모터 캘리브레이션이나 DOA 보정 파일을 다른 조립체에 복사하지
   않는다. 둘 다 장착 상태에 종속된다.

### 3.2 소스 준비

PR이 병합되기 전에는 이 브랜치를, 병합된 뒤에는 병합 대상 브랜치를 사용한다.

```bash
git clone --branch codex/jetson-pi-middleware \
  https://github.com/PIXS-Physical-aI-eXperience-System/talking-lamp.git \
  /home/asdf/talking-lamp-integration
```

Pi가 인터넷에 직접 연결되지 않으면 Jetson에서 코드를 복사한다. 기존 디렉터리를
삭제하거나 `--delete`로 동기화하지 말고, 변경 파일을 먼저 확인한다.

```bash
rsync -av --exclude .git \
  /home/asdf/talking-lamp-integration/ \
  pixs@192.168.100.2:/home/pixs/talking-lamp/
```

Pi의 오디오·USB 시스템 패키지와 LeLamp 하드웨어 환경은 다음과 같이 준비한다.

```bash
sudo apt update
sudo apt install -y alsa-utils usbutils libusb-1.0-0 \
  gstreamer1.0-tools gstreamer1.0-alsa \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad

cd /home/pixs/talking-lamp/lelamp_runtime
uv sync
uv pip install --python .venv pyusb 'ruckig>=0.14'
```

`uv`가 없는 현 장비처럼 이미 `.venv`가 준비되어 있으면 재생성하지 말고
`/home/pixs/talking-lamp/lelamp_runtime/.venv/bin/python`이 실행 가능한지만
확인한다. `rpi-ws281x`는 실제 LED를 승인하기 전에는 필요하지 않으며 현재
장비에도 설치되어 있지 않다. 모터 캘리브레이션 절차는
[LeLamp Runtime README](../lelamp_runtime/README.md)를 따른다.

### 3.3 유선 IP 설정

NetworkManager의 유선 프로필 이름을 먼저 확인한다.

```bash
nmcli connection show
```

Jetson에는 `192.168.100.1/24`, Pi에는 `192.168.100.2/24`를 고정 설정한다.
게이트웨이와 DNS는 이 전용 링크에 넣지 않는다. 아래 명령은 해당 장비의 로컬
콘솔에서 실행하고 `<wired-profile>`을 실제 프로필 이름으로 바꾼다.

```bash
sudo nmcli connection modify '<wired-profile>' \
  ipv4.method manual ipv4.addresses 192.168.100.1/24 \
  ipv4.gateway '' ipv4.dns ''
sudo nmcli connection up '<wired-profile>'
```

Pi에서는 주소만 `192.168.100.2/24`로 바꾼다. 양쪽에서 확인한다.

```bash
ping -c 3 192.168.100.2  # Jetson에서
ping -c 3 192.168.100.1  # Pi에서
```

### 3.4 XVF3800과 DOA 보정

마이크 시험은 `origin/feat/voice-bench`의
`voice-bench/bench/MIC-ARRIVAL.md`를 먼저 읽고 진행한다. Linear-4는 전면
반평면만 유효하며, 장착된 램프에서 정면 오프셋과 부호를 다시 측정해야 한다.

필수 결과 파일:

```text
/home/pixs/talking-lamp/voice-bench/out/doa/calibration.json
```

현재 장비의 측정값은 `offset=90.49852890666492`, `sign=1`이며 다른 조립체의
초깃값으로 사용하면 안 된다. 설치 전 USB와 오디오를 확인한다.

```bash
lsusb -d 2886:0022
cd /home/pixs/talking-lamp
deploy/pi/check-audio.sh
```

XVF USB 권한 규칙은 다음과 같아야 한다.

```udev
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0022", GROUP="plugdev", MODE="0660", TAG+="uaccess"
```

규칙을 `/etc/udev/rules.d/99-respeaker-flex.rules`에 설치한 뒤
`sudo udevadm control --reload-rules && sudo udevadm trigger`를 실행하고, 서비스
사용자 `pixs`가 `plugdev` 그룹에 속하는지 확인한다.

### 3.5 Pi 서비스 설치

설치 스크립트는 기본적으로 서비스를 설치만 하고 자동 시작을 활성화하지 않는다.

```bash
cd /home/pixs/talking-lamp
sudo deploy/pi/install-motion-service.sh
sudo deploy/pi/install-device-service.sh
systemctl is-enabled talking-lamp-motion.service   # disabled
systemctl is-enabled talking-lamp-device.service   # disabled
```

생성되는 비밀 파일은 다음과 같다.

```text
/etc/talking-lamp/motion.env  TALKING_LAMP_TOKEN=<motion-secret>
/etc/talking-lamp/device.env  TALKING_LAMP_TOKEN=<device-secret>
```

파일은 `root:root`, 모드 `0600`이어야 한다. 토큰을 명령행 인수, Git, 채팅,
로그에 넣지 않는다.

### 3.6 Jetson ROS 2 브리지 설치

Jetson에는 Ubuntu 24.04/L4T R39 계열과 ROS 2 Jazzy를 사용한다. 설치
스크립트가 ROS base, rosdep, colcon과 GStreamer 플러그인을 설치하고 워크스페이스를
빌드한다. 인터넷 연결이 필요하다.

```bash
cd /home/asdf/talking-lamp-integration
sudo deploy/jetson/install-ros-bridges.sh \
  --repo /home/asdf/talking-lamp-integration
```

설치 중 Ubuntu 저장소가 `NOSPLIT` 또는 서명 오류를 내면 프록시·캡티브 포털을
먼저 제거한다. 스크립트가 HTTP 저장소를 HTTPS로 정규화하지만, 인터넷 연결
자체가 인증 페이지로 가로채지는 상태는 해결하지 못한다.

Pi와 Jetson의 토큰 값은 아래처럼 각각 일치시킨다. 키 이름은 서로 다르다.

| Pi 파일/키 | Jetson 파일/키 |
| --- | --- |
| `motion.env` / `TALKING_LAMP_TOKEN` | `motion-bridge.env` / `TALKING_LAMP_MOTION_TOKEN` |
| `device.env` / `TALKING_LAMP_TOKEN` | `device-bridge.env` / `TALKING_LAMP_DEVICE_TOKEN` |

Jetson 파일 위치:

```text
/etc/talking-lamp/motion-bridge.env
/etc/talking-lamp/device-bridge.env
```

`sudoedit`으로 값을 옮기고 `sudo chmod 0600`을 적용한다. 토큰을 화면에 출력하는
`cat`, 셸 히스토리에 남기는 `echo TOKEN=...`, 메신저 복사를 사용하지 않는다.

빌드를 다시 확인한다.

```bash
cd /home/asdf/talking-lamp-integration/jetson_ws
source /opt/ros/jazzy/setup.bash
rosdep check --from-paths src --ignore-src
colcon build --symlink-install
```

### 3.7 최초 시작과 확인

램프를 받치고 서보 전원을 켠 다음 Pi, Jetson 순서로 시작한다.

```bash
# Pi
sudo systemctl start talking-lamp-motion.service
sudo systemctl start talking-lamp-device.service
systemctl --no-pager --full status \
  talking-lamp-motion.service talking-lamp-device.service

# Jetson
sudo systemctl start talking-lamp-bridges.service
systemctl --no-pager --full status talking-lamp-bridges.service
```

Jetson에서 ROS 상태를 확인한다.

```bash
source /opt/ros/jazzy/setup.bash
source /home/asdf/talking-lamp-integration/jetson_ws/install/setup.bash
ros2 node list
ros2 topic echo --once /lamp/motion_status
ros2 topic hz /lamp/audio/capture
```

정상 기준은 브리지 노드 2개, `motion_status.connected: true`, fault 없음,
마이크 약 50 Hz다. `/lamp/audio_status`는 Pi capture pipeline의 fault와 감독
재시작 때만 발행되는 event-driven topic이다. 정상 시작과 재생 전환은 발행하지
않으므로 확인에 `--once`로 사용하지 않는다. 이 릴리스에서 sequence와 buffer
필드는 예약값이다.

## 4. 운영 방법

### 서비스 제어

```bash
# Pi
sudo systemctl restart talking-lamp-motion.service talking-lamp-device.service
journalctl -u talking-lamp-motion.service -u talking-lamp-device.service -f

# Jetson
sudo systemctl restart talking-lamp-bridges.service
journalctl -u talking-lamp-bridges.service -f
```

정지는 Jetson 브리지 → Pi device → Pi motion 순서로 한다. 모션 서비스가
정상 종료할 때 램프는 sleep pose를 확인한 뒤 토크를 해제하므로 헤드를 받친다.

```bash
sudo systemctl stop talking-lamp-bridges.service                    # Jetson
sudo systemctl stop talking-lamp-device.service talking-lamp-motion.service  # Pi
```

재부팅 자동 시작은 남은 실물 시험을 통과한 뒤에만 활성화한다.

```bash
sudo systemctl enable talking-lamp-motion.service talking-lamp-device.service  # Pi
sudo systemctl enable talking-lamp-bridges.service                              # Jetson
```

### ROS에서 모션 확인

Jetson에서 환경을 source한 뒤 사용한다.

```bash
ros2 service call /lamp/list_motions lamp_interfaces/srv/ListMotions '{}'

ros2 action send_goal --feedback /lamp/play_motion \
  lamp_interfaces/action/PlayMotion \
  "{name: nod, replace_current: true, intensity: 0.2, repeat: 1}"

ros2 action send_goal --feedback /lamp/return_center \
  lamp_interfaces/action/ReturnCenter '{}'

ros2 service call /lamp/interrupt_motion \
  lamp_interfaces/srv/InterruptMotion '{}'
```

실물 최초 호출은 낮은 `intensity`와 1회 반복으로 시작한다. `idle`은 약 60초인
유한 모션이며 한 번 끝난다. 계속 대기시키려면 정책 계층이 필요할 때 다시
호출한다.

### Jetson 대화 애플리케이션 연결

`jetson_ws/src/lamp_interaction/lamp_interaction/turn.py`의 `TurnOrchestrator`를
사용하고 `TurnPorts`를 실제 ROS 클라이언트로 구현한다.

1. `/lamp/audio/capture`를 STT 입력에 전달한다.
2. `/lamp/orientation_status`에서 해당 발화의 canonical `speech_id`가
   `state=aligned`가 될 때까지 기다린다.
3. 대화 결과로 TTS PCM과 표현 모션 이름을 만든다.
4. `PlayAudio` goal을 먼저 수락시키고 `/lamp/audio/playback_frames`에 같은
   `stream_id`의 프레임을 보낸다.
5. `PlayMotion`을 오디오와 동시에 실행한다.
6. 둘 다 `success=true`로 끝난 뒤에만 `ReturnCenter`, 그 뒤 `idle`을 실행한다.

오디오 프레임 규약:

- 16 kHz, mono, `pcm_s16le`
- 20 ms마다 640 bytes
- sequence는 0부터 1씩 증가
- 마지막은 data가 비어 있고 `end_of_stream=true`인 EOS 프레임
- 발화와 방향 이벤트는 시간 추정이 아니라 정확히 같은 `speech_id`로 결합

`TurnOrchestrator.run(TurnResponse(...))`가 이 순서를 강제한다. 사용자가 램프
응답 중 끼어들면 `TurnOrchestrator.barge_in()`으로 오디오와 모션을 취소하고
모션 interrupt를 호출한다. Jetson 애플리케이션은 실패를 숨기지 말고 반환되는
`code`를 기록해 재시도·중앙 복귀 여부를 명시적으로 결정한다.

### LED API

ROS 인터페이스는 준비되어 있다.

- `/lamp/led/set_solid`: 단색과 brightness 요청
- `/lamp/led/frame`: 정확히 8×8 `rgb8` 프레임
- `/lamp/led/clear`: 전체 끄기
- `/lamp/led/status`: 적용 밝기, clamp, fault

현재 production unit에는 `--enable-led-hardware`가 없으므로 실제 LED는 켜지지
않는다. [Pi 장치 커미셔닝](pi-device-commissioning.md)의 매핑·전원 단계가 모두
통과하기 전에는 기본 unit을 변경하지 않는다. 실물 시험을 시작할 때만 Pi에서
`uv pip install --python lelamp_runtime/.venv rpi-ws281x`로 드라이버를 설치한다.

## 5. 이번 작업에서 구현한 내용

### Pi

- 인증된 모션 TCP 서버와 로컬 Unix socket
- 이름 기반 모션, interrupt, task-light, 상태·heartbeat
- XVF3800 firmware/USB 검증과 재발견
- 20 Hz DOA 읽기, 보정·안정화, `base_yaw` 정렬과 중앙 복귀
- GStreamer 기반 마이크 Opus/RTP 송신과 스피커 Opus/RTP 수신
- WS2812B-64 8×8 매핑·밝기 제한·fault API와 기본 null sink
- 모션과 device 프로세스 분리 및 systemd 안전 종료

### Jetson

- ROS 2 Jazzy 인터페이스 패키지
- motion/device TCP 재연결 브리지
- 마이크 `AudioFrame` publish와 TTS frame 검증·재생 action
- 방향·모션·오디오·LED 상태와 action/service/topic API
- 방향 정렬 후 응답, 응답 종료 후 중앙 복귀와 idle을 강제하는
  `TurnOrchestrator`

### 실기에서 발견해 수정한 문제

- single-thread ROS executor 때문에 오디오 action 중 playback frame이 처리되지
  않던 문제
- XVF3800 재생 형식과 GStreamer appsrc caps 불일치
- 긴 모션의 요청 TTL과 action 완료 timeout이 섞이던 문제
- 스피커 출력이 새 VAD/DOA 이벤트로 되먹임되던 문제
- idle 중 motion status/service가 막히던 문제
- 약 60초 idle이 60초 timeout 직전에 실패하던 문제

### 통과한 실기 검증

- 마이크 캡처 16 kHz mono, 20 ms, 약 50 Hz
- Jetson→Pi 440 Hz 시험음 청취 및 `PlayAudio code=drained`
- 임의 방향의 반복 발화에 `base_yaw`가 지속 추종
- 오디오와 `nod` 동시 실행 후 둘 다 종료, 중앙 복귀, idle 순서
- idle 실행 중 status/list/interrupt 응답 및 전체 60초 idle 완료
- Pi 전원 저하 없음(`throttled=0x0`), 제한된 full-duplex 시험 중 USB reset 없음

## 6. 장애 진단

| 증상 | 먼저 확인할 것 |
| --- | --- |
| Jetson에서 Pi 접속 불가 | 양쪽 고정 IP, 케이블/link, `ping`, Pi 방화벽, 허용 호스트 `192.168.100.1` |
| bridge가 연결되지 않음 | Pi 서비스 active 여부, 8765/8766 listen, 양쪽 토큰 값과 키 이름 |
| 마이크 topic이 없음 | `lsusb -d 2886:0022`, udev 권한, `deploy/pi/check-audio.sh`, device journal |
| 스피커 소리가 없음 | XVF가 playback device인지, UDP 5006, Jetson/Pi의 GStreamer `not-negotiated` 로그. `PlayAudio code=drained`는 Pi pipeline의 EOS 처리·종료만 뜻하며 UDP 전달·실제 가청 여부는 보장하지 않음 |
| 램프가 자기 소리에 회전 | speaker가 XVF 출력인지, playback active/drain 상태, self-playback guard 로그 |
| 방향이 반대로 움직임 | 다른 장착물의 calibration을 복사했는지, `doa_direction_sign`, Linear-4 정면 방향 |
| 모션 명령 실패 | 서보 전원, `/dev/ttyACM0`, motion fault, 모터 캘리브레이션, 헤드 기계 간섭 |
| idle 중 ROS가 멈춤 | 최신 multi-thread bridge 배포 여부, Jetson bridge journal |
| Pi가 재부팅·USB 재연결 | `vcgencmd get_throttled`, kernel journal, LED 5 V 부하를 즉시 제거 |

현재 서비스 시작 이후의 경고만 보려면 각 호스트에서 다음 패턴을 사용한다.

```bash
unit=talking-lamp-bridges.service
started=$(systemctl show "$unit" -p ActiveEnterTimestamp --value)
journalctl -u "$unit" --since "$started" -p warning --no-pager
```

## 7. 다음 작업자가 이어갈 순서

1. **Jetson 팀 애플리케이션 어댑터**
   - `TurnPorts`를 rclpy action/service/topic 클라이언트로 구현한다.
   - STT 결과와 orientation을 exact `speech_id`로 결합한다.
   - 대화 정책이 표현 모션 이름과 TTS 스트림을 만든 뒤
     `TurnOrchestrator`만 호출하게 한다.
2. **WS2812B-64 실물 커미셔닝**
   - DIN/DOUT·첫 픽셀·색 순서를 확인한다.
   - 10% mapping 이후 10/25/50/75/100% 순으로 전원 시험한다.
   - USB reset, 전압 저하, 발열이 생기기 전 마지막 통과 밝기를 ceiling으로
     승인한다. 별도 5 V 환경이 없으므로 100% 가능을 가정하지 않는다.
3. **복구 시험**
   - XVF3800 분리·재연결, Jetson/Pi 링크 단절·복구, 각 장비 재부팅을 시험한다.
   - 재연결은 이전 action이나 오디오 프레임을 자동 재실행하지 않는 것이 정상이다.
4. **부팅 자동 시작 승인**
   - 위 시험 후 세 서비스를 enable하고, 전원 투입부터 ROS 상태 정상까지 확인한다.
5. **회귀 검증**
   - 변경 후 아래 명령과 Jetson `colcon build`를 모두 실행한다.

```bash
cd <talking-lamp-repo>
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" .venv/bin/python -m compileall -q \
  src jetson_ws/src
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" .venv/bin/pytest -q
git diff --check

cd jetson_ws
source /opt/ros/jazzy/setup.bash
rosdep check --from-paths src --ignore-src
colcon build --symlink-install
```

## 8. 변경 시 지켜야 할 경계

- Jetson은 고수준 ROS 명령만 내리고 Pi의 100 Hz 모션·안전 제어를 우회하지
  않는다.
- 오디오 codec/RTP/jitter는 GStreamer가 담당한다. Python에서 Opus/RTP를 다시
  구현하지 않는다.
- TCP 재연결 후 이전 non-idempotent motion/audio 요청을 자동 재전송하지 않는다.
- 장치 토큰을 저장소·launch 파일·로그에 넣지 않는다.
- 실제 LED 활성화는 명시적인 커미셔닝 결과와 운영자 승인 없이는 기본 서비스에
  추가하지 않는다.
- 오디오나 표현 모션이 실패하면 자동 중앙 복귀와 idle로 성공처럼 덮지 않는다.
