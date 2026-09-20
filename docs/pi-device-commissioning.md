# Raspberry Pi 장치 서비스 초기 설치 검증

reSpeaker Flex XVF3800 Linear-4와 WS2812B-64 제어 서비스를 검증하는 절차다. LED 검증은 2026-09-16에 통과했다. 배선·드라이버·전원·매트릭스를 변경하면 아래 안전 절차를 다시 수행한다. Raspberry Pi는 마이크·스피커·방향 계산·LED·베이스 yaw 정렬을 담당하고 ROS는 Jetson에서 실행한다.

## 고정 네트워크와 USB 식별 정보

- Pi 유선 주소: `192.168.100.2`
- Jetson 유선 주소: `192.168.100.1`
- 장치 제어: 인증된 TCP `192.168.100.2:8766`
- XVF3800 USB 식별자: `2886:0022`
- 검증한 펌웨어 버전: `1.0.3`
- 모션 연결 소켓: `/run/talking-lamp/motion-control.sock`

런타임 어댑터는 읽기 전용이다. 펌웨어 쓰기·초기화·범용 USB 제어 메서드는 제공하지 않는다. 펌웨어가 일치하지 않으면 장치 데몬을 종료하지만 `talking-lamp-motion.service`는 중단하지 않는다.

설치된 udev 규칙은 다음과 같아야 한다.

```udev
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0022", GROUP="plugdev", MODE="0660", TAG+="uaccess"
```

다음 명령으로 확인한다.

```bash
lsusb -d 2886:0022
ls -l /dev/bus/usb/$(lsusb -d 2886:0022 | awk '{gsub(":", "", $4); print $2 "/" $4}')
```

## DOA 보정 확인

배포 시 사용하는 파일은 `/home/pixs/talking-lamp/voice-bench/out/doa/calibration.json`이다. 로더는 아래 장치 기본 형식 또는 `voice-bench` v2 형식을 허용하며 각 형식의 필드 집합이 정확히 일치해야 한다.

```json
{
  "doa_zero_deg": 0.0,
  "doa_direction_sign": 1,
  "front_half_angle_deg": 90.0
}
```

위 숫자는 형식 설명용이며 실제 설치값이 아니다. `voice-bench`로 해당 장착 상태에서 측정한 결과를 사용하고 다른 장착 상태의 보정값을 복사하지 않는다. 이번에 수록한 [실측 보정 파일](../.hardware-results/lamp-pi/doa/calibration.json)은 `conv=2` 형식으로 `offset_deg`, `sign`, `n`, `std`, `side_raw`, `side_delta`, `side_std`, `side`를 포함한다. 로더는 `offset_deg`와 `sign`을 영점·방향 부호로 읽고 전방 반각을 90°로 설정한다. 설치 전 파일 해시를 기록한다.

```bash
sha256sum /home/pixs/talking-lamp/voice-bench/out/doa/calibration.json
```

초기 설치 검증 기록:

```text
보정 SHA-256: acaec65d96d31137d27a58beab947f1b75fdfc35529a328458797d940d5fe797
측정 영점·부호·날짜: offset=90.49852890666492, sign=1, 2026-09-16
```

## 소프트웨어만 설치하고 시험하기

WS2812B의 전원선과 데이터선을 물리적으로 분리한다. 부팅 자동 시작을 활성화하지 않고 설치한다.

```bash
cd /home/pixs/talking-lamp
sudo deploy/pi/install-device-service.sh
systemctl is-enabled talking-lamp-device.service   # 예상: disabled
systemctl is-active talking-lamp-device.service    # 예상: inactive
```

저장소의 서비스 유닛은 실기 검증을 완료해 `--enable-led-hardware`를 포함하지만 기본적으로 부팅 자동 시작은 비활성이다. 소프트웨어만 전경 실행으로 시험할 때는 매트릭스를 분리하고 아래처럼 해당 플래그를 생략한다. 보호된 토큰을 출력하지 않고 읽어 실행한다.

```bash
sudo -u pixs bash -c 'set -a; . /etc/talking-lamp/device.env; set +a; \
  PYTHONPATH=/home/pixs/talking-lamp/src:/home/pixs/talking-lamp/lelamp_runtime \
  /home/pixs/talking-lamp/lelamp_runtime/.venv/bin/python -m device.daemon \
  --bind 192.168.100.2 --port 8766 --allow-host 192.168.100.1 \
  --motion-socket /run/talking-lamp/motion-control.sock \
  --calibration /home/pixs/talking-lamp/voice-bench/out/doa/calibration.json \
  --sample-rate 20 --gpio-pin 12 --xvf-vid 0x2886 --xvf-pid 0x0022 \
  --led-rotation 180 --max-brightness 0.08'
```

`SIGTERM`을 보내 종료 코드가 0인지 확인한다. 종료 순서는 폴링 중지 → TCP 세션 종료 → LED 출력 지우기·닫기 → USB 제어 핸들 해제다. 독립된 모션 서비스는 계속 활성 상태여야 한다.

## LED 배선 확인

매트릭스 전원을 프로그래밍 가능한 GPIO나 3.3 V 레일에서 공급하지 않는다. Pi의 5 V 헤더를 모듈 5 V에, Pi GND를 모듈 GND에, GPIO 12를 모듈 **DIN**에 연결한다. 실제 보드에서 DIN/DOUT·5 V·GND·첫 번째 물리 픽셀을 식별한 뒤 연결한다.

운영 서비스 유닛은 2026-09-16 실기 검증 후 `--enable-led-hardware`를 포함하지만 부팅 자동 시작은 비활성이다. 정상 운영에서는 `--led-rotation 180`과 `--max-brightness 0.08`을 유지한다. 배선 변경 후 다음 순서로 확인한다.

1. 전체 출력 지우기·끄기.
2. 논리 좌표 `(0, 0)`에 저밝기 빨간 픽셀 하나 표시.
3. 첫 번째 논리 행 표시.
4. 첫 번째 논리 열 표시.
5. 빨강·초록 체크무늬 표시.

관측 결과에 따라 `layout`, `origin`, `rotation_deg`, `color_order`를 조정한다. 3.3 V 데이터가 불안정하면 프레임 재전송으로 보완하지 말고 적절한 74AHCT 계열 레벨 시프터를 사용한다.

## 공용 5 V 전원 검증

배치 검증을 통과한 뒤에만 전체 흰색 출력을 10 → 25 → 50 → 75 → 100% 순서로 시험한다. 운영 서비스는 모든 요청을 8%로 제한하므로 이는 통제된 초기 설치 시험이다. 운영 서비스를 중지하고 각 단계에 명시적인 임시 밝기 상한을 둔 시간 제한 전경 시험을 실행한다. 보고된 `applied_brightness`가 목표 단계와 같은지 확인하고 매 표본 후 출력을 지운다. 재시작 전에 운영 서비스의 8% 제한을 복원한다. 각 밝기에서 XVF3800 녹음·재생을 동시에 실행하며 다음 결과를 기록한다.

```bash
vcgencmd get_throttled
vcgencmd measure_volts core
journalctl -k --since '-2 minutes' | grep -Ei 'under-voltage|usb|reset|disconnect'
```

새 스로틀링 비트·전압 강하·USB 재연결·재부팅·깜빡임·커넥터/케이블/모듈의 비정상 발열이 관찰되면 즉시 출력을 지우고 중단한다. 임시 상한은 직전에 통과한 단계로 둔다. 운영 `--max-brightness`를 바꾸기 전에는 반복 시험으로 확인한 최종 상한을 운영자가 승인해야 한다. Pi 5가 공식 어댑터를 사용한다는 이유만으로 100%가 안전하다고 간주하지 않는다.

## 인수 검증 기록

```text
XVF 식별자/버전: 2886:0022 / 1.0.3
가상 LED 출력 SIGTERM 종료: 통과 (2026-09-16; 정상 종료, 자동 시작 비활성 유지)
녹음 형식: S16_LE, 16000 Hz, 6채널 (ALSA 하드웨어)
Jetson 녹음 출력: pcm_s16le, 16000 Hz, 모노, 20 ms 프레임
녹음 ROS 주기: 통과 (49.989~50.003 Hz, 50프레임 구간, 2026-09-16)
재생 형식: S16_LE, 16000 Hz, 2채널 (ALSA 하드웨어)
Jetson → Pi 재생 액션: 통과 (success=true, code=drained, 2026-09-16)
전이중 USB 초기화/연결 끊김: 시간 제한 시험음 재생 중 관찰되지 않음
전이중 시험 후 Pi 스로틀링: throttled=0x0
운영자 청취 확인: 통과 (시간 제한 440 Hz 시험음 청취, 2026-09-16)
픽셀 배치: 통과 (GPIO 12 / RP1 PIO; 180° 회전 육안 확인, 2026-09-16)
25/50/75% 전체 흰색 전원 표본: 통과 (8초 / 8초 / 5초, throttled=0x0)
100% 전체 흰색 전원 표본: 제한된 시험만 통과 (3초, 장시간 사용 승인 아님)
최저 관측 EXT5V: 5.0987 V; 최종 SoC 온도: 31.8 °C
커널 저전압/USB 초기화/연결 끊김 경고: 관찰되지 않음
승인된 운영 밝기 상한: 0.08
```
