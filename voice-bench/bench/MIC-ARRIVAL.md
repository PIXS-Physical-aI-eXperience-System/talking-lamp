# 마이크 도착 후 측정 절차

reSpeaker Flex XVF3800 Circular-4 가 도착하면 이 순서로 진행한다.
E(이수혁)에게 넘겨야 할 값들이 여기서 나온다 — 인수인계 7번(C → E: 소리 방향).

## 0. 환경

```bash
./bench/setup.sh record          # sounddevice·soundfile
venvs/vad/bin/pip install ten-vad numpy soundfile sounddevice
```

`xvf_host` 가 필요하다. 없으면 DOA 를 읽을 수 없다:
https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY → `host_control/`

## 1. 전제 확인 — 여기서 걸리면 나머지가 무의미하다

```bash
venvs/vad/bin/python bench/mic_check.py
```

확인하는 것:

| 항목 | 통과 기준 |
| --- | --- |
| 장치 인식 | XVF3800 이 입력 장치로 잡힘 |
| **채널 수** | **6채널 이상** — 2채널이면 처리음 펌웨어라 원음에 접근 못 함 |
| `xvf_host` | 실행되고 방위각을 반환함 |
| **출력 장치** | **XVF3800 이 출력으로도 잡힘** — 아니면 하드웨어 AEC 가 죽는다 |

6채널이 아니면 펌웨어를 바꾼다:

```bash
dfu-util -R -e -a 1 -D respeaker_flex_ua-io16-6ch-cir.bin
```

## 2. DOA 정면 보정

어레이를 어떻게 장착하든 물리적 회전이 생긴다. 보정 없이 재면 모든 각도가
일정하게 틀어진 채로 나온다.

```bash
venvs/vad/bin/python bench/doa_measure.py calibrate
```

## 3. DOA 8방향 측정 — 조건별로

```bash
venvs/vad/bin/python bench/doa_measure.py measure --label quiet      # 조용한 환경
venvs/vad/bin/python bench/doa_measure.py measure --label fan        # Jetson 팬 켠 상태
venvs/vad/bin/python bench/doa_measure.py measure --label elevated   # 책상+45cm 고각
venvs/vad/bin/python bench/doa_measure.py measure --label servo      # 서보 동작 중
venvs/vad/bin/python bench/doa_measure.py report
```

각 방향에서 **1 m 거리, 3초간 발화**. 0° = 램프 정면, 반시계 방향 증가
(lamp_base 규약, E 의 안건 0절).

`fan` 은 계획서 146행의 검토 항목이고, `servo` 는 진동이 DOA 에 미치는 영향을 본다.

**E 에게 넘길 값**: 가장 나쁜 조건의 평균 절대오차 → L1 칼만의 측정 노이즈 σ.

## 4. AEC 와 barge-in

```bash
venvs/vad/bin/python bench/aec_check.py echo
venvs/vad/bin/python bench/aec_check.py bargein --trials 5
```

`echo` 는 램프 자기 목소리가 AEC 통과 후 얼마나 남는지 잰다.
**사용자 발화 대비 여유가 15 dB 이상이면 내장 AEC 로 충분**하고,
6 dB 미만이면 이격 확대·음량 축소·소프트웨어 보강을 검토해야 한다.

`bargein` 은 사람 반응시간이 섞이므로 TTS 를 끈 상태를 기준선으로 삼아 뺀다.
그 차이가 "램프가 말하는 중이라서 늦어진 몫" 이며 그것이 E 에게 줄 값이다.

## 5. 회신할 것 (E 안건 3절)

- DOA 오차표와 σ — 3-2
- 내장 AEC 충분 여부 — 3-1
- barge-in 감지 지연 확정치 (잠정 150~200 ms 로 회신했음) — 3-3

## 주의

- **각도는 원형 데이터다.** 359° 와 1° 는 2° 차이다. `doa_measure.py` 는 벡터
  평균으로 계산한다(산술평균이면 180° 라는 엉뚱한 값이 나온다).
- **XVF3800 은 방위각만 준다.** 고도(elevation)는 평면 원형 어레이의 원리적
  한계로 얻을 수 없다. `dz` 처리 방식은 E 와 협의 중이다.
- 측정은 조건마다 **3회 이상 반복**할 것. 이 종류의 측정은 편차가 크다.
