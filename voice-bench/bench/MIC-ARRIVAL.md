# 마이크 DOA 측정 절차 — Jetson 인수인계

작성: 파트 C(음성) 최승원 · 2026-09-15
수행: 파트 E(이수혁), Jetson + 램프 본체
산출물: **L1 칼만에 넣을 "θ ± 오차"** — 인수인계 7번(C → E: 소리 방향)

맥에서 도구를 검증하고 규약을 확정한 상태다. **측정값 자체는 램프에 장착한
뒤 Jetson 에서 새로 내야 한다** — 보정은 장착 방향에 따라 완전히 달라진다.

---

## 도착한 것: Linear-4 (원형 아님)

발주는 Circular-4 였으나 **Linear-4** 가 왔고, 시간 때문에 그대로 간다.

| | Circular-4 | **Linear-4 (실물)** |
| --- | --- | --- |
| 마이크 간격 | 44 mm | **33 mm** |
| 집음 | 360° 전방향 | **전면 약 180°, 후면 억제** |

**선형 배열은 앞뒤를 물리적으로 구분할 수 없다.** 축을 기준으로 대칭인 두
방향이 같은 시간차를 만들기 때문이고, XVF3800 은 그래서 후면을 아예 억제한다.

→ **S6(소리 방향 추종)는 전면 반평면에서만 성립한다.** 램프 뒤에서 부르면
방향을 줄 수 없다. 책상에서 사용자를 향해 두는 배치라면 실용상 문제는 없다.

---

## 0. 준비 (Jetson)

```bash
cd ~/talking-lamp && git pull        # 브랜치 feat/voice-bench
cd voice-bench
./bench/xvf_setup.sh
venvs/vad/bin/pip install pyusb sounddevice
```

**펌웨어는 이미 선형 6채널이 들어 있다**(장치 이름에 `L16K6Ch`). 2채널로
잡히면 원음에 접근할 수 없으므로 그때만 다시 굽는다. XMOS USB-C 포트
(3.5mm 잭 쪽)에 연결할 것.

```bash
sudo apt install dfu-util && sudo dfu-util -l
sudo dfu-util -R -e -a 1 -D \
  tools/respeaker-flex/xmos_firmwares/usb/respeaker_flex_usb_l16k6ch_v1.0.3.bin
```

`c16k6ch`(원형)를 넣으면 방향이 엉뚱하게 나온다. **`l` 인지 확인할 것.**

6채널 구성 (16 kHz / 32 bit): `0` 처리음(회의) · `1` 처리음(음성인식) ·
`2~5` 마이크 0~3 원음.

> Flex 는 `reSpeaker_XVF3800_USB_4MIC_ARRAY` 와 **다른 제품, 다른 저장소**다.
> 4MIC 저장소의 펌웨어를 넣으면 안 된다. → [respeaker/reSpeaker_Flex](https://github.com/respeaker/reSpeaker_Flex)

---

## 1. 전제 확인

```bash
venvs/vad/bin/python bench/mic_check.py
```

| 항목 | 통과 기준 |
| --- | --- |
| 장치 인식 | 입력 장치로 잡힘 |
| **채널 수** | **6채널 이상** |
| DOA 경로 | pyusb 로 장치가 보임 (VID 0x2886) |
| **출력 장치** | **XVF3800 이 출력으로도 잡힘** |

**출력 장치로 안 잡히면 스피커가 이 보드에 물려 있지 않다는 뜻이고, 하드웨어
AEC 가 동작하지 않는다.** 램프가 자기 목소리를 듣고 반응하게 된다.

---

## 2. 값이 쓸 만한지 먼저 본다

```bash
venvs/vad/bin/python bench/doa_measure.py live
```

한 자리에서 가만히 말하면 막대가 한 봉우리로 모여야 한다. 그다음 좌우로
옮기면 봉우리가 따라와야 한다. 판정은 **가장 안정적이었던 5초**로 한다.

- `±10° 안 80% 이상` → 정상. 다음으로
- 퍼지면 → 자세·거리(1 m)·발화 연속성을 먼저 의심할 것

맥에서는 가만히 있을 때 **산포 2.2°, ±10° 안 100%** 가 나왔다.

---

## 3. 보정 — 정면과 부호

```bash
venvs/vad/bin/python bench/doa_measure.py calibrate
```

두 지점을 잰다. 6초씩이며, 그중 **가장 안정적인 3초를 자동으로 고른다** —
6초 내내 완벽히 멈춰 있을 필요는 없다.

1. **정면** (램프가 바라보는 쪽, 1 m)
2. **한쪽 옆** — 45° 이상이면 충분. `오른손=r / 왼손=l`

### 각도 규약 (틀리면 램프가 소리 반대쪽으로 돈다)

- 원시값 범위 **0~180°**, 정면이 **약 90°** (선형 배열은 축 방향이 0°, 정면이 broadside)
- 램프 기준 **0° = 정면, 반시계 방향 +** → **램프의 왼쪽이 +90°**
- **마이크를 마주 본 사람의 오른손 쪽 = 램프의 왼쪽(+)**

도구는 사람의 **손 기준**으로만 묻는다. "램프 기준 왼쪽" 같은 표현은 마주 선
사람이 시점을 뒤집어야 하고, 거기서 틀리면 부호가 반대로 저장된다. 오차표는
멀쩡해 보이므로 잡히지도 않는다.

---

## 4. 측정 → 대응표 → 검증

**오프셋 하나로는 못 맞춘다.** 맥 실측에서 실제 180° 를 훑는 동안 원시값은
108.5° 만 움직였고, 구간 기울기가 0.13~1.20 으로 제각각이었다. 다만 순서는
뒤집히지 않아서(단조) 표로 펴면 된다.

```bash
venvs/vad/bin/python bench/doa_measure.py measure --label quiet
venvs/vad/bin/python bench/doa_measure.py table   --label quiet     # 대응표 생성
venvs/vad/bin/python bench/doa_measure.py measure --label quiet-verify
```

7지점(왼손 쪽 90·60·30 / 정면 / 오른손 쪽 30·60·90), 각 1 m 거리 6초 발화.

**세 번째가 진짜 검증이다.** 표를 만든 데이터에 그 표를 적용하면 오차가 0 이
나오는 게 당연하므로, 새로 재야 효과를 알 수 있다.

맥 기준 성적: 표 적용 전 **28.2°** → 적용 후 **6.6°**.

> **바닥에 각도 표시를 하고 할 것.** 맥에서 남은 오차 6.6° 는 대부분 눈대중
> 위치 탓으로 보인다. −30° 한 점만 −18.6° 로 튀었고 나머지는 0~10° 였다.

### 조건별로 반복

```bash
venvs/vad/bin/python bench/doa_measure.py measure --label fan     # Jetson 팬 켠 상태
venvs/vad/bin/python bench/doa_measure.py measure --label servo   # 서보 동작 중
venvs/vad/bin/python bench/doa_measure.py report
```

**`servo` 가 가장 중요하다.** 램프가 움직이면서 소리를 따라가야 하는데 자기
모터 소리에 방향을 잃으면 S6 가 성립하지 않는다.

**칼만의 측정 노이즈 σ 는 가장 나쁜 조건의 평균 절대오차로 잡는다.**

---

## 5. AEC 와 barge-in

```bash
venvs/vad/bin/python bench/aec_check.py echo
venvs/vad/bin/python bench/aec_check.py bargein --trials 5
```

`echo` 는 램프 자기 목소리가 AEC 통과 후 얼마나 남는지 잰다.
**사용자 발화 대비 여유 15 dB 이상이면 내장 AEC 로 충분**, 6 dB 미만이면
이격 확대·음량 축소·소프트웨어 보강을 검토해야 한다.

`bargein` 은 사람 반응시간이 섞이므로 TTS 를 끈 상태를 기준선으로 삼아 뺀다.
그 차이가 "램프가 말하는 중이라서 늦어진 몫" 이고, 그것이 넘길 값이다.

---

## 알려진 고장 모드

- **값이 멈춘다.** 맥 검증에서 한 지점이 58개 표본 전부 정확히 48.0° 로
  나왔다. 실제 추정값은 표본마다 1° 안팎으로 흔들리므로 소수점까지 같으면
  갱신이 안 된 것이다. 그 한 점이 평균오차를 6.6° → 20.3° 로 올렸다.
  도구가 감지해서 다시 잴지 묻는다 — **건너뛰지 말고 다시 잴 것.**
- **각도는 원형 데이터다.** 359° 와 1° 는 2° 차이다. 도구는 벡터 평균으로
  계산한다(산술평균이면 180° 라는 엉뚱한 값이 나온다).
- **고도(elevation)는 얻을 수 없다.** 평면 배열의 원리적 한계다. `dz` 처리는
  E 와 협의 중.
- 조건마다 **3회 이상 반복**할 것. 이 종류의 측정은 편차가 크다.

---

## 넘길 것

| 항목 | 어디서 | 받는 곳 |
| --- | --- | --- |
| DOA 오차표와 σ | `doa_measure.py report` | E, L1 칼만 (3-2) |
| 내장 AEC 충분 여부 | `aec_check.py echo` | E (3-1) |
| barge-in 감지 지연 | `aec_check.py bargein` | E (3-3), 잠정 150~200 ms 로 회신함 |
| **전면 180° 제약** | 이 문서 | E, S6 설계 |

결과 파일은 `out/doa/` 에 남는다. `out/` 은 커밋되지 않으므로 **JSON 을 따로
공유할 것.** 보정 파일(`out/doa/calibration.json`)은 장착 상태에 종속되므로
맥에서 만든 것을 가져다 쓰면 안 된다.
