# sim/ — MuJoCo 모션 개발 환경

파트 E(움직임/IK)가 하드웨어 없이 IK·궤적 생성기·모션 블렌더를 개발·검증하는 환경.
LeLamp 벤더링 자산([../LeLamp/simulation/](../LeLamp/simulation/))에서 팔 구조를 가져온다.

## 설치

프로젝트 루트에 `.venv`가 이미 있다. 없으면 `make venv`.

## 실행

리포 루트에서 (ROS가 셸에 소스돼 있어도 `make`가 `PYTHONPATH=`로 막아준다):

```bash
make check    # 헤드리스 씬 검증 (서버에서도 됨)
make build    # 팔 XML 재생성 (아래 참조)
make demo     # 모션 스택 시나리오 데모 -> out/motion_*.png
```

디스플레이가 있는 데스크탑에서:

```bash
make drive     # 모션 스택 라이브 뷰어 - 실행한 터미널에서 W/A/S/D 로 타겟 이동
make goto ARGS="0.30 0.12 0.02"        # (아무 터미널) 타겟을 정확한 좌표로
make goto ARGS="0.30 0.12 0.0 light"   # + 모드 전환 (track / light / reach)
PYTHONPATH= .venv/bin/python sim/view.py   # 관절 슬라이더만 있는 소박한 뷰어
```

`make drive` 조종: **실행한 터미널에 포커스를 두고** `W`/`S` 앞뒤, `A`/`D` 좌우, `R`/`F` 위아래
(화살표·PageUp/Down도 됨). 누르고 있으면 계속 이동, 떼면 멈춤. `+`/`-` 속도, `1`/`2`/`3` 모드,
`0` 홈, `Q` 종료. 뷰어 창에서 초록 공을 `Ctrl`+우클릭 드래그해도 됨 — 창 안에서 엉뚱한 키를
눌러 카메라가 고정되면 `Esc`.

> ROS Jazzy가 소스된 셸에서 `.venv/bin/python`을 직접 쓰면 `/opt/ros/*`의 패키지가
> venv를 덮어써 깨진다. `make`를 쓰거나 `PYTHONPATH=`를 앞에 붙일 것. 각 sim 스크립트도
> `sim/_bootstrap.py`로 `/opt/ros` 경로를 제거한다.

`check.py` 출력: 관절 맵, 관절별 스윕 시 헤드 팁 이동량, 대략적 도달 범위, home 자세 유지 안정성, `out/`에 렌더 3장.

**모션 스택**(L0~L3 레이어·IK·궤적 생성기·칼만·블렌더·100Hz 런타임)은 [../src/motion/](../src/motion/)에 있다.
`sim/`은 그 스택이 돌아가는 씬·검증 도구이고, 스택 자체 문서는 [../src/motion/README.md](../src/motion/README.md).
`sim/motion_demo.py`가 idle→wake→얼굴 추종→끄덕임→작업 조명→barge-in 시나리오를 dynamics 백엔드로 돌려
스냅샷과 레이어 활동 타임라인 플롯을 그린다.

헤드리스 렌더가 안 되면 `MUJOCO_GL=egl` + 시스템 `libgl1`/`libegl1` 설치.

## 파일

| 파일 | 용도 |
| --- | --- |
| `build_arm.py` | 벤더 자산 → `lelamp_arm.xml` **재생성 스크립트** (아래 "팔 재구성") |
| `lelamp_arm.xml` | **생성물.** 손으로 고치지 말고 `build_arm.py`를 고칠 것 |
| `world.xml` | 씬 — 팔 + 책상(윗면 z=0) + 조명 + `home` 키프레임 + `work_target` 사이트(S1용) + `closeup` 카메라 |
| `lamp.py` | 헬퍼 — 로드, 관절/액추에이터/qpos 인덱스, 관절 범위, `head` 사이트 위치 |
| `check.py` | 헤드리스 검증 |
| `view.py` | 인터랙티브 뷰어 |

## 관절

| 서보 ID | 관절 이름 | 범위 (MuJoCo, 벤더 URDF 기준 · 잠정) |
| --- | --- | --- |
| 1 | `base_yaw` | -287.7° … 72.3° |
| 2 | `base_pitch` | -62.1° … 117.9° |
| 3 | `elbow_pitch` | -161.6° … 18.4° |
| 4 | `wrist_roll` | -217.0° … 143.0° |
| 5 | `wrist_pitch` | -48.9° … 131.1° |

관절 = 액추에이터 이름. `ctrl`/`qpos` 인덱스는 `lamp.actuator_order()` / `lamp.qpos_order()`로.
`head` 사이트 = 헤드 셸(디퓨저+램프헤드) 바운딩박스 중심, IK 타겟 기준점. `work_target` 사이트 = 책상 위 예시 목표점(런타임에 옮겨서 S1 테스트).

home 자세에서 헤드 팁 ≈ `(-0.166, 0.097, 0.259) m`, 단일 관절 스윕 도달 범위 ≈
x∈[-0.23, 0.36], y∈[-0.19, 0.19], z∈[0.00, 0.60] m.

## 팔 재구성 — 왜 `build_arm.py`가 있나

**벤더 익스포트(`robot.xml`·`robot.urdf`)는 운동학 트리가 깨져 있다.** onshape-to-robot이
5축을 직렬 체인이 아니라 베이스에서 두 갈래로 분기시켜, 관절을 돌려도 팔이 안 움직였다
(base_yaw 전 범위 돌려도 헤드 2 mm 이동).

`build_arm.py`가 하는 일:

1. 깨진 모델을 로드해 **home 자세(전 관절 0)에서** 각 관절의 월드 앵커·축·가동범위와
   각 구조 메시의 월드 트랜스폼을 읽는다 (home 자세의 메시 배치 자체는 CAD와 일치함).
2. 이 값들로 **직렬 체인**을 새로 생성: `base → base_yaw → base_pitch → elbow_pitch →
   wrist_roll → wrist_pitch → head`. 각 관절은 실제 home 앵커에 실제 축으로 배치,
   각 링크 프레임은 home에서 월드 정렬 → 메시는 지금과 똑같은 자리에 붙고 home 렌더가 동일.
3. 서보·PCB 클러터 메시는 버리고 구조 셸 5개 + 베이스만 유지.
4. 링크마다 박스 관성 프록시(질량 = 대응 벤더 바디 질량) 부여.

### 한계 / TODO

- **관성은 근사치** (세그먼트 박스). 동역학 정밀도가 필요하면 CAD에서 다시.
- **관절 범위는 벤더 URDF 값** — 실물 캘리브레이션 전까지 잠정.
- `kp=120, kv=8`은 12 V에서 자세 유지되도록 임의로 올린 값 — 실물 서보로 재튜닝.
- **근본 해결: OnShape에서 재익스포트.** `../LeLamp/simulation/config.json`의
  `documentId: 16c9706360b5ad34f9c8db49` + LeLamp README의 CAD 링크. OnShape 계정·API 키
  발급 후 [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot)으로 assembly mate를
  직렬 체인으로 잡아 재생성 → `lelamp_arm.xml`을 그걸로 교체. (초기 설정 3번 "CAD 원본 복사"와 같은 작업.)

지금 상태로 IK·궤적 생성기·블렌더 개발 착수 가능 (E 파트 순번 6~10).
