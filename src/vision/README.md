# src/vision — D(비전) 런타임 모듈

파트 D / 김아현 · 측정과 근거는 [vision-bench/](../../vision-bench/README.md)

카메라 프레임 한 장을 받아 **책상 위 물체**와 **얼굴**의 3D 위치를 `lamp_base` 좌표로 낸다.
[2.4절 공통 규약](../../docs/파트-분배.md)을 그대로 따른다. 단위는 m, 모든 결과에 `time.monotonic()` 스탬프가 붙는다.

```python
from vision import Intrinsics, Pose, VisionPipeline
from vision.detector import ObjectDetector, FaceDetector

pipe = VisionPipeline(
    Intrinsics.load("vision-bench/calib/intrinsics.json"),
    Pose.load("vision-bench/calib/pose.json"),
    ObjectDetector("vision-bench/models/yolox_s.onnx"),
    FaceDetector("vision-bench/models/yunet_2023mar.onnx"),
)
vf = pipe.process(frame_bgr)
vf.objects        # [ObjectTarget(label, conf, pos, box, stamp)]  → S1
vf.faces          # [FaceTarget(conf, pos, box, stamp)]           → S2·S4
vf.labels_ko()    # ['책', '키보드', '의자', ...]                   → A 인지
```

## 구성

| 파일 | 역할 | cv2 필요 |
| --- | --- | --- |
| `camera.py` | 핀홀 + 5항 왜곡 모델. OpenCV `calibrateCamera` 출력과 형식 동일 | 아니오 |
| `geometry.py` | 픽셀 → `lamp_base` 3D. 물체는 책상 평면, 얼굴은 눈 사이 거리 | 아니오 |
| `labels.py` | COCO 클래스, 책상 물체 집합, **한국어 매핑** | 아니오 |
| `pipeline.py` | 검출 → 필터 → 3D. D의 출력 경계 | 아니오 |
| `detector.py` | YOLOX(사물), YuNet(얼굴). ONNX Runtime **CPU 전용** | 예 |
| `calibration.py` | ChArUco 보드로 내부 파라미터와 책상 등록 | 예 |
| `head_camera.py` | 헤드→카메라 변환(`HeadCameraMount`), 휴식 자세 FK로 카메라 자세 계산 | FK 부분만 MuJoCo |
| `follow.py` | S2 얼굴 추종 보정 루프. 헤드 관절값 없이 화면 오차로 목표점을 고침 | 아니오 |
| `tracking.py` | 여러 프레임 평균으로 흔들림 제거, 재실 판정 히스테리시스 | 아니오 |
| `runtime.py` | 프레임마다 무엇을 낼지 정하는 상태기계. ROS 없음 | 아니오 |

기하 계산은 numpy만 쓴다. 그래서 cv2가 없는 개발 PC에서도 테스트가 돈다.

## 물체: 책상 평면 역투영

램프 베이스가 책상에 놓이므로 `lamp_base`에서 **책상면은 z = 0**이다. 박스 하단 중심(물체가 책상에 닿는 점)을 지나는
광선을 z = 0 평면과 교차시키면 그 점이 조명 타겟점이다. 깊이 센서나 깊이 추정 모델이 필요 없다.

다음 셋은 타겟에서 빠진다. 빠진 이유는 `VisionFrame.rejected`에 남는다.

- 조명 대상이 아닌 클래스. 대상은 `labels.TASK_LIGHT` = 책·노트북·키보드뿐이다.
  헤드폰(→mouse), 펜(→remote), 어댑터(→chair) 같은 COCO 오인이 여기서 빠진다
- 광선이 수평선 위를 향해 책상과 만나지 않는 경우
- 작업 영역 밖: 베이스 반경 12 cm 이내(램프 자기 몸), 80 cm 초과, 베이스 뒤쪽

## 얼굴: 눈 사이 거리 **[제안 — E 확인 필요]**

**얼굴은 책상면 위에 있지 않다.** 앉은 사용자의 눈은 책상에서 35~45 cm 위다.
[협의안건 4-3](../../docs/협의/2026-09-05_파트별-제안_이수혁.md)의 "2D 박스 + 책상평면 역투영"을 얼굴에 적용하면
얼굴이 책상 위 엉뚱한 곳에 찍힌다.

대신 YuNet이 주는 두 눈 좌표로 거리를 잰다.

- 성인 눈 사이 거리(IPD)는 평균 63 mm이고 편차가 작다. 화면상 눈 간격이 거리에 반비례한다
- 왜곡을 보정한 좌표로 계산하므로 광각 화면 가장자리에서도 유효하다
- 출력 점은 **두 눈의 중간점**이다. 램프가 바라봐야 할 곳이 거기다. 박스 중심은 코끝 근처로 내려가 있다
- **옆모습이면 얼굴 폭(14 cm)으로 계산한다.** 고개를 돌리면 눈 간격이 박스보다 훨씬 빨리 좁아져서,
  눈 방식이 거리를 두 배로 본다(실측: 눈 기준 0.97 m, 폭 기준 0.44 m, 실제 약 0.5 m).
  눈 간격/얼굴 폭 비율이 0.30 미만이면 옆모습으로 본다(`PROFILE_RATIO`, 정면은 약 0.45)
- 0.25~2.5 m 밖이면 버린다

**한계**: 사람마다 IPD가 달라서 거리 오차가 약 ±8%다. 옆모습 판정 경계 근처(비율 0.3 부근)에서는 추정이 거칠다.
L1은 **방향**이 핵심이고 거리가 몇 cm 틀려도 고개 각도는 거의 변하지 않으므로, 추종 용도로는 충분하다고 본다.
E(이수혁)가 칼만 측정 노이즈를 잡을 때 거리 방향 분산을 방향 분산보다 크게 두면 된다.

## 인지 라벨: D가 매핑을 소유한다 **[A 확인 필요]**

A가 `docs/인지-연계-가이드.md` 6절(`feat/cognition-llm` 브랜치)에 남긴 열린 항목("한국어 매핑 누가?")에 대한 D의 제안이다.

- `labels.KO`는 A의 `vision_to_cognition.py`에 있던 표를 **그대로** 옮겼다. A가 튜닝할 때 쓴 프롬프트가 같은 단어를 받는다
- D가 소유하는 이유: 검출기가 어떤 클래스를 내는지는 D가 정한다. 모델을 바꾸면 라벨 집합이 바뀌고, 그때 A 코드가 깨지지 않아야 한다
- `labels_ko()`는 **장면 전체**를 낸다. S1 타겟(책상 물체)과 다르다. A의 E2E 예시에 "의자"가 들어 있어서다
- 한국어 표에 없는 클래스는 영어로 넘기지 않고 **버린다.** 프롬프트가 한국어 전용이고 C의 TTS가 혼용 문자열을 읽지 못한다

A 쪽 전환은 이렇게 하면 된다:

```python
# 전: vision_to_cognition.py 안의 COCO, KO, preproc, decode, nms 복사본
# 후:
from vision import VisionPipeline
labels = pipe.process(frame).labels_ko()
r = respond(labels, user_text)
```

## 검출기 선택

| | 모델 | 근거 |
| --- | --- | --- |
| 사물 | YOLOX-s | S1이 이벤트성이라 CPU 281 ms도 충분하다. 초점 수정 후 재비교에서 tiny와 노트 검출은 같고 오탐이 적다 |
| 얼굴 | YuNet | CPU 16 ms(초당 62회), 99 MB |
| 실행 장치 | **CPU** | Jetson에서 CUDA 세션은 모델 크기와 무관하게 피크 900 MB가 나와 예산을 넘는다 |

실물 카메라 기준 한 프레임 전체(YOLOX-s + YuNet + 3D)가 **335 ms**다.
얼굴만 30 Hz로 돌리는 경로는 사물 검출과 분리해야 한다. 아직 남은 작업이다.

## 헤드 카메라

카메라는 **램프 헤드에 붙어 있어** 팔과 함께 움직인다. 그래서 `Pose`는 상수가 아니라
`FK(관절값) × 헤드→카메라 변환`이다. 헤드 관절값이 Pi 밖으로 나오지 않으므로(base_yaw만 나옴),
E 코드를 고치지 않는 방식으로 쓴다.

- **S1**: `/lamp/return_center`로 휴식 자세에 두고 `busy=false`가 된 뒤 찍는다. 휴식 자세는 상수라 FK로 헤드 자세가 나온다
- **S2**: 얼굴은 마지막으로 보낸 목표점 기준으로 화면 오차만큼 조금씩 고쳐 보낸다
- 책상은 언제나 `lamp_base`의 z = 0 이므로 **설치할 때 책상을 등록하는 단계가 없다**

구현: `head_camera.HeadKinematics.camera_pose_at_rest(mount, base_yaw)`가 휴식 자세 카메라 자세를,
`follow.FaceFollower`가 S2 보정 루프를 맡는다.

**S2 보정 루프는 헤드가 멈췄을 때만 고친다.** 멈춤은 화면으로 판정한다(얼굴 오차가 3프레임 동안 0.5° 안).
헤드가 움직이는 중에 고치면 아직 실행되지 않은 보정 위에 보정이 쌓여 넘어간다. 지연 있는 가상 헤드로 돌린
폐루프 테스트에서 처음 방식은 1.9° 오버슈트가 났고, 멈춤 대기 방식은 지연과 무관하게 0°였다
(38° 떨어진 얼굴을 0.8~3.5초에 2.5° 안으로).

### 휴식 자세라도 머리는 멈춰 있지 않다

E의 L0 대기 계층은 어떤 자세 위에도 호흡 흔들림을 얹는다(`motion.idle`: 호흡 2.4°, 시선 1.3°, 배회 0.6°).
D는 휴식 자세를 가정하고 역투영하므로 이 흔들림이 답에 그대로 들어간다. **대기 오프셋을 E의 FK로 재생해 보면
책상 위 한 점이 중앙값 1.1 cm, 최악 위상에서 17.7 cm 움직인다.**

흔들림은 주기적이고 평균이 0에 가깝다. 그래서 호흡 한 주기(4.2초)만큼 평균 내면 대부분 상쇄된다.

| 평균 구간 | 중앙값 | p95 | 최악 |
| --- | --- | --- | --- |
| 한 프레임 | 1.2 cm | 7.7 | 15.8 |
| 2.0 s | 0.9 cm | 5.9 | 12.6 |
| **4.2 s (호흡 1주기)** | **0.7 cm** | **3.9** | **7.4** |
| 8.4 s | 0.4 cm | 2.6 | 4.6 |

D의 카메라는 계속 돌기 때문에 **S1 명령이 올 때는 평균이 이미 쌓여 있다. 촬영 지연이 붙지 않는다.**
`tracking.ObjectAverager`가 휴식 자세에 있는 동안 물체마다 이 평균을 유지하고,
머리가 휴식 자세를 벗어나면 버린다(그 자세로 계산한 값이라 더는 유효하지 않다).

## ROS 노드

`jetson_ws/src/lamp_vision` — `src/vision`을 감싸는 껍데기다. 판단은 전부 `runtime.py`에 있고 노드는 전달만 한다.

| 방향 | 이름 | 형식 | 쓰임 |
| --- | --- | --- | --- |
| 구독 | `/lamp/motion_status` | MotionStatus | `busy` 면 휴식 자세가 깨진 것으로 본다 |
| 구독 | `/lamp/orientation_status` | OrientationStatus | `current_yaw`. E가 내보내는 유일한 관절값 |
| 발행 | `/lamp/track_point` | PointStamped | S2. 볼 지점 |
| 발행 | `/lamp/vision/presence` | Bool | S4. 재실 변화할 때만 |
| 발행 | `/lamp/vision/labels` | String(JSON) | 인지용 한국어 라벨. 변할 때만. **[A 확인 필요]** |
| 발행 | `/lamp/vision/status` | String(JSON) | 진단. 1 Hz |
| 호출 | `/lamp/return_center` | 액션 | 휴식 자세 복귀. D가 카메라 자세를 아는 유일한 자세 |
| 호출 | `/lamp/place_task_light` | 액션 | S1. 조명할 책상 위 지점 |
| 서비스 | `/lamp/vision/center` | Trigger | 휴식 자세 복귀 + 카메라 자세 재고정 |
| 서비스 | `/lamp/vision/place_light` | Trigger | S1 실행 |
| 서비스 | `/lamp/vision/follow_face` | SetBool | S2 on/off. **기본 off** — 부르지 않았는데 쳐다보지 않게 |

라벨 토픽만 D 이름공간(`/lamp/vision/`)에 둔다. A와 메시지 타입을 합의하기 전까지 E의 `lamp_interfaces`를 건드리지 않기 위해서다.

**검출기는 두 속도로 돈다.** 젯슨 CPU에서 YOLOX-s는 한 장에 611 ms, YuNet은 13 ms다. 둘 다 매 프레임 돌리면
얼굴 추적이 1.6 Hz로 떨어진다. 책상 위 물체는 움직이지 않으므로 사물은 0.5초마다, 얼굴은 매 프레임 본다.
실측으로 **얼굴 33 fps, 사물 1.8 Hz**가 나온다. 추종 중에는 어차피 휴식 자세가 아니라 물체 좌표가 무효이므로
사물 검출은 라벨 갱신용으로만 2초마다 돈다.

젯슨에서 띄우는 방법. `rclpy`는 ROS에서, `cv2`·`onnxruntime`은 `vision-venv`에서 온다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/talking-lamp-integration/jetson_ws/install/setup.bash
PYTHONPATH=<repo>/jetson_ws/src/lamp_vision ~/vision-venv/bin/python -m lamp_vision.node
```

## 캘리브레이션

`vision-bench/tools/calib.py`로 한다. 7×5 ChArUco 보드(칸 35 mm) 하나로 두 단계를 끝낸다.

1. **내부 파라미터**: 보드를 여러 각도로 30장. 카메라당 한 번. **초점 판정도 겸한다**.
   인쇄하지 않고 노트북 화면에 띄워도 된다 — 크기와 무관하기 때문이다
2. **헤드→카메라 변환**: 램프를 휴식 자세(`/lamp/return_center`)에 두고, 보드를 책상에 **테이프로 평평하게**
   붙이고 40초쯤 찍는다. 카메라를 다시 붙이지 않는 한 한 번이면 된다

두 번째는 **여러 장을 한꺼번에 맞춘다**(`head_camera.fit_mount`). 자세를 평균 내는 방식은 시도했다가
버렸다 — 평균 자세는 어느 순간과도 맞지 않아서, 한 장이 0.07 cm 일 때 45초 평균이 2.57 cm 였다.
모서리가 다 잡힌 장만 쓴다(일부만 잡힌 장을 섞으면 재투영 오차가 16 px → 45 px 로 나빠진다).

실측 정확도는 **한 장 0.87 cm, 4.2초 평균 0.57 cm**(45~62 cm 구간)다.
근거는 [results/head-camera-2026-09-28.md](../../vision-bench/results/head-camera-2026-09-28.md).

일반 체커보드가 아니라 ChArUco를 쓰는 이유가 있다. 체커보드는 180° 돌려서 읽힐 수 있어서 책상 등록이 거울상으로 뒤집혀도
알아챌 방법이 없다. ChArUco는 코너마다 마커 ID가 붙어 있어서 원점이 모호하지 않다.
이 방향 규약은 `tests/model/test_vision_calibration.py`가 합성 이미지로 검증한다(카메라 위치 5 mm 이내 복원).

## 테스트

```bash
make test                                   # cv2·MuJoCo 없으면 해당 테스트는 건너뜀
```

Jetson(`vision-venv`)에서는 캘리브레이션과 E의 실제 FK를 쓰는 테스트까지 포함해 67개가 돈다.

ROS 노드는 젯슨에서 실물 카메라와 실제 ROS로 확인했다. 토픽·서비스가 다 뜨고, 라벨이 나오고
(`["의자","모니터","키보드"]`), 휴식 자세가 아닐 때 `place_light`가 이유를 붙여 거절한다.
