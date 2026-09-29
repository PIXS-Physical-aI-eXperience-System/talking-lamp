"""카메라 캘리브레이션과 책상 등록 — ChArUco 보드 하나로 전부 한다.

순서:
    python calib.py board                        # 인쇄용 보드 이미지 생성
    python calib.py collect                      # 보드를 여러 각도로 보여주면 자동 저장 (초점 판정 겸)
    python calib.py intrinsics --square-mm 35.0  # 내부 파라미터 → ../calib/intrinsics.json
    python calib.py desk --near-cm 20 --square-mm 35.0   # 책상 기준 카메라 자세 → ../calib/pose.json
    python calib.py mount --near-cm 20 --square-mm 35.0  # 헤드→카메라 변환 → ../calib/head_camera.json
    python calib.py check                        # 저장된 캘리브레이션으로 지금 화면의 물체·얼굴 3D 위치

카메라는 램프 헤드에 달려 있다. 그래서 필요한 건 책상 기준 카메라 자세(desk)가 아니라
헤드 기준 카메라 위치(mount)다. mount 는 램프가 **휴식 자세**(/lamp/return_center)에 있을 때 찍는다.
휴식 자세 관절값은 motion.config.REST_POSE 상수라 헤드 자세를 FK 로 알 수 있기 때문이다.

--square-mm 은 **인쇄된 보드의 칸을 자로 잰 값**이다. 프린터가 크기를 조금씩
바꾸므로 공칭 35 mm 를 그대로 쓰지 않는다.

계산은 src/vision 에 있고, 이 스크립트는 카메라와 파일을 다루는 껍데기다.
"""
import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # vision-bench/
sys.path.insert(0, str(ROOT.parent / "src"))           # 저장소의 src/
from vision import Intrinsics, Pose, VisionPipeline      # noqa: E402
from vision import calibration as cal                    # noqa: E402
from vision.detector import (ObjectDetector, FaceDetector,  # noqa: E402
                             DEFAULT_OBJECT_MODEL, DEFAULT_FACE_MODEL)
from vision.head_camera import HeadCameraMount, HeadKinematics, fit_mount  # noqa: E402

CALIB = ROOT / "calib"
VIEWS = CALIB / "views"
INTR = CALIB / "intrinsics.json"
POSE = CALIB / "pose.json"
MOUNT = CALIB / "head_camera.json"
MODELS = ROOT / "models"


def open_cam(device=0, w=1920, h=1080):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    # MJPG 가 아니면 HD 가 안 나온다 (vision-bench/README.md)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        sys.exit(f"/dev/video{device} 를 열 수 없다")
    return cap


def sharpest(cap, seconds=1.5):
    best, bs = None, -1.0
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, f = cap.read()
        if not ok:
            continue
        s = cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if s > bs:
            best, bs = f, s
    return best


# -- board -------------------------------------------------------------------

def cmd_board(a):
    CALIB.mkdir(parents=True, exist_ok=True)
    pps, margin = 210, 40
    img = cal.board_image(pps, margin)
    png = CALIB / "charuco_7x5_35mm.png"
    cv2.imwrite(str(png), img)
    # 인쇄 크기를 mm 로 고정한 SVG. 브라우저에서 "실제 크기(100%)"로 인쇄한다.
    mm_per_px = 35.0 / pps
    w_mm, h_mm = img.shape[1] * mm_per_px, img.shape[0] * mm_per_px
    import base64
    b64 = base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_mm:.2f}mm" height="{h_mm:.2f}mm" '
           f'viewBox="0 0 {img.shape[1]} {img.shape[0]}">'
           f'<image width="{img.shape[1]}" height="{img.shape[0]}" href="data:image/png;base64,{b64}"/></svg>')
    (CALIB / "charuco_7x5_35mm.svg").write_text(svg)
    # 인쇄용은 PDF 를 쓴다. 벡터라 배율 다이얼로그를 건드릴 필요가 없고,
    # 바닥 여백의 100 mm 기준선으로 프린터가 축소했는지 바로 확인된다.
    pdf = CALIB / f"charuco_7x5_{a.square_mm:.0f}mm_a4.pdf"
    pdf.write_bytes(cal.board_pdf(a.square_mm))
    print(f"저장: {png}")
    print(f"저장: {CALIB / 'charuco_7x5_35mm.svg'}  ({w_mm:.0f} x {h_mm:.0f} mm, A4 가로에 들어감)")
    print(f"저장: {pdf}  (A4 가로, 칸 {a.square_mm:.1f} mm, 배율 100%)")
    print("인쇄 후 바닥의 100 mm 기준선을 자로 잰다. 100 mm 가 아니면 그 비율로 --square-mm 을 고친다.")
    print("종이가 휘지 않게 판자에 붙일 것.")


# -- collect -----------------------------------------------------------------

def cmd_collect(a):
    """보드가 보일 때마다, 이전 저장과 충분히 다른 자세면 자동 저장한다.

    초점 판정도 겸한다. 0.4~0.6 m 에서 코너가 안 잡히거나 매번 개수가 크게
    흔들리면 렌즈가 작업 거리에 초점을 못 맞추는 것이다.
    """
    VIEWS.mkdir(parents=True, exist_ok=True)
    cap = open_cam()
    saved = sorted(VIEWS.glob("v*.jpg"))
    centers = []
    print(f"목표 {a.n}장. 보드를 30~70 cm 거리에서 기울이고 옮겨가며 보여준다. 화면 가장자리도 채울 것.")
    print("Ctrl+C 로 중단해도 그때까지 저장된 건 남는다.\n")
    try:
        while len(saved) < a.n:
            ok, f = cap.read()
            if not ok:
                continue
            found = cal.detect(f, a.square_mm / 1000)
            if found is None:
                print("\r  보드 안 보임           ", end="", flush=True)
                continue
            _, img = found
            c = img.mean(0)
            spread = np.ptp(img, axis=0).max()
            s = cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            print(f"\r  코너 {len(img):2d}개  선명도 {s:5.0f}   ", end="", flush=True)
            # 직전 저장들과 중심이 충분히 멀거나 크기가 다를 때만 저장
            if any(np.linalg.norm(c - pc) < 120 and abs(spread - ps) < 80 for pc, ps in centers):
                continue
            centers.append((c, spread))
            p = VIEWS / f"v{len(saved):02d}.jpg"
            cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(p)
            print(f"\n  저장 {len(saved)}/{a.n}: {p.name}  (코너 {len(img)}, 선명도 {s:.0f})")
            time.sleep(0.4)
    except KeyboardInterrupt:
        pass
    cap.release()
    print(f"\n{len(saved)}장 저장됨 → {VIEWS}")


# -- intrinsics --------------------------------------------------------------

def cmd_intrinsics(a):
    files = sorted(VIEWS.glob("v*.jpg"))
    views, size = [], None
    for p in files:
        img = cv2.imread(str(p))
        size = (img.shape[1], img.shape[0])
        found = cal.detect(img, a.square_mm / 1000)
        print(f"  {p.name}: {'코너 %d' % len(found[0]) if found else '보드 못 찾음 — 제외'}")
        if found:
            views.append(found)
    cam = cal.calibrate(views, *size)
    cam.save(INTR)
    print(f"\n저장: {INTR}")
    print(f"  fx={cam.fx:.1f} fy={cam.fy:.1f} cx={cam.cx:.1f} cy={cam.cy:.1f}")
    print(f"  왜곡 {tuple(round(d, 4) for d in cam.dist)}")
    print(f"  재투영 오차 RMS {cam.rms:.3f} px  ({len(views)}장)")
    if cam.rms > 1.0:
        print("  ⚠ 1 px 넘음. 흐린 장을 빼고 다시 하거나, 광각이라 fisheye 모델이 필요할 수 있다.")


# -- desk --------------------------------------------------------------------

def cmd_desk(a):
    if not INTR.exists():
        sys.exit("intrinsics.json 이 없다. intrinsics 먼저.")
    cam = Intrinsics.load(INTR)
    if a.image:
        img = cv2.imread(a.image)
    else:
        cap = open_cam()
        print("보드를 램프 앞 책상에 평평하게 놓았는지 확인. 인쇄물 윗변이 램프에서 먼 쪽.")
        img = sharpest(cap)
        cap.release()
        CALIB.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(CALIB / "desk.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    placement = cal.DeskPlacement(a.near_cm / 100, a.lateral_cm / 100, a.square_mm / 1000,
                                  a.board_mm / 1000)
    pose = cal.register_desk(img, cam, placement)
    pose.save(POSE)
    print(f"저장: {POSE}")
    print(f"  카메라 위치 (lamp_base) x={pose.t[0]:+.3f} y={pose.t[1]:+.3f} z={pose.t[2]:.3f} m")
    print(f"  높이 {pose.height * 100:.1f} cm, 아래로 {pose.tilt_deg:.1f}° 기울어짐")
    print(f"  재투영 오차 RMS {pose.rms_px:.3f} px")
    if pose.height < 0.25:
        print("  ⚠ 25 cm 미만. 납작한 물체 검출이 급격히 나빠지는 높이다 (results/mount-height).")


# -- mount -------------------------------------------------------------------

def cmd_mount(a):
    """헤드→카메라 변환. 카메라를 다시 붙이지 않는 한 한 번이면 된다."""
    if not INTR.exists():
        sys.exit("intrinsics.json 이 없다. intrinsics 먼저.")
    cam = Intrinsics.load(INTR)
    placement = cal.DeskPlacement(a.near_cm / 100, a.lateral_cm / 100, a.square_mm / 1000,
                                  a.board_mm / 1000)
    kin = HeadKinematics()
    q = kin.rest_q(a.yaw)
    head_R, head_t = kin.head(q)
    R_bb, t_bb = placement.board_in_base()
    obs = []
    if a.image:
        f = cv2.imread(a.image)
        found = cal.detect(f, placement.square_m)
        if found is None:
            sys.exit("보드를 못 찾았다.")
        obs.append((found[0] @ R_bb.T + t_bb, found[1]))
        best = f
    else:
        # 한 장으로 끝내지 않는다. E 의 대기 계층(motion.idle)은 휴식 자세에서도
        # 멈추지 않고 헤드를 약 3° RMS 흔든다. 자세를 평균 내는 것으로는 안 된다
        # (평균 자세는 어느 순간과도 맞지 않아 오히려 더 틀린다). 대신 여러 장의
        # 픽셀 오차를 한꺼번에 최소화하는 마운트 하나를 찾는다.
        print("램프가 휴식 자세에서 멈춰 있는지 확인. 보드는 책상에 평평하게.")
        print(f"{a.seconds:.0f}초 동안 여러 장을 찍어 한꺼번에 맞춘다 (대기 모션 흔들림 분산).")
        cap = open_cam()
        CALIB.mkdir(parents=True, exist_ok=True)
        best, bs, t0 = None, -1.0, time.time()
        while time.time() - t0 < a.seconds:
            ok, f = cap.read()
            if not ok:
                continue
            found = cal.detect(f, placement.square_m)
            if found is None:
                continue
            obs.append((found[0] @ R_bb.T + t_bb, found[1]))
            s_ = cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if s_ > bs:
                best, bs = f, s_
            print(f"\r  {len(obs):4d}장", end="", flush=True)
        cap.release()
        if len(obs) < 10:
            sys.exit(f"\n보드를 찍은 장이 {len(obs)}장뿐이다. 보드가 화면에 다 들어오는지 확인.")
        # 모서리가 일부만 잡힌 장은 자세가 크게 흔들린다. 가장 많이 잡힌 수만 남긴다.
        best_n = max(len(o) for o, _ in obs)
        kept = [o for o in obs if len(o[0]) >= best_n]
        print(f"  {len(obs)}장 중 모서리 {best_n}개짜리 {len(kept)}장만 쓴다")
        obs = kept
        cv2.imwrite(str(CALIB / "mount.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print()
    # 한 장만 쓴 결과와 비교해 얼마나 흔들렸는지 알려준다
    mount, rms = fit_mount(obs, cam, head_R, head_t)
    mount.save(MOUNT)
    print(f"저장: {MOUNT}")
    print(f"  사용한 관절값 (rad): {np.round(q, 3).tolist()}")
    print(f"  헤드 기준 카메라 위치: x={mount.t[0]*100:+.1f} y={mount.t[1]*100:+.1f} z={mount.t[2]*100:+.1f} cm")
    print(f"  렌즈 축과 헤드 조준축(+x) 사이 각: {mount.axis_offset_deg:.1f}°")
    print(f"  {len(obs)}장 동시 맞춤, 재투영 오차 RMS {rms:.2f} px")
    # 램프가 흔들리거나 안정되기 전이면 RMS 가 커진다. 실측 정상치는 20 px 근처였다.
    if rms > 35.0:
        print("  ⚠ 재투영 오차가 크다. 램프가 아직 안정되지 않았거나 보드 위치 입력이 틀렸을 수 있다.")
        print("    return_center 뒤 몇 초 기다렸다가 다시 돌린다.")
    if not 0.20 < mount.camera_pose(head_R, head_t).height < 0.50:
        print("  ⚠ 복원된 카메라 높이가 책상 위 20~50 cm 밖이다. 보드 위치 입력을 확인.")


# -- check -------------------------------------------------------------------

def cmd_check(a):
    cam = Intrinsics.load(INTR) if INTR.exists() else Intrinsics.from_fov(1920, 1080, 120)
    if MOUNT.exists():
        pose = HeadKinematics().camera_pose_at_rest(HeadCameraMount.load(MOUNT), a.yaw)
        print("헤드→카메라 변환 + 휴식 자세 FK 로 계산한다. 램프가 휴식 자세여야 맞다.")
    elif POSE.exists():
        pose = Pose.load(POSE)
    else:
        print("캘리브레이션이 없어 가정 자세(높이 30 cm)로 계산한다. 숫자는 참고만.")
        pose = Pose.look_at([-0.05, 0.20, 0.30], [0.45, 0.0, 0.0])
    print(f"내부 파라미터: {cam.source}, 자세: {pose.source}")
    pipe = VisionPipeline(cam, pose,
                          ObjectDetector(MODELS / a.model, a.conf),
                          FaceDetector(MODELS / DEFAULT_FACE_MODEL))
    cap = open_cam()
    img = sharpest(cap)
    cap.release()
    t0 = time.perf_counter()
    vf = pipe.process(img)
    dt = (time.perf_counter() - t0) * 1000
    print(f"처리 {dt:.0f} ms\n")
    for o in vf.objects:
        print(f"  물체 {o.label:10s} {o.conf:.2f}  ({o.pos[0]:+.3f}, {o.pos[1]:+.3f}, {o.pos[2]:+.3f}) m")
    for f in vf.faces:
        print(f"  얼굴 {'':10s} {f.conf:.2f}  ({f.pos[0]:+.3f}, {f.pos[1]:+.3f}, {f.pos[2]:+.3f}) m"
              f"   거리 {np.linalg.norm(f.pos - pose.t):.2f} m")
    for label, why in vf.rejected:
        print(f"  제외 {label:10s} {why}")
    if not (vf.objects or vf.faces or vf.rejected):
        print("  검출 없음")
    print(f"\n인지용 라벨: {vf.labels_ko()}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("board"); p.add_argument("--square-mm", type=float, default=35.0)
    p = sub.add_parser("collect"); p.add_argument("--n", type=int, default=20)
    p.add_argument("--square-mm", type=float, default=35.0)
    p = sub.add_parser("intrinsics"); p.add_argument("--square-mm", type=float, required=True)
    p = sub.add_parser("desk")
    p.add_argument("--near-cm", type=float, required=True, help="램프 베이스 중심 → 보드 격자 가까운 변")
    p.add_argument("--lateral-cm", type=float, default=0.0, help="보드 중심의 좌우 오프셋 (+ 가 램프 왼쪽)")
    p.add_argument("--square-mm", type=float, required=True)
    p.add_argument("--board-mm", type=float, default=0.0, help="보드 면이 책상보다 높은 높이 (받침 두께)")
    p.add_argument("--image", help="카메라 대신 저장된 이미지 사용")
    p = sub.add_parser("mount")
    p.add_argument("--near-cm", type=float, required=True, help="램프 베이스 중심 → 보드 격자 가까운 변")
    p.add_argument("--lateral-cm", type=float, default=0.0)
    p.add_argument("--square-mm", type=float, required=True)
    p.add_argument("--board-mm", type=float, default=0.0, help="보드 면이 책상보다 높은 높이 (받침 두께)")
    p.add_argument("--seconds", type=float, default=45.0, help="촬영 시간. 길수록 대기 모션이 고르게 섞인다")
    p.add_argument("--yaw", type=float, help="base_yaw (rad). /lamp/orientation_status 의 current_yaw. 생략하면 REST_POSE 값")
    p.add_argument("--image", help="카메라 대신 저장된 이미지 사용")
    p = sub.add_parser("check")
    p.add_argument("--yaw", type=float, help="base_yaw (rad), mount 사용 시")
    p.add_argument("--model", default=DEFAULT_OBJECT_MODEL)
    p.add_argument("--conf", type=float, default=0.3)
    a = ap.parse_args()
    {"board": cmd_board, "collect": cmd_collect, "intrinsics": cmd_intrinsics,
     "desk": cmd_desk, "mount": cmd_mount, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
