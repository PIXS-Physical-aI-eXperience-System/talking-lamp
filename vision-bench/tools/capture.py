"""높이별 책상 촬영 — 카메라 마운트 위치 결정용.

같은 장면을 여러 높이에서 찍어 두고, detect.py 로 높이에 따라 박스 하단
중심이 실제 접점과 얼마나 어긋나는지 비교한다.

사용:
    python capture.py 29                 # 29 cm 높이에서 촬영
    python capture.py 29 --rotate 90     # 카메라를 눕혀 들었을 때 보정
    python capture.py 29 --delay 10      # 자리 잡을 시간을 더 준다
"""
import argparse, time, sys
from pathlib import Path
import cv2

OUT = Path(__file__).resolve().parent.parent / "images"

# 라플라시안 분산. 또렷한 실내 사진은 보통 수백 이상이고,
# 한 자리수면 초점이 안 맞았거나 대비가 없는 면을 보고 있는 것이다.
SHARP_WARN = 60.0
SETTLE_SEC = 3.0          # 자동초점이 수렴할 시간


def sharpness(frame):
    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("height_cm", type=float, help="책상면에서 렌즈 중심까지 높이 (cm)")
    ap.add_argument("--delay", type=int, default=5, help="자리 잡을 카운트다운 초")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="저장 전 회전. 카메라를 눕혀 들었으면 지정한다")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"/dev/video{a.device} 를 열 수 없다")
    # MJPG 가 아니면 HD 가 안 나온다. 압축하지 않은 YUYV 로는 USB 대역폭이
    # 모자라 1280x720 이상에서 select() 타임아웃이 난다 (실측 60초).
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)

    print(f"높이 {a.height_cm:g} cm — 카메라를 자리에 두고 기다린다")
    for s in range(a.delay, 0, -1):
        print(f"  {s}...", flush=True)
        cap.read()
        time.sleep(1)

    # 이 카메라는 자동초점이고 수동 조절이 막혀 있다. 흰 벽처럼 대비가 없는
    # 면에서는 초점이 늦게 잡히므로, 잠시 계속 읽으면서 가장 선명한 장을 고른다.
    print(f"  초점 수렴 대기 {SETTLE_SEC:g}초...", flush=True)
    best, best_s, n = None, -1.0, 0
    t0 = time.time()
    while time.time() - t0 < SETTLE_SEC:
        ok, frame = cap.read()
        if not ok:
            continue
        n += 1
        s = sharpness(frame)
        if s > best_s:
            best, best_s = frame, s
    cap.release()

    if best is None:
        sys.exit("프레임을 읽지 못했다")

    if a.rotate:
        code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}[a.rotate]
        best = cv2.rotate(best, code)

    path = OUT / f"h{int(round(a.height_cm)):03d}.jpg"
    cv2.imwrite(str(path), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = best.shape[:2]
    print(f"저장: {path}  ({w}x{h})  {n}장 중 가장 선명한 것, 선명도 {best_s:.0f}")
    if best_s < SHARP_WARN:
        print(f"  ⚠ 선명도가 {SHARP_WARN:.0f} 미만이다. 초점이 안 맞았거나 흔들렸다.")
        print(f"    물건이 화면 가운데 오게 하고, 카메라를 받쳐서 다시 찍을 것.")

if __name__ == "__main__":
    main()
