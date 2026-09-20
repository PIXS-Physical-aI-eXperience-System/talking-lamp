"""초점 맞추기 보조 — 선명도를 실시간으로 찍어준다.

이 카메라는 소프트웨어로 초점을 조절할 수 없다(FOCUS 설정이 거부된다).
렌즈를 손으로 돌리면서 이 숫자가 커지는 쪽을 찾으면 된다.

사용:
    python focus.py            # Ctrl+C 로 종료
    python focus.py --save     # 종료할 때 가장 선명했던 장을 저장
"""
import argparse, time, sys
from pathlib import Path
import cv2

def sharpness(frame):
    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def bar(v, lo=0, hi=800, width=40):
    n = int(min(max(v - lo, 0) / (hi - lo), 1.0) * width)
    return "█" * n + "·" * (width - n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"/dev/video{a.device} 를 열 수 없다")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("렌즈를 천천히 돌리면서 숫자가 커지는 쪽을 찾는다. Ctrl+C 로 끝낸다.")
    print("또렷한 실내 장면은 보통 수백 이상 나온다.\n")
    best, best_s = None, -1.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            s = sharpness(frame)
            if s > best_s:
                best, best_s = frame, s
            print(f"\r  선명도 {s:7.1f}  {bar(s)}  (최고 {best_s:.0f})", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(f"\n\n최고 선명도: {best_s:.0f}")
        if best_s < 60:
            print("  여전히 낮다. 렌즈 보호 필름, 최소 초점거리(30~50cm)를 확인할 것.")
        if a.save and best is not None:
            p = Path(__file__).resolve().parent.parent / "images" / "_focus_best.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  저장: {p}")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
