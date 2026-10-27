#!/usr/bin/env python3
"""라이브 카메라 E2E (③) — 카메라 프레임 → YOLOX 검출 → 한국어 라벨 → 인지모듈.

capture.py(D)의 카메라 캡처 + vision_to_cognition.detect_labels + cognition_run.respond
를 결합한다. 진짜 카메라로 찍어 바로 대사·행동태그까지 뽑는 전 구간 관통.

실행 (detector-venv, 네 파일이 같은 폴더에):
    /mnt/ssd/detector-venv/bin/python live_cognition.py --text "지금 뭐 하는 것 같아?"
    ... --save        # 검출 프레임을 저장해 눈으로 확인
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import onnxruntime as ort

from vision_to_cognition import detect_labels
from cognition_run import respond

SETTLE = 2.0            # 자동초점 수렴 시간 (capture.py 기준)
TMP = "/tmp/live_frame.jpg"


def grab_frame(device=0, width=1280, height=720, settle=SETTLE):
    """카메라에서 가장 선명한 프레임 한 장. capture.py 방식."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"/dev/video{device} 를 열 수 없다 (카메라 연결/사용중 확인)")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    best, best_s = None, -1.0
    t0 = time.time()
    while time.time() - t0 < settle:
        ok, f = cap.read()
        if not ok:
            continue
        s = cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if s > best_s:
            best, best_s = f, s
    cap.release()
    if best is None:
        sys.exit("프레임을 읽지 못했다")
    return best, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="사용자 발화(STT 대신 직접 입력)")
    ap.add_argument("--model", default="/home/asdf/vision-bench/models/yolox_tiny.onnx")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="캡처 프레임 저장")
    a = ap.parse_args()

    print("카메라 켜는 중... (자동초점 수렴 대기)")
    frame, sharp = grab_frame(a.device)
    cv2.imwrite(TMP, frame)
    if a.save:
        out = str(Path.home() / "live_capture.jpg")
        cv2.imwrite(out, frame)
        print(f"  프레임 저장: {out}  (선명도 {sharp:.0f})")

    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    size = sess.get_inputs()[0].shape[2]

    ko, en = detect_labels(TMP, sess, size, a.conf)
    print(f"\n[D 검출기] 라이브 프레임 ({frame.shape[1]}x{frame.shape[0]})")
    print(f"  영어 COCO : {en}")
    print(f"  한국어 라벨: {ko}")

    r, raw = respond(ko, a.text)
    print(f"\n[A 인지모듈] 사용자=\"{a.text}\"")
    print(f"  → say    : {r.say!r}   (C 음성으로)")
    print(f"  → actions: {r.actions}   (E 모터로)")
    print(f"  → end_turn: {r.end_turn}")
    if r.notes:
        print(f"  (파서 방어: {r.notes})")


if __name__ == "__main__":
    main()
