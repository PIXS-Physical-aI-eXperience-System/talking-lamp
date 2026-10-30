#!/usr/bin/env python3
"""A↔D 브릿지 (③ 분류기 연계) — 이미지 → YOLOX 검출 → 한국어 라벨 → 인지모듈.

    이미지 →[D의 YOLOX]→ 영어 COCO 라벨 →[매핑]→ 한국어 라벨 →[respond()]→ {say, actions}

D(김아현)의 vision-bench/tools/detect.py 검출 로직(preproc/decode/nms)을 그대로
재사용한다. detect.py 는 결과를 print 만 하므로, 여기서 '라벨 리스트'를 뽑아
A파트 인지모듈(cognition_run.respond)에 넘긴다.

실행 (detector-venv 로, 세 파일이 같은 폴더에 있어야 함):
    /mnt/ssd/detector-venv/bin/python vision_to_cognition.py \
        ~/vision-bench/images/h029.jpg --text "지금 뭐 하는 것 같아?"

주의: 검출기는 영어 COCO 라벨을 낸다. 한국어 매핑(KO)은 A쪽 임시 처리이며,
최종 인터페이스(누가 매핑·어떤 형식)는 D와 합의할 항목이다.
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort

from cognition_run import respond

COCO = (
 "person bicycle car motorcycle airplane bus train truck boat traffic_light "
 "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
 "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
 "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
 "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
 "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
 "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
 "cell_phone microwave oven toaster sink refrigerator book clock vase "
 "scissors teddy_bear hair_drier toothbrush").split()

# COCO(영어) → 한국어 라벨. 인지모듈 프롬프트가 한국어를 쓰므로 변환한다.
KO = {
    "person": "사람", "laptop": "노트북", "cup": "컵", "bottle": "물병",
    "book": "책", "keyboard": "키보드", "mouse": "마우스", "cell_phone": "휴대폰",
    "remote": "리모컨", "bowl": "그릇", "clock": "시계", "vase": "꽃병",
    "scissors": "가위", "chair": "의자", "tv": "모니터", "wine_glass": "와인잔",
    "potted_plant": "화분", "teddy_bear": "인형", "dining_table": "책상",
    "fork": "포크", "knife": "칼", "spoon": "숟가락", "banana": "바나나",
    "apple": "사과", "backpack": "가방", "handbag": "가방",
}


def preproc(img, size):
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    r = min(size / img.shape[0], size / img.shape[1])
    rh, rw = int(img.shape[0] * r), int(img.shape[1] * r)
    canvas[:rh, :rw] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32), r


def decode(out, size):
    grids, strides_all = [], []
    for s in (8, 16, 32):
        n = size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
        strides_all.append(np.full((1, n * n, 1), s))
    g = np.concatenate(grids, 1); st = np.concatenate(strides_all, 1)
    out[..., :2] = (out[..., :2] + g) * st
    out[..., 2:4] = np.exp(out[..., 2:4]) * st
    return out


def detect_labels(path, sess, size, conf_thr=0.3):
    """이미지 → 검출된 한국어 라벨 리스트(중복 제거, 신뢰도순)."""
    img = cv2.imread(str(path))
    if img is None:
        return [], []
    blob, r = preproc(img, size)
    out = decode(sess.run(None, {sess.get_inputs()[0].name: blob})[0], size)[0]
    scores = out[:, 4:5] * out[:, 5:]
    cls = scores.argmax(1); conf = scores[np.arange(len(cls)), cls]
    m = conf > conf_thr
    en, seen = [], set()
    for i in conf[m].argsort()[::-1]:
        name = COCO[cls[m][i]]
        if name not in seen:
            seen.add(name); en.append(name)
    ko = [KO.get(n, n) for n in en]        # 매핑 없으면 영어 그대로
    return ko, en


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--text", required=True, help="사용자 발화(STT 대신 직접 입력)")
    ap.add_argument("--model", default="/home/asdf/vision-bench/models/yolox_tiny.onnx")
    ap.add_argument("--conf", type=float, default=0.3)
    a = ap.parse_args()

    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    size = sess.get_inputs()[0].shape[2]

    ko, en = detect_labels(a.image, sess, size, a.conf)
    print(f"[D 검출기] {Path(a.image).name}")
    print(f"  영어 COCO : {en}")
    print(f"  한국어 라벨: {ko}")

    r, raw = respond(ko, a.text)
    print(f"\n[A 인지모듈] 사용자=\"{a.text}\"")
    print(f"  LLM 원문 : {raw.strip()[:150]}")
    print(f"  → say    : {r.say!r}   (C 음성으로)")
    print(f"  → actions: {r.actions}   (E 모터로)")
    print(f"  → end_turn: {r.end_turn}")
    if r.notes:
        print(f"  (파서 방어: {r.notes})")


if __name__ == "__main__":
    main()
