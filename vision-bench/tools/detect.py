"""촬영한 이미지에 YOLOX를 돌려 검출 결과를 본다.

박스 하단 중심(x, y_bottom)을 함께 찍는다. 2D→3D 환산을 책상 평면 역투영으로
하므로, 물체가 책상에 닿는 지점이 곧 조명 타겟점이 된다. 높이별 이미지를
비교해 이 지점이 얼마나 안정적인지 보는 것이 목적이다.

사용:
    python detect.py ../images/h029.jpg --model ../models/yolox_tiny.onnx
    python detect.py ../images/*.jpg --save      # 박스 그린 이미지도 남긴다
"""
import argparse, sys
from pathlib import Path
import numpy as np, cv2, onnxruntime as ort

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

# 책상 위에 실제로 놓이는 것들. S1 조명 타겟 후보로 이것만 본다.
DESK = {"book","keyboard","mouse","laptop","cell_phone","cup","bottle",
        "scissors","remote","bowl","clock","vase","person"}


def preproc(img, size):
    """YOLOX 공식 전처리 — 114로 채운 캔버스에 비율 유지 축소, 정규화 없음."""
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    r = min(size / img.shape[0], size / img.shape[1])
    rh, rw = int(img.shape[0] * r), int(img.shape[1] * r)
    canvas[:rh, :rw] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32), r


def decode(out, size):
    """격자·스트라이드를 곱해 실제 픽셀 좌표로 되돌린다."""
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


def nms(boxes, scores, thr=0.45):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thr]
    return keep


def run(path, sess, size, conf_thr, save):
    img = cv2.imread(str(path))
    if img is None:
        print(f"{path}: 읽기 실패"); return
    blob, r = preproc(img, size)
    out = decode(sess.run(None, {sess.get_inputs()[0].name: blob})[0], size)[0]

    scores = out[:, 4:5] * out[:, 5:]
    cls = scores.argmax(1); conf = scores[np.arange(len(cls)), cls]
    m = conf > conf_thr
    if not m.any():
        print(f"\n{Path(path).name}: 검출 없음"); return
    box = out[m, :4]; cls = cls[m]; conf = conf[m]
    xy = np.stack([box[:,0]-box[:,2]/2, box[:,1]-box[:,3]/2,
                   box[:,0]+box[:,2]/2, box[:,1]+box[:,3]/2], 1) / r

    print(f"\n{Path(path).name}  ({img.shape[1]}x{img.shape[0]})")
    print(f"  {'클래스':12s} {'신뢰도':>6s} {'박스 하단 중심(px)':>20s}  {'책상물체':>8s}")
    for i in nms(xy, conf):
        name = COCO[cls[i]]
        bx, by = (xy[i,0]+xy[i,2])/2, xy[i,3]      # 책상과 닿는 지점
        print(f"  {name:12s} {conf[i]:6.2f} {f'({bx:.0f}, {by:.0f})':>20s}  "
              f"{'O' if name in DESK else '':>8s}")
        if save:
            cv2.rectangle(img, (int(xy[i,0]),int(xy[i,1])), (int(xy[i,2]),int(xy[i,3])), (0,200,0), 2)
            cv2.circle(img, (int(bx),int(by)), 5, (0,0,255), -1)
            cv2.putText(img, f"{name} {conf[i]:.2f}", (int(xy[i,0]),int(xy[i,1])-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)
    if save:
        p = Path(path).with_name(Path(path).stem + "_det.jpg")
        cv2.imwrite(str(p), img); print(f"  → {p.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=str(Path(__file__).parent.parent/"models"/"yolox_tiny.onnx"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--save", action="store_true", help="박스를 그린 이미지를 저장")
    a = ap.parse_args()

    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    size = sess.get_inputs()[0].shape[2]
    print(f"모델 {Path(a.model).name} (입력 {size}x{size}), 신뢰도 임계 {a.conf}")
    print("빨간 점 = 박스 하단 중심. 물체가 책상에 닿는 지점이며 조명 타겟점의 기준이다.")
    for p in a.images:
        run(p, sess, size, a.conf, a.save)

if __name__ == "__main__":
    main()
