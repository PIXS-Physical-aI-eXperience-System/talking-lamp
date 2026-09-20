"""브라우저로 카메라를 보면서 초점을 맞추고 그 자리에서 촬영한다.

젯슨에 화면이 없으므로 MJPEG 로 흘려보낸다. 렌즈를 돌리면서 선명도 숫자가
커지는 쪽을 찾고, 원하는 장면이 되면 페이지에서 바로 저장한다.

`--detect` 를 주면 검출 박스를 실시간으로 겹쳐 그린다. 카메라를 옮겨가며
"이 위치에서 키보드가 잡히는지"를 눈으로 바로 확인할 수 있다. 검출은 영상과
별도 스레드로 돌아 미리보기 프레임률을 떨어뜨리지 않는다.

사용:
    python live.py                         # 영상만
    python live.py --detect                # 검출 박스 겹쳐 보기
    python live.py --detect --model ../models/yolox_nano.onnx
"""
import argparse, threading, time, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect import preproc, decode, nms, COCO, DESK  # noqa: E402

import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "images"
ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
PREVIEW_W = 960

state = {
    "frame": None,      # 최신 원본 프레임
    "sharp": 0.0,
    "dets": [],         # [(x1,y1,x2,y2,name,conf), ...] 원본 해상도 좌표
    "dropped": [],      # 자기 몸으로 판정해 버린 것들 (이름만)
    "det_ms": 0.0,
    "self_pct": 0.0,    # 화면에서 램프 자기 몸이 차지하는 비율 %
    "mask": None,       # 미리보기용 마스크 (축소본)
    "model": None,      # 현재 쓰는 검출 모델 경로. 페이지에서 바꿀 수 있다
    "conf": 0.25,
    "filter_self": False,
    "lock": threading.Lock(),
    "stop": False,
}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# 램프 자기 몸 색. 3D 출력물이 주황색이라 색으로 구분할 수 있다.
# 이것은 임시 방편이다. 제대로 된 해법은 4절 주석 참고.
SELF_HSV = {"hue_lo": 3, "hue_hi": 22, "sat_min": 110, "val_min": 60}
SELF_OVERLAP = 0.35     # 박스의 이 비율 이상이 자기 몸이면 버린다


def self_mask(frame, cfg):
    """램프 자기 몸(주황 출력물)을 색으로 골라낸다.

    카메라를 램프에 달면 팔과 헤드가 항상 화면에 들어온다. 그대로 두면
    헤드가 book/cup 등으로 오검출되어 조명이 자기 머리를 겨냥하게 된다.

    색 기반이라 한계가 뚜렷하다. 책상에 주황색 물건이 있으면 같이 지워지고,
    기구를 다른 색으로 출력하면 못 쓴다. 근본 해법은 둘 중 하나다.
      - 기하: 책상 평면 역투영 결과가 책상면 위에 안 떨어지는 검출을 버린다
      - 자기 가림 마스크: 관절값으로 램프 자세를 알아 자기 몸을 렌더해 지운다
        (관절값 발행이 선결이라 지금은 불가)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = np.array([cfg["hue_lo"], cfg["sat_min"], cfg["val_min"]], np.uint8)
    hi = np.array([cfg["hue_hi"], 255, 255], np.uint8)
    m = cv2.inRange(hsv, lo, hi)
    # 잡티를 없애고 덩어리만 남긴다
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def grabber(device, width, height):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    while not state["stop"]:
        ok, fr = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        s = cv2.Laplacian(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        with state["lock"]:
            state["frame"], state["sharp"] = fr, s
    cap.release()


def detector():
    """최신 프레임을 계속 물어 검출한다. 영상 스레드와 독립적으로 돈다.

    모델은 페이지에서 바꿀 수 있다. 세션은 한 번 만든 것을 캐시해 두므로
    되돌아와도 다시 로드하지 않는다.
    """
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sessions = {}

    while not state["stop"]:
        with state["lock"]:
            fr = None if state["frame"] is None else state["frame"].copy()
            model_path = state["model"]
            conf_thr = state["conf"]
            filter_self = state["filter_self"]
        if fr is None or model_path is None:
            time.sleep(0.05)
            continue

        if model_path not in sessions:
            s = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
            sessions[model_path] = (s, s.get_inputs()[0].name, s.get_inputs()[0].shape[2])
            print(f"모델 적재: {Path(model_path).name} (입력 {sessions[model_path][2]})")
        sess, name_in, size = sessions[model_path]

        t0 = time.perf_counter()
        mask = self_mask(fr, SELF_HSV) if filter_self else None
        self_pct = float(mask.mean() / 255 * 100) if mask is not None else 0.0

        blob, r = preproc(fr, size)
        out = decode(sess.run(None, {name_in: blob})[0], size)[0]
        sc = out[:, 4:5] * out[:, 5:]
        cls = sc.argmax(1)
        conf = sc[np.arange(len(cls)), cls]
        m = conf > conf_thr

        dets, dropped = [], []
        if m.any():
            b = out[m, :4]
            xy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                           b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) / r
            cls, conf = cls[m], conf[m]
            H, W = fr.shape[:2]
            for i in nms(xy, conf):
                x1, y1, x2, y2 = xy[i]
                name = COCO[cls[i]]
                if mask is not None:
                    a, bb = int(max(0, x1)), int(max(0, y1))
                    c, d2 = int(min(W, x2)), int(min(H, y2))
                    if c > a and d2 > bb:
                        # 박스 안에서 자기 몸이 차지하는 비율
                        frac = mask[bb:d2, a:c].mean() / 255
                        if frac > SELF_OVERLAP:
                            dropped.append(f"{name} {conf[i]:.2f}")
                            continue
                dets.append((float(x1), float(y1), float(x2), float(y2), name, float(conf[i])))

        dt = (time.perf_counter() - t0) * 1000
        small_mask = None
        if mask is not None:
            sw = PREVIEW_W
            small_mask = cv2.resize(mask, (sw, int(mask.shape[0] * sw / mask.shape[1])),
                                    interpolation=cv2.INTER_NEAREST)
        with state["lock"]:
            state["dets"], state["dropped"] = dets, dropped
            state["det_ms"], state["self_pct"], state["mask"] = dt, self_pct, small_mask


def draw(frame, dets, scale):
    """미리보기 크기에 맞춰 박스와 박스 하단 중심을 그린다."""
    for x1, y1, x2, y2, name, conf in dets:
        desk = name in DESK
        col = (0, 200, 0) if desk else (140, 140, 140)
        p1 = (int(x1 * scale), int(y1 * scale))
        p2 = (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(frame, p1, p2, col, 2)
        # 박스 하단 중심 = 물체가 책상에 닿는 지점. 조명 타겟점의 기준이다.
        cv2.circle(frame, (int((x1 + x2) / 2 * scale), int(y2 * scale)), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"{name} {conf:.2f}", (p1[0], max(14, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return frame


PAGE = """<!doctype html><meta charset=utf-8><title>카메라</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
 .wrap{max-width:1100px;margin:0 auto;padding:12px}
 img{width:100%;border-radius:8px;display:block;background:#000}
 .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}
 .s{font-size:30px;font-variant-numeric:tabular-nums;min-width:5ch}
 .bar{flex:1;height:14px;background:#333;border-radius:7px;overflow:hidden}
 .fill{height:100%;background:#4ade80;width:0%}
 label{font-size:14px;color:#aaa}
 input,select,button{font-size:15px;padding:7px 10px;border-radius:6px;border:1px solid #444;background:#222;color:#eee}
 button{background:#2563eb;border-color:#2563eb;cursor:pointer}
 #msg{font-size:14px;color:#4ade80;min-height:1.3em}
 .hint{font-size:13px;color:#888}
 #dets{font-size:13px;color:#ddd;min-height:2.6em;line-height:1.5}
 .chip{display:inline-block;background:#1f3d24;color:#86efac;border-radius:5px;padding:2px 8px;margin:2px 4px 2px 0}
 .chip.off{background:#2a2a2a;color:#999}
 .chip.drop{background:#3a2020;color:#fca5a5;text-decoration:line-through}
 .self{font-size:14px;color:#fbbf24}
</style>
<div class=wrap>
 <div class=row>
   <span class=s id=sharp>—</span>
   <div class=bar><div class=fill id=fill></div></div>
   <span class=hint id=dms></span>
   <span class=self id=selfp></span>
 </div>
 <div class=hint>렌즈를 돌리면서 숫자가 커지는 쪽을 찾는다. 또렷하면 보통 수백 이상.</div>
 <img id=v src="/stream">
 <div id=dets></div>
 <div class=row>
   <label>모델 <select id=m onchange=setmodel()></select></label>
   <label>임계 <input id=c type=number value=0.25 step=0.05 min=0.05 max=0.9 style=width:7ch onchange=setmodel()></label>
   <label><input id=fs type=checkbox onchange=setmodel()> 자기 몸 필터</label>
 </div>
 <div class=row>
   <label>높이(cm) <input id=h type=number value=29 step=1 style=width:7ch></label>
   <label>회전 <select id=r><option>0</option><option>90</option><option>180</option><option>270</option></select></label>
   <button onclick=shot()>이 장면 저장</button>
   <span id=msg></span>
 </div>
 <div class=hint>초록 박스 = 책상 물체, 회색 = 그 외. 빨간 점 = 박스 하단 중심(책상과 닿는 지점).</div>
</div>
<script>
(async()=>{
  const d=await (await fetch('/models')).json();
  m.innerHTML=d.models.map(x=>`<option value="${x.path}"${x.path===d.current?' selected':''}>${x.name}</option>`).join('');
  c.value=d.conf; fs.checked=d.filter_self;
})();
async function setmodel(){
  await fetch(`/model?path=${encodeURIComponent(m.value)}&conf=${c.value}&filter_self=${fs.checked?1:0}`);
}
setInterval(async()=>{
  const d=await (await fetch('/stat')).json();
  sharp.textContent=d.sharp.toFixed(0);
  fill.style.width=Math.min(100,d.sharp/800*100)+'%';
  fill.style.background=d.sharp<60?'#f87171':d.sharp<200?'#fbbf24':'#4ade80';
  dms.textContent=d.det_ms?`검출 ${d.det_ms.toFixed(0)}ms`:'';
  selfp.textContent=d.self_pct?`램프 자기 몸이 화면의 ${d.self_pct.toFixed(0)}%`:'';
  const a=d.dets.map(x=>`<span class="chip${x.desk?'':' off'}">${x.name} ${x.conf.toFixed(2)}</span>`);
  const b=d.dropped.map(x=>`<span class="chip drop">${x}</span>`);
  dets.innerHTML=(a.length||b.length)?a.concat(b).join(''):'<span class=hint>검출 없음</span>';
},300);
async function shot(){
  msg.textContent='저장 중...';
  const d=await (await fetch(`/shot?h=${h.value}&rotate=${r.value}`)).json();
  msg.textContent=d.ok?`저장됨 ${d.name} (선명도 ${d.sharp.toFixed(0)})`:'실패';
}
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)

        if u.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif u.path == "/stat":
            with state["lock"]:
                out = {
                    "sharp": state["sharp"],
                    "det_ms": state["det_ms"],
                    "self_pct": state["self_pct"],
                    "dropped": list(state["dropped"]),
                    "dets": [{"name": d[4], "conf": d[5], "desk": d[4] in DESK}
                             for d in state["dets"]],
                }
            self._json(out)

        elif u.path == "/models":
            # yunet 은 출력 형식이 달라 이 디코더로 못 읽는다. 사물 검출기만 고른다.
            names = sorted(p for p in MODELS_DIR.glob("yolox_*.onnx"))
            with state["lock"]:
                cur, conf, fs = state["model"], state["conf"], state["filter_self"]
            self._json({
                "models": [{"name": p.name, "path": str(p)} for p in names],
                "current": cur, "conf": conf, "filter_self": fs,
            })

        elif u.path == "/model":
            q = parse_qs(u.query)
            with state["lock"]:
                if "path" in q:
                    state["model"] = q["path"][0]
                if "conf" in q:
                    state["conf"] = float(q["conf"][0])
                if "filter_self" in q:
                    state["filter_self"] = q["filter_self"][0] == "1"
                state["dets"], state["dropped"] = [], []
            self._json({"ok": True})

        elif u.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with state["lock"]:
                        fr = None if state["frame"] is None else state["frame"].copy()
                        dets = list(state["dets"])
                        msk = None if state["mask"] is None else state["mask"].copy()
                    if fr is not None:
                        # 미리보기는 줄여 보낸다. 전송량을 아끼고 지연을 줄인다.
                        scale = PREVIEW_W / fr.shape[1]
                        small = cv2.resize(fr, (PREVIEW_W, int(fr.shape[0] * scale)))
                        if msk is not None and msk.shape[:2] == small.shape[:2]:
                            # 자기 몸으로 판정한 영역을 어둡게 덮어 가림 범위를 보이게 한다
                            small[msk > 0] = (small[msk > 0] * 0.35).astype(small.dtype)
                        if dets:
                            small = draw(small, dets, scale)
                        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ok:
                            self.wfile.write(
                                b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                            )
                    time.sleep(0.06)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif u.path == "/shot":
            q = parse_qs(u.query)
            hcm = int(float(q.get("h", ["0"])[0]))
            rot = int(q.get("rotate", ["0"])[0])
            with state["lock"]:
                fr = None if state["frame"] is None else state["frame"].copy()
                s = state["sharp"]
            if fr is None:
                self._json({"ok": False})
                return
            if rot in ROT:
                fr = cv2.rotate(fr, ROT[rot])
            OUT.mkdir(parents=True, exist_ok=True)
            name = f"h{hcm:03d}.jpg"
            # 저장은 박스를 그리지 않은 원본으로 한다. 나중에 detect.py 로 다시 돌린다.
            cv2.imwrite(str(OUT / name), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"저장: {name}  선명도 {s:.0f}")
            self._json({"ok": True, "name": name, "sharp": s})

        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--detect", action="store_true", help="검출 박스를 겹쳐 그린다")
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent.parent / "models" / "yolox_tiny.onnx"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--filter-self", action="store_true",
                    help="램프 자기 몸(주황 출력물)을 색으로 걸러낸다")
    ap.add_argument("--hue", default="3,22", help="자기 몸 색상 범위 H_lo,H_hi (OpenCV 0~179)")
    ap.add_argument("--sat-min", type=int, default=110)
    ap.add_argument("--val-min", type=int, default=60)
    a = ap.parse_args()

    lo, hi = (int(v) for v in a.hue.split(","))
    SELF_HSV.update(hue_lo=lo, hue_hi=hi, sat_min=a.sat_min, val_min=a.val_min)

    state["model"] = a.model if a.detect else None
    state["conf"] = a.conf
    state["filter_self"] = a.filter_self

    threading.Thread(target=grabber, args=(a.device, a.width, a.height), daemon=True).start()
    time.sleep(1.5)
    if a.detect:
        threading.Thread(target=detector, daemon=True).start()

    print(f"브라우저에서 열 것:  http://100.79.117.124:{a.port}")
    print("Ctrl+C 로 종료")
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        state["stop"] = True
        print("\n종료")


if __name__ == "__main__":
    main()
