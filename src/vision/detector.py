"""Object (YOLOX) and face (YuNet) detectors on ONNX Runtime, CPU only.

CPU is a deliberate choice, not a fallback. On the Jetson a CUDA session costs
~900 MB of peak RSS regardless of model size (nano 897 MB vs s 968 MB), which
breaks the 0.3-0.7 GB budget in docs/파트-분배.md 2.3. The same models on CPU
take 84-190 MB. Measurements: vision-bench/results/detector-2026-09-20.md.

``decode`` and ``nms`` are pure numpy so they can be unit-tested without
OpenCV; everything that touches images imports cv2 lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import COCO, DESK, TASK_LIGHT

# Default models. YOLOX-s (vision-bench/README.md 3): S1 is event-driven, so
# its 281 ms CPU latency costs nothing. With the lens focused, s and tiny both
# boxed a flat notepad equally well; the earlier "s misses the notebook" was a
# blurred frame with the pad cut off at the edge. tiny additionally called an
# empty basket "laptop" at 0.84, which would send the light at nothing.
DEFAULT_OBJECT_MODEL = "yolox_s.onnx"
DEFAULT_FACE_MODEL = "yunet_2023mar.onnx"


@dataclass(frozen=True)
class Detection:
    label: str
    conf: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in full-frame pixels

    @property
    def on_desk_class(self) -> bool:
        return self.label in DESK

    @property
    def task_light_class(self) -> bool:
        return self.label in TASK_LIGHT

    @property
    def ground_uv(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the object touches the desk."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)


@dataclass(frozen=True)
class FaceDetection:
    conf: float
    box: tuple[float, float, float, float]
    right_eye: tuple[float, float]
    left_eye: tuple[float, float]


# -- YOLOX post-processing (pure numpy) --------------------------------------

def decode(out: np.ndarray, size: int) -> np.ndarray:
    """Grid/stride decode of raw YOLOX output ``[1, N, 85]`` to pixel boxes."""
    out = out.copy()
    grids, strides = [], []
    for s in (8, 16, 32):
        n = size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
        strides.append(np.full((1, n * n, 1), s))
    g = np.concatenate(grids, 1)
    st = np.concatenate(strides, 1)
    out[..., :2] = (out[..., :2] + g) * st
    out[..., 2:4] = np.exp(out[..., 2:4]) * st
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.45) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        order = order[1:][iou <= thr]
    return keep


def postprocess(raw: np.ndarray, size: int, scale: float, conf_thr: float,
                nms_thr: float = 0.45) -> list[Detection]:
    out = decode(raw, size)[0]
    scores = out[:, 4:5] * out[:, 5:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(cls)), cls]
    m = conf > conf_thr
    if not m.any():
        return []
    b = out[m, :4]
    xy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                   b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) / scale
    cls, conf = cls[m], conf[m]
    return [Detection(COCO[cls[i]], float(conf[i]), tuple(float(v) for v in xy[i]))
            for i in nms(xy, conf, nms_thr)]


# -- detectors ---------------------------------------------------------------

def _cpu_session(path: str | Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class ObjectDetector:
    """YOLOX on CPU. Input size is read from the model (416 or 640)."""

    def __init__(self, model_path: str | Path, conf_thr: float = 0.3):
        self.sess = _cpu_session(model_path)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        self.conf_thr = conf_thr
        self.name = Path(model_path).name

    def _preproc(self, img):
        import cv2
        s = self.size
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        r = min(s / img.shape[0], s / img.shape[1])
        rh, rw = int(img.shape[0] * r), int(img.shape[1] * r)
        canvas[:rh, :rw] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32), r

    def __call__(self, bgr: np.ndarray) -> list[Detection]:
        blob, r = self._preproc(bgr)
        raw = self.sess.run(None, {self.input_name: blob})[0]
        return postprocess(raw, self.size, r, self.conf_thr)


class FaceDetector:
    """YuNet through OpenCV. Runs on a downscaled copy for speed.

    The measured 16.2 ms/frame is at 640 px wide; feeding it the full 1920x1080
    frame would cost several times that for no gain at desk distances.
    """

    def __init__(self, model_path: str | Path, conf_thr: float = 0.6, input_width: int = 640):
        import cv2
        self._cv2 = cv2
        self.model_path = str(model_path)
        self.conf_thr = conf_thr
        self.input_width = input_width
        self._det = None
        self._size = None

    def __call__(self, bgr: np.ndarray) -> list[FaceDetection]:
        cv2 = self._cv2
        h, w = bgr.shape[:2]
        k = min(1.0, self.input_width / w)
        small = cv2.resize(bgr, (int(w * k), int(h * k))) if k < 1 else bgr
        size = (small.shape[1], small.shape[0])
        if self._det is None or self._size != size:
            self._det = cv2.FaceDetectorYN.create(self.model_path, "", size, self.conf_thr)
            self._size = size
        _, faces = self._det.detect(small)
        if faces is None:
            return []
        out = []
        for f in faces:
            x, y, fw, fh = (float(v) / k for v in f[:4])
            out.append(FaceDetection(
                conf=float(f[14]),
                box=(x, y, x + fw, y + fh),
                right_eye=(float(f[4]) / k, float(f[5]) / k),
                left_eye=(float(f[6]) / k, float(f[7]) / k),
            ))
        return out
