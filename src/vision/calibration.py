"""Camera calibration and one-time desk registration with a ChArUco board.

ChArUco rather than a plain chessboard: every corner carries a marker ID, so
the board's origin is unambiguous. A plain chessboard can be read rotated by
180 deg, which would silently flip the desk registration and send the task
light to the mirror-image spot.

Two steps, both with the same printed board:

1. **Intrinsics** — 15-30 views of the board at varied angles and positions.
   Once per camera. Also the focus check: corners that do not resolve at
   0.4-0.6 m mean the lens cannot serve S1 at all.
2. **Desk registration** — the board lying flat in front of the lamp at a
   measured spot, one frame. Once per installation (docs/진행-순서.md D-3).

Board placement for step 2 (``DeskPlacement``): face up, centred on the
lamp's forward axis, printed image upright as seen from behind the lamp, i.e.
the top edge of the print is the far edge. ``near_edge_m`` is the distance
from the centre of the lamp base to the board's near edge.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .camera import Intrinsics
from .geometry import Pose

SQUARES_X = 7
SQUARES_Y = 5
SQUARE_M = 0.035          # nominal; always pass the measured print size
MARKER_RATIO = 26 / 35
DICTIONARY = "DICT_5X5_100"
MIN_CORNERS = 8


def _cv2():
    import cv2
    return cv2


def make_board(square_m: float = SQUARE_M):
    cv2 = _cv2()
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    return cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), square_m, square_m * MARKER_RATIO, d)


def board_image(px_per_square: int = 210, margin_px: int = 40) -> np.ndarray:
    """Printable board, grey8, square grid starting exactly at ``margin_px``.

    Drawn square by square instead of with ``CharucoBoard.generateImage``:
    in OpenCV 5.0 that call fails an ROI assertion whenever the square size
    comes out to a whole number of pixels, and otherwise re-centres the grid,
    so neither the test render nor the printed scale would be exact.
    ``px_per_square`` should be a multiple of 35 so markers are whole pixels.
    """
    cv2 = _cv2()
    b = make_board(1.0)                       # board units = squares
    pps, m = px_per_square, margin_px
    img = np.full((SQUARES_Y * pps + 2 * m, SQUARES_X * pps + 2 * m), 255, np.uint8)

    marker_square = set()
    side = int(round(MARKER_RATIO * pps))
    for corners, mid in zip(b.getObjPoints(), b.getIds().ravel()):
        c = np.asarray(corners).reshape(-1, 3)
        x0, y0 = c[:, 0].min(), c[:, 1].min()
        marker_square.add((int(x0), int(y0)))
        mk = cv2.aruco.generateImageMarker(b.getDictionary(), int(mid), side, borderBits=1)
        px, py = m + int(round(x0 * pps)), m + int(round(y0 * pps))
        img[py:py + side, px:px + side] = mk
    # ChArUco puts a marker in every white square; the rest are black.
    for j in range(SQUARES_Y):
        for i in range(SQUARES_X):
            if (i, j) not in marker_square:
                img[m + j * pps:m + (j + 1) * pps, m + i * pps:m + (i + 1) * pps] = 0
    return img


def detect(bgr: np.ndarray, square_m: float = SQUARE_M):
    """Board corners in one image -> (object points [N,3], image points [N,2]) or None."""
    cv2 = _cv2()
    board = make_board(square_m)
    cc, ci, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(bgr)
    if cc is None or len(cc) < MIN_CORNERS:
        return None
    obj, img = board.matchImagePoints(cc, ci)
    return obj.reshape(-1, 3).astype(np.float64), img.reshape(-1, 2).astype(np.float64)


def calibrate(views, width: int, height: int) -> Intrinsics:
    """Intrinsics from several ``detect`` results.

    Uses OpenCV's 5-term model. If RMS stays above ~1 px on the 120 deg lens,
    the next thing to try is ``cv2.fisheye`` rather than more views.
    """
    cv2 = _cv2()
    if len(views) < 5:
        raise ValueError(f"need at least 5 board views, got {len(views)}")
    obj = [v[0].astype(np.float32) for v in views]
    img = [v[1].astype(np.float32) for v in views]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, (width, height), None, None)
    d = tuple(float(x) for x in np.ravel(dist)[:5])
    return Intrinsics(width, height, float(K[0, 0]), float(K[1, 1]),
                      float(K[0, 2]), float(K[1, 2]), d, "charuco", float(rms))


@dataclass(frozen=True)
class DeskPlacement:
    near_edge_m: float        # base centre -> board near edge, along +x
    lateral_m: float = 0.0    # board centre offset along +y (left)
    square_m: float = SQUARE_M
    # How far the printed face sits above the desk. Paper taped straight down
    # is ~0.1 mm and can be ignored; a clipboard or a book under it cannot,
    # because the error goes straight into the camera height and from there
    # into every desk point. Measure the backing and pass it.
    height_m: float = 0.0

    def board_in_base(self) -> tuple[np.ndarray, np.ndarray]:
        """Rotation and origin of the board frame in ``lamp_base``.

        OpenCV's board frame: origin at the print's top-left corner, +x along
        the top edge, +y down the print, +z into the board. Lying face up with
        the top edge far from the lamp, that is +x -> lamp's right (-y_base),
        +y -> toward the lamp (-x_base), +z -> into the desk (-z_base).
        """
        R = np.array([[0.0, -1.0, 0.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0]])
        w = SQUARES_X * self.square_m
        h = SQUARES_Y * self.square_m
        t = np.array([self.near_edge_m + h, self.lateral_m + w / 2, self.height_m])
        return R, t


def register_desk(bgr: np.ndarray, cam: Intrinsics, placement: DeskPlacement) -> Pose:
    """Camera pose in ``lamp_base`` from one image of the board on the desk."""
    cv2 = _cv2()
    found = detect(bgr, placement.square_m)
    if found is None:
        raise RuntimeError("board not found or too few corners; check focus and lighting")
    obj, img = found
    cam = cam.scaled(bgr.shape[1], bgr.shape[0])
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, np.array(cam.dist), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, cam.K, np.array(cam.dist))
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))

    R_cb, _ = cv2.Rodrigues(rvec)          # board -> camera
    R_bb, t_bb = placement.board_in_base()  # board -> base
    R = R_bb @ R_cb.T                        # camera -> base
    t = t_bb - R @ tvec.ravel()
    return Pose(R, t, source="charuco", rms_px=rms)


# -- printable PDF -----------------------------------------------------------

A4_LANDSCAPE_MM = (297.0, 210.0)
_MM = 72.0 / 25.4                        # PDF units are points


def board_pdf(square_mm: float = SQUARE_M * 1000, page_mm=A4_LANDSCAPE_MM,
              ruler_mm: float = 100.0) -> bytes:
    """The board as a vector PDF at true physical size, centred on the page.

    Vector rather than a rasterised image so it prints crisp at any printer
    resolution, and so the square edges land exactly where the geometry says.
    ``square_mm`` must leave a margin: at the 35 mm default the board is
    245 x 175 mm on A4 landscape, clear of every printer's dead border.

    A ``ruler_mm`` long reference line is drawn in the bottom margin. Measuring
    it is the check that the printer did not rescale: the calibration is only
    as good as the assumed square size.
    """
    cv2 = _cv2()
    board = make_board(1.0)                          # board units = one square
    pw, ph = (v * _MM for v in page_mm)
    sq = square_mm * _MM
    bw, bh = SQUARES_X * sq, SQUARES_Y * sq
    if bw > pw or bh > ph:
        raise ValueError(f"{square_mm} mm squares do not fit on a {page_mm[0]}x{page_mm[1]} mm page")
    x0, y_top = (pw - bw) / 2, (ph + bh) / 2         # PDF y grows upward

    def rect(x, y, w, h):
        return f"{x:.4f} {y:.4f} {w:.4f} {h:.4f} re f\n"

    with_marker = set()
    parts = ["1 g\n", rect(x0, y_top - bh, bw, bh), "0 g\n"]
    markers = []
    for corners, mid in zip(board.getObjPoints(), board.getIds().ravel()):
        c = np.asarray(corners).reshape(-1, 3)
        mx, my = float(c[:, 0].min()), float(c[:, 1].min())
        with_marker.add((int(math.floor(mx)), int(math.floor(my))))
        markers.append((mx, my, cv2.aruco.generateImageMarker(
            board.getDictionary(), int(mid), 7, borderBits=1) > 127))
    # ChArUco puts a marker in every white square; the others are solid black.
    for j in range(SQUARES_Y):
        for i in range(SQUARES_X):
            if (i, j) not in with_marker:
                parts.append(rect(x0 + i * sq, y_top - (j + 1) * sq, sq, sq))
    for mx, my, bits in markers:
        side = MARKER_RATIO * sq
        cell = side / bits.shape[0]
        for r in range(bits.shape[0]):
            for c_ in range(bits.shape[1]):
                if not bits[r, c_]:
                    parts.append(rect(x0 + mx * sq + c_ * cell,
                                      y_top - my * sq - (r + 1) * cell, cell, cell))

    ruler = ruler_mm * _MM
    rx, ry = (pw - ruler) / 2, (ph - bh) / 4
    parts += ["0.6 w 0 G\n",
              f"{rx:.4f} {ry:.4f} m {rx + ruler:.4f} {ry:.4f} l S\n",
              f"{rx:.4f} {ry - 3:.4f} m {rx:.4f} {ry + 3:.4f} l S\n",
              f"{rx + ruler:.4f} {ry - 3:.4f} m {rx + ruler:.4f} {ry + 3:.4f} l S\n",
              "BT /F1 8 Tf 1 0 0 1 {:.2f} {:.2f} Tm ({}) Tj ET\n".format(
                  rx, ry + 6,
                  f"ChArUco {SQUARES_X}x{SQUARES_Y}  {DICTIONARY}  "
                  f"square {square_mm:.1f} mm  -  this line is exactly "
                  f"{ruler_mm:.0f} mm; measure it before use")]
    return _pdf_document("".join(parts), pw, ph)


def _pdf_document(content: str, width_pt: float, height_pt: float) -> bytes:
    """Minimal single-page PDF. Base-14 Helvetica, so no font to embed."""
    stream = content.encode("ascii")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width_pt:.2f} {height_pt:.2f}] "
        f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode("ascii"),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)
