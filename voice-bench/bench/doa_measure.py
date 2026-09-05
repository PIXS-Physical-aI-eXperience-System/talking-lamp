"""DOA(소리 방향) 실측 — E 의 L1 칼만에 넣을 'θ ± 오차' 를 만든다.

인수인계 7번(C → E: 소리 방향). 이 값이 없으면 S6(시야 밖 소리 추종)를 못 만든다.

    python bench/doa_measure.py calibrate            # 정면 기준 맞추기 (먼저)
    python bench/doa_measure.py measure --label quiet
    python bench/doa_measure.py measure --label fan       # Jetson 팬 켠 상태
    python bench/doa_measure.py measure --label elevated  # 책상+45cm 고각
    python bench/doa_measure.py measure --label servo     # 서보 동작 중
    python bench/doa_measure.py report

각도 규약: lamp_base 기준, 0° = 램프 정면(사용자 방향, +x), 반시계 방향 증가.
XVF3800 은 방위각만 주고 고도는 주지 않는다(평면 원형 어레이의 한계).
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "out", "doa")
CAL = os.path.join(OUT, "calibration.json")
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


# ── 원형 통계 ───────────────────────────────────────────────────────────
# 각도는 359° 와 1° 가 2° 차이다. 산술평균·표준편차를 그대로 쓰면 틀린 값이
# 조용히 나오므로 벡터 평균으로 계산한다.

def circ_mean(deg):
    r = np.radians(np.asarray(deg, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def circ_std(deg):
    r = np.radians(np.asarray(deg, dtype=float))
    R = np.hypot(np.sin(r).mean(), np.cos(r).mean())
    return float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12)))))


def ang_err(measured, truth):
    """부호 있는 최단 각도차 (-180 ~ +180)."""
    return (measured - truth + 180) % 360 - 180


# ── 장치 ────────────────────────────────────────────────────────────────

def read_doa():
    """xvf_host 로 방위각을 읽는다. 실패하면 None.

    출력은 집중빔1·집중빔2·자유빔·자동선택빔의 4개 각도이며,
    문서상 마지막(자동선택빔)이 사용 대상이다.
    """
    exe = shutil.which("xvf_host") or shutil.which("xvf_host.py")
    if not exe:
        return None, "xvf_host 없음"
    try:
        r = subprocess.run([exe, "AEC_AZIMUTH_VALUES"], capture_output=True,
                           text=True, timeout=5)
        raw = (r.stdout or r.stderr).strip()
        nums = [float(t) for t in raw.replace(",", " ").split()
                if _isfloat(t)]
        if not nums:
            return None, raw[:120]
        # 라디안·도가 섞여 나오므로, 0~360 범위 값들 중 마지막을 자동선택빔으로 본다
        degs = [n for n in nums if -360.0 <= n <= 360.0 and abs(n) > 6.3] or nums
        return float(degs[-1]) % 360, raw
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _isfloat(t):
    try:
        float(t)
        return True
    except ValueError:
        return False


def sample_doa(seconds=3.0, hz=10):
    """말하는 동안 방위각을 반복 측정해 모은다."""
    vals, raws = [], []
    t_end = time.time() + seconds
    while time.time() < t_end:
        v, raw = read_doa()
        if v is not None:
            vals.append(v)
        raws.append(raw)
        time.sleep(1.0 / hz)
    return vals, raws[:3]


# ── 절차 ────────────────────────────────────────────────────────────────

def cmd_calibrate(args):
    """보드의 0° 와 램프 정면이 어디서 어긋나는지 잰다.

    어레이를 어떻게 장착하든 물리적 회전이 생기므로, 이 보정 없이는
    모든 각도가 일정하게 틀어진 채로 나온다.
    """
    os.makedirs(OUT, exist_ok=True)
    print("정면 기준 보정")
    print("  램프 '정면'(사용자가 앉는 방향)에서 1 m 떨어져 서세요.")
    input("  준비되면 Enter → 3초간 계속 말해주세요 ")
    vals, raws = sample_doa()
    if not vals:
        print(f"  ! DOA 를 못 읽었다: {raws[:1]}")
        return 1
    offset = circ_mean(vals)
    json.dump({"offset_deg": offset, "n": len(vals), "std": circ_std(vals)},
              open(CAL, "w"), ensure_ascii=False, indent=2)
    print(f"  보정값 {offset:.1f}°  (표본 {len(vals)}개, 산포 {circ_std(vals):.1f}°)")
    print(f"  저장: {CAL}")
    return 0


def cmd_measure(args):
    if not os.path.exists(CAL):
        print("먼저 calibrate 를 실행할 것")
        return 1
    offset = json.load(open(CAL))["offset_deg"]
    os.makedirs(OUT, exist_ok=True)

    print(f"[{args.label}] 8방향 측정 — 각 방향에서 1 m 거리, 3초간 발화")
    print(f"보정값 {offset:.1f}° 적용. 0° = 램프 정면, 반시계 방향 증가\n")

    rows = []
    for truth in ANGLES:
        input(f"  {truth:>3}° 위치로 이동 → Enter 후 3초간 말하기 ")
        vals, raws = sample_doa()
        if not vals:
            print(f"       DOA 읽기 실패: {raws[:1]}")
            continue
        meas = (circ_mean(vals) - offset) % 360
        err = ang_err(meas, truth)
        rows.append({"truth": truth, "measured": round(meas, 1),
                     "error": round(err, 1), "spread": round(circ_std(vals), 1),
                     "n": len(vals)})
        print(f"       측정 {meas:>6.1f}°   오차 {err:>+6.1f}°   산포 {circ_std(vals):>5.1f}°")

    path = os.path.join(OUT, f"{args.label}.json")
    json.dump({"label": args.label, "offset_deg": offset, "rows": rows},
              open(path, "w"), ensure_ascii=False, indent=2)
    if rows:
        errs = [abs(r["error"]) for r in rows]
        print(f"\n  평균 절대오차 {statistics.mean(errs):.1f}°   최대 {max(errs):.1f}°")
    print(f"  저장: {path}")
    return 0


def cmd_report(args):
    files = sorted(f for f in os.listdir(OUT) if f.endswith(".json") and f != "calibration.json")
    if not files:
        print("측정 결과가 없다. measure 를 먼저 실행할 것")
        return 1
    print(f"{'조건':<12}{'평균오차':>9}{'최대오차':>9}{'산포':>8}{'방향수':>7}")
    print("-" * 46)
    for f in files:
        d = json.load(open(os.path.join(OUT, f)))
        rows = d["rows"]
        if not rows:
            continue
        errs = [abs(r["error"]) for r in rows]
        spread = statistics.mean(r["spread"] for r in rows)
        print(f"{d['label']:<12}{statistics.mean(errs):>8.1f}°{max(errs):>8.1f}°"
              f"{spread:>7.1f}°{len(rows):>7}")

    print("\nE 에게 넘길 값 — L1 칼만의 측정 노이즈 표준편차(σ)")
    print("  가장 나쁜 조건의 '평균오차' 를 σ 로 잡는 것이 안전하다.")
    print("  칼만이 DOA 를 얼마나 믿을지가 이 값으로 정해진다.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate")
    m = sub.add_parser("measure")
    m.add_argument("--label", required=True,
                   help="조건 이름 (quiet / fan / elevated / servo)")
    sub.add_parser("report")
    args = ap.parse_args()
    return {"calibrate": cmd_calibrate, "measure": cmd_measure,
            "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
