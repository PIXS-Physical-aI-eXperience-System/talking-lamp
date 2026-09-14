"""DOA(소리 방향) 실측 — E 의 L1 칼만에 넣을 'θ ± 오차' 를 만든다.

인수인계 7번(C → E: 소리 방향). 이 값이 없으면 S6(시야 밖 소리 추종)를 못 만든다.

    python bench/doa_measure.py calibrate            # 정면 기준 맞추기 (먼저)
    python bench/doa_measure.py measure --label quiet
    python bench/doa_measure.py measure --label fan       # Jetson 팬 켠 상태
    python bench/doa_measure.py measure --label elevated  # 책상+45cm 고각
    python bench/doa_measure.py measure --label servo     # 서보 동작 중
    python bench/doa_measure.py report

각도 규약: lamp_base 기준, 0° = 램프 정면(사용자 방향, +x), 반시계 방향 증가.
XVF3800 은 방위각만 주고 고도는 주지 않는다(평면 어레이의 한계).

배열 형태에 따라 잴 수 있는 범위가 다르다. --geometry 로 지정한다.

  circular  마이크 4개 원형 44mm. 360° 전방향.
  linear    마이크 4개 직선 33mm. 전면 약 180°, 후면은 펌웨어가 억제한다.
            직선 배열은 앞뒤를 물리적으로 구분할 수 없기 때문이다 — 축을 기준으로
            대칭인 두 방향이 같은 시간차를 만든다. 그래서 뒤쪽 소리는 방향을
            줄 수 없고, S6(소리 방향 추종)는 전면 반평면에서만 성립한다.
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
# 잴 각도. 선형은 후면을 펌웨어가 억제하므로 전면 반평면만 돈다.
ANGLES_CIRCULAR = [0, 45, 90, 135, 180, 225, 270, 315]
# 정면 ±90° 를 30° 간격으로. 왼쪽 끝에서 오른쪽 끝으로 한 방향으로 훑도록
# 늘어놓는다 — 측정할 때 사람이 왔다갔다 하지 않아도 된다.
ANGLES_LINEAR = [270, 300, 330, 0, 30, 60, 90]
ANGLES = ANGLES_LINEAR


# ── 원형 통계 ───────────────────────────────────────────────────────────
# 각도는 359° 와 1° 가 2° 차이다. 산술평균·표준편차를 그대로 쓰면 틀린 값이
# 조용히 나오므로 벡터 평균으로 계산한다.


def find_xvf_host():
    """xvf_host 를 찾는다. bench/xvf_setup.sh 가 받아둔 것을 먼저 본다.

    소스에서 빌드하는 물건이 아니라 미리 빌드된 바이너리로 배포되며,
    jetson 용이 따로 들어 있다. PATH 에 넣는 것이 아니라 저장소 안에
    두는 구조라 여기서 직접 찾아야 한다.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(root, "tools", "respeaker-flex", "host_control", "jetson", "xvf_host")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    return shutil.which("xvf_host") or shutil.which("xvf_host.py")

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

# reSpeaker Flex 는 USB 제어 전송으로 값을 직접 읽을 수 있다. 공식
# python_control/respeaker_get_doa.py 와 같은 방식이며, xvf_host 바이너리가
# 없어도 되고 speech_detected 플래그까지 같이 온다.
#   DOA_VALUE: resid 20, cmdid 18, 4바이트 (+ 상태 1바이트)
_VID = 0x2886
_DOA = (20, 18, 4)
_dev = None


def usb_device():
    """장치 핸들을 한 번만 잡아 재사용한다. 없으면 None."""
    global _dev
    if _dev is not None:
        return _dev
    try:
        import usb.core
    except ImportError:
        return None
    devs = sorted(usb.core.find(find_all=True, idVendor=_VID) or [],
                  key=lambda d: getattr(d, "idProduct", 0))
    _dev = devs[0] if devs else None
    return _dev


def read_doa():
    """방위각(도)과 speech_detected 를 읽는다. 실패하면 (None, 사유).

    반환: (각도 또는 None, 원문/사유, speech_detected 또는 None)
    """
    dev = usb_device()
    if dev is not None:
        try:
            import usb.util
            resid, cmdid, length = _DOA
            r = dev.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0, 0x80 | cmdid, resid, length + 1, 100000).tolist()
            # r[0] 은 상태. 각도는 리틀엔디언 2바이트, r[3] 이 발화 감지 플래그다.
            return float(r[1] + r[2] * 256) % 360, f"usb {r}", bool(r[3])
        except Exception as e:
            return None, f"USB 읽기 실패: {type(e).__name__}: {e}", None

    # 폴백: xvf_host 바이너리
    exe = find_xvf_host()
    if not exe:
        return None, ("장치를 못 찾았다. pyusb 설치(pip install pyusb)와 "
                      "USB 연결·권한(udev)을 확인할 것"), None
    try:
        r = subprocess.run([exe, "AEC_AZIMUTH_VALUES"], capture_output=True,
                           text=True, timeout=5)
        raw = (r.stdout or r.stderr).strip()
        nums = [float(t) for t in raw.replace(",", " ").split() if _isfloat(t)]
        if not nums:
            return None, raw[:120], None
        degs = [n for n in nums if -360.0 <= n <= 360.0 and abs(n) > 6.3] or nums
        return float(degs[-1]) % 360, raw, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", None


def _isfloat(t):
    try:
        float(t)
        return True
    except ValueError:
        return False


def sample_doa(seconds=3.0, hz=10):
    """말하는 동안 방위각을 반복 측정해 모은다.

    펌웨어가 발화 감지 플래그를 같이 주므로, 그게 켜진 표본만 쓴다. 조용할 때의
    각도는 직전 값이나 잡음 방향이라 섞으면 산포가 부풀려진다. 플래그를 못 읽는
    경로(xvf_host 폴백)에서는 전부 쓴다.
    """
    vals, raws, gated = [], [], 0
    t_end = time.time() + seconds
    while time.time() < t_end:
        v, raw, speech = read_doa()
        if v is not None:
            if speech is False:
                gated += 1
            else:
                vals.append(v)
        raws.append(raw)
        time.sleep(1.0 / hz)
    if gated:
        raws.append(f"(발화 없음으로 버린 표본 {gated}개)")
    return vals, raws[:3]


# ── 절차 ────────────────────────────────────────────────────────────────

def cmd_live(args):
    """원시 DOA 를 실시간으로 찍는다. 각도 규약을 눈으로 확인하는 용도다.

    펌웨어가 어느 방향을 0° 로 삼는지는 문서에 없다. 선형 배열이면 축 방향이
    0° 이고 정면(broadside)이 90° 일 가능성이 크지만, 확인 없이 가정하면
    오차표가 통째로 틀어진다. 좌·정면·우로 옮겨 다니며 값을 보면 규약이 드러난다.
    """
    print("원시 DOA 실시간 (Ctrl+C 로 종료)")
    print("  좌 → 정면 → 우 로 옮겨 다니며 말해서, 값이 어느 쪽으로 늘어나는지 볼 것")
    print("  선형이면 0~180 범위에 머물 것으로 예상된다\n")
    lo, hi, seen = 360.0, 0.0, 0
    try:
        while True:
            v, raw, speech = read_doa()
            if v is None:
                print(f"  읽기 실패: {raw}")
                time.sleep(1.0)
                continue
            if speech:
                seen += 1
                lo, hi = min(lo, v), max(hi, v)
            mark = "발화" if speech else "  · "
            print(f"  {mark}  {v:6.1f}°     (발화 중 범위 {lo:.0f}~{hi:.0f}°, "
                  f"표본 {seen})   ", end="\r", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
        if seen:
            print(f"\n발화 중 관측 범위: {lo:.1f}° ~ {hi:.1f}°  (표본 {seen}개)")
            print("  → 0~180 범위로 보인다. 선형 배열의 반평면 규약이 맞다."
                  if hi <= 181 else
                  "  → 180 을 넘는다. 0~360 규약이거나 후면 값도 나온다.")
        else:
            print("발화로 인식된 표본이 없다. 더 크게, 더 길게 말해볼 것")
    return 0



def cmd_calibrate(args):
    """보드의 각도 규약을 램프 기준으로 옮긴다. 두 가지를 잰다.

    1) 정면 오프셋 — 어레이를 어떻게 장착하든 물리적 회전이 생긴다. 이 보정
       없이는 모든 각도가 일정하게 틀어진 채로 나온다. 선형 배열에서는 정면이
       0° 가 아니라 90° 근처로 나온다(축 방향이 0°, 정면이 broadside).

    2) 회전 방향 — 원시값이 커지는 쪽이 램프의 왼쪽인지 오른쪽인지. 문서에
       없고 장착 방향에 따라 뒤집힌다. 이걸 틀리면 부호가 반대로 나와서
       **램프가 소리 반대쪽으로 돈다.** 오차표는 멀쩡해 보이므로 조용히 틀린다.
    """
    os.makedirs(OUT, exist_ok=True)
    print("각도 보정 — 두 지점에서 잰다\n")

    print("[1/2] 정면")
    print("  램프 정면(사용자가 앉는 방향)에서 1 m 떨어져 서세요.")
    input("  준비되면 Enter → 3초간 계속 말해주세요 ")
    front, raws = sample_doa()
    if not front:
        print(f"  ! DOA 를 못 읽었다: {raws[:1]}")
        return 1
    offset = circ_mean(front)
    sd = circ_std(front)
    print(f"  정면 원시값 {offset:.1f}°  (표본 {len(front)}개, 산포 {sd:.1f}°)")
    # 한 자리에 서서 3초 말한 값의 산포다. 이게 크면 보정값 자체가 흔들리고,
    # 그 오차가 이후 모든 각도에 그대로 깔린다. 안정된 조건에서는 1° 안쪽이었다.
    if sd > 10:
        print(f"  ! 산포 {sd:.1f}° 는 너무 크다. 보정값을 믿을 수 없다.")
        print("    보드가 고정돼 있는지, 1 m 거리에서 끊지 않고 말했는지 확인하고")
        print("    다시 실행할 것. 안정된 조건에서는 1° 안쪽이 나온다.")
        return 1
    if sd > 3:
        print(f"  · 산포 {sd:.1f}° — 다소 흔들린다. 결과 해석에 감안할 것")
    print()

    print("[2/2] 오른쪽")
    print("  램프를 마주 본 채로, 램프에서 볼 때 오른쪽 90° 위치로 이동하세요.")
    print("  (정면에 선 사람이 램프를 축으로 왼쪽으로 걸어간 자리다)")
    input("  준비되면 Enter → 3초간 계속 말해주세요 ")
    right, raws = sample_doa()
    if not right:
        print(f"  ! DOA 를 못 읽었다: {raws[:1]}")
        return 1
    r = circ_mean(right)
    delta = ang_err(r, offset)        # 정면 대비 원시값이 어느 쪽으로 움직였나
    sign = 1 if delta > 0 else -1     # +1 이면 원시값 증가 = 램프 오른쪽
    print(f"  오른쪽 원시값 {r:.1f}°  (정면 대비 {delta:+.1f}°)")

    sd_r = circ_std(right)
    if sd_r > 10:
        print(f"  ! 산포 {sd_r:.1f}° 는 너무 크다. 다시 실행할 것")
        return 1
    if abs(delta) < 20:
        print("  ! 정면과 거의 같다. 위치를 제대로 옮겼는지 확인할 것")
        return 1
    # 90° 를 옮겼으면 원시값도 90° 근처로 움직여야 한다. 크게 어긋나면 실제
    # 이동각이 90° 가 아니었거나, 원시 각도가 물리 각도에 선형으로 대응하지
    # 않는 것이다. 어느 쪽이든 단순 오프셋 보정으로는 각도가 맞지 않는다.
    if abs(abs(delta) - 90) > 25:
        print(f"  ! 90° 를 옮겼는데 원시값은 {abs(delta):.0f}° 만 변했다.")
        print("    실제로 90° 를 이동했는지 먼저 확인할 것 (바닥에 표시를 두면 좋다).")
        print("    위치가 맞는데도 이렇게 나오면 원시 각도가 물리 각도에 비례하지")
        print("    않는 것이고, 오프셋 보정만으로는 못 맞춘다 — 대응표가 필요하다.")
        print("    일단 저장은 하되, 이 보정으로 잰 값은 신뢰도가 낮다.")
    print(f"  → 원시값이 {'커지는' if sign > 0 else '작아지는'} 쪽이 램프의 오른쪽\n")

    json.dump({"offset_deg": offset, "sign": sign,
               "n": len(front), "std": sd,
               "right_raw": r, "right_delta": delta, "right_std": sd_r},
              open(CAL, "w"), ensure_ascii=False, indent=2)
    print(f"  저장: {CAL}")
    return 0


def to_lamp(raw, cal):
    """원시 각도를 램프 기준(0° = 정면, 반시계 +)으로 옮긴다."""
    return (cal["sign"] * ang_err(raw, cal["offset_deg"])) % 360


def cmd_measure(args):
    if not os.path.exists(CAL):
        print("먼저 calibrate 를 실행할 것")
        return 1
    cal = json.load(open(CAL))
    if "sign" not in cal:
        print("보정 파일이 예전 형식이다(회전 방향 없음). calibrate 를 다시 실행할 것")
        return 1
    offset = cal["offset_deg"]
    os.makedirs(OUT, exist_ok=True)

    print(f"[{args.label}] {len(ANGLES)}방향 측정 — 각 방향에서 1 m 거리, 3초간 발화")
    print(f"보정 {offset:.1f}°, 회전 {'정방향' if cal['sign'] > 0 else '역방향'} 적용. "
          f"0° = 램프 정면, 반시계 방향 증가\n")

    rows = []
    for truth in ANGLES:
        input(f"  {truth:>3}° 위치로 이동 → Enter 후 3초간 말하기 ")
        vals, raws = sample_doa()
        if not vals:
            print(f"       DOA 읽기 실패: {raws[:1]}")
            continue
        meas = to_lamp(circ_mean(vals), cal)
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
    global ANGLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["linear", "circular"], default="linear",
                    help="마이크 배열 형태. 선형은 후면을 못 재므로 전면 반평면만 돈다")
    ap.add_argument("--angles",
                    help="잴 각도를 직접 지정 (쉼표 구분). --geometry 기본값을 덮는다")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate")
    sub.add_parser("live")
    m = sub.add_parser("measure")
    m.add_argument("--label", required=True,
                   help="조건 이름 (quiet / fan / elevated / servo)")
    sub.add_parser("report")
    args = ap.parse_args()

    ANGLES = (ANGLES_CIRCULAR if args.geometry == "circular" else ANGLES_LINEAR)
    if args.angles:
        ANGLES = [float(x) % 360 for x in args.angles.split(",")]

    return {"calibrate": cmd_calibrate, "measure": cmd_measure,
            "live": cmd_live, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
