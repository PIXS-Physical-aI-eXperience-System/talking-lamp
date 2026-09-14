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
_USE_STREAM = True


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


def sample_doa(seconds=6.0, hz=10, show=True):
    """말하는 동안 방위각을 반복 측정해 모은다.

    펌웨어가 발화 감지 플래그를 같이 주므로, 그게 켜진 표본만 쓴다. 조용할 때의
    각도는 직전 값이나 잡음 방향이라 섞으면 산포가 부풀려진다. 플래그를 못 읽는
    경로(xvf_host 폴백)에서는 전부 쓴다.
    """
    vals, raws, stamped, gated = [], [], [], 0
    width = 61
    hist = [0] * width
    stream = MicStream(enabled=_USE_STREAM)
    stream.__enter__()
    if show:
        print("\n" * 4, end="")
    t_end = time.time() + seconds
    while time.time() < t_end:
        v, raw, speech = read_doa()
        if v is not None:
            if speech is False:
                gated += 1
            else:
                vals.append(v)
                stamped.append((time.time(), v))
                hist[min(int(v / 180 * (width - 1)), width - 1)] += 1
        raws.append(raw)
        if show:
            # 6초 동안 화면이 멈춰 있으면 제대로 잡히는지 모른 채 말해야 한다.
            left = max(0.0, t_end - time.time())
            print("\033[4F" + "\n".join(
                ln + "\033[K" for ln in
                render(v, bool(speech), hist, width,
                       f"{left:.1f}초 남음   표본 {len(vals)}").split("\n")))
        time.sleep(1.0 / hz)
    stream.__exit__()
    if show:
        print()
    if gated:
        raws.append(f"(발화 없음으로 버린 표본 {gated}개)")
    return vals, raws[:3], stamped


class MicStream:
    """XVF3800 의 오디오 입력을 열어 둔다.

    DOA 는 펌웨어의 음성 처리 파이프라인이 만들어내는 값이다. 아무도 오디오를
    받아가지 않으면 그 파이프라인이 돌지 않아 DOA 가 갱신되지 않을 수 있다.
    공식 예제는 스트림을 열지 않지만, 그 예제가 맞다는 보장은 없다.
    열고 재는 것과 안 열고 재는 것을 비교할 수 있게 해 둔다.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.stream = None

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            import sounddevice as sd
            idx = next((i for i, d in enumerate(sd.query_devices())
                        if d["max_input_channels"] >= 6
                        and any(k in d["name"].lower()
                                for k in ("xvf", "respeaker", "xmos"))), None)
            if idx is None:
                print("  ! XVF3800 입력 장치를 못 찾아 스트림 없이 진행한다")
                return self
            d = sd.query_devices(idx)
            self.stream = sd.InputStream(
                device=idx, channels=int(d["max_input_channels"]),
                samplerate=int(d["default_samplerate"]))
            self.stream.start()
            print(f"  오디오 스트림 열림: [{idx}] {d['name']} "
                  f"{int(d['max_input_channels'])}ch")
        except Exception as e:
            print(f"  ! 스트림을 못 열었다({type(e).__name__}). 스트림 없이 진행한다")
        return self

    def __exit__(self, *a):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


def best_window(stamped, seconds=5.0):
    """가장 안정적이었던 구간을 찾는다.

    한 번의 실행 안에 '가만히 말하기' 와 '좌우로 옮기기' 가 섞이면 전체 산포는
    당연히 크다. 그걸로 장치를 판정하면 멀쩡한 것도 실패로 나온다 —
    실제로 그렇게 오판했다. 움직이지 않았던 구간만 골라서 본다.
    """
    best = None
    for i, (t0, _) in enumerate(stamped):
        w = [v for t, v in stamped[i:] if t - t0 <= seconds]
        if len(w) < 10:
            continue
        st = stability(w)
        if best is None or st["near10"] > best["near10"]:
            best = st
    return best


def stability(vals):
    """방향값이 쓸 만한지 한 줄로 판정할 수 있게 요약한다.

    최소~최대만 보면 튄 값 하나가 범위를 다 잡아먹어서 아무것도 알 수 없다.
    중앙값 주변에 얼마나 모여 있는지를 봐야 한다.
    """
    if not vals:
        return None
    med = circ_mean(vals)
    near = sum(1 for v in vals if abs(ang_err(v, med)) <= 10) / len(vals)
    return {"mean": med, "std": circ_std(vals), "near10": near, "n": len(vals)}


def print_histogram(vals, width=50):
    """20° 칸으로 나눈 막대. 한 방향에 모이는지 흩어지는지가 바로 보인다."""
    bins = [0] * 9          # 0~180 을 20° 씩
    over = 0
    for v in vals:
        if v > 180:
            over += 1
        else:
            bins[min(int(v // 20), 8)] += 1
    top = max(bins) or 1
    for i, c in enumerate(bins):
        bar = "█" * int(width * c / top)
        print(f"  {i*20:>3}~{i*20+19:<3}° {c:>5}  {bar}")
    if over:
        print(f"  180° 초과  {over:>5}  ← 선형 배열에서 나오면 안 되는 값")


# ── 절차 ────────────────────────────────────────────────────────────────

SPARK = " ▁▂▃▄▅▆▇█"


def render(cur, speech, hist, width=61, note=""):
    """0~180° 눈금 위에 현재 방향과 누적 분포를 그린다.

    숫자만 보면 값이 튀는지 몰린지 판단이 안 된다. 축 위에 찍으면 한눈에 보인다.
    """
    # 현재 위치 표시자
    axis = ["─"] * width
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        axis[min(int(frac * (width - 1)), width - 1)] = "┼"
    if cur is not None:
        axis[min(int(cur / 180 * (width - 1)), width - 1)] = "●" if speech else "○"

    # 누적 분포 (칸당 스파크라인 한 글자)
    top = max(hist) or 1
    spark = "".join(SPARK[min(int(len(SPARK) * c / top), len(SPARK) - 1)] for c in hist)

    cur_txt = f"{cur:5.1f}°" if cur is not None else "  -  "
    mark = "\033[32m● 발화\033[0m" if speech else "\033[90m○ 조용\033[0m"
    return (f"  {mark}   현재 {cur_txt}   {note}\n"
            f"  왼쪽 {''.join(axis)} 오른쪽\n"
            f"   0°  {spark}  180°\n"
            f"       ↑정면은 보정 후 결정 (보통 90° 부근)")


def cmd_live(args):
    """방향을 실시간으로 그려 본다. 값이 쓸 만한지 먼저 가리는 단계다.

    한 자리에서 계속 말하면 한 칸에 몰려야 한다. 전 구간에 퍼지면 그 값으로는
    보정도 오차표도 의미가 없다.
    """
    width = 61
    hist = [0] * width
    vals, stamped = [], []
    print("실시간 방향 (Ctrl+C 로 종료)")
    print("  먼저 한 자리에서 가만히 말해 안정되는지 보고, 그다음 좌우로 옮겨")
    print("  막대가 따라오는지 보세요. 판정은 가장 안정적이었던 5초 구간으로 합니다.")
    with MicStream(enabled=not args.no_stream):
        print("\n" * 4, end="")
        try:
            while True:
                v, raw, speech = read_doa()
                if v is None:
                    print(f"\033[4F  읽기 실패: {raw}\033[K\n\n\n")
                    time.sleep(1.0)
                    continue
                if speech:
                    vals.append(v)
                    stamped.append((time.time(), v))
                    hist[min(int(v / 180 * (width - 1)), width - 1)] += 1
                print("\033[4F" + "\n".join(
                    line + "\033[K" for line in
                    render(v, speech, hist, width).split("\n")))
                time.sleep(0.08)
        except KeyboardInterrupt:
            print()

    if not vals:
        print("발화로 인식된 표본이 없다. 더 크게, 더 길게 말해볼 것")
        return 1
    whole = stability(vals)
    best = best_window(stamped) or whole
    print(f"\n전체   표본 {whole['n']}개   중앙 {whole['mean']:.1f}°   "
          f"산포 {whole['std']:.1f}°   ±10° 안 {whole['near10']*100:.0f}%")
    print(f"       (좌우로 옮겨 다녔다면 이 값이 큰 것은 정상이다)")
    print(f"가장 안정적이었던 5초   중앙 {best['mean']:.1f}°   "
          f"산포 {best['std']:.1f}°   ±10° 안 {best['near10']*100:.0f}%\n")
    if best["near10"] >= 0.8:
        print("  ✔ 가만히 있을 때 한 방향에 모인다. 장치는 정상이다 — 보정으로 진행.")
    elif best["near10"] >= 0.5:
        print("  · 절반쯤만 모인다. 보정은 되겠지만 오차가 클 것이다.")
    else:
        print("  ✗ 가만히 있어도 값이 퍼진다. 이 상태로는 방향을 못 쓴다.")
        print("    live --no-stream 과 비교해 볼 것 (스트림이 필요한지 확인)")
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
    input("  준비되면 Enter → 6초간 계속 말해주세요 (몸을 고정할 것) ")
    front, raws, st_front = sample_doa()
    if not front:
        print(f"  ! DOA 를 못 읽었다: {raws[:1]}")
        return 1
    # 사람이 6초 내내 완벽히 멈춰 있기를 기대하지 않는다. 가장 안정적이었던
    # 3초를 골라 쓴다 — 그게 실제로 가만히 있던 구간이다.
    w = best_window(st_front, seconds=3.0)
    if w is None:
        offset, sd = circ_mean(front), circ_std(front)
    else:
        offset, sd = w["mean"], w["std"]
    print(f"  정면 원시값 {offset:.1f}°  (표본 {len(front)}개 중 가장 안정된 3초, "
          f"산포 {sd:.1f}°)")
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

    print("[2/2] 한쪽 옆")
    print("  왼쪽이든 오른쪽이든 한쪽으로 확실히 옮기세요 (45° 이상이면 충분).")
    print("  정확히 90° 일 필요는 없습니다 — 여기서는 '어느 쪽이 +인지'만 정합니다.")
    side = ""
    while side not in ("l", "r"):
        side = input("  어느 쪽으로 가시겠습니까? 램프에서 볼 때 왼쪽=l / 오른쪽=r : ").strip().lower()
    input("  그 자리에서 Enter → 6초간 계속 말해주세요 (몸을 고정할 것) ")
    right, raws, st_right = sample_doa()
    if not right:
        print(f"  ! DOA 를 못 읽었다: {raws[:1]}")
        return 1
    wr = best_window(st_right, seconds=3.0)
    r, sd_r = (circ_mean(right), circ_std(right)) if wr is None else (wr["mean"], wr["std"])
    delta = ang_err(r, offset)        # 정면 대비 원시값이 어느 쪽으로 움직였나
    # 규약: 0° = 램프 정면, 반시계 방향 증가. 위에서 내려다볼 때 반시계이므로
    # 램프의 왼쪽이 +90°, 오른쪽이 -90° 다. 그래서 '왼쪽으로 갔을 때 원시값이
    # 커진다' 면 부호가 +1 이다. 여기를 뒤집으면 오차표는 멀쩡해 보이는데
    # 램프가 소리 반대쪽으로 돈다.
    sign = (1 if delta > 0 else -1) * (1 if side == "l" else -1)
    print(f"  원시값 {r:.1f}°  (정면 대비 {delta:+.1f}°, 산포 {sd_r:.1f}°)")

    if sd_r > 10:
        print(f"  ! 산포 {sd_r:.1f}° 는 너무 크다. 다시 실행할 것")
        return 1
    if abs(delta) < 20:
        print("  ! 정면과 거의 같다. 위치를 제대로 옮겼는지 확인할 것")
        return 1
    print(f"  → 원시값이 {'커지는' if sign > 0 else '작아지는'} 쪽이 램프의 왼쪽(+)")

    # 참고용. 이동각을 정확히 모르므로 판정하지 않고 알려만 준다.
    # 원시 각도가 실제 각도에 비례하는지는 measure 결과로 확인한다.
    print(f"  (참고: {'오른' if side == 'r' else '왼'}쪽으로 옮겼을 때 원시값이 "
          f"{abs(delta):.0f}° 움직였다. 실제 이동각과 크게 다르면 원시 각도가"
          f" 실제에 비례하지 않는 것이고, 그건 measure 결과에서 드러난다)\n")

    json.dump({"conv": 2, "offset_deg": offset, "sign": sign,
               "n": len(front), "std": sd,
               "side_raw": r, "side_delta": delta, "side_std": sd_r,
               "side": side},
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
    if cal.get("conv") != 2:
        print("보정 파일이 예전 규약이다. calibrate 를 다시 실행할 것")
        print("  (왼쪽이 + 인 반시계 규약으로 바뀌었다 — 예전 파일은 부호가 반대다)")
        return 1
    offset = cal["offset_deg"]
    os.makedirs(OUT, exist_ok=True)

    print(f"[{args.label}] {len(ANGLES)}방향 측정 — 각 방향에서 1 m 거리, 3초간 발화")
    print(f"보정 {offset:.1f}°, 회전 {'정방향' if cal['sign'] > 0 else '역방향'} 적용. "
          f"0° = 램프 정면, 반시계 방향 증가\n")

    rows = []
    for truth in ANGLES:
        signed = ang_err(truth, 0)
        where = "정면" if signed == 0 else (
            f"왼쪽 {abs(signed):.0f}°" if signed > 0 else f"오른쪽 {abs(signed):.0f}°")
        input(f"  {where:<9} ({signed:+.0f}°) 로 이동 → Enter 후 6초간 말하기 (몸 고정) ")
        vals, raws, stmp = sample_doa()
        if not vals:
            print(f"       DOA 읽기 실패: {raws[:1]}")
            continue
        w = best_window(stmp, seconds=3.0)
        raw_mean, spread = ((circ_mean(vals), circ_std(vals)) if w is None
                            else (w["mean"], w["std"]))
        meas = to_lamp(raw_mean, cal)
        err = ang_err(meas, truth)
        rows.append({"truth": truth, "measured": round(meas, 1),
                     "raw": round(raw_mean, 1), "error": round(err, 1),
                     "spread": round(spread, 1), "n": len(vals)})
        print(f"       측정 {meas:>6.1f}°   오차 {err:>+6.1f}°   산포 {spread:>5.1f}°"
              f"   (원시 {raw_mean:.1f}°)")

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
    ap.add_argument("--no-stream", action="store_true",
                    help="오디오 스트림을 열지 않고 읽는다 (펌웨어가 스트림 없이도 "
                         "방향을 갱신하는지 비교용)")
    ap.add_argument("--angles",
                    help="잴 각도를 직접 지정 (쉼표 구분). --geometry 기본값을 덮는다")
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(q):
        # 하위 명령 뒤에 써도 받도록 양쪽에 단다. 앞에만 두면
        # "live --no-stream" 이 오류가 나는데, 그 순서가 더 자연스럽다.
        q.add_argument("--no-stream", action="store_true",
                       dest="no_stream_sub", help=argparse.SUPPRESS)
        return q

    common(sub.add_parser("calibrate"))
    common(sub.add_parser("live"))
    m = common(sub.add_parser("measure"))
    m.add_argument("--label", required=True,
                   help="조건 이름 (quiet / fan / elevated / servo)")
    common(sub.add_parser("report"))
    args = ap.parse_args()

    global _USE_STREAM
    args.no_stream = args.no_stream or getattr(args, "no_stream_sub", False)
    _USE_STREAM = not args.no_stream
    ANGLES = (ANGLES_CIRCULAR if args.geometry == "circular" else ANGLES_LINEAR)
    if args.angles:
        ANGLES = [float(x) % 360 for x in args.angles.split(",")]

    return {"calibrate": cmd_calibrate, "measure": cmd_measure,
            "live": cmd_live, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
