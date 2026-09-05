"""마이크 어레이 도착 직후 확인 — 측정 전에 전제부터 맞는지 본다.

여기서 걸리면 뒤의 측정이 전부 무의미하므로 가장 먼저 돌린다.
아무것도 바꾸지 않고 읽기만 한다.

    python bench/mic_check.py
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def section(t):
    print(f"\n── {t}")


def main() -> int:
    ok = True

    section("오디오 장치")
    try:
        import sounddevice as sd
    except ImportError:
        print("  ! sounddevice 없음:  pip install sounddevice soundfile numpy")
        return 1

    found = []
    for i, d in enumerate(sd.query_devices()):
        name = d["name"]
        if any(k in name.lower() for k in ("xvf", "respeaker", "xmos", "seeed")):
            found.append((i, d))
        if d["max_input_channels"] > 0:
            mark = "★" if any(k in name.lower() for k in ("xvf", "respeaker", "xmos")) else " "
            print(f"  {mark} [{i}] {name}  입력 {d['max_input_channels']}ch  "
                  f"{int(d['default_samplerate'])}Hz")

    if not found:
        print("  ✗ XVF3800 을 못 찾았다. USB 연결과 전원을 확인할 것")
        ok = False
    else:
        i, d = found[0]
        ch = d["max_input_channels"]
        print(f"\n  찾음: [{i}] {d['name']}  입력 {ch}ch")
        if ch >= 6:
            print("  ✔ 6채널 이상 — 처리음 2 + 원음 4 로 보인다 (빔포밍·DOA 자체 구현 가능)")
        else:
            print(f"  ⚠ {ch}채널뿐 — 2채널 처리음 펌웨어일 수 있다.")
            print("    6채널 펌웨어로 전환해야 원음에 접근할 수 있다:")
            print("      dfu-util -R -e -a 1 -D respeaker_flex_ua-io16-6ch-cir.bin")
            ok = False

    section("xvf_host (DOA 읽기 도구)")
    exe = shutil.which("xvf_host") or shutil.which("xvf_host.py")
    if exe:
        print(f"  ✔ {exe}")
        try:
            r = subprocess.run([exe, "AEC_AZIMUTH_VALUES"], capture_output=True,
                               text=True, timeout=10)
            print(f"  출력: {(r.stdout or r.stderr).strip()[:200]}")
        except Exception as e:
            print(f"  ! 실행 실패: {type(e).__name__} {e}")
            ok = False
    else:
        print("  ✗ xvf_host 없음 — DOA 를 읽을 수 없다.")
        print("    https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY")
        print("    host_control/ 에서 빌드하거나 xvf_host.py 를 쓴다")
        ok = False

    section("스피커 출력 경로 (AEC 의 전제)")
    outs = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
            if d["max_output_channels"] > 0]
    for i, n in outs:
        mark = "★" if any(k in n.lower() for k in ("xvf", "respeaker", "xmos")) else " "
        print(f"  {mark} [{i}] {n}")
    if not any(any(k in n.lower() for k in ("xvf", "respeaker", "xmos")) for _, n in outs):
        print("\n  ⚠ XVF3800 이 출력 장치로 안 잡힌다.")
        print("    스피커를 보드의 JST 커넥터에 연결하고, TTS 출력을 이 장치로 보내야")
        print("    하드웨어 AEC 가 참조신호를 얻는다. Jetson 에 직결하면 AEC 가 죽는다.")
        ok = False
    else:
        print("\n  ✔ 출력 장치로 잡힌다 — TTS 를 이쪽으로 보낼 것")

    print("\n" + ("전제 확인 완료. bench/doa_measure.py 로 진행" if ok
                  else "위 ✗ 항목을 먼저 해결할 것"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
