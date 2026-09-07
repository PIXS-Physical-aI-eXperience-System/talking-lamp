#!/usr/bin/env bash
# XVF3800 호스트 도구를 미리 받아 둔다. 마이크 도착 당일에 이걸 하느라
# 시간을 쓰지 않기 위한 사전 작업이다.
#
#   ./bench/xvf_setup.sh
#
# xvf_host 는 소스에서 빌드하는 물건이 아니라 미리 빌드된 바이너리로
# 배포되며, jetson 용이 따로 들어 있다. 그래서 하는 일은 내려받기와
# 실행 권한 부여뿐이다.
#
# 저장소: https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY
# Flex Circular-4 도 같은 저장소가 커버한다 (같은 XVF3800 코어).
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=tools/xvf3800
REPO=https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY

# 플랫폼 폴더가 5개(jetson, linux_x86_64, mac_arm64, rpi_64bit, win32)이고
# 펌웨어도 여러 벌이라 전부 받으면 낭비다. 필요한 것만 잘라 온다.
if [ -d "$DEST/.git" ]; then
  echo "── 이미 있다. 갱신만 한다"
  git -C "$DEST" pull --ff-only
else
  echo "── 내려받기 (필요한 폴더만)"
  git clone --depth 1 --filter=blob:none --sparse "$REPO" "$DEST"
  git -C "$DEST" sparse-checkout set host_control/jetson xmos_firmwares doc
fi

PLAT="$DEST/host_control/jetson"
chmod +x "$PLAT/xvf_host" "$PLAT/xvf_dfu" 2>/dev/null || true

echo
echo "── 확인"
if [ ! -x "$PLAT/xvf_host" ]; then
  echo "  ✗ $PLAT/xvf_host 가 없다. 저장소 구조가 바뀌었는지 확인할 것"
  exit 1
fi
echo "  ✔ $PLAT/xvf_host"
# --help 는 장치 없이도 돌아야 한다. 여기서 죽으면 아키텍처가 안 맞는 것이다
# (jetson 폴더는 aarch64 용이라 맥이나 x86 리눅스에서는 실행되지 않는다).
if "$PLAT/xvf_host" --help >/dev/null 2>&1; then
  echo "  ✔ 실행 가능"
else
  echo "  ⚠ 실행되지 않는다 — 이 폴더는 aarch64(Jetson) 용이다."
  echo "    맥·x86 에서는 정상이며, Jetson 에서 다시 확인할 것"
fi

echo
echo "── 6채널 펌웨어 (원음 접근에 필요)"
ls -1 "$DEST/xmos_firmwares/usb"/*6ch*.bin 2>/dev/null | sed 's/^/  /' || \
  echo "  ! 6채널 펌웨어를 못 찾았다. xmos_firmwares/usb 를 직접 확인할 것"

cat <<'NEXT'

준비 끝. 마이크가 도착하면:

  1) 전제 확인
     venvs/vad/bin/python bench/mic_check.py

  2) 2채널로 잡히면 6채널 펌웨어로 전환 (XMOS USB-C 포트에 연결할 것)
     sudo apt install dfu-util
     sudo dfu-util -l                       # 장치가 보이는지
     sudo dfu-util -R -e -a 1 -D tools/xvf3800/xmos_firmwares/usb/<6ch 펌웨어>

  3) DOA 측정
     tools/xvf3800/host_control/jetson/xvf_host AEC_AZIMUTH_VALUES
     venvs/vad/bin/python bench/doa_measure.py calibrate

6채널 펌웨어의 채널 구성 (16 kHz / 32 bit):
  0 처리음(회의용)  1 처리음(음성인식용)  2~5 마이크 0~3 원음
NEXT
