#!/usr/bin/env bash
# reSpeaker Flex(XVF3800) 호스트 도구와 펌웨어를 받아 둔다.
#
#   ./bench/xvf_setup.sh
#
# 주의: Flex 는 reSpeaker_XVF3800_USB_4MIC_ARRAY 와 다른 제품이고 저장소도
# 다르다. 4MIC 저장소의 펌웨어를 Flex 에 넣으면 안 된다.
#   Flex   https://github.com/respeaker/reSpeaker_Flex
#   4MIC   https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY
#
# Flex 펌웨어 이름 규칙: respeaker_flex_usb_<c|l><표본율><채널>_v<버전>.bin
#   c = circular(원형, 44mm)   l = linear(선형, 33mm)
#   16k6ch = 16 kHz 6채널      48k2ch = 48 kHz 2채널
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=tools/respeaker-flex
REPO=https://github.com/respeaker/reSpeaker_Flex

if [ -d "$DEST/.git" ]; then
  echo "── 이미 있다. 갱신만 한다"
  git -C "$DEST" pull --ff-only
else
  echo "── 내려받기"
  git clone --depth 1 "$REPO" "$DEST"
fi

echo
echo "── USB 펌웨어 (6채널만 원음에 접근할 수 있다)"
for g in l c; do
  [ "$g" = l ] && name="선형(linear)" || name="원형(circular)"
  f=$(ls -1 "$DEST/xmos_firmwares/usb/respeaker_flex_usb_${g}16k6ch"*.bin 2>/dev/null | sort | tail -1 || true)
  [ -n "$f" ] && echo "  $name  $f" || echo "  $name  없음"
done

echo
echo "── DOA 읽기"
if [ -f "$DEST/python_control/respeaker_get_doa.py" ]; then
  echo "  ✔ $DEST/python_control/respeaker_get_doa.py (공식 예제)"
  echo "    bench/doa_measure.py 는 같은 USB 제어 전송을 직접 쓴다 — 바이너리 불필요"
else
  echo "  ! 공식 예제를 못 찾았다. 저장소 구조가 바뀌었는지 확인할 것"
fi
python3 -c "import usb.core" 2>/dev/null \
  && echo "  ✔ pyusb 설치됨" \
  || echo "  ✗ pyusb 없음 —  venvs/vad/bin/pip install pyusb"

cat <<'NEXT'

── 다음 순서

  1) 6채널 펌웨어 굽기. XMOS USB-C 포트(3.5mm 잭 쪽)에 연결할 것.
     sudo apt install dfu-util
     sudo dfu-util -l                     # 장치가 보이는지 먼저
     sudo dfu-util -R -e -a 1 -D tools/respeaker-flex/xmos_firmwares/usb/<위 파일>

  2) 전제 확인
     venvs/vad/bin/python bench/mic_check.py

  3) DOA — 선형이면 --geometry linear (기본값)
     venvs/vad/bin/python bench/doa_measure.py calibrate
     venvs/vad/bin/python bench/doa_measure.py measure --label quiet

6채널 구성 (16 kHz / 32 bit):
  0 처리음(회의)  1 처리음(음성인식)  2~5 마이크 0~3 원음
NEXT
