#!/usr/bin/env bash
# Read-only XVF3800/GStreamer commissioning checks.
set -euo pipefail

repo=/home/pixs/talking-lamp
card=L16K6Ch
device=plughw:CARD=$card,DEV=0

for command in gst-inspect-1.0 arecord aplay lsusb ss; do
    command -v "$command" >/dev/null || {
        echo "missing command: $command" >&2
        exit 2
    }
done

gst-inspect-1.0 alsasrc alsasink audioconvert audioresample \
    opusenc opusdec rtpopuspay rtpopusdepay rtpjitterbuffer \
    udpsink udpsrc >/dev/null

lsusb -d 2886:0022 >/dev/null
arecord -L | grep -F "CARD=$card" >/dev/null
aplay -L | grep -F "CARD=$card" >/dev/null

# One second of discarded capture confirms the commissioned six-channel mode.
timeout 3 arecord -q -D "$device" -t raw -f S32_LE -r 16000 -c 6 -d 1 /dev/null

cd "$repo"
PYTHONPATH="$repo/src:$repo/lelamp_runtime" \
    "$repo/lelamp_runtime/.venv/bin/python" - <<'PY'
from device.xvf3800 import Xvf3800

device = Xvf3800.discover()
try:
    version = device.read_version()
finally:
    device.close()
if version != (1, 0, 3):
    raise SystemExit(f"unexpected XVF3800 version: {version}")
print("XVF3800 2886:0022 version 1.0.3; ALSA capture/playback and GStreamer elements OK")
PY

ss -lun | grep -E ':(5004|5006)[[:space:]]' || true
