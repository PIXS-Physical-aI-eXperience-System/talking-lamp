# Raspberry Pi Device Service Commissioning

This procedure commissions the reSpeaker Flex XVF3800 Linear-4 and the
WS2812B-64 control service without allowing unverified LED hardware to turn on.
The Raspberry Pi owns the microphone, speaker, direction calculation, LED and
base-yaw alignment; ROS remains on the Jetson.

## Fixed network and USB identity

- Pi wired address: `192.168.100.2`
- Jetson wired address: `192.168.100.1`
- Device control: authenticated TCP `192.168.100.2:8766`
- XVF3800 USB identity: `2886:0022`
- Commissioned firmware version: `1.0.3`
- Motion boundary: `/run/talking-lamp/motion-control.sock`

The runtime adapter is read-only. It does not expose firmware write, reset, or
generic USB-control methods. A firmware mismatch stops the device daemon and
does not stop `talking-lamp-motion.service`.

The installed udev rule must remain:

```udev
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0022", GROUP="plugdev", MODE="0660", TAG+="uaccess"
```

Verify it with:

```bash
lsusb -d 2886:0022
ls -l /dev/bus/usb/$(lsusb -d 2886:0022 | awk '{gsub(":", "", $4); print $2 "/" $4}')
```

## DOA calibration gate

The required file is
`/home/pixs/talking-lamp/voice-bench/out/doa/calibration.json`. It has exactly
these fields:

```json
{
  "doa_zero_deg": 0.0,
  "doa_direction_sign": 1,
  "front_half_angle_deg": 90.0
}
```

The numbers above illustrate the schema and are not installation values. Use
the result measured with `voice-bench`; never copy a calibration from another
mounting. Record the exact artifact before install:

```bash
sha256sum /home/pixs/talking-lamp/voice-bench/out/doa/calibration.json
```

Commissioning record:

```text
calibration SHA256: acaec65d96d31137d27a58beab947f1b75fdfc35529a328458797d940d5fe797
measured zero/sign/date/operator: offset=90.49852890666492, sign=1, 2026-09-16
```

## Software-only install and test

Keep the WS2812B power and data wires physically disconnected. Install without
boot enablement:

```bash
cd /home/pixs/talking-lamp
sudo deploy/pi/install-device-service.sh
systemctl is-enabled talking-lamp-device.service   # expected: disabled
systemctl is-active talking-lamp-device.service    # expected: inactive
```

The checked-in unit intentionally omits `--enable-led-hardware`; therefore it
uses the in-memory `NullPixelSink` even if the module is accidentally wired.
For a foreground dry run, load the protected token without printing it and run:

```bash
sudo -u pixs bash -c 'set -a; . /etc/talking-lamp/device.env; set +a; \
  PYTHONPATH=/home/pixs/talking-lamp/src:/home/pixs/talking-lamp/lelamp_runtime \
  /home/pixs/talking-lamp/lelamp_runtime/.venv/bin/python -m device.daemon \
  --bind 192.168.100.2 --port 8766 --allow-host 192.168.100.1 \
  --motion-socket /run/talking-lamp/motion-control.sock \
  --calibration /home/pixs/talking-lamp/voice-bench/out/doa/calibration.json \
  --sample-rate 20 --gpio-pin 12 --xvf-vid 0x2886 --xvf-pid 0x0022 \
  --max-brightness 0.10'
```

Send `SIGTERM` and verify an exit code of zero. The shutdown order is: stop
polling, close TCP sessions, clear and close the LED sink, then dispose the USB
control handle. The independent motion service must remain active.

## LED wiring gate

Do not power the matrix from a programmable GPIO or the 3.3 V rail. The planned
wiring is Pi 5 V header to module 5 V, Pi GND to module GND, and GPIO 12 to the
module **DIN**. Do not connect until DIN/DOUT, 5 V, GND and the physical first
pixel have been identified from the actual board.

After wiring is inspected, make a temporary copy of the unit with
`--enable-led-hardware`; do not add that flag to the default checked-in unit.
Keep `--max-brightness 0.10` for mapping checks. Verify, in this order:

1. Clear/all off.
2. One low-brightness red pixel at logical `(0, 0)`.
3. First logical row.
4. First logical column.
5. Red/green checkerboard.

Adjust `layout`, `origin`, `rotation_deg`, and `color_order` from observed
results. Do not compensate for unstable 3.3 V data by retrying frames; use an
appropriate 74AHCT-family level shifter if the electrical level is unreliable.

## Shared-5 V power gate

Only after mapping passes, test full white at 10%, 25%, 50%, 75%, then 100%.
At every level run XVF3800 capture and playback concurrently and record:

```bash
vcgencmd get_throttled
vcgencmd measure_volts core
journalctl -k --since '-2 minutes' | grep -Ei 'under-voltage|usb|reset|disconnect'
```

Immediately clear the matrix and stop if a new throttling bit, voltage drop,
USB reconnect, reboot, flicker, or abnormal connector/cable/module heating is
observed. The temporary ceiling is the previous passing level. An operator must
approve the final repeated-test ceiling before changing the production
`--max-brightness`; software never assumes 100% is safe merely because the Pi 5
uses an official adapter.

## Acceptance record

```text
XVF identity/version: 2886:0022 / 1.0.3
Null-sink SIGTERM clear: PASS (2026-09-16; service exited cleanly and remained disabled)
capture format: S16_LE, 16000 Hz, 6 channels (ALSA hardware)
Jetson capture output: pcm_s16le, 16000 Hz, mono, 20 ms frames
capture ROS rate: PASS (49.989-50.003 Hz, 50-frame window, 2026-09-16)
playback format: S16_LE, 16000 Hz, 2 channels (ALSA hardware)
Jetson-to-Pi playback action: PASS (success=true, code=drained, 2026-09-16)
full-duplex USB reset/disconnect: none observed during bounded tone test
Pi throttling after bounded full-duplex test: throttled=0x0
operator audibility confirmation: PASS (bounded 440 Hz tone heard, 2026-09-16)
pixel mapping: PENDING (LED disconnected)
10/25/50/75/100% power results: PENDING (LED disconnected)
approved max brightness: 0.10 unverified default
```
