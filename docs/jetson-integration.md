# Jetson Integration Guide

## Ownership boundary

The Jetson owns STT, dialogue, TTS, response policy and the ROS 2 API. The
Raspberry Pi owns all physical devices and safety enforcement: XVF3800 USB
capture/playback/control, DOA stabilization, WS2812B output, Feetech motion and
the 100 Hz owner loop. Do not install ROS on the Pi and do not send raw joint
angles, servo registers, GPIO waveforms, Opus packets assembled in Python, or
firmware operations from Jetson application code.

The direct wired link is fixed:

| Endpoint | Address/port | Purpose |
| --- | --- | --- |
| Jetson | `192.168.100.1` | ROS and GStreamer bridge host |
| Pi motion | `192.168.100.2:8765/TCP` | Named motions/tracking/task light |
| Pi device | `192.168.100.2:8766/TCP` | Direction, audio lifecycle and LED |
| Jetson capture | `192.168.100.1:5004/UDP` | Pi microphone Opus/RTP |
| Pi playback | `192.168.100.2:5006/UDP` | Jetson TTS Opus/RTP |

TCP sessions are independent and each permits one authenticated Jetson owner.
The device bridge must never connect to the motion port and vice versa.

## Jetson platform

The commissioned Jetson reports Ubuntu 24.04.4, L4T 39.2, aarch64 and Python
3.12, so it uses ROS 2 Jazzy. Install `ros-jazzy-ros-base`, not the desktop
variant, using the official ROS apt instructions:
<https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html>.

The deployed integration checkout is `/home/asdf/talking-lamp-integration` and
its ROS workspace is `jetson_ws`. The original `/home/asdf/talking-lamp`
checkout remains untouched. After installation:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/asdf/talking-lamp-integration/jetson_ws
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y
colcon build --symlink-install
source install/setup.bash
```

## Secrets

Use two root/operator-provisioned mode-0600 files on Jetson:

```text
/etc/talking-lamp/motion-bridge.env  TALKING_LAMP_MOTION_TOKEN=...
/etc/talking-lamp/device-bridge.env  TALKING_LAMP_DEVICE_TOKEN=...
```

Their values must equal the corresponding Pi files. Never commit, print or
place a token in a ROS launch file. A reconnect creates a new TCP session and
does not replay the prior request, motion, TTS stream or LED frame.

The system service loads both files and starts both bridge nodes:

```bash
sudo systemctl start talking-lamp-bridges.service
systemctl status talking-lamp-bridges.service
```

Keep it disabled during commissioning. Enable it at boot only after the live
audio, motion, return-center and idle-ordering checks pass.

## ROS API

Motion bridge:

- `/lamp/play_motion` — `lamp_interfaces/action/PlayMotion`
- `/lamp/place_task_light` — `lamp_interfaces/action/PlaceTaskLight`
- `/lamp/interrupt_motion` — `lamp_interfaces/srv/InterruptMotion`
- `/lamp/list_motions` — `lamp_interfaces/srv/ListMotions`
- `/lamp/track_point`, `/lamp/track_bearing` — latest-value topics
- `/lamp/motion_status` — controller/transport health

Device bridge:

- `/lamp/audio/capture` — 20 ms `AudioFrame`, 16 kHz mono `pcm_s16le`
- `/lamp/audio/playback_frames` — TTS producer frames for one accepted stream
- `/lamp/play_audio` — owns playback start, EOS and Pi drain completion
- `/lamp/audio_status` — capture/playback/sequence/buffer state
- `/lamp/orientation_status` — Pi DOA/alignment status with `speech_id`
- `/lamp/return_center` — completes only at Pi `centered`
- `/lamp/led/frame` — `sensor_msgs/Image`, exactly 8×8 `rgb8`
- `/lamp/led/set_solid`, `/lamp/led/clear`, `/lamp/led/status`

Only the first release format is accepted: `sample_rate=16000`, `channels=1`,
`encoding=pcm_s16le`, and 640 data bytes for every non-EOS frame. Sequence
starts at zero, increments by one and has one empty EOS frame. Invalid, stale,
duplicate or wrong-stream frames are rejected before GStreamer.

## Required conversation ordering

Application code supplies a `TurnResponse(speech_id, motion_name,
audio_stream)` to `TurnOrchestrator`. The helper enforces:

```text
matching orientation state == aligned
        ↓
PlayAudio and PlayMotion start concurrently
        ↓ wait for both terminal success
ReturnCenter
        ↓ wait for centered
PlayMotion("idle")
```

There is no automatic center or idle if TTS or expression motion fails. The
policy layer must inspect the machine-readable result and may explicitly
interrupt, retry or return to center. This prevents a failed or still-running
expression from being hidden by an early center request.

For barge-in, call `TurnOrchestrator.barge_in()`. It cancels audio and motion,
then calls motion interrupt. The current direction anchor is retained; the
next VAD rising edge receives a new `speech_id` and can replace it.

## Team application adapter

The STT/dialogue/TTS implementation remains outside this repository. Its
adapter must implement the injected `TurnPorts` methods in
`lamp_interaction.turn`; it must not import Pi modules. Capture STT results and
orientation events are joined by exact canonical `speech_id`, not wall time or
an assumed "latest" utterance.

## Failure behavior

- Wired TCP loss: all pending Actions fail; nothing is replayed on reconnect.
- Capture RTP loss: short gaps use Opus concealment; status faults after the
  configured long gap while motion remains responsive.
- XVF removal: Pi keeps device TCP/LED alive, marks XVF fault and rediscovers
  with exponential backoff. A reconnect begins a new capture stream.
- Playback loss: PlayAudio fails, so automatic center/idle does not run.
- Device-session change: event sequence restarts at one; the bridge discards
  events from older session IDs.
- LED remains software-only until the physical mapping and shared-5 V power
  tests in `docs/pi-device-commissioning.md` are completed.

## Commissioning record (2026-09-16)

- ROS 2 Jazzy workspace: four packages built successfully; `rosdep check`
  reported all system dependencies satisfied.
- Pi microphone to Jetson: 16 kHz mono `pcm_s16le`, 20 ms frames at
  49.989-50.003 Hz.
- Jetson playback to Pi: bounded 440 Hz tone was audible; `PlayAudio` returned
  `success=true`, `code=drained`.
- Direction alignment: live XVF3800 event produced a canonical `speech_id` and
  terminal `aligned` state; the base reached the reported target yaw.
- Response ordering: audio and low-intensity `nod` were accepted together and
  both completed (`drained` / `completed`) before `ReturnCenter` completed at
  yaw `0.261799` rad; only then was `idle` accepted.
- Looping idle responsiveness: while `active_motion=idle`, motion status and
  list services remained responsive, interrupt succeeded, and idle could be
  started again.
- Self-playback guard: Pi ignores XVF VAD/DOA while playback is active and for
  0.3 seconds after drain, preventing the speaker response from becoming a new
  orientation request.
- Services remain disabled at boot during commissioning. Physical WS2812B
  mapping/power acceptance is still pending because the module is disconnected.
