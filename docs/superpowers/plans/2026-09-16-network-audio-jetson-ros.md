# Network Audio and Jetson ROS Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream XVF3800 microphone audio to Jetson and TTS audio back to the Pi, expose device and motion functions through ROS 2 Jazzy, and enforce the aligned → simultaneous response/motion → centered → idle turn sequence.

**Architecture:** GStreamer owns ALSA, Opus, RTP and jitter buffering; application code only validates stream metadata, supervises pipelines, and exchanges 20 ms PCM frames with ROS. The Pi remains ROS-free and exposes separate authenticated motion and device TCP sessions. Pure-Python Jetson transports stay independent of `rclpy`; ROS nodes only adapt those transports to generated interfaces, and a separately tested interaction helper joins Action results in the required order.

**Tech Stack:** Python 3.12 on Pi and Jetson, asyncio, GStreamer 1.24, Opus/RTP/UDP, ROS 2 Jazzy/rclpy, rosidl, pytest, systemd

**Spec:** `docs/superpowers/specs/2026-09-15-pi-audio-doa-led-jetson-ros-design.md`

## Global Constraints

- Raspberry Pi has no ROS 2 installation and remains the sole owner of XVF3800 ALSA/control, WS2812B, and Feetech hardware.
- Pi capture is processed 16 kHz mono PCM encoded as Opus/RTP to `192.168.100.1:5004`; playback is Opus/RTP from Jetson to Pi `192.168.100.2:5006` with a 40 ms jitter buffer.
- GStreamer performs all codec, RTP, resampling, jitter and ALSA work; Python must not implement Opus or RTP packet assembly.
- ALSA uses the stable card ID `L16K6Ch`, never a numeric card index.
- Device TCP remains `192.168.100.2:8766`, motion TCP remains `:8765`, and both accept only Jetson `192.168.100.1`.
- `AudioFrame` supports only `pcm_s16le`, 16 kHz, mono, 20 ms/640-byte frames in the first release.
- TTS playback success is returned only after the Pi receiver receives stop, sends EOS/SIGINT to GStreamer, and the process exits successfully after ALSA drain.
- Capture/playback loss does not block the 100 Hz motion owner; stale streams or commands are never replayed after reconnect.
- Jetson is Ubuntu 24.04/L4T 39.2 and therefore uses ROS 2 Jazzy, installed from the official ROS package repository.
- TTS and expression motion start concurrently only after matching `speech_id` is `aligned`; both must succeed before return-center, and `idle` starts only after `centered`.
- The LED hardware flag remains absent and physical LED/power tests remain blocked until the module is wired and identified.

---

### Task 1: Pi GStreamer Audio Supervisor

**Files:**
- Create: `src/device/audio.py`
- Create: `tests/test_device_audio.py`
- Modify: `src/device/protocol.py`
- Modify: `src/device/server.py`
- Modify: `tests/test_device_protocol.py`
- Modify: `tests/test_device_server.py`

**Interfaces:**
- Consumes: stable ALSA card ID, Pi/Jetson wired addresses, capture/playback ports, injected async process factory.
- Produces: `AudioConfig`, `AudioStatus`, `capture_pipeline(config)`, `playback_pipeline(config)`, and async `AudioSupervisor.start()`, `play_start(metadata)`, `play_stop(stream_id)`, `close()`.

- [x] **Step 1: Write failing pure pipeline and lifecycle tests**

Assert capture contains `alsasrc device=plughw:CARD=L16K6Ch,DEV=0`, mono/16 kHz conversion, `opusenc frame-size=20`, `rtpopuspay`, host `192.168.100.1`, port 5004 and bind `192.168.100.2`. Assert playback contains a source-bound `udpsrc` on 5006, exact Opus caps, `rtpjitterbuffer latency=40 drop-on-latency=true`, decode/resample and `alsasink device=plughw:CARD=L16K6Ch,DEV=0`.

Use a fake process with `returncode`, `send_signal`, and `wait`. Assert a second stream is rejected as `audio_busy`, wrong PCM metadata is rejected before process creation, mismatched stop is `unknown_stream`, and valid stop sends SIGINT then waits before returning `drained`.

- [x] **Step 2: Run focused tests and verify the missing module failure**

Run: `pytest -q tests/test_device_audio.py`

Expected: collection fails with `ModuleNotFoundError: device.audio`.

- [x] **Step 3: Implement strict configuration, pipeline builders, and supervisor**

```python
@dataclass(frozen=True)
class AudioConfig:
    alsa_card: str = "L16K6Ch"
    pi_host: str = "192.168.100.2"
    jetson_host: str = "192.168.100.1"
    capture_port: int = 5004
    playback_port: int = 5006
    jitter_ms: int = 40

@dataclass(frozen=True)
class AudioStatus:
    capture_running: bool
    playback_running: bool
    stream_id: str | None
    state: str
    code: str
    message: str
```

Build argument arrays for `gst-launch-1.0 -q -e`; never invoke a shell. Keep one continuous capture process. Playback begins on `audio.play.start` and `audio.play.stop` sends SIGINT and awaits a bounded exit, killing only after timeout and returning `drain_timeout`.

- [x] **Step 4: Extend strict device commands**

Add exact commands:

```text
audio.play.start {stream_id, sample_rate, channels, encoding}
audio.play.stop  {stream_id}
audio.status     {}
```

`stream_id` is canonical UUID, sample rate is exactly 16000, channels exactly 1, encoding exactly `pcm_s16le`. `DeviceCommandHandler` accepts an audio supervisor; `device.status` includes audio status and disconnect stops active playback without stopping capture.

- [x] **Step 5: Run protocol/audio/server tests and commit**

Run: `pytest -q tests/test_device_audio.py tests/test_device_protocol.py tests/test_device_server.py`

Commit: `feat(audio): supervise Pi RTP capture and playback`

### Task 2: Pi Audio Daemon and Software Commissioning

**Files:**
- Modify: `src/device/daemon.py`
- Modify: `tests/test_device_daemon.py`
- Modify: `deploy/pi/talking-lamp-device.service`
- Modify: `tests/test_device_deploy.py`
- Create: `deploy/pi/check-audio.sh`
- Modify: `docs/pi-device-commissioning.md`

**Interfaces:**
- Consumes: Task 1 `AudioSupervisor`, existing XVF adapter and device server.
- Produces: capture autostart before XVF polling, independent audio status/fault events, bounded shutdown, and exact Pi diagnostic commands.

- [ ] **Step 1: Write failing daemon integration tests**

Assert order `server.start → led.open → audio.start → xvf.open → poll`, and shutdown `server.close → audio.close → led.clear/close → xvf.close`. A capture process crash emits `audio.status/fault` and is restarted with bounded exponential backoff without stopping device TCP. Runtime XVF loss emits a fault and rediscovers USB without restarting motion.

- [ ] **Step 2: Implement audio lifecycle and XVF rediscovery**

Add daemon CLI options `--alsa-card`, `--capture-port`, `--playback-port`, `--jitter-ms`. Poll process state every device tick. Backoff sequence is 0.25, 0.5, 1, 2, 4 seconds capped at 5 seconds; a newly discovered XVF must report `(1,0,3)` before DOA resumes. Reset utterance collection on device loss.

- [ ] **Step 3: Add exact deployment and diagnostic contract**

The unit includes audio arguments but still omits `--enable-led-hardware`. `check-audio.sh` verifies `gst-inspect-1.0` elements, stable ALSA capture/playback names, 6-channel device capability, ports, XVF firmware and GStreamer loopback without writing firmware.

- [ ] **Step 4: Run full tests, deploy disabled, and commission over wired LAN**

Run the full local suite, sync only changed files after a recoverable Pi backup, install disabled, start temporarily, receive capture RTP on Jetson with a bounded `gst-launch-1.0` fakesink test, send a generated tone from Jetson to Pi, obtain user audibility confirmation, then stop the service and confirm it remains disabled.

- [ ] **Step 5: Commit measured evidence**

Record exact ALSA card string, negotiated formats, packet counters, process exit codes, and whether full-duplex caused USB resets. Commit: `feat(audio): integrate and commission Pi network audio`.

### Task 3: ROS Interfaces

**Files:**
- Create: `jetson_ws/src/lamp_interfaces/msg/AudioFrame.msg`
- Create: `jetson_ws/src/lamp_interfaces/msg/AudioStatus.msg`
- Create: `jetson_ws/src/lamp_interfaces/msg/OrientationStatus.msg`
- Create: `jetson_ws/src/lamp_interfaces/msg/LedStatus.msg`
- Create: `jetson_ws/src/lamp_interfaces/action/PlayAudio.action`
- Create: `jetson_ws/src/lamp_interfaces/action/ReturnCenter.action`
- Create: `jetson_ws/src/lamp_interfaces/srv/SetLedSolid.srv`
- Modify: `jetson_ws/src/lamp_interfaces/CMakeLists.txt`
- Modify: `jetson_ws/src/lamp_interfaces/package.xml`
- Modify: `tests/test_deploy_scripts.py`

**Interfaces:**
- Consumes: generated ROS 2 built-in time and `sensor_msgs/Image`/`std_srvs/Trigger` at node boundaries.
- Produces: exact interfaces listed in spec sections 8-9.

- [ ] **Step 1: Write failing static contract tests**

Assert exact field order/types. `PlayAudio` goal is stream ID/sample rate/channels/encoding; result is success/code/message; feedback is received sequence, played sequence, buffered ms. `ReturnCenter` result includes success/code/message/current yaw. Orientation status includes speech ID, raw/relative/target/current angles, clamped, state/code/message.

- [ ] **Step 2: Add interface files and ament dependencies**

Add `sensor_msgs` and `std_srvs` dependencies where used by packages; register all messages, services and actions with `rosidl_generate_interfaces`.

- [ ] **Step 3: Run static tests and commit**

Run: `pytest -q tests/test_deploy_scripts.py`

Commit: `feat(ros): define lamp audio orientation and LED interfaces`.

### Task 4: Pure Jetson Motion and Device Transports

**Files:**
- Create: `jetson_ws/src/lamp_motion_bridge/lamp_motion_bridge/transport.py`
- Create: `jetson_ws/src/lamp_device_bridge/lamp_device_bridge/transport.py`
- Create: `jetson_ws/src/lamp_device_bridge/lamp_device_bridge/audio.py`
- Create: `tests/test_jetson_motion_transport.py`
- Create: `tests/test_jetson_device_transport.py`
- Create: `tests/test_jetson_audio.py`

**Interfaces:**
- Consumes: authenticated NDJSON TCP and GStreamer GI appsink/appsrc.
- Produces: reconnecting `MotionTransport`, `DeviceTransport`, ordered event callbacks, `CaptureReceiver`, and `PlaybackSender`; none imports `rclpy`.

- [ ] **Step 1: Write failing transport tests**

Test canonical IDs, receive-time independent TTL, accepted/terminal correlation, 16 KiB cap, heartbeat every 1 second, one reconnect loop, pending-request failure on disconnect, no replay, new device session resets event sequence, old session events are discarded, and callbacks never run on the socket reader while holding its write lock.

- [ ] **Step 2: Implement shared Python-3.12 asyncio behavior**

Each transport has `connect()`, `request(kind,payload,ttl_ms)`, `events()`, and `close()`. The device transport validates server-pushed `session_id/sequence`. Token is provided by constructor and never logged.

- [ ] **Step 3: Write failing audio frame tests and implement GI adapters**

Validate exactly 640 bytes per non-EOS frame, monotonically increasing sequence, canonical stream ID, 16000/1/`pcm_s16le`, duplicate/stale rejection and one EOS. `CaptureReceiver` uses appsink with max-buffers=10/drop=true; `PlaybackSender` uses appsrc and emits Opus RTP to Pi port 5006. GI imports are lazy.

- [ ] **Step 4: Run pure tests and commit**

Run: `pytest -q tests/test_jetson_motion_transport.py tests/test_jetson_device_transport.py tests/test_jetson_audio.py`

Commit: `feat(jetson): add pure motion device and audio transports`.

### Task 5: ROS Bridge Nodes and Packages

**Files:**
- Create: `jetson_ws/src/lamp_motion_bridge/lamp_motion_bridge/node.py`
- Create: package/launch/setup metadata under `jetson_ws/src/lamp_motion_bridge/`
- Create: `jetson_ws/src/lamp_device_bridge/lamp_device_bridge/node.py`
- Create: package/launch/setup metadata under `jetson_ws/src/lamp_device_bridge/`
- Create: `tests/test_jetson_bridge_contract.py`

**Interfaces:**
- Consumes: Task 3 generated interfaces and Task 4 pure transports.
- Produces: every ROS name in spec sections 9.1-9.2.

- [ ] **Step 1: Write failing static node/launch contract tests**

Assert launch parameters, exact topic/action/service names, `KEEP_LAST(1)` for tracking/LED latest-value paths, bounded capture queue, reliable Actions/Services, and absence of Pi implementation imports.

- [ ] **Step 2: Implement motion node**

Map `/lamp/play_motion`, `/lamp/place_task_light`, `/lamp/interrupt_motion`, `/lamp/list_motions`, tracking topics and `/lamp/motion_status` to one persistent motion transport. Action cancellation sends the correlated cancel command and waits for terminal cancellation.

- [ ] **Step 3: Implement device node**

Publish capture frames, audio/orientation/LED status, accept LED Image/solid/clear calls, and implement PlayAudio/ReturnCenter Actions. PlayAudio validates the goal, starts Pi receiver, streams appsrc frames, sends stop at EOS, and succeeds only on Pi `drained`. ReturnCenter succeeds only for terminal `centered`.

- [ ] **Step 4: Add packaging and launch files**

Install console scripts `lamp_motion_bridge` and `lamp_device_bridge`. Launch defaults match fixed wired addresses/ports but read tokens from environment, never source.

- [ ] **Step 5: Run static/pure tests and commit**

Commit: `feat(ros): bridge lamp motion audio orientation and LED`.

### Task 6: Turn Orchestration Helper

**Files:**
- Create: `jetson_ws/src/lamp_interaction/lamp_interaction/turn.py`
- Create: package metadata under `jetson_ws/src/lamp_interaction/`
- Create: `tests/test_lamp_interaction.py`
- Create: `docs/jetson-integration.md`

**Interfaces:**
- Consumes: a conversation result with `speech_id`, TTS frame async iterator, and motion name plus injected Action clients.
- Produces: `TurnOrchestrator.run(response) -> TurnResult` with explicit cancellation/failure policy.

- [ ] **Step 1: Write ordering and failure tests**

Use timestamped fakes. Assert wait-aligned completes before TTS/motion start; TTS and motion overlap; return-center begins after both successful terminal results; idle begins only after centered. If either response task fails, assert no automatic center or idle. On barge-in, assert audio/motion cancellation and interrupt occur while orientation is retained.

- [ ] **Step 2: Implement the minimal helper**

Use `asyncio.TaskGroup` or `gather` with explicit sibling cancellation. Match orientation by `speech_id`, enforce timeouts, and return machine-readable codes instead of swallowing exceptions.

- [ ] **Step 3: Document the Jetson team boundary and commit**

Document topic/action examples, failure policy, token locations, launch commands and the required call ordering. Commit: `feat(interaction): enforce aligned response center idle sequence`.

### Task 7: Jetson Install and End-to-End Acceptance

**Files:**
- Create: `deploy/jetson/detect-platform.sh`
- Create: `deploy/jetson/install-ros-bridges.sh`
- Create: `deploy/jetson/talking-lamp-bridges.service`
- Create: `tests/test_jetson_deploy.py`
- Modify: `docs/jetson-integration.md`

**Interfaces:**
- Consumes: Ubuntu/L4T facts, official ROS 2 repository, colcon workspace, two Pi token files.
- Produces: disabled-by-default Jetson service and recorded E2E acceptance evidence.

- [ ] **Step 1: Write failing platform/installer tests**

Map Ubuntu 24.04 plus L4T 39.x to Jazzy and reject unknown combinations. Staged install preserves mode-0600 token files, never starts services, and requires explicit `--enable`. Unit binds to the wired Pi addresses and sources `/opt/ros/jazzy/setup.bash` plus workspace install.

- [ ] **Step 2: Implement installer using official ROS Jazzy apt procedure**

Install `ros-jazzy-ros-base`, `python3-colcon-common-extensions`, `python3-rosdep`, GStreamer GI/plugins, initialize rosdep only when needed, run rosdep and `colcon build --symlink-install`. Do not install ROS desktop.

- [ ] **Step 3: Build and run contract smoke tests on Jetson**

Run `colcon build`, source the workspace, inspect all interface types, launch bridges, verify two authenticated TCP owners, heartbeat/status topics and capture frame flow. Keep both Pi services disabled until this passes.

- [ ] **Step 4: Execute controlled hardware E2E**

With the user speaking from random front directions, verify matching `speech_id`, `aligned`, simultaneous TTS/expression, no early return, `centered`, then `idle`. Test LAN disconnect, playback loss and XVF unplug/replug without stale replay. Leave LED physical tests pending until connected.

- [ ] **Step 5: Run verification, record remaining physical blocker, and commit**

Run the full repository suite, ROS build/tests, and service-state checks. Record exact commands/results and keep services disabled unless the user explicitly chooses boot enablement. Commit: `docs: record Jetson Pi end-to-end acceptance`.

## Completion Check

- Full local Python suite passes in the project virtual environment.
- Pi service is installed but disabled/inactive after tests; LED hardware flag is absent.
- Jetson ROS 2 Jazzy workspace builds on the actual Ubuntu 24.04/L4T 39.2 host.
- RTP capture and playback pass over the dedicated wired link, using stable ALSA card ID and no USB reset.
- Device and motion transports reconnect without replaying old work.
- Automated tests prove aligned → concurrent response/motion → centered → idle ordering.
- Live E2E proves the same ordering; only WS2812B wiring/mapping/power acceptance may remain blocked by the disconnected module.
