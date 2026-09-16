# Pi Device DOA and LED Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Raspberry Pi device service that reads XVF3800 speech/DOA, stabilizes and calibrates one direction per utterance, drives the existing local orientation socket, and exposes a power-limited WS2812B-64 control surface without enabling unverified LED hardware.

**Architecture:** A new dependency-light `device` package separates pure DOA/LED policy from USB, Unix-socket, TCP, and GPIO adapters. The device daemon owns XVF3800 USB control and the LED driver, while the existing motion daemon remains the only motor owner; direction changes cross the protected Unix socket. Hardware libraries are imported only by concrete adapters so all policy and transport tests run on non-Pi hosts.

**Tech Stack:** Python 3.12+, asyncio, PyUSB (Pi runtime extra), rpi-ws281x (Pi runtime extra), pytest, systemd

**Spec:** `docs/superpowers/specs/2026-09-15-pi-audio-doa-led-jetson-ros-design.md`

## Global Constraints

- Do not install ROS 2 on Raspberry Pi.
- Read XVF3800 `DOA_VALUE` at 20 Hz and create exactly one canonical UUID on each speech-detected rising edge.
- Require at least 6 valid samples and 400 ms before accepting; extend collection only to 1.0 s and reject unstable or insufficient input.
- Use circular medoid and circular median absolute deviation; default stability threshold is 12 degrees.
- Apply calibrated `doa_zero_deg` and `doa_direction_sign`; reject directions outside the default front half-angle of 90 degrees.
- The device service never assumes centre is 0 rad. It receives `center_yaw` and safe yaw limits from the motion service status/configuration boundary.
- The motion service remains the sole motor owner; device code sends only validated local Unix RPC.
- LED logical frames are 8x8 `rgb8`, top-left row-major; physical mapping is configured by layout, origin, rotation, and color order.
- Unverified LED brightness defaults to 0.10, every request is clamped, and startup, shutdown, fault, and Jetson disconnect clear all pixels.
- No production path may instantiate the GPIO LED adapter unless an explicit `--enable-led-hardware` flag is present.
- Audio RTP, ALSA playback, ROS 2 bridges, and turn orchestration are separate later plans.

---

### Task 1: Circular DOA Math and Calibrated Target Conversion

**Files:**
- Create: `src/device/__init__.py`
- Create: `src/device/doa.py`
- Create: `tests/test_device_doa.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: raw DOA degrees, `DoaCalibration`, motion-owned `center_yaw` and safe yaw limits.
- Produces: `circular_distance_deg(a, b) -> float`, `circular_medoid_deg(samples) -> float`, `circular_mad_deg(samples, center) -> float`, and `calibrate_target(doa_deg, calibration, center_yaw, safe_limits) -> DoaTarget`.

- [x] **Step 1: Write failing circular-statistics tests**

```python
def test_circular_medoid_does_not_average_across_180():
    samples = [358.0, 359.0, 0.0, 1.0, 2.0]
    center = circular_medoid_deg(samples)
    assert circular_distance_deg(center, 0.0) <= 1.0
    assert circular_mad_deg(samples, center) == 1.0
```

- [x] **Step 2: Run the new test and verify it fails**

Run: `PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_device_doa.py -q`
Expected: FAIL during collection because `device.doa` does not exist.

- [x] **Step 3: Implement finite input validation and circular statistics**

```python
def circular_distance_deg(a: float, b: float) -> float:
    return abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)

def circular_medoid_deg(samples: Sequence[float]) -> float:
    values = _validated_degrees(samples)
    return min(enumerate(values), key=lambda item: (
        sum(circular_distance_deg(item[1], other) for other in values), item[0]))[1]

def circular_mad_deg(samples: Sequence[float], center: float) -> float:
    distances = sorted(circular_distance_deg(value, center) for value in _validated_degrees(samples))
    return statistics.median(distances)
```

- [x] **Step 4: Add calibrated-transform tests**

Cover zero offset, both direction signs, wrap at 359/1 degrees, front-half rejection, exact safe-limit clamping, non-finite values, invalid signs, inverted limits, and the commissioned nonzero `center_yaw=math.radians(15)`.

- [x] **Step 5: Implement immutable calibration and target records**

```python
@dataclass(frozen=True)
class DoaCalibration:
    zero_deg: float
    direction_sign: int
    front_half_angle_deg: float = 90.0

@dataclass(frozen=True)
class DoaTarget:
    raw_doa_deg: float
    relative_rad: float
    target_yaw: float
    clamped: bool

def calibrate_target(doa_deg: float, calibration: DoaCalibration, *,
                     center_yaw: float, safe_limits: tuple[float, float]) -> DoaTarget:
    relative_deg = _wrap_signed_deg(calibration.direction_sign * (doa_deg - calibration.zero_deg))
    if abs(relative_deg) > calibration.front_half_angle_deg:
        raise DoaError("rear_direction", "direction is outside the configured front half-plane")
    unclamped = center_yaw + math.radians(relative_deg)
    target = min(max(unclamped, safe_limits[0]), safe_limits[1])
    return DoaTarget(doa_deg % 360.0, math.radians(relative_deg), target,
                     not math.isclose(target, unclamped, abs_tol=1e-12))
```

- [x] **Step 6: Export the package and run focused tests**

Add `src/device` to the Hatch wheel package list, then run:
`PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_device_doa.py -q`
Expected: all Task 1 tests pass.

- [x] **Step 7: Commit**

```bash
git add pyproject.toml src/device tests/test_device_doa.py
git commit -m "feat(device): add calibrated circular DOA math"
```

### Task 2: Utterance DOA Stabilizer

**Files:**
- Modify: `src/device/doa.py`
- Modify: `tests/test_device_doa.py`

**Interfaces:**
- Consumes: `DoaSample(timestamp: float, doa_deg: float, speech_detected: bool)` at nominal 20 Hz.
- Produces: `DoaStabilizer.observe(sample) -> DoaDecision | None`, with terminal `ready` or `rejected` emitted once per rising-edge speech UUID.

- [x] **Step 1: Write failing lifecycle tests**

```python
def test_stabilizer_waits_400_ms_then_emits_one_stable_direction():
    ids = iter([UUID("00000000-0000-0000-0000-000000000001")])
    stabilizer = DoaStabilizer(id_factory=lambda: next(ids))
    assert stabilizer.observe(DoaSample(0.00, 359.0, True)) is None
    for index, angle in enumerate([0, 1, 359, 2, 0, 1, 359], start=1):
        result = stabilizer.observe(DoaSample(index * 0.05, angle, True))
    assert result is None
    result = stabilizer.observe(DoaSample(0.40, 0.0, True))
    assert result.state == "ready"
    assert result.speech_id == "00000000-0000-0000-0000-000000000001"
    assert result.sample_count == 9
```

- [x] **Step 2: Run the focused lifecycle tests and verify red**

Expected: FAIL because `DoaStabilizer` is undefined.

- [x] **Step 3: Implement rising-edge ownership and collection timing**

`DoaStabilizer` validates monotonic finite timestamps, creates a UUID only on `False -> True`, keeps only samples observed while speech is true, waits until `min_window_sec`, and closes the utterance after one terminal decision. It does not retarget after `ready`.

- [x] **Step 4: Add rejection and reset tests**

Cover fewer than 6 valid samples at 1.0 s, MAD above 12 degrees through 1.0 s, early speech falling edge, repeated true samples without a new UUID, falling then rising creating a new UUID, out-of-order time, and invalid DOA outside 0 through 359.

- [x] **Step 5: Implement deterministic decisions**

```python
@dataclass(frozen=True)
class DoaDecision:
    state: Literal["ready", "rejected"]
    speech_id: str
    doa_deg: float | None
    dispersion_deg: float | None
    sample_count: int
    code: str
    timestamp: float
```

At or after 400 ms, emit `ready/stable` when count and MAD pass. At or after 1.0 s, emit `rejected/insufficient_samples` or `rejected/unstable`. Never emit two terminal decisions for the same speech ID.

- [x] **Step 6: Run Task 1-2 tests and commit**

```bash
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_device_doa.py -q
git add src/device/doa.py tests/test_device_doa.py
git commit -m "feat(device): stabilize one DOA per utterance"
```

### Task 3: XVF3800 USB Control Adapter

**Files:**
- Create: `src/device/xvf3800.py`
- Create: `tests/test_xvf3800.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: an injected object exposing `ctrl_transfer`, or PyUSB discovery for VID `0x2886`, PID `0x0022`.
- Produces: `Xvf3800.read_version() -> tuple[int, int, int]`, `Xvf3800.read_doa() -> XvfDoa(doa_deg, speech_detected)`, and idempotent `close()`.

- [x] **Step 1: Write failing USB framing tests**

Assert VERSION uses IN vendor request, command `0x80`, resource 48, length 4; DOA uses command `0x92`, resource 20, length 5; `[0,126,0,1,0]` decodes to `(126, True)`; status 64 retries at most 100 times; any other status raises `XvfError("device_status", ...)`.

- [x] **Step 2: Verify tests fail, then implement the injected adapter**

Use only stdlib in module import scope. Import `usb.core` and `usb.util` inside `discover()`/`close()` so unit tests and non-Pi installs do not require PyUSB.

- [x] **Step 3: Add discovery and firmware-validation tests**

Require the exact commissioned USB identity `2886:0022`, reject missing/multiple matches, and expose version without writing firmware. Confirm no write control transfer is available from this adapter.

- [x] **Step 4: Implement discovery and validate on fake USB**

The production factory accepts optional bus/address selectors, claims no audio interface, and disposes only resources it opened. It never resets or flashes the device.

- [x] **Step 5: Run tests and commit**

```bash
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_xvf3800.py -q
git add pyproject.toml src/device/xvf3800.py tests/test_xvf3800.py
git commit -m "feat(device): read XVF3800 version and DOA"
```

### Task 4: Protected Motion Unix Client and Direction Coordinator

**Files:**
- Create: `src/device/motion_client.py`
- Create: `src/device/coordinator.py`
- Create: `tests/test_device_motion_client.py`
- Create: `tests/test_device_coordinator.py`
- Modify: `src/motion/orientation.py`
- Modify: `src/motion/controller.py`
- Modify: `tests/test_orientation.py`
- Modify: `tests/test_motion_controller.py`

**Interfaces:**
- Consumes: terminal `DoaDecision`, `DoaCalibration`, motion `orientation.status`, and the local socket `/run/talking-lamp/motion-control.sock`.
- Produces: correlated `orientation.acquire`, `orientation.return_center`, and `orientation.status` calls plus immutable device-level orientation events.

- [x] **Step 1: Write failing motion-boundary tests**

Extend the orientation snapshot contract with `center_yaw`, `safe_yaw_min`, and
`safe_yaw_max`. Assert that `orientation.status` and normal controller status
publish the same finite values, that `center_yaw` is inside the safe interval,
and that `orientation.return_center` targets the published `center_yaw`.

- [x] **Step 2: Expose motion-owned centre and safe limits**

Add the three immutable fields to `OrientationSnapshot`, sourcing them from
`OrientationConfig.center_yaw` and `BaseYawOrientationLayer.safe_yaw_limits`.
Continue using `asdict()` in controller responses so the local status boundary
gains the fields without a second configuration source.

- [x] **Step 3: Write failing Unix client tests**

Use a temporary asyncio Unix server. Assert exact five-field token-free envelopes, canonical request UUIDs, fresh request IDs, receive-time TTL, accepted then terminal correlation, response size limit, timeout, EOF, and no automatic replay.

- [x] **Step 4: Implement `MotionUnixClient`**

Expose `request(kind, payload, ttl_ms=1000) -> list[dict]`; open one local connection per logical call, cap lines at 16 KiB, and close on terminal response or error.

- [x] **Step 5: Write coordinator tests**

Assert stable DOA reads the motion status boundary, converts using returned `center_yaw`/safe limits, sends one acquire with the same speech UUID, preserves motion's `clamped` result, rejects rear/unstable input without motion calls, and forwards explicit return-center only after requested by Jetson policy.

- [x] **Step 6: Implement `DirectionCoordinator`**

The coordinator owns no motor state. It maps collector states to `collecting`, `rejected`, `orienting`, `aligned`, `timeout`, `returning`, and `centered`, retaining raw DOA, relative direction, target/current yaw, clamp flag, code, message, and timestamp.

- [x] **Step 7: Run tests and commit**

```bash
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_orientation.py tests/test_motion_controller.py tests/test_device_motion_client.py tests/test_device_coordinator.py -q
git add src/motion/orientation.py src/motion/controller.py src/device/motion_client.py src/device/coordinator.py tests/test_orientation.py tests/test_motion_controller.py tests/test_device_motion_client.py tests/test_device_coordinator.py
git commit -m "feat(device): connect stable DOA to local orientation"
```

### Task 5: WS2812B-64 Logical Mapping and Power-Limited Controller

**Files:**
- Create: `src/device/led.py`
- Create: `tests/test_device_led.py`

**Interfaces:**
- Consumes: exactly 192 RGB bytes or 64 RGB tuples, `LedMapping`, and an injected `PixelSink`.
- Produces: mapped physical pixels, clamped brightness status, `solid`, `frame`, `clear`, and idempotent `close`.

- [x] **Step 1: Write failing coordinate-mapping tests**

Cover row-major and serpentine layout, four origins, four rotations, and RGB/GRB color order using uniquely numbered 8x8 fixtures.

- [x] **Step 2: Implement pure `map_frame`**

Reject any size other than 8x8x3, booleans, non-integer channels, and channels outside 0..255 before touching the sink. Mapping returns a new 64-element tuple and never mutates the caller's frame.

- [x] **Step 3: Write power-policy and lifecycle tests**

Assert default `max_brightness=0.10`, request clamping, finite range validation, clear before first frame, clear on normal close, clear after sink exception, and no hardware adapter construction without explicit enable.

- [x] **Step 4: Implement `LedController` and lazy `Ws281xSink`**

The concrete adapter imports `rpi_ws281x` only inside its constructor, defaults to GPIO 12 and 64 pixels, and requires `enable_hardware=True`. The controller reports requested and applied brightness separately.

- [x] **Step 5: Run tests and commit**

```bash
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_device_led.py -q
git add src/device/led.py tests/test_device_led.py
git commit -m "feat(device): add safe WS2812B matrix control"
```

### Task 6: Authenticated Device TCP Protocol and Server

**Files:**
- Create: `src/device/protocol.py`
- Create: `src/device/server.py`
- Create: `tests/test_device_protocol.py`
- Create: `tests/test_device_server.py`

**Interfaces:**
- Consumes: authenticated NDJSON on port 8766 and coordinator/LED methods from Tasks 4-5.
- Produces: strict request results for `orientation.return_center`, `orientation.status`, `led.frame`, `led.solid`, `led.clear`, `led.status`, `device.status`, and `system.heartbeat`, plus ordered server-pushed orientation/LED status events.

- [x] **Step 1: Write strict-schema tests**

Reuse motion protocol envelope rules without importing private validators. Test exact fields, canonical UUID, finite JSON, 1..10000 ms TTL, constant-time token comparison, 16 KiB line cap, exact payload schemas, 192-byte RGB limit, and rejection before coordinator/sink calls.

- [x] **Step 2: Implement decoder/encoder and verify red-to-green**

Use distinct `DeviceRequest` and `DeviceProtocolError` types. Do not accept audio commands in this milestone; return `unknown_type` until the RTP/audio plan adds them.

- [x] **Step 3: Write async server tests**

Cover Jetson allowlist, one authenticated owner, heartbeat expiry, accepted/terminal correlation, 64 pending-request cap, session UUID, monotonic event sequence, new session reset, disconnect LED clear, and no request replay after reconnect.

- [x] **Step 4: Implement server lifecycle**

Bind before starting hardware polling, serialize writes with a lock, cancel forwarders before closing adapters, clear LEDs on every exit path, and leave the independent motion TCP connection untouched.

- [x] **Step 5: Run tests and commit**

```bash
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_device_protocol.py tests/test_device_server.py -q
git add src/device/protocol.py src/device/server.py tests/test_device_protocol.py tests/test_device_server.py
git commit -m "feat(device): serve authenticated device control"
```

### Task 7: Device Daemon, Disabled-by-Default Deployment, and Pi Commissioning

**Files:**
- Create: `src/device/daemon.py`
- Create: `deploy/pi/talking-lamp-device.service`
- Create: `deploy/pi/install-device-service.sh`
- Create: `tests/test_device_daemon.py`
- Create: `tests/test_device_deploy.py`
- Create: `docs/pi-device-commissioning.md`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: environment token, calibration JSON, motion socket, XVF adapter, optional LED adapter, bind/allowlist arguments.
- Produces: one supervised device process, disabled by default, with explicit `--enable-led-hardware` commissioning gate.

- [ ] **Step 1: Write daemon startup-order and shutdown tests**

Assert config and TCP bind validation occur before USB/GPIO open; polling begins only after the server binds; SIGINT/SIGTERM stop polling, close server, clear/close LED, and dispose XVF in that order; adapter failure returns nonzero without affecting motion service.

- [ ] **Step 2: Implement CLI and 20 Hz poll loop**

Required physical arguments are XVF VID/PID and calibration path. Defaults are bind `192.168.100.2`, port `8766`, allow host `192.168.100.1`, motion socket `/run/talking-lamp/motion-control.sock`, sample rate 20 Hz, GPIO 12, and LED disabled.

- [ ] **Step 3: Write staged installer tests**

Mirror the motion installer contract: preserve existing mode-0600 token/config files, create the `talking-lamp` group, default disabled, never start during install, support `--destdir` and `--dry-run`, and require explicit `--enable` only for boot enablement—not LED hardware enablement.

- [ ] **Step 4: Implement unit and installer**

Use `User=pixs`, `Group=talking-lamp`, `UMask=0007`, `Restart=on-failure`, bounded restart rate, `After=talking-lamp-motion.service`, and no `Requires=` so XVF failure cannot stop motion. Keep the LED hardware flag absent in the installed unit until commissioning approval.

- [ ] **Step 5: Document exact commissioning gates**

Record the commissioned XVF identity/version, udev rule, calibration SHA256, dry-run/null LED procedure, single-pixel/row/column/checkerboard order, and the separate 10/25/50/75/100% full-white power test. State that the LED must remain disconnected for software-only verification.

- [ ] **Step 6: Run focused and full regression tests**

```bash
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_device_*.py tests/test_xvf3800.py -q
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest -q
```

Expected: all device tests and every existing motion regression pass.

- [ ] **Step 7: Pi software-only commissioning**

Deploy with LED hardware disabled. Verify XVF version `(1, 0, 3)`, observe live DOA without moving motors, run the device server against a fake LED sink, and confirm SIGTERM clears the fake sink. Do not connect or power the WS2812B in this task.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/device/daemon.py deploy/pi tests/test_device_daemon.py tests/test_device_deploy.py docs/pi-device-commissioning.md
git commit -m "feat(device): deploy Pi DOA and LED service safely"
```

## Completion Check

- Run the device-focused tests and the complete existing suite from the clean implementation worktree.
- Confirm imports of `device.doa`, `device.protocol`, and fake LED tests do not import PyUSB or rpi-ws281x.
- Confirm the installed device service is disabled/inactive and its unit has no LED hardware-enable flag.
- Confirm XVF runtime never exposes firmware write/reset operations.
- Confirm orientation requests cross only the Unix socket and the device server never opens `/dev/ttyACM0`.
- Confirm LED remains physically disconnected until the separate mapping and staged power commissioning session.
