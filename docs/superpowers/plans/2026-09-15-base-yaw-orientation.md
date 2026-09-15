# Base Yaw Speaker Orientation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a locally commanded, settle-aware `base_yaw` orientation anchor so Raspberry Pi can face a stabilized speaker direction before Jetson starts response audio and expressive motion.

**Architecture:** A focused orientation module owns the yaw-only blender layer and its state machine. `MotionController` remains the sole runtime owner, accepts validated local orientation requests through a Unix socket, and reports completion only after position and velocity settle. Existing relative primitives run on top of the anchor, with only their yaw component scaled when the anchored clip would exceed calibrated limits.

**Tech Stack:** Python 3.12+, NumPy, asyncio Unix sockets, existing MotionRuntime/MotionController, pytest

**Spec:** `docs/superpowers/specs/2026-09-15-pi-audio-doa-led-jetson-ros-design.md`

## Global Constraints

- Raspberry Pi owns Feetech motion and keeps the 100 Hz runtime free of ROS, ALSA, libusb and GStreamer callbacks.
- Only `MotionController` may mutate `MotionRuntime`; socket tasks submit mailbox requests and never step the runtime.
- Orientation owns only `base_yaw` at blender priority 15; primitive priority 20 remains additive and TaskLight priority 30 remains authoritative.
- Defaults are: 5° mechanical margin, 5° no-move deadband, 3° settle error, 0.08 rad/s settle velocity, 150 ms settle duration, 2 s acquisition timeout and 10 s disconnected hold.
- All absolute angles are finite radians in the calibrated simulation/base frame.
- Remote TCP keeps one authenticated Jetson owner. `orientation.acquire`, `orientation.return_center` and `orientation.status` are accepted only on the protected local Unix socket.
- The Unix socket path is `/run/talking-lamp/motion-control.sock`, owned by the service user/group with mode `0660`.
- Expired, malformed, duplicate and out-of-range requests never reach the hardware runtime.
- Lost connections never replay an orientation or motion request automatically.
- Existing motion behavior and automated tests must remain green.

## Roadmap boundary

The approved design contains four independently reviewable subsystems. This is plan 1 of 4 and ends with a simulated, locally callable orientation milestone. It does not read XVF3800 DOA, drive LEDs, transport audio or expose new Jetson ROS nodes. Those become separate plans after this milestone: (2) Pi device/DOA/LED service, (3) RTP audio, and (4) Jetson ROS bridges, orchestration and deployment. This boundary lets every plan finish with working tests instead of leaving four partially connected stacks.

---

## File map

- Create `src/motion/orientation.py`: configuration, yaw-only layer, immutable status, state transitions and settle/timeout rules.
- Modify `src/motion/runtime.py`: register orientation at priority 15 and expose narrow orientation methods/properties.
- Modify `src/motion/primitives.py`: clone a primitive with one joint's offsets scaled.
- Modify `src/motion/layers.py`: return primitive playback metadata and accept per-joint clip scaling.
- Modify `src/motion/controller.py`: local orientation request lifecycle, tickets, status and disconnect fallback.
- Modify `src/motion/protocol.py`: channel-specific command allowlists and strict local request decoding.
- Create `src/motion/local_control.py`: Unix NDJSON request server using the existing controller ticket contract.
- Modify `src/motion/middleware_server.py`: start/stop TCP, Unix server and the one motion owner together.
- Modify `deploy/pi/talking-lamp-motion.service`: runtime directory, local socket argument and permissions.
- Modify `deploy/pi/install-motion-service.sh`: create/validate the service group and local socket configuration.
- Create `tests/test_orientation.py`: layer, state machine, settling, timeout and disconnect behavior.
- Modify `tests/test_primitives.py`: yaw-only clip scaling.
- Modify `tests/test_runtime.py`: blender priority and anchored primitive behavior.
- Modify `tests/test_motion_controller.py`: orientation tickets, busy rules, result data and fallback.
- Modify `tests/test_motion_protocol.py`: local/remote command boundary.
- Modify `tests/test_motion_middleware.py`: Unix server framing, lifecycle and sole runtime owner.
- Modify `tests/test_deploy_scripts.py`: systemd runtime directory and group checks.
- Modify `src/motion/README.md`: local diagnostic examples and state semantics.

---

### Task 1: Orientation layer and deterministic state machine

**Files:**
- Create: `src/motion/orientation.py`
- Create: `tests/test_orientation.py`

**Interfaces:**
- Consumes: `BlendContext`, `LayerOutput`, `NJ`, `REST_POSE` and a `(5, 2)` calibrated joint-limit array.
- Produces: `OrientationConfig`, `OrientationSnapshot`, `OrientationError`, `BaseYawOrientationLayer` and `OrientationCoordinator`.

- [ ] **Step 1: Write failing tests for yaw-only ownership and clamping**

```python
def test_orientation_layer_claims_only_base_yaw_and_clamps_with_margin():
    limits = np.array([[-1.0, 1.0], [-2, 2], [-2, 2], [-2, 2], [-2, 2]], float)
    cfg = OrientationConfig(center_yaw=0.1, yaw_margin=np.deg2rad(5))
    layer = BaseYawOrientationLayer(cfg, limits)
    clamped = layer.acquire(2.0)
    out = layer.update(BlendContext(np.zeros(5), 0.01, 0.01))
    assert clamped
    assert out.value[0] == pytest.approx(1.0 - np.deg2rad(5))
    np.testing.assert_array_equal(out.weight, [1, 0, 0, 0, 0])
    assert out.additive is False
```

Also cover `release()` returning an inactive output and constructor rejection of non-finite center, malformed limits, non-positive thresholds and a margin that eliminates the yaw range.

- [ ] **Step 2: Run the focused test and verify red**

Run: `PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_orientation.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'motion.orientation'`.

- [ ] **Step 3: Implement immutable config/status and the yaw-only layer**

Use these public shapes:

```python
@dataclass(frozen=True)
class OrientationConfig:
    center_yaw: float = float(REST_POSE[0])
    yaw_margin: float = np.deg2rad(5.0)
    deadband: float = np.deg2rad(5.0)
    settle_error: float = np.deg2rad(3.0)
    settle_velocity: float = 0.08
    settle_duration: float = 0.15
    acquire_timeout: float = 2.0
    disconnect_hold: float = 10.0

@dataclass(frozen=True)
class OrientationSnapshot:
    state: str
    speech_id: str | None
    target_yaw: float | None
    current_yaw: float
    clamped: bool
    code: str

class OrientationError(ValueError):
    def __init__(self, code: str, message: str): ...

class BaseYawOrientationLayer:
    name = "orientation"
    priority = 15
    def acquire(self, target_yaw: float) -> bool: ...
    def return_center(self) -> None: ...
    def release(self) -> None: ...
    @property
    def safe_yaw_limits(self) -> tuple[float, float]: ...
    def update(self, ctx: BlendContext) -> LayerOutput: ...
```

The layer clamps to `joint_limits[0] ± yaw_margin`, stores whether clamping occurred and emits an absolute output with weight `[1, 0, 0, 0, 0]` while active. It does not interpolate; the shared trajectory generator supplies velocity/acceleration/jerk limits.

- [ ] **Step 4: Add failing state-machine tests**

```python
def test_coordinator_settles_only_after_continuous_position_and_velocity_window():
    coordinator = make_coordinator(settle_duration=.15)
    first = coordinator.acquire("speech-1", target_yaw=.4, now=1.0,
                                current_yaw=0.0, task_light_busy=False)
    assert first.state == "orienting"
    for index in range(14):
        state = coordinator.observe(now=1.01 + index * .01,
                                    current_yaw=.39, velocity=.01)
        assert state.state == "orienting"
    state = coordinator.observe(now=1.15, current_yaw=.39, velocity=.01)
    assert state.state == "aligned"
```

Add tests for: immediate `aligned` inside deadband; settle timer reset after one fast tick; `timeout` after 2 s; `blocked_by_task_light`; duplicate `speech_id` returning the existing snapshot without retargeting; return while motion busy raising `OrientationError("busy", ...)`; `returning -> centered` using the same settle window; and disconnected hold changing `aligned -> returning` at exactly 10 s.

- [ ] **Step 5: Implement `OrientationCoordinator` minimally**

```python
class OrientationCoordinator:
    def __init__(self, layer: BaseYawOrientationLayer, cfg: OrientationConfig): ...
    def acquire(self, speech_id: str, target_yaw: float, *, now: float,
                current_yaw: float, task_light_busy: bool) -> OrientationSnapshot: ...
    def return_center(self, *, now: float, current_yaw: float,
                      motion_busy: bool) -> OrientationSnapshot: ...
    def observe(self, *, now: float, current_yaw: float,
                velocity: float) -> OrientationSnapshot: ...
    def disconnected(self, *, now: float) -> None: ...
```

Use explicit states `idle`, `orienting`, `aligned`, `timeout`, `returning`, `centered`. Keep `speech_id` latched through centered. Validate speech IDs as non-empty strings here; UUID syntax belongs at the protocol boundary. Track the start of an uninterrupted settle window rather than counting ticks so tests remain correct if a supported `dt` changes.

- [ ] **Step 6: Run orientation tests**

Run: `PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_orientation.py -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit the orientation unit**

```bash
git add src/motion/orientation.py tests/test_orientation.py
git commit -m "feat(motion): add base yaw orientation state machine"
```

---

### Task 2: MotionRuntime integration and blender authority

**Files:**
- Modify: `src/motion/runtime.py:66-133`
- Modify: `tests/test_runtime.py`

**Interfaces:**
- Consumes: Task 1 `BaseYawOrientationLayer`, `OrientationConfig`, `OrientationCoordinator`.
- Produces: `MotionRuntime.orientation`, `MotionRuntime.orientation_control`, `acquire_orientation()`, `return_center()`, `release_orientation()` and `orientation_snapshot()`.

- [ ] **Step 1: Write failing runtime integration tests**

```python
def test_orientation_anchor_overrides_tracking_yaw_but_not_other_tracking_joints(rt):
    rt.observe_point([.4, .3, .3])
    rt.acquire_orientation("speech-1", .5, now=0.0)
    states = rt.run(1.5)
    assert states[-1].trace.per_layer["orientation"][0] == 1.0
    assert states[-1].trace.per_layer["orientation"][1:].sum() == 0.0
    assert states[-1].q_cmd[0] == pytest.approx(.5, abs=.06)
```

Also assert blender layer order is `idle`, `track`, `orientation`, `primitive`, `task_light`; task light can still override yaw; and releasing orientation restores existing behavior.

- [ ] **Step 2: Run the tests and verify red**

Run: `PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_runtime.py -k orientation -v`

Expected: FAIL because `MotionRuntime.acquire_orientation` does not exist.

- [ ] **Step 3: Wire orientation into `MotionRuntime`**

Extend the constructor with optional `orientation_cfg: OrientationConfig | None = None`. Reuse the already loaded `HardwareAlignment.load().joint_limits`, construct the layer and coordinator once, and insert the layer into `MotionBlender` between track and primitive.

Expose:

```python
def acquire_orientation(self, speech_id: str, target_yaw: float, *, now: float):
    return self.orientation_control.acquire(
        speech_id, target_yaw, now=now, current_yaw=float(self.traj.pos[0]),
        task_light_busy=self.task_light.busy,
    )

def return_center(self, *, now: float, motion_busy: bool): ...
def release_orientation(self) -> None: ...
def orientation_snapshot(self) -> OrientationSnapshot: ...
def orientation_safe_yaw_limits(self) -> tuple[float, float]: ...
```

Do not call coordinator `observe` here; the controller must do it after each hardware-backed step using the actual commanded yaw and velocity.

- [ ] **Step 4: Run runtime and existing blender tests**

Run: `PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_runtime.py tests/test_blender_layers.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit runtime integration**

```bash
git add src/motion/runtime.py tests/test_runtime.py
git commit -m "feat(motion): anchor runtime to speaker yaw"
```

---

### Task 3: Orientation-aware primitive yaw scaling

**Files:**
- Modify: `src/motion/primitives.py:20-128`
- Modify: `src/motion/layers.py:150-212`
- Modify: `src/motion/runtime.py:100-103`
- Modify: `tests/test_primitives.py`
- Modify: `tests/test_runtime.py`

**Interfaces:**
- Consumes: active orientation target and calibrated yaw safe limits from Tasks 1–2.
- Produces: `Primitive.scaled_joint(index: int, factor: float) -> Primitive`, `PrimitivePlayInfo(name: str, yaw_scale: float)` and `PrimitiveLayer.play(...) -> PrimitivePlayInfo`.

- [ ] **Step 1: Write failing primitive scaling tests**

```python
def test_scaled_joint_returns_copy_and_changes_only_requested_offsets():
    original = Primitive("p", np.array([0., 1.]),
                         np.array([[0, 0, 0, 0, 0], [.4, .2, .3, .4, .5]]))
    scaled = original.scaled_joint(0, .25)
    np.testing.assert_allclose(scaled.offsets[:, 0], original.offsets[:, 0] * .25)
    np.testing.assert_array_equal(scaled.offsets[:, 1:], original.offsets[:, 1:])
    np.testing.assert_array_equal(original.offsets[-1], [.4, .2, .3, .4, .5])
```

Validate index `0 <= index < NJ` and finite factor in `[0, 1]`.

- [ ] **Step 2: Run focused tests and verify red**

Run: `PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_primitives.py -k scaled_joint -v`

Expected: FAIL with missing `scaled_joint`.

- [ ] **Step 3: Implement copy-based joint scaling**

Return a new `Primitive` whose `times` and `offsets` are copies. Never mutate the library cache.

- [ ] **Step 4: Write failing anchored playback tests**

```python
def test_anchored_motion_scales_only_yaw_to_fit_safe_range(rt):
    rt.acquire_orientation("speech-1", rt.traj.position_limits[0, 1] - np.deg2rad(6), now=0.)
    info = rt.play_primitive("headshake")
    assert 0.0 <= info.yaw_scale < 1.0
    for state in rt.run(4.0):
        assert state.q_blend[0] <= rt.traj.position_limits[0, 1] - np.deg2rad(5) + 1e-9
```

Add a regression test proving playback with no orientation produces byte-for-byte equal commands to the current behavior and reports `yaw_scale=1.0`.

- [ ] **Step 5: Implement the yaw scale calculation**

For clip yaw minimum `negative <= 0`, maximum `positive >= 0`, anchor `a` and safe bounds `[lo, hi]`, calculate:

```python
factor = 1.0
if positive > 0:
    factor = min(factor, (hi - a) / positive)
if negative < 0:
    factor = min(factor, (lo - a) / negative)
factor = float(np.clip(factor, 0.0, 1.0))
```

`PrimitiveLayer.play` accepts keyword-only `yaw_anchor: float | None` and `yaw_limits: tuple[float, float] | None`. Both must be provided together. Load and resample first, scale joint 0 only, then compute claimed joints. Return `PrimitivePlayInfo`.

`MotionRuntime.play_primitive` passes the active absolute orientation target and `orientation.safe_yaw_limits` whenever the layer is active, including `timeout`, `returning`, and `centered`. Anchor changes must refit the current primitive (including release tails) from an unscaled source before blending; repeats use the current anchor. Preserve every existing load keyword and return the playback info.

- [ ] **Step 6: Run primitive/runtime tests**

Run: `PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_primitives.py tests/test_runtime.py -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit safe anchored playback**

```bash
git add src/motion/primitives.py src/motion/layers.py src/motion/runtime.py tests/test_primitives.py tests/test_runtime.py
git commit -m "feat(motion): fit primitive yaw around speaker anchor"
```

---

### Task 4: Controller tickets, busy rules and disconnect fallback

**Files:**
- Modify: `src/motion/controller.py:28-349`
- Modify: `tests/test_motion_controller.py`

**Interfaces:**
- Consumes: runtime orientation methods and `PrimitivePlayInfo`.
- Produces: terminal tickets for `orientation.acquire`, `orientation.return_center`, immediate `orientation.status`, and extended `ControllerStatus`.

- [ ] **Step 1: Write failing controller lifecycle tests**

```python
def test_orientation_ticket_completes_only_after_settle(controller):
    ticket = controller.submit(request(
        "orientation.acquire", speech_id="00000000-0000-0000-0000-000000000001",
        target_yaw=.4,
    ))
    controller.tick_once(now=1.)
    assert ticket.accepted.result().state == "accepted"
    assert not ticket.completed.done()
    for index in range(300):
        controller.tick_once(now=1.01 + index / 100)
        if ticket.completed.done():
            break
    assert ticket.completed.result().code == "aligned"
```

Add tests for deadband completion, acquisition timeout, duplicate speech ID, task-light rejection, return-center busy rejection, centered completion, yaw-scale result data, active-orientation status fields, and disconnect triggering return only after 10 s.

- [ ] **Step 2: Run controller orientation tests and verify red**

Run: `PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_motion_controller.py -k orientation -v`

Expected: FAIL because orientation commands are unknown.

- [ ] **Step 3: Extend immutable controller status**

Append fields with stable defaults so existing construction sites remain readable:

```python
orientation_state: str = "idle"
orientation_speech_id: str | None = None
orientation_target_yaw: float | None = None
orientation_current_yaw: float = 0.0
orientation_clamped: bool = False
primitive_yaw_scale: float = 1.0
```

Store at most one active orientation ticket and one return ticket. `orientation.status` returns `asdict(runtime.orientation_snapshot())` without starting a layer.

- [ ] **Step 4: Implement command acceptance and completion**

- `orientation.acquire`: reject while task light is active, call runtime acquire, accept the ticket, and complete immediately only for deadband-aligned state.
- `orientation.return_center`: reject if a primitive or task light is busy; start return and complete at `centered`.
- After every `runtime.step()`, call coordinator `observe` with `step.q_cmd[0]`, `step.vel[0]` and controller monotonic `now`, then resolve aligned/timeout/centered tickets.
- When motion starts, retain `PrimitivePlayInfo.yaw_scale` for status and terminal result data.
- Change `_safe_disconnect` to accept the current tick's `now`. Remote disconnect still interrupts expressive/task motion immediately, but `_safe_disconnect(now)` calls `orientation_control.disconnected(now=now)` so an anchor returns only after the 10 s hold.
- Controller shutdown releases orientation after resolving tickets.

- [ ] **Step 5: Run controller and runtime suites**

Run: `PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_motion_controller.py tests/test_runtime.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit controller behavior**

```bash
git add src/motion/controller.py tests/test_motion_controller.py
git commit -m "feat(motion): coordinate speaker orientation requests"
```

---

### Task 5: Strict local protocol and Unix control server

**Files:**
- Modify: `src/motion/protocol.py:14-202`
- Create: `src/motion/local_control.py`
- Modify: `tests/test_motion_protocol.py`
- Modify: `tests/test_motion_middleware.py`

**Interfaces:**
- Consumes: `MotionController.submit(Request)` and `CommandTicket` futures.
- Produces: `REMOTE_COMMAND_TYPES`, `LOCAL_COMMAND_TYPES`, `decode_local_request(...)`, and `MotionUnixServer`.

- [ ] **Step 1: Write failing protocol-boundary tests**

```python
def test_remote_decoder_rejects_local_orientation_acquire(valid_message):
    valid_message.update(type="orientation.acquire", payload={
        "speech_id": "00000000-0000-0000-0000-000000000001",
        "target_yaw": 0.4,
    })
    with pytest.raises(ProtocolError, match="local_only"):
        decode_request(wire(valid_message), token="secret", received_at=1.)

def test_local_decoder_accepts_exact_orientation_payload_without_token():
    request = decode_local_request(wire({
        "version": 1,
        "id": "00000000-0000-0000-0000-000000000002",
        "type": "orientation.acquire",
        "ttl_ms": 1000,
        "payload": {"speech_id": "00000000-0000-0000-0000-000000000001",
                    "target_yaw": .4},
    }), received_at=1.)
    assert request.payload["target_yaw"] == .4
```

Also cover non-canonical speech UUID, non-finite yaw, yaw outside `[-pi, pi]`, unexpected token in local messages, remote commands on the local socket and exact empty payloads for return/status.

- [ ] **Step 2: Run protocol tests and verify red**

Run: `PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_motion_protocol.py -k orientation -v`

Expected: FAIL with missing `decode_local_request`.

- [ ] **Step 3: Refactor validation without weakening remote auth**

Keep `decode_request`'s current six-field authenticated envelope. Add a five-field local envelope `{version,id,type,ttl_ms,payload}` and shared `_decode(..., allowed_types, token_required)` helper. Define:

```python
LOCAL_COMMAND_TYPES = frozenset({
    "orientation.acquire", "orientation.return_center", "orientation.status",
    "system.heartbeat",
})
REMOTE_COMMAND_TYPES = frozenset({  # the current public motion commands only
    "motion.play", "motion.cancel", "motion.interrupt", "motion.status",
    "motion.list", "track.point", "track.bearing", "track.clear",
    "task_light.place", "task_light.cancel", "task_light.clear",
    "system.heartbeat",
})
COMMAND_TYPES = REMOTE_COMMAND_TYPES  # compatibility export
```

Do not add local-only types to `REMOTE_COMMAND_TYPES`. Validate acquire payload exactly as `{speech_id, target_yaw}` and return/status as empty objects.

- [ ] **Step 4: Write failing Unix server tests**

Use a pytest temporary directory, `NullBackend`, the real controller and a server path below `tmp_path`.

```python
async def test_unix_server_returns_accepted_then_aligned(tmp_path, controller_runner):
    server = MotionUnixServer(controller_runner.controller, tmp_path / "motion.sock")
    await server.start()
    reader, writer = await asyncio.open_unix_connection(server.path)
    writer.write(local_wire("orientation.acquire", speech_id=SPEECH_ID, target_yaw=.2))
    await writer.drain()
    assert json.loads(await reader.readline())["state"] == "accepted"
    assert json.loads(await reader.readline())["code"] == "aligned"
```

Also test socket mode `0660`, stale socket replacement only when it is actually a socket, refusal to unlink a regular file, maximum line size, three-schema-error close, expired request, duplicate ID and clean socket removal on close.

- [ ] **Step 5: Implement `MotionUnixServer`**

Mirror the TCP server's bounded NDJSON framing and future forwarding but omit token and remote-owner semantics. Multiple local diagnostic/device clients may connect; the controller still serializes actual commands. Create with mode `0660` after bind. Never unlink an arbitrary path: use `lstat`, require `stat.S_ISSOCK`, and reject symlinks/regular files.

- [ ] **Step 6: Run protocol and middleware tests**

Run: `PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_motion_protocol.py tests/test_motion_middleware.py -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit local control transport**

```bash
git add src/motion/protocol.py src/motion/local_control.py tests/test_motion_protocol.py tests/test_motion_middleware.py
git commit -m "feat(motion): expose protected local orientation control"
```

---

### Task 6: Daemon and systemd lifecycle integration

**Files:**
- Modify: `src/motion/middleware_server.py:158-276`
- Modify: `deploy/pi/talking-lamp-motion.service`
- Modify: `deploy/pi/install-motion-service.sh`
- Modify: `tests/test_motion_middleware.py`
- Modify: `tests/test_deploy_scripts.py`

**Interfaces:**
- Consumes: `MotionUnixServer` from Task 5.
- Produces: one process that starts TCP and Unix listeners before the motion owner, then closes both before parking hardware.

- [ ] **Step 1: Write failing daemon lifecycle tests**

Add assertions that `_serve` starts both servers, a failure to bind either server prevents hardware stepping, shutdown closes both listeners, joins the motion owner and removes the socket before backend close.

```python
def test_parser_defaults_to_runtime_unix_socket():
    args = build_parser().parse_args(["--null-backend"])
    assert args.local_socket == Path("/run/talking-lamp/motion-control.sock")
```

- [ ] **Step 2: Run focused middleware tests and verify red**

Run: `PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_motion_middleware.py -k 'unix or local_socket' -v`

Expected: FAIL because the parser and `_serve` do not know the Unix server.

- [ ] **Step 3: Integrate both servers with explicit startup ordering**

Add `--local-socket` as a `Path`. Construct both servers after catalog/backend/runtime/controller creation. Start both listeners before starting the owner thread. In `finally`, stop controller, join owner, cancel serving tasks, close TCP, close Unix, then let `main` close the backend. A partial bind failure closes the listener that did start.

- [ ] **Step 4: Write failing deployment assertions**

Require these unit properties:

```ini
[Service]
RuntimeDirectory=talking-lamp
RuntimeDirectoryMode=0750
UMask=0007
ExecStart=... --local-socket /run/talking-lamp/motion-control.sock
```

Require the installer to create or validate a system group named `talking-lamp`, add the service user to it without overwriting token files and keep enablement opt-in.

- [ ] **Step 5: Update service and installer**

Preserve existing restart limits, environment file, stop timeout, serial port arguments and non-overwrite guarantees. Do not use world-writable socket permissions.

- [ ] **Step 6: Run deployment and complete test suite**

Run:

```bash
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest tests/test_motion_middleware.py tests/test_deploy_scripts.py -v
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit daemon integration**

```bash
git add src/motion/middleware_server.py deploy/pi/talking-lamp-motion.service deploy/pi/install-motion-service.sh tests/test_motion_middleware.py tests/test_deploy_scripts.py
git commit -m "feat(motion): serve orientation over local socket"
```

---

### Task 7: Operator documentation and milestone verification

**Files:**
- Modify: `src/motion/README.md`
- Create: `docs/base-yaw-orientation.md`

**Interfaces:**
- Consumes: completed Tasks 1–6.
- Produces: exact local diagnostic commands, calibration handoff and evidence for the next Pi device-service plan.

- [ ] **Step 1: Document the local request examples**

Include a Python/socat example for each exact local message, expected accepted/terminal events, state fields, task-light `busy` behavior, 10-second disconnect return and the fact that DOA degrees are not accepted here—the next device service performs stabilization and conversion before submitting radians.

- [ ] **Step 2: Document calibration inputs for the next milestone**

Record that the device plan must supply canonical `speech_id`, finite target radians, `clamped` is calculated by motion, and `doa_zero_deg`/`doa_direction_sign` belong to device configuration rather than motion configuration.

- [ ] **Step 3: Run documentation and compile checks**

Run:

```bash
git diff --check
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/python -m compileall -q src
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest -q
```

Expected: no diff errors, compile errors or test failures.

- [ ] **Step 4: Review the milestone against the spec**

Verify with concrete test names that orientation claims only yaw, settles before success, survives primitive playback as an anchor, scales yaw safely, rejects TaskLight conflict, returns after explicit request, falls back after disconnect and never opens a second hardware owner.

- [ ] **Step 5: Commit documentation**

```bash
git add src/motion/README.md docs/base-yaw-orientation.md
git commit -m "docs: explain base yaw orientation control"
```

- [ ] **Step 6: Record clean milestone state**

Run: `git status --short --branch && git log --oneline -8`

Expected: clean worktree with the orientation milestone commits above the approved design and plan.
