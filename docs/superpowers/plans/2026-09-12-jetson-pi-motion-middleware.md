# Jetson–Pi Motion Middleware Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a ROS 2-facing Jetson bridge and a safe Raspberry Pi TCP daemon so Jetson Orin Nano 8GB applications can execute, cancel, discover, and monitor high-level lamp motions over the wired link.

**Architecture:** ROS 2 remains on the Jetson and exposes generic Actions, Topics, and Services. A persistent NDJSON/TCP bridge sends versioned high-level events to a Raspberry Pi daemon; only the daemon's 100 Hz control thread mutates `MotionRuntime` or accesses `FeetechBackend`. A validated TOML catalog is the executable motion whitelist, so adding a motion does not change either network interface.

**Tech Stack:** Python standard library (`asyncio`, `json`, `queue`, `threading`, `tomllib`), NumPy, existing `MotionRuntime`, pytest, ROS 2 Humble/Jazzy (`rclpy`, `rosidl`, `geometry_msgs`), systemd.

**Spec:** `docs/superpowers/specs/2026-09-12-jetson-pi-motion-middleware-design.md`

## Global Constraints

- Raspberry Pi remains the sole owner of the Feetech bus, safety limits, blending, trajectory generation, and 100 Hz deadlines.
- Network clients may send behavior names and spatial targets; they may not send raw joint angles, serial settings, calibration values, or PID registers.
- Raspberry Pi stays on Debian 13/Python 3.13 without ROS 2.
- JetPack 6.x/L4T 36.x/Ubuntu 22.04 uses ROS 2 Humble; JetPack 7.x/L4T 39.x/Ubuntu 24.04 uses ROS 2 Jazzy.
- The ROS bridge uses APIs common to Humble and Jazzy and remains compatible with Python 3.10.
- Jetson installs `ros-base`, `lamp_interfaces`, and `lamp_motion_bridge`; desktop ROS tools and continuous rosbag recording are excluded.
- TCP binds to the dedicated wired interface at `192.168.100.2`; messages are UTF-8 NDJSON with a 16 KiB maximum line length.
- Command TTL starts from the Pi's monotonic receive time, so correctness never depends on synchronized wall clocks.
- Tracking uses latest-value semantics and `keep_last=1`; expressive commands use bounded FIFO semantics.
- Every normal, signal, and exception shutdown path parks in sleep and disables torque through `FeetechBackend.close()`.
- No physical motion test runs until unit, TCP loopback, and simulation tests pass.

## File Structure

### Raspberry Pi runtime

- `src/motion/catalog.py`: parse `catalog.toml`, resolve enabled recordings, and run preflight validation.
- `src/motion/protocol.py`: decode and encode the versioned Pi wire contract.
- `src/motion/controller.py`: bounded mailbox, command arbitration, status, Action completion, and the single-owner 100 Hz loop.
- `src/motion/middleware_server.py`: asyncio TCP connections and controller lifecycle.
- `src/motion/remote_client.py`: dependency-free client used by diagnostics and contract tests.
- `lelamp_runtime/lelamp/recordings/catalog.toml`: executable motion whitelist and semantic metadata.
- `deploy/pi/talking-lamp-motion.service`: Pi systemd unit.
- `deploy/pi/install-motion-service.sh`: idempotent service installer.

### Jetson ROS workspace

- `jetson_ws/src/lamp_interfaces/action/PlayMotion.action`: generic named-motion Action.
- `jetson_ws/src/lamp_interfaces/action/PlaceTaskLight.action`: task-light placement Action.
- `jetson_ws/src/lamp_interfaces/msg/MotionStatus.msg`: bridge/Pi status.
- `jetson_ws/src/lamp_interfaces/srv/ListMotions.srv`: catalog query.
- `jetson_ws/src/lamp_interfaces/srv/InterruptMotion.srv`: immediate interruption.
- `jetson_ws/src/lamp_interfaces/CMakeLists.txt`, `package.xml`: interface build metadata.
- `jetson_ws/src/lamp_motion_bridge/lamp_motion_bridge/transport.py`: reconnecting TCP client with request/result correlation.
- `jetson_ws/src/lamp_motion_bridge/lamp_motion_bridge/node.py`: ROS Action, Topic, and Service adapters.
- `jetson_ws/src/lamp_motion_bridge/launch/motion_bridge.launch.py`: runtime parameters.
- `jetson_ws/src/lamp_motion_bridge/setup.py`, `setup.cfg`, `package.xml`, `resource/lamp_motion_bridge`: Python package metadata.
- `deploy/jetson/detect-platform.sh`: map JetPack/L4T/Ubuntu to the supported ROS distribution.
- `deploy/jetson/install-motion-bridge.sh`: install dependencies and build the two packages.

### Tests and documentation

- `tests/test_motion_catalog.py`: catalog validation and extension behavior.
- `tests/test_motion_protocol.py`: wire schema, authentication, TTL, and limits.
- `tests/test_motion_controller.py`: arbitration, latest tracking, completion, faults, and deadlines.
- `tests/test_motion_middleware.py`: TCP round trips, duplicate requests, reconnect, and disconnect safety.
- `tests/test_jetson_bridge_contract.py`: golden vectors shared by Pi and the Python 3.10-compatible Jetson transport.
- `tests/test_deploy_scripts.py`: platform detection and service/launch file invariants.
- `docs/motion-middleware.md`: operator setup, commands, new-motion workflow, and recovery.

---

### Task 1: Validated Motion Catalog

**Files:**
- Create: `src/motion/catalog.py`
- Create: `lelamp_runtime/lelamp/recordings/catalog.toml`
- Create: `tests/test_motion_catalog.py`
- Modify: `src/motion/primitives.py`

**Interfaces:**
- Consumes: `Primitive.load(name: str, recordings_dir: Path)`, `PrimitiveLibrary.get(name: str)`, `HardwareAlignment.load()`.
- Produces: `MotionCatalog.load(path: Path) -> MotionCatalog`, `MotionCatalog.names() -> list[str]`, `MotionCatalog.validate() -> CatalogReport`, `MotionCatalog.library() -> PrimitiveLibrary`.

- [ ] **Step 1: Write failing catalog tests**

```python
def test_catalog_is_an_explicit_sorted_whitelist(tmp_path):
    write_recording(tmp_path / "wave.csv", valid_rows())
    write_recording(tmp_path / "draft.csv", valid_rows())
    write_catalog(tmp_path / "catalog.toml", {"wave": {"file": "wave.csv", "enabled": True}})
    catalog = MotionCatalog.load(tmp_path / "catalog.toml")
    assert catalog.names() == ["wave"]

def test_catalog_preflight_rejects_unsafe_recording_before_hardware(tmp_path):
    write_recording(tmp_path / "unsafe.csv", rows_with_non_finite_value())
    write_catalog(tmp_path / "catalog.toml", {"unsafe": {"file": "unsafe.csv", "enabled": True}})
    report = MotionCatalog.load(tmp_path / "catalog.toml").validate()
    assert not report.ok
    assert report.errors[0].code == "non_finite"
```

- [ ] **Step 2: Run the focused tests and confirm the missing module failure**

Run: `.venv/bin/pytest tests/test_motion_catalog.py -v`

Expected: FAIL during import with `ModuleNotFoundError: No module named 'motion.catalog'`.

- [ ] **Step 3: Implement immutable catalog entries and validation**

```python
@dataclass(frozen=True)
class MotionEntry:
    name: str
    file: Path
    enabled: bool
    tags: tuple[str, ...]

@dataclass(frozen=True)
class CatalogReport:
    errors: tuple[CatalogError, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

class MotionCatalog:
    def __init__(self, path: Path, entries: tuple[MotionEntry, ...]):
        self.path = path
        self.recordings_dir = path.parent
        self.entries = entries

    @classmethod
    def load(cls, path: Path) -> "MotionCatalog":
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        entries = tuple(
            MotionEntry(
                name=name,
                file=(path.parent / values["file"]).resolve(),
                enabled=values["enabled"],
                tags=tuple(values.get("tags", ())),
            )
            for name, values in document["motions"].items()
        )
        return cls(path.resolve(), entries)

    def names(self) -> list[str]:
        return sorted(entry.name for entry in self.entries if entry.enabled)

    def validate(self) -> CatalogReport:
        errors = []
        for entry in self.entries:
            if not entry.enabled:
                continue
            try:
                Primitive.load(entry.name, recordings_dir=self.recordings_dir)
            except (OSError, ValueError, KeyError) as exc:
                errors.append(CatalogError(entry.name, "invalid_recording", str(exc)))
        return CatalogReport(tuple(errors))

    def library(self) -> PrimitiveLibrary:
        report = self.validate()
        if not report.ok:
            raise CatalogValidationError(report)
        return PrimitiveLibrary(
            recordings_dir=self.recordings_dir,
            allowed_names=frozenset(self.names()),
        )
```

Add `CatalogError(name: str, code: str, message: str)` and `CatalogValidationError`. Before constructing each `MotionEntry`, require names to match `^[a-z][a-z0-9_]{0,63}$`, reject unknown TOML keys, and verify `file.resolve().parent == path.parent.resolve()`. Map non-finite samples and malformed columns to stable error codes rather than the generic code shown in the structural snippet. Extend `PrimitiveLibrary` with an optional `allowed_names: frozenset[str] | None`; `get()` must reject names outside that set before touching the filesystem.

- [ ] **Step 4: Add the initial explicit whitelist**

```toml
[motions.nod]
file = "nod.csv"
enabled = true
tags = ["agreement"]

[motions.headshake]
file = "headshake.csv"
enabled = true
tags = ["disagreement"]
```

Include the remaining verified recordings `curious`, `excited`, `happy_wiggle`, `sad`, `shy`, `shock`, `scanning`, `wake_up`, and `idle` with their exact filenames and `enabled = true`.

- [ ] **Step 5: Run catalog and primitive regression tests**

Run: `.venv/bin/pytest tests/test_motion_catalog.py tests/test_primitives.py -v`

Expected: all tests PASS; an unlisted CSV remains unavailable.

- [ ] **Step 6: Commit the catalog slice**

```bash
git add src/motion/catalog.py src/motion/primitives.py tests/test_motion_catalog.py lelamp_runtime/lelamp/recordings/catalog.toml
git commit -m "feat(motion): add validated motion catalog"
```

### Task 2: Versioned Wire Protocol

**Files:**
- Create: `src/motion/protocol.py`
- Create: `tests/test_motion_protocol.py`
- Create: `tests/fixtures/motion_protocol_vectors.json`

**Interfaces:**
- Consumes: no motion or hardware objects.
- Produces: `decode_request(line: bytes, *, token: str, received_at: float) -> Request`, `encode_message(message: Mapping[str, object]) -> bytes`, `Request.expires_at: float`, `ProtocolError(code, message)`.

- [ ] **Step 1: Write failing schema and TTL tests**

```python
def test_decode_uses_pi_receive_time_for_ttl():
    request = decode_request(valid_line(ttl_ms=750), token="secret", received_at=12.5)
    assert request.expires_at == pytest.approx(13.25)

@pytest.mark.parametrize("line,code", [
    (b"{}\n", "missing_field"),
    (request_line(version=2), "unsupported_version"),
    (request_line(type="motor.raw"), "unknown_type"),
    (request_line(payload={"point": [1.0, float("nan"), 2.0]}), "non_finite"),
])
def test_invalid_requests_are_rejected(line, code):
    with pytest.raises(ProtocolError, match=code):
        decode_request(line, token="secret", received_at=1.0)
```

- [ ] **Step 2: Run the protocol tests and confirm failure**

Run: `.venv/bin/pytest tests/test_motion_protocol.py -v`

Expected: FAIL during import because `motion.protocol` does not exist.

- [ ] **Step 3: Implement the protocol codec**

```python
PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 16 * 1024
COMMAND_TYPES = frozenset({
    "motion.play", "motion.cancel", "motion.interrupt", "motion.status", "motion.list",
    "track.point", "track.bearing", "track.clear",
    "task_light.place", "task_light.cancel", "task_light.clear", "system.heartbeat",
})

@dataclass(frozen=True)
class Request:
    id: str
    type: str
    payload: dict[str, object]
    expires_at: float
```

Parse with `json.loads`, compare tokens with `hmac.compare_digest`, validate UUID text, bound `ttl_ms` to `1..10_000`, recursively reject non-finite floats, and validate each command payload against an explicit field set. Motion names use `^[a-z][a-z0-9_]{0,63}$`; `replace_current` is a JSON boolean; `intensity` is `0.0..1.0`; `repeat` is `1..3`; points and vectors contain exactly three finite numbers whose absolute values do not exceed 2 metres. A bearing direction must have nonzero norm. `encode_message` must use compact separators and append exactly one newline.

- [ ] **Step 4: Add stable golden request/response vectors**

The fixture must contain one valid example for every command type plus `accepted`, `completed`, `cancelled`, and `failed` server events. Tests decode and re-encode each vector and compare its parsed JSON object, allowing object-key order to differ.

- [ ] **Step 5: Run focused and full tests**

Run: `.venv/bin/pytest tests/test_motion_protocol.py -v && .venv/bin/pytest -q`

Expected: protocol tests and the existing suite PASS.

- [ ] **Step 6: Commit the protocol slice**

```bash
git add src/motion/protocol.py tests/test_motion_protocol.py tests/fixtures/motion_protocol_vectors.json
git commit -m "feat(motion): define remote command protocol"
```

### Task 3: Single-Owner Motion Controller

**Files:**
- Create: `src/motion/controller.py`
- Create: `tests/test_motion_controller.py`
- Modify: `src/motion/runtime.py`

**Interfaces:**
- Consumes: `MotionRuntime`, `MotionCatalog`, validated `Request`.
- Produces: `MotionController.submit(request: Request) -> CommandTicket`, `MotionController.snapshot() -> ControllerStatus`, `MotionController.run()`, `MotionController.stop(reason: str)`, `CommandTicket.accepted` and `CommandTicket.completed` futures.

- [ ] **Step 1: Write failing arbitration and ownership tests**

```python
def test_latest_tracking_value_replaces_pending_value(controller):
    controller.submit(point_request("a", [0.2, 0.0, 0.3]))
    controller.submit(point_request("b", [0.3, 0.1, 0.4]))
    controller.tick_once(now=1.0)
    assert controller.runtime.track.filter.position.tolist() == pytest.approx([0.3, 0.1, 0.4])

def test_duplicate_motion_id_cannot_start_twice(controller):
    first = controller.submit(play_request("same", "nod"))
    second = controller.submit(play_request("same", "nod"))
    assert first is second

def test_expired_request_never_reaches_runtime(controller):
    ticket = controller.submit(play_request("late", "nod", expires_at=1.0))
    controller.tick_once(now=1.1)
    assert ticket.completed.result().code == "expired"
    assert not controller.runtime.primitive.busy
```

- [ ] **Step 2: Run the controller tests and confirm failure**

Run: `.venv/bin/pytest tests/test_motion_controller.py -v`

Expected: FAIL during import because `motion.controller` does not exist.

- [ ] **Step 3: Add observable primitive progress without changing motion output**

Expose read-only `PrimitiveLayer.active_name: str | None`, `started_at: float | None`, and `progress(now: float) -> float`. Keep blend, trajectory, sign, scale, and smoothing behavior unchanged. Add runtime tests proving the added properties do not change the produced `q_cmd` sequence.

- [ ] **Step 4: Implement the mailbox and controller state machine**

```python
@dataclass(frozen=True)
class ControllerStatus:
    state: str
    active_motion: str | None
    busy: bool
    fault: str | None
    sent_ticks: int
    deadline_misses: int

@dataclass
class CommandTicket:
    request_id: str
    accepted: Future
    completed: Future

class MotionController:
    def submit(self, request: Request) -> CommandTicket:
        with self._lock:
            cached = self._recent.get(request.id)
            if cached is not None:
                return cached
            ticket = CommandTicket(request.id, Future(), Future())
            self._remember(request.id, ticket)
            if request.type == "track.point":
                self._latest_point = (request, ticket)
            elif request.type == "track.bearing":
                self._latest_bearing = (request, ticket)
            else:
                try:
                    self._commands.put_nowait((request, ticket))
                except queue.Full:
                    self._finish(ticket, "failed", "queue_full")
            return ticket

    def tick_once(self, *, now: float) -> None:
        self._apply_latest_tracking(now)
        self._apply_one_discrete_command(now)
        step = self.runtime.step()
        self._update_active_ticket(step)

    def snapshot(self) -> ControllerStatus:
        with self._lock:
            return self._status

    def stop(self, reason: str) -> None:
        self._stop_reason = reason
        self._stop_event.set()
```

Implement `_remember` with an `OrderedDict` capped at 256 entries. `_apply_latest_tracking` atomically removes each latest slot, rejects expired values, and calls the matching TrackLayer method. `_apply_one_discrete_command` consumes at most one FIFO entry per tick and maps it using Step 5. `_finish` resolves accepted and completed futures exactly once. `run()` repeats `tick_once` on the monotonic deadline grid until `_stop_event` is set. Use a bounded `queue.Queue(maxsize=32)` and reuse `DEADLINE_JITTER_SECONDS` from `hardware_run.py`; expired slots are counted and skipped. Only the controller thread invokes runtime methods.

- [ ] **Step 5: Implement command mapping and completion rules**

Map play, interrupt, tracking, and task-light commands only to existing public `MotionRuntime` methods. A play ticket completes when the primitive layer is inactive and trajectory velocity is below the configured settling tolerance for five consecutive ticks. A disconnect command clears tracking, calls `barge_in()`, and completes active remote tickets as `cancelled`.

- [ ] **Step 6: Run controller, runtime, trajectory, and hardware-run tests**

Run: `.venv/bin/pytest tests/test_motion_controller.py tests/test_runtime.py tests/test_trajectory.py tests/test_hardware_run.py -v`

Expected: all tests PASS and the existing 100 Hz deadline policy remains unchanged.

- [ ] **Step 7: Commit the controller slice**

```bash
git add src/motion/controller.py src/motion/runtime.py tests/test_motion_controller.py tests/test_runtime.py
git commit -m "feat(motion): add remote command controller"
```

### Task 4: Pi TCP Server and Diagnostic Client

**Files:**
- Create: `src/motion/middleware_server.py`
- Create: `src/motion/remote_client.py`
- Create: `tests/test_motion_middleware.py`

**Interfaces:**
- Consumes: `decode_request`, `encode_message`, `MotionController.submit`, `CommandTicket`.
- Produces: `MotionTcpServer`, `MotionClient.request(type, payload, ttl_ms)`, CLI subcommands `list`, `status`, `play`, `interrupt`, `track-point`, and `task-light`.

- [ ] **Step 1: Write failing loopback tests**

```python
async def test_play_emits_ack_then_terminal_result(server, client):
    events = await client.request("motion.play", {"name": "nod", "replace_current": True,
                                                   "intensity": 1.0, "repeat": 1})
    assert events[0]["state"] == "accepted"
    assert events[-1]["state"] == "completed"

async def test_disconnect_invokes_controller_safe_wait(server, raw_connection):
    await raw_connection.close()
    await eventually(lambda: server.controller.snapshot().state == "safe_wait")
```

- [ ] **Step 2: Run the middleware tests and confirm failure**

Run: `.venv/bin/pytest tests/test_motion_middleware.py -v`

Expected: FAIL during import because the server and client modules do not exist.

- [ ] **Step 3: Implement the asyncio server**

Use `asyncio.start_server` with `limit=MAX_LINE_BYTES + 1`. Permit one authenticated control connection; a second connection receives `{"state":"failed","code":"controller_connected"}` and closes. For each request, send the accepted future result, then schedule the terminal future response without blocking reads. Heartbeat timeout is 2.5 seconds. On EOF or timeout, call `controller.remote_disconnected()` exactly once.

- [ ] **Step 4: Implement the dependency-free client**

Use `asyncio.open_connection`, a background reader task, UUID request IDs, and dictionaries of pending accepted/terminal futures. Reconnect with delays `0.25, 0.5, 1, 2, 5` seconds capped at 5 seconds. Do not replay motion requests automatically; callers decide whether to submit a new request ID.

- [ ] **Step 5: Add CLI argument validation**

```text
python -m motion.remote_client --host 192.168.100.2 --port 8765 list
python -m motion.remote_client --host 192.168.100.2 --port 8765 play nod
python -m motion.remote_client --host 192.168.100.2 --port 8765 interrupt
```

Read the token from `TALKING_LAMP_TOKEN`; never accept the token as a CLI argument so it does not appear in process listings.

- [ ] **Step 6: Run TCP failure and regression tests**

Run: `.venv/bin/pytest tests/test_motion_middleware.py tests/test_motion_protocol.py -v && .venv/bin/pytest -q`

Expected: loopback, duplicate, bad token, over-size, reconnect, heartbeat, and full regression tests PASS.

- [ ] **Step 7: Commit the TCP slice**

```bash
git add src/motion/middleware_server.py src/motion/remote_client.py tests/test_motion_middleware.py
git commit -m "feat(motion): serve high-level commands over TCP"
```

### Task 5: Hardware Entry Point and Pi Service

**Files:**
- Modify: `src/motion/middleware_server.py`
- Create: `deploy/pi/talking-lamp-motion.service`
- Create: `deploy/pi/install-motion-service.sh`
- Modify: `tests/test_motion_middleware.py`
- Create: `tests/test_deploy_scripts.py`

**Interfaces:**
- Consumes: valid `MotionCatalog`, `FeetechBackend`, `MotionController`, `MotionTcpServer`.
- Produces: `python -m motion.middleware_server` and an enabled Pi systemd unit.

- [ ] **Step 1: Write failing fail-closed startup tests**

```python
def test_invalid_catalog_prevents_backend_construction(monkeypatch, tmp_path):
    opened = []
    monkeypatch.setattr(server, "FeetechBackend", lambda **kw: opened.append(kw))
    assert server.main(["--catalog", str(bad_catalog(tmp_path)), "--port", "/dev/ttyACM0",
                        "--lamp-id", "lelamp"]) == 2
    assert not opened
```

- [ ] **Step 2: Run the startup tests and confirm failure**

Run: `.venv/bin/pytest tests/test_motion_middleware.py::test_invalid_catalog_prevents_backend_construction -v`

Expected: FAIL because the hardware CLI is not implemented.

- [ ] **Step 3: Implement fail-closed construction order**

Parse configuration, require `TALKING_LAMP_TOKEN`, load and validate the catalog, verify Ruckig, then create `FeetechBackend`, `MotionRuntime`, controller, and TCP server in that order. Put controller stop and `backend.close()` in a single idempotent `finally` path. Handle SIGINT and SIGTERM by requesting controller shutdown and waiting for park completion.

- [ ] **Step 4: Add the systemd unit and installer**

```ini
[Service]
Type=simple
User=pixs
WorkingDirectory=/home/pixs/talking-lamp
EnvironmentFile=/etc/talking-lamp/motion.env
Environment=PYTHONPATH=/home/pixs/talking-lamp/src:/home/pixs/talking-lamp/lelamp_runtime
ExecStart=/home/pixs/talking-lamp/lelamp_runtime/.venv/bin/python -m motion.middleware_server --bind 192.168.100.2 --tcp-port 8765 --port /dev/ttyACM0 --lamp-id lelamp
KillSignal=SIGTERM
TimeoutStopSec=20
Restart=on-failure
RestartSec=3
```

Add `StartLimitIntervalSec=60` and `StartLimitBurst=3` under `[Unit]`. The installer validates paths, creates `/etc/talking-lamp/motion.env` mode `0600` without overwriting an existing token, installs the unit, runs `systemctl daemon-reload`, and leaves the unit disabled until the operator passes `--enable`.

- [ ] **Step 5: Test service invariants without systemd mutation**

Run: `.venv/bin/pytest tests/test_deploy_scripts.py tests/test_motion_middleware.py -v`

Expected: unit has the correct user, interpreter, bind address, signal, and stop timeout; installer dry-run is idempotent.

- [ ] **Step 6: Commit the Pi deployment slice**

```bash
git add src/motion/middleware_server.py tests/test_motion_middleware.py tests/test_deploy_scripts.py deploy/pi
git commit -m "feat(motion): add safe Pi daemon service"
```

### Task 6: ROS 2 Interface Package

**Files:**
- Create: `jetson_ws/src/lamp_interfaces/action/PlayMotion.action`
- Create: `jetson_ws/src/lamp_interfaces/action/PlaceTaskLight.action`
- Create: `jetson_ws/src/lamp_interfaces/msg/MotionStatus.msg`
- Create: `jetson_ws/src/lamp_interfaces/srv/ListMotions.srv`
- Create: `jetson_ws/src/lamp_interfaces/srv/InterruptMotion.srv`
- Create: `jetson_ws/src/lamp_interfaces/CMakeLists.txt`
- Create: `jetson_ws/src/lamp_interfaces/package.xml`
- Modify: `tests/test_deploy_scripts.py`

**Interfaces:**
- Consumes: standard `builtin_interfaces` and `geometry_msgs`.
- Produces: generated ROS types under `lamp_interfaces.action`, `.msg`, and `.srv` on Humble and Jazzy.

- [ ] **Step 1: Write failing static contract tests**

```python
def test_play_motion_action_is_generic_and_cancellable():
    sections = read_ros_sections("jetson_ws/src/lamp_interfaces/action/PlayMotion.action")
    assert sections.goal == ["string name", "bool replace_current", "float32 intensity", "uint32 repeat"]
    assert "float32 progress" in sections.feedback
```

- [ ] **Step 2: Run the contract tests and confirm missing files**

Run: `.venv/bin/pytest tests/test_deploy_scripts.py -v`

Expected: FAIL with missing `lamp_interfaces` files.

- [ ] **Step 3: Create exact ROS interfaces**

`PlayMotion.action` uses the fields from Task 1's approved design. `PlaceTaskLight.action` goal contains `geometry_msgs/PointStamped target`; both results contain `bool success`, `string code`, and `string message`; both feedback sections contain `string state` and `float32 progress`. `MotionStatus.msg` contains timestamp, connected, state, active motion, busy, fault, sent ticks, and deadline misses.

- [ ] **Step 4: Add minimal ament metadata**

Use `rosidl_generate_interfaces`, declare `geometry_msgs` and `builtin_interfaces`, export `rosidl_default_runtime`, and mark the package as a member of `rosidl_interface_packages`. Do not add runtime code to the interface package.

- [ ] **Step 5: Build on both supported ROS images**

Run on an aarch64 Jetson or equivalent supported containers:

```bash
source /opt/ros/humble/setup.bash && colcon build --packages-select lamp_interfaces
source /opt/ros/jazzy/setup.bash && colcon build --packages-select lamp_interfaces
```

Expected: both builds finish with zero failed packages and `ros2 interface show lamp_interfaces/action/PlayMotion` reports the exact contract.

- [ ] **Step 6: Commit the interface slice**

```bash
git add jetson_ws/src/lamp_interfaces tests/test_deploy_scripts.py
git commit -m "feat(jetson): define ROS motion interfaces"
```

### Task 7: Jetson ROS Bridge

**Files:**
- Create: `jetson_ws/src/lamp_motion_bridge/lamp_motion_bridge/transport.py`
- Create: `jetson_ws/src/lamp_motion_bridge/lamp_motion_bridge/adapter.py`
- Create: `jetson_ws/src/lamp_motion_bridge/lamp_motion_bridge/node.py`
- Create: `jetson_ws/src/lamp_motion_bridge/lamp_motion_bridge/__init__.py`
- Create: `jetson_ws/src/lamp_motion_bridge/launch/motion_bridge.launch.py`
- Create: `jetson_ws/src/lamp_motion_bridge/setup.py`
- Create: `jetson_ws/src/lamp_motion_bridge/setup.cfg`
- Create: `jetson_ws/src/lamp_motion_bridge/package.xml`
- Create: `jetson_ws/src/lamp_motion_bridge/resource/lamp_motion_bridge`
- Create: `tests/test_jetson_bridge_contract.py`

**Interfaces:**
- Consumes: `lamp_interfaces`, `geometry_msgs`, Pi protocol golden vectors.
- Produces: `PiTransport`, `BridgeAdapter`, `RosMotionBridge`, console script `lamp_motion_bridge`.

- [ ] **Step 1: Write failing Python 3.10-compatible transport tests**

```python
def test_jetson_encoder_matches_pi_golden_vectors():
    for vector in golden_requests():
        encoded = transport.encode_request(**vector["arguments"])
        assert json.loads(encoded) == vector["json"]

def test_status_mapping_preserves_fault_and_counters():
    status = BridgeAdapter.status_from_event(status_event(fault="overrun", sent_ticks=42))
    assert status.fault == "overrun"
    assert status.sent_ticks == 42
```

- [ ] **Step 2: Run tests and confirm missing bridge package**

Run: `PYTHONPATH=jetson_ws/src/lamp_motion_bridge .venv/bin/pytest tests/test_jetson_bridge_contract.py -v`

Expected: FAIL because the package modules do not exist.

- [ ] **Step 3: Implement the standalone transport and adapter**

Keep `transport.py` free of ROS imports so it can be tested in the repository venv. It owns the persistent socket, heartbeat, request correlation, terminal events, and reconnect policy. It never automatically retries a `motion.play`. `adapter.py` converts simple dataclasses to/from wire dictionaries so ROS callbacks contain no protocol parsing.

- [ ] **Step 4: Implement the ROS node**

Create one Action server for each Action, two Services, two tracking subscriptions, and one status publisher. Use a mutually exclusive callback group for Action state changes and a reentrant group for tracking/status. Forward Action cancel callbacks to the matching Pi request ID. Publish disconnected status immediately when the transport reports EOF.

- [ ] **Step 5: Apply exact QoS settings**

Tracking subscriptions use `KEEP_LAST`, depth 1, `BEST_EFFORT`, and `VOLATILE`. Motion status uses depth 5, `RELIABLE`, and `VOLATILE`. Services and Actions use ROS defaults. The direct wired link still uses TCP reliability between the bridge and Pi.

- [ ] **Step 6: Add launch and package metadata**

The launch file declares `pi_host=192.168.100.2`, `pi_port=8765`, `token_env=TALKING_LAMP_TOKEN`, `tracking_ttl_ms=250`, and `heartbeat_interval_ms=500`. `setup.py` uses syntax valid on Python 3.10 and exposes `lamp_motion_bridge = lamp_motion_bridge.node:main`.

- [ ] **Step 7: Test pure transport, then ROS builds**

Run:

```bash
PYTHONPATH=jetson_ws/src/lamp_motion_bridge .venv/bin/pytest tests/test_jetson_bridge_contract.py -v
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --packages-select lamp_interfaces lamp_motion_bridge --symlink-install
colcon test --packages-select lamp_motion_bridge
colcon test-result --verbose
```

Expected: golden vectors pass, both packages build, and no ROS tests fail on Humble and Jazzy.

- [ ] **Step 8: Commit the bridge slice**

```bash
git add jetson_ws/src/lamp_motion_bridge tests/test_jetson_bridge_contract.py
git commit -m "feat(jetson): bridge ROS commands to Pi motion server"
```

### Task 8: JetPack Detection, Documentation, and End-to-End Verification

**Files:**
- Create: `deploy/jetson/detect-platform.sh`
- Create: `deploy/jetson/install-motion-bridge.sh`
- Create: `docs/motion-middleware.md`
- Modify: `README.md`
- Modify: `tests/test_deploy_scripts.py`

**Interfaces:**
- Consumes: completed Pi daemon and ROS packages.
- Produces: deterministic installation, operator documentation, and verified wired end-to-end workflow.

- [ ] **Step 1: Write failing platform matrix tests**

```python
@pytest.mark.parametrize("l4t,ubuntu,expected", [
    ("36.5.2", "22.04", "humble"),
    ("39.2.1", "24.04", "jazzy"),
])
def test_detect_platform_supported_matrix(l4t, ubuntu, expected):
    result = run_detector(l4t=l4t, ubuntu=ubuntu)
    assert result.stdout.strip() == expected

def test_detect_platform_rejects_mismatched_os():
    result = run_detector(l4t="36.5.2", ubuntu="24.04")
    assert result.returncode != 0
    assert "unsupported JetPack/Ubuntu combination" in result.stderr
```

- [ ] **Step 2: Run the deployment tests and confirm failure**

Run: `.venv/bin/pytest tests/test_deploy_scripts.py -v`

Expected: FAIL because Jetson scripts do not exist.

- [ ] **Step 3: Implement detection and installation**

`detect-platform.sh` accepts `L4T_VERSION_OVERRIDE` and `UBUNTU_VERSION_OVERRIDE` only for tests; production reads `nvidia-l4t-core` and `/etc/os-release`. `install-motion-bridge.sh` installs `ros-$ROS_DISTRO-ros-base`, `python3-colcon-common-extensions`, initializes rosdep when needed, resolves package dependencies, and builds only `lamp_interfaces` and `lamp_motion_bridge`.

- [ ] **Step 4: Document exact operator workflows**

Document JetPack detection, static Ethernet addresses, token provisioning, Pi service installation, Jetson build/launch, status inspection, play/cancel examples, fault recovery, and the six-step new-motion workflow. Include commands for both Humble and Jazzy by using the detector output rather than duplicating instructions.

- [ ] **Step 5: Run all non-hardware verification**

Run:

```bash
.venv/bin/pytest -q
.venv/bin/python -m compileall -q src jetson_ws/src/lamp_motion_bridge/lamp_motion_bridge
git diff --check
MUJOCO_GL=egl .venv/bin/python -m sim.check
```

Expected: the full suite passes, compilation and diff checks are silent, and simulation check exits 0.

- [ ] **Step 6: Deploy Pi daemon without enabling motion and verify read-only calls**

Copy the committed tree to `/home/pixs/talking-lamp`, run the Pi installer without `--enable`, then start the daemon with a `NullBackend` test option bound to `192.168.100.2`. From the Jetson or development host call `list` and `status`; confirm the catalog and `connected=true` without opening `/dev/ttyACM0`.

- [ ] **Step 7: Build and launch on the actual Jetson**

Run the platform detector, install the matching ROS distribution, build the two packages, and launch the bridge. Confirm `ros2 action list`, `ros2 topic list`, and `ros2 service list` expose every contract in the spec. Record L4T, Ubuntu, ROS, Python, and package versions in the verification log.

- [ ] **Step 8: Run controlled hardware acceptance**

After Tasks 1–7 pass, enable the Pi hardware service. Verify `status`, play `nod` once, cancel one motion, allow tracking TTL to expire, unplug/reconnect Ethernet, then run each enabled catalog motion once. Confirm the final sleep pose, torque disable after service stop, zero out-of-range commands, and bounded deadline misses.

- [ ] **Step 9: Commit documentation and deployment support**

```bash
git add deploy/jetson docs/motion-middleware.md README.md tests/test_deploy_scripts.py
git commit -m "docs: add Jetson Pi middleware deployment guide"
```

- [ ] **Step 10: Push and update the pull request**

Run:

```bash
git status --short
git log --oneline --decorate -10
git push origin feat/motion-stack-e2e-sim
```

Expected: the worktree is clean, the remote branch contains every middleware commit, and PR #1 shows the Pi server, Jetson ROS bridge, tests, and deployment guide.
