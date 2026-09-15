# Base-yaw orientation operator guide

This document covers the completed **motion-only** base-yaw orientation
milestone. It provides a protected Pi-local way for a future device service to
anchor `base_yaw`, inspect the anchor, and request a return to centre. It does
not implement XVF3800 access or DOA collection/stabilization, audio transport or
playback, WS2812B control, `talking-lamp-device.service`, Jetson ROS bridges, or
turn orchestration. Those are later plans.

## Preconditions and boundary

Run the Pi motion daemon through `talking-lamp-motion.service`. It creates the
protected Unix socket at `/run/talking-lamp/motion-control.sock`.

The socket mode is `0660`; only the service user/group may use it. This is a
Pi-local control surface, not a replacement for the one authenticated Jetson
TCP owner on port 8765. `orientation.acquire`, `orientation.return_center`, and
`orientation.status` are rejected on that remote TCP surface. The Unix transport
only submits controller mailbox tickets; it never steps the runtime or opens
another hardware owner.

Every request is one UTF-8 NDJSON object followed by exactly one newline:

```json
{"version":1,"id":"canonical-request-uuid","type":"...","ttl_ms":1000,"payload":{}}
```

`id` must be canonical lower-case UUID text, `ttl_ms` is an integer from 1 to
10000, and local messages must contain exactly these five fields—there is no
`token` field. Generate a fresh request UUID for each logical request.

## Exact local requests

Set a socket variable once:

```bash
SOCKET=/run/talking-lamp/motion-control.sock
```

The `socat` examples keep stdin open briefly so the server can send its reply;
for an orientation that moves, make that delay long enough to receive its
terminal event (or use the Python example).

### Acquire an orientation anchor

`target_yaw` is an absolute, finite angle in calibrated base-frame **radians**,
and must be in `[-pi, pi]`. `speech_id` is a canonical UUID created by the
future device service for this utterance.

```bash
{ printf '%s\n' \
  '{"version":1,"id":"10000000-0000-0000-0000-000000000001","type":"orientation.acquire","ttl_ms":1000,"payload":{"speech_id":"20000000-0000-0000-0000-000000000001","target_yaw":0.40}}'; sleep 3; } \
  | socat - UNIX-CONNECT:"$SOCKET"
```

```python
import json
import socket

request = {
    "version": 1,
    "id": "10000000-0000-0000-0000-000000000001",
    "type": "orientation.acquire",
    "ttl_ms": 1000,
    "payload": {
        "speech_id": "20000000-0000-0000-0000-000000000001",
        "target_yaw": 0.40,
    },
}
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect("/run/talking-lamp/motion-control.sock")
    client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
    for line in client.makefile("rb"):
        event = json.loads(line)
        print(event)
        if event.get("state") != "accepted":
            break
```

For an accepted move, the first event is `{"id": ..., "state":"accepted",
"code":"accepted", ...}`. The terminal event is `state:"completed"` with
either `code:"aligned"` after the settle rule succeeds, or `code:"timeout"`
after the two-second acquisition deadline. A target already inside the 5°
deadband retains the actual current yaw as its anchor and completes immediately
as `aligned` when yaw is stationary and inside the safe interval. It does not
subsequently turn to the nearby requested angle. If yaw is already moving, it
settles at that retained anchor first. If the current pose lies inside the
mechanical margin, safety takes priority: motion clamps the current anchor into
the safe interval, reports `clamped:true`, and waits for settling. The motion
target is held after `aligned`; another sample for the same `speech_id` does not
retarget it. A second request with the same currently latched `speech_id`
returns terminal `code:"duplicate"` instead.

An active task light rejects this request with the single terminal event
`state:"failed", code:"blocked_by_task_light"`; it does not modify the current
anchor. Release the task light, then acquire for a new utterance as the policy
requires.

### Request an explicit return to centre

```bash
{ printf '%s\n' \
  '{"version":1,"id":"10000000-0000-0000-0000-000000000002","type":"orientation.return_center","ttl_ms":1000,"payload":{}}'; sleep 3; } \
  | socat - UNIX-CONNECT:"$SOCKET"
```

```python
import json
import socket

request = {
    "version": 1,
    "id": "10000000-0000-0000-0000-000000000002",
    "type": "orientation.return_center",
    "ttl_ms": 1000,
    "payload": {},
}
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect("/run/talking-lamp/motion-control.sock")
    client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
    for line in client.makefile("rb"):
        event = json.loads(line)
        print(event)
        if event.get("state") != "accepted":
            break
```

The expected sequence is `accepted/accepted`, then terminal
`completed/centered` only after the centre target satisfies the settle rule.
While a primitive motion or task light is busy, this request is rejected as the
single terminal `failed/busy` event. The caller must wait for (or explicitly
cancel according to its policy) the busy work; motion deliberately does not cut
it short to hide an incorrect orchestration order.

### Read orientation status

```bash
{ printf '%s\n' \
  '{"version":1,"id":"10000000-0000-0000-0000-000000000003","type":"orientation.status","ttl_ms":1000,"payload":{}}'; sleep 1; } \
  | socat - UNIX-CONNECT:"$SOCKET"
```

```python
import json
import socket

request = {
    "version": 1,
    "id": "10000000-0000-0000-0000-000000000003",
    "type": "orientation.status",
    "ttl_ms": 1000,
    "payload": {},
}
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect("/run/talking-lamp/motion-control.sock")
    client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
    for line in client.makefile("rb"):
        event = json.loads(line)
        print(event)
        if event.get("state") != "accepted":
            break
```

An unexpired status request claimed by the controller replies `accepted/accepted`
followed by `completed/completed`. If an otherwise valid request expires before
that claim, its only terminal reply is `failed/expired`; it does not reach
runtime command handling or hardware control. The completed event's `data` is
the orientation snapshot:

| Field | Meaning |
| --- | --- |
| `state` | `idle`, `orienting`, `aligned`, `timeout`, `returning`, or `centered` |
| `speech_id` | latched utterance UUID, or `null` while idle |
| `target_yaw` | anchor/centre target in radians, or `null` while idle |
| `current_yaw` | latest commanded base yaw in radians |
| `clamped` | whether motion clamped the requested target to its safe yaw interval |
| `code` | machine-readable state/result code |

The normal controller status additionally reports `busy`, `orientation_state`,
`orientation_speech_id`, `orientation_target_yaw`, `orientation_current_yaw`,
`orientation_clamped`, and `primitive_yaw_scale`. `busy` is true while
orientation is `orienting` or `returning`, as well as while primitive or task
light motion is active. An already `aligned` anchor is held but is not by itself
busy.

### Local heartbeat

`system.heartbeat` has no orientation side effect, but is the fourth allowed
local message and is useful for a transport diagnostic.

```bash
{ printf '%s\n' \
  '{"version":1,"id":"10000000-0000-0000-0000-000000000004","type":"system.heartbeat","ttl_ms":1000,"payload":{}}'; sleep 1; } \
  | socat - UNIX-CONNECT:"$SOCKET"
```

```python
import json
import socket

request = {
    "version": 1,
    "id": "10000000-0000-0000-0000-000000000004",
    "type": "system.heartbeat",
    "ttl_ms": 1000,
    "payload": {},
}
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect("/run/talking-lamp/motion-control.sock")
    client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
    for line in client.makefile("rb"):
        event = json.loads(line)
        print(event)
        if event.get("state") != "accepted":
            break
```

An unexpired heartbeat claimed by the controller returns `accepted/accepted`
then `completed/completed`. If it expires before claim, it instead returns the
single terminal `failed/expired` event without runtime command handling or
hardware control.

## Settle, primitive, and disconnect behavior

Motion owns only `base_yaw` at priority 15. The shared trajectory generator
enforces trajectory limits, while completion requires both yaw error no greater
than 3° and yaw speed no greater than 0.08 rad/s continuously for 150 ms. The
safe yaw interval is the calibrated base-yaw joint range inset by the configured
5° margin. Motion calculates `clamped`; callers must report rather than
pre-compute it.

While orientation controls yaw, the safe interval also constrains the trajectory
path on both backends. TaskLight keeps its higher priority and calibrated hard
limits. A starting pose outside the margin recovers continuously into the safe
interval; software does not make an instantaneous position jump to clamp it.

The anchor persists while a relative primitive plays. Before primitive playback,
motion computes a yaw-only scale in `[0, 1]` so `anchor_yaw + primitive_offset`
remains inside the safe range; it preserves the other joint offsets and requested
overall intensity. This applies whenever the absolute anchor is active,
including `timeout`, `returning`, and `centered`, and to each repeated clip.
Acquisition is allowed during primitive playback so idle/body motion can
continue while a future device service acquires a direction. An anchor change
refits yaw from the current clip's original resampled offsets before the next
blend, including an interrupted release tail. It preserves clip time, envelope,
and all other joint offsets; successive refits do not compound scaling.
`primitive_yaw_scale` reports the current fitted scale in controller status;
a primitive terminal result's `data.yaw_scale` reports its last applied scale.
A replaced motion retains its own scale in its terminal result.

On an **authenticated remote TCP owner disconnect**, controller work is safely
interrupted, but an existing orientation anchor stays in place for exactly 10
seconds. It then starts the same low-level return-to-centre path and eventually
becomes `centered`; it does not replay a request after reconnect. This is not a
local Unix client EOF rule: an EOF can cancel only that client's still-unstarted
local ticket, and cannot cancel an orientation that the controller already
claimed or another local client's request.

## Inputs required from the later Pi device service

This motion milestone intentionally does no microphone conversion. The next
device-service plan must:

- generate and retain one canonical UUID `speech_id` for each utterance;
- stabilize and validate DOA before issuing exactly one orientation request for
  that utterance;
- convert calibrated DOA into a finite absolute `target_yaw` in radians within
  `[-pi, pi]` before sending `orientation.acquire`;
- keep `doa_zero_deg` and `doa_direction_sign` in the **device configuration**,
  not in motion configuration; and
- use the returned `clamped` field as the motion authority's result instead of
  claiming its own clamp decision.

DOA degrees, `doa_zero_deg`, and `doa_direction_sign` are not local motion
request fields. The later device service owns DOA sampling, stabilization,
device-frame calibration, front-hemisphere rejection, and degree-to-radian
conversion before it calls this socket.

## Real microphone bench handoff (required before mic testing)

Before any real microphone test, update/check `origin/feat/voice-bench` and
read `voice-bench/bench/MIC-ARRIVAL.md` from that ref. At this milestone's
verification time it resolved to `91ce0a14f6f21bfe4aee5e2cd6183bb02c6ec187`;
do not rely on that old revision when commissioning. Use, without checking out
or copying its files into this motion worktree:

```bash
git fetch origin feat/voice-bench
git rev-parse origin/feat/voice-bench
git show origin/feat/voice-bench:voice-bench/bench/MIC-ARRIVAL.md
```

The delivered array is **Linear-4**, not Circular-4. It is front-hemisphere
only (approximately 180°) and must use the Linear 16 kHz/6-channel firmware and
device configuration (`L16K6Ch`, not a circular `C16K6Ch` image). After the
array is mounted in the lamp, recalibrate it in that installed orientation: make
a new seven-point table at left 90/60/30°, front, and right 30/60/90°, then use a
separate verification measurement. Do not copy an old calibration file—an
existing `out/doa/calibration.json` is mounting-specific.

Run the bench under the servo-noise condition as well as quiet/fan conditions;
servo behavior is the critical condition for tracking. Also verify that the XVF
output device is present and use the board's playback/AEC path, then run the XVF
output and AEC checks before treating microphone results as usable. These bench
results (including noise/error data and AEC outcome) are inputs to the next
device-service plan, not evidence that this motion-only milestone has
implemented audio or DOA.

## Verification evidence for this milestone

The automated checks map the motion guarantees in the approved design to these
concrete tests:

| Guarantee | Concrete tests |
| --- | --- |
| Orientation claims only absolute base yaw and obeys a safe margin | `tests/test_orientation.py::test_orientation_layer_claims_only_base_yaw_and_clamps_with_margin` |
| Success waits for the continuous position/velocity settle window | `tests/test_orientation.py::test_coordinator_settles_only_after_continuous_position_and_velocity_window`; `tests/test_motion_controller.py::test_orientation_ticket_completes_only_after_settle` |
| Relative primitives keep the anchor and reduce yaw safely | `tests/test_runtime.py::test_orientation_anchor_overrides_tracking_yaw_but_not_other_tracking_joints`; `tests/test_runtime.py::test_anchored_motion_scales_only_yaw_to_fit_safe_range` |
| Anchor changes and release tails preserve safe commands | `tests/test_motion_controller.py::test_acquiring_during_primitive_refits_yaw_before_commanding`; `tests/test_runtime.py::test_anchor_refits_release_tail_without_restarting_or_scaling_body`; `tests/test_runtime.py::test_autonomous_return_refits_primitive_release_before_next_blend` |
| Timeout, returning, and centered anchors fit repeated playback at both bounds | `tests/test_motion_controller.py::test_retained_anchor_fits_repeated_primitive_commands` |
| Deadband holds actual yaw; unsafe or moving starts settle first | `tests/test_motion_controller.py::test_deadband_requests_never_command_the_requested_displacement`; `tests/test_motion_controller.py::test_deadband_outside_margin_moves_to_safe_anchor_before_success`; `tests/test_motion_controller.py::test_deadband_retarget_during_return_waits_for_existing_velocity_to_settle` |
| Replaced motion results keep outgoing playback metadata | `tests/test_motion_controller.py::test_replaced_motion_result_retains_outgoing_yaw_scale` |
| One-shot diagnostics exit on successful or failed terminal replies | `tests/test_orientation_diagnostics.py::test_one_shot_diagnostic_exits_at_first_terminal_event` |
| Task-light conflict leaves orientation untouched | `tests/test_orientation.py::test_coordinator_reports_task_light_conflict_without_acquiring_layer`; `tests/test_motion_controller.py::test_orientation_acquire_rejects_active_task_light` |
| Explicit return finishes only at centre | `tests/test_orientation.py::test_coordinator_centers_after_return_settle_window_and_latches_speech_id`; `tests/test_motion_controller.py::test_orientation_return_center_completes_after_center_settle` |
| Remote disconnect holds then falls back to centre | `tests/test_orientation.py::test_coordinator_returns_after_exact_disconnected_hold`; `tests/test_motion_controller.py::test_disconnect_starts_center_return_only_after_orientation_hold` |
| Local path never creates a second hardware owner | `tests/test_motion_middleware.py::test_daemon_binds_both_listeners_before_starting_motion_owner`; `tests/test_motion_middleware.py::test_unix_disconnect_invalidates_unstarted_orientation_request` |
| Local-only protocol remains separate from authenticated TCP | `tests/test_motion_protocol.py::test_remote_decoder_rejects_local_orientation_acquire_with_stable_code`; `tests/test_motion_protocol.py::test_local_decoder_accepts_exact_orientation_payload_without_token` |
| A local request expired before controller claim is terminal and does not execute | `tests/test_motion_middleware.py::test_unix_server_forwards_expired_request_to_controller`; `tests/test_motion_controller.py::test_expired_request_never_reaches_runtime` |

Run the complete software check from the repository root:

```bash
git diff --check
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/python -m compileall -q src
PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest -q
```

Passing these checks verifies the simulated motion/protocol milestone only. It
is not a substitute for later Pi hardware, microphone, audio, LED, or Pi–Jetson
end-to-end validation in the approved design.
