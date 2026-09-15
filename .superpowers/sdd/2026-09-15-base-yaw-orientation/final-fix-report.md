# Final integration fixes

## Investigation before implementation

Read the complete final review, implementation plan, approved design, debugging/TDD skills and testing reference, and current orientation/runtime/primitive/controller source and tests.

### Finding 1: anchor changes during playback

Phase 1 data flow: `motion.play -> _start_clip -> runtime.play_primitive -> PrimitiveLayer.play` fits a resampled clip once. `orientation.acquire -> runtime.acquire_orientation -> coordinator.acquire -> layer.acquire` changes only the absolute yaw. `runtime.step -> blender.compute` then adds the stale scaled offsets. `PrimitiveLayer.busy` includes an interrupted release envelope; it continues sampling until closed. Automatic disconnect return changes the layer inside `coordinator.observe`, after the hardware step, so synchronizing only explicit acquisition would miss it.

Phase 2: the existing copy-based `Primitive.scaled_joint` and play-time extrema calculation provide the safe fitting pattern, but there is no retained unscaled playback source or anchor-change synchronization. Phase 3 hypothesis: retaining the current resampled source and refitting yaw before the next blend whenever the anchor changes will preserve the margins and all other joints, including release tails, without accumulating scaling. Fit the whole current clip conservatively (no resampling/restart or time discontinuity), allowing idle/body motion to continue during acquisition.

Phase 4 RED: both controller acquisition cases commanded 1.4662815/1.4662813 rad above 1.39762694; interrupted release blended 1.4248157 rad; retained-source test showed original 0.1202837 rad amplitude instead of fitting. GREEN: four focused acquisition/refit cases pass after retaining the source, refitting on acquisition, and syncing controller playback scale from the layer after hardware steps. The release fixture needed a positive-yaw timestamp (1.5 s); the initial 0.8 s negative offset did not expose the upper-margin bug.

### Finding 2: retained active anchors bypass safety

Phase 1 data flow: runtime play checks coordinator state `orienting/aligned`, although `timeout`, `returning`, and `centered` all retain the absolute layer target. Repeats re-enter this same play path. Automatic return changes that target after `runtime.step`; explicit return changes it through runtime. Phase 2: the layer target itself is the authority for whether absolute yaw is active; state labels describe ticket lifecycle and must not gate safety. Phase 3 hypothesis: use the active layer target for playback and synchronize changed anchor context before every blend, caching the fitted context to avoid per-tick array copies. This also catches automatic returns and direct layer activation/release.

Phase 4 RED: all six state/endpoint repeated-playback cases violated actual command margins (upper about 1.466282 rad; lower about -1.604544 rad versus safe -1.51727744), plus automatic-return release blended 1.4248157 rad. GREEN: all seven new cases plus four finding-1 cases pass. Runtime reads the layer's actual target and refits before blending; fitting is cached by anchor/limits and invalidated for a newly loaded source.

Additional Phase 1–3 investigation: the full default Ruckig suite passed 428 tests, but running the new command invariants with `TALKING_LAMP_NO_RUCKIG=1` exposed six remaining failures. Safe blended targets still produced analytic commands up to 1.39955167 rad (safe upper 1.39762694) or down to -1.51928442 rad (safe lower -1.51727744). The analytic spring intentionally allows small setpoint overshoot and only its position-limit braking reserve enforces a path boundary. The equivalent Ruckig path-extrema check likewise knows only hard limits. Therefore fitting offsets alone cannot guarantee commanded margins on every backend. Extend trajectory stepping with optional tighter bounds, intersected with hard bounds and scoped to the active orientation when TaskLight is not authoritative. A pose initially outside the safe interval gets a recovery interval spanning its current pose and the safe interval. Do not clip commands or reset trajectory velocity. Ruckig must invalidate/check its retained path when the effective bounds change, rather than reuse a path feasible only under old hard limits.

### Finding 3: deadband assigns a moving target

Phase 1: coordinator calls `layer.acquire(requested)` before its deadband branch, then publishes aligned. Runtime's following trajectory steps still consume that changed layer target. Phase 2: the settle path correctly waits for actual yaw and velocity; only immediate success bypasses it. Phase 3 hypothesis: resolve deadband first, use the safe current commanded yaw as the anchor for a no-move request, and choose the settle path if current yaw lies outside the margin. Report clamping when safety requires moving an unsafe current pose into the interval.

Phase 4 RED: stationary yaw changed from 0.2617993878 to 0.2618393878 within two ticks after immediate aligned; both outside-margin initial poses falsely completed immediately. GREEN: three regression cases and existing orientation cases pass. An additional return-in-progress test reproduced immediate aligned at nonzero yaw velocity; pass actual trajectory velocity to acquisition, retaining the current target but waiting for the ordinary settle path when already moving. This test also passes after that change. Existing unit expectation changed from requested .04 rad to retained actual 0 rad.

### Finding 4: replacement result metadata

Phase 1: `_start_clip` sets `_primitive_yaw_scale` to the incoming clip's value before `_cancel_motion` uses that field for the outgoing ticket. Phase 2: replacement loading precedes cancellation intentionally; unknown clips leave active work intact and a real load exception faults the controller, finishing pending tickets as fault. Phase 3 hypothesis: return the new scale from `_start_clip` and publish it only after cancelling the outgoing ticket; repeat playback explicitly stores its returned scale too. This preserves existing load-before-cancel failure semantics.

Phase 4 RED: outgoing terminal scale was 1.0 instead of 0.14241747648734046. GREEN: 14 replacement/repeat-focused cases pass, including a real missing-recording failure that retains the existing fault result for both tickets.

### Finding 5 (Minor): one-shot diagnostics wait forever

Phase 1: status and heartbeat examples iterate a socket file until EOF, whereas the Unix server keeps the connection open for further requests after a result. Phase 2: acquire and return examples already break when state is no longer accepted. Phase 3 hypothesis: use the same terminal break in both diagnostics, handling successful accepted/result sequences and single failed/expired or failed/fault replies. The regression executes the actual documented Python code in a subprocess against a real persistent Unix socket peer; it checks printed events and client closure, not source text.

Phase 4 RED: all six examples/cases exceeded the subprocess deadline because the server remained open. GREEN: all six pass after breaking at the first non-accepted event. No protocol or transport behavior changed.

## Final design and regression coverage

- Keep acquisition available during expressive and idle playback; no new busy rejection contract. The current clip is refitted conservatively using its full original resampled yaw extrema. This may reduce more than a remaining-tail-only fit, but preserves timing/envelopes and avoids cumulative scaling. Tests additionally cover the real idle recording and acquisition during returning.
- Use the absolute layer target as the safety activation condition, not ticket state. Play/repeat and pre-blend synchronization cover retained timeout/return/center anchors, direct activation, and autonomous return.
- Enforce the safe path interval through trajectory bounds, never by clipping `q_cmd` or resetting motion state. Bounds are intersected with hard limits for the step and restored afterward; TaskLight retains priority. Ruckig replans from current position/velocity/acceleration when bounds change and does not reuse a path certified only under different bounds.
- Deadband retains actual current yaw (or a safety-clamped current yaw), not the requested displacement. A moving yaw or unsafe initial yaw waits for settling. Duplicate speech handling remains unchanged.
- Controller status follows the current fitted yaw scale. Motion terminals report their last scale; replacing a clip loads successfully before the outgoing ticket is cancelled, then publishes incoming metadata. Missing-file loading retains the prior fault behavior.
- Plan Task 3 now requires fitting for every active absolute anchor and changed-anchor/repeat refits. Operator examples and the concrete test-evidence table reflect these fixes while retaining the motion-only milestone and future microphone handoff.

## Files

- `src/motion/layers.py`: retain unscaled current clip, cached yaw refitting, current scale metadata.
- `src/motion/runtime.py`: active-layer safety context, pre-blend refits, velocity-aware acquire, safe trajectory bounds.
- `src/motion/orientation.py`: layer target accessor and true no-move deadband with safe recovery/settling.
- `src/motion/controller.py`: outgoing replacement scale ownership and updated live scale.
- `src/motion/trajectory.py`: optional per-step bounds intersected with hard limits and Ruckig path recertification.
- `tests/test_motion_controller.py`: actual backend commands and ticket regressions for acquisition/playback/return/timeout/repeats/deadband/replacement/load failures.
- `tests/test_runtime.py`: interrupted release tails, unscaled restoration, and autonomous return refitting.
- `tests/test_orientation.py`: deadband layer output/target expectation.
- `tests/test_orientation_diagnostics.py`: execute documented scripts against persistent Unix sockets.
- `docs/base-yaw-orientation.md`: terminal diagnostics, corrected behavior, test evidence, unchanged hardware scope.
- `docs/superpowers/plans/2026-09-15-base-yaw-orientation.md`: correct the safety-state restriction.
- `.superpowers/sdd/2026-09-15-base-yaw-orientation/final-fix-report.md`: this evidence report.

## Deviations and risks

- Expanded into `trajectory.py` because real fallback command tests proved clip fitting alone insufficient for the requested mechanical margin. The public hard-limit array remains unchanged outside each step.
- An initial pose already inside the mechanical margin cannot instantly become safe; the recovery interval allows its existing pose and contracts toward the safe interval. Existing momentum can make a newly tightened path infeasible for Ruckig; it faults before issuing a newly unsafe command rather than reuse an uncertified path. Hardware braking/commissioning remains a later validation step.
- Analytic mode remains the existing acceleration-limited, non-jerk-limited fallback. No new hardware smoothness claims are made.
- Whole-current-clip fitting is intentionally conservative versus scanning only future frames. Body offsets and requested intensity remain untouched; trajectory synchronization can still affect commanded joint timing as before.
- Default Ruckig suite passed before the extra fallback investigation; that first full run was superseded by final checks after the trajectory fix. All failures and fixture corrections are recorded above rather than treating that initial pass as final evidence.
- No Pi, XVF3800, DOA stabilization, audio, LED, Jetson ROS, network transport, or hardware commissioning work was performed. No pushes or merges were made.

## Full verification

All commands ran in `/home/slihump/projects/talking-lamp/.worktrees/jetson-pi-middleware` with `/home/slihump/projects/talking-lamp/.venv/bin/python` and its matching pytest.

1. Final full suite (exit 0):

   ```bash
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest -p no:cacheprovider -o addopts='' -q
   ```

   **432 passed in 33.16 s** with the default Ruckig backend.

2. Final fallback safety/replacement/trajectory regression run (exit 0):

   ```bash
   PYTHONDONTWRITEBYTECODE=1 TALKING_LAMP_NO_RUCKIG=1 PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/pytest -p no:cacheprovider -o addopts='' -q tests/test_motion_controller.py tests/test_runtime.py tests/test_trajectory.py -k 'acquiring_during_primitive or anchor_refit or autonomous_return_refits or retained_anchor or deadband or retains_outgoing or replacement_load_failure or position_limits'
   ```

   **23 passed, 89 deselected in 6.62 s**. This includes the six cases that initially exposed residual trajectory overshoot.

3. Default focused integration/trajectory/orientation/diagnostic run after trajectory fix: **145 passed in 15.18 s**. Subsequent idle/return acquisition expansion: **6 passed in 2.60 s**, then included in the final full run above.
4. `git diff --check`: exit 0, no whitespace errors, repeated immediately before the implementation commit.
5. `PYTHONPATH="$PWD/src:$PWD/lelamp_runtime" /home/slihump/projects/talking-lamp/.venv/bin/python -m compileall -q src`: exit 0, no compile errors, repeated immediately before the implementation commit.
6. Additional in-memory Ruckig stress sweep: 26 acquisition timing points during a near-upper-bound headshake, 250 subsequent ticks each, **zero measured margin violations**. This supplements, rather than replaces, the committed regressions.

## Commits

- `a9fe8a374d093071769accee7c70db33cd25f41b` — `fix(motion): preserve safe yaw through orientation transitions` (all source, tests, operator documentation, and plan correction).
- This report is committed in the following audit-only commit, `docs: record final orientation integration fix evidence`; its SHA is supplied in the task completion handoff (a commit cannot contain its own SHA).

All four Important findings and the Minor diagnostic finding are addressed. Software verification is complete; later hardware milestones remain outside this task.

DONE
