"""Thread-safe command mailbox; one owner advances runtime and motor output.

Submission, disconnect and stop only change mailbox state. The first call to
``run`` or ``tick_once`` binds the runtime to that thread for this controller's
lifetime. Backend construction/parking remain the daemon's responsibility.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, InvalidStateError
from dataclasses import asdict, dataclass, field
import math
import queue
import threading
import time
from typing import Callable

import numpy as np

from .catalog import MotionCatalog
from .hardware_run import DEADLINE_JITTER_SECONDS
from .primitives import DEFAULT_SCALE
from .protocol import Request
from .runtime import MotionRuntime, StepState


@dataclass(frozen=True)
class ControllerStatus:
    state: str
    active_motion: str | None
    busy: bool
    fault: str | None
    sent_ticks: int
    deadline_misses: int
    progress: float = 0.0


@dataclass(frozen=True)
class CommandResult:
    request_id: str
    state: str
    code: str
    message: str = ""
    data: dict[str, object] = field(default_factory=dict)


@dataclass
class CommandTicket:
    request_id: str
    accepted: Future[CommandResult] = field(default_factory=Future)
    completed: Future[CommandResult] = field(default_factory=Future)


class MotionController:
    def __init__(
        self, runtime: MotionRuntime, catalog: MotionCatalog, *,
        settling_tolerance: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not math.isfinite(settling_tolerance) or settling_tolerance <= 0:
            raise ValueError("settling_tolerance must be finite and positive")
        self.runtime = runtime
        self.catalog = catalog
        self.settling_tolerance = settling_tolerance
        self.clock = clock
        self._stop_event = threading.Event()
        self.sleep = sleep or self._stop_event.wait
        self._lock = threading.RLock()
        self._commands: queue.Queue[tuple[Request, CommandTicket]] = queue.Queue(maxsize=32)
        self._recent: OrderedDict[str, CommandTicket] = OrderedDict()
        # In-flight IDs must survive churn in the bounded recent-result cache.
        self._pending: dict[str, CommandTicket] = {}
        self._latest_point: tuple[Request, CommandTicket] | None = None
        self._latest_bearing: tuple[Request, CommandTicket] | None = None
        self._tracking_expires: float | None = None
        self._disconnect = False
        self._stop_reason = "stopped"
        self._owner: threading.Thread | None = None
        self._ticking = False
        self._active: tuple[Request, CommandTicket] | None = None
        self._repeats_left = 0
        self._settled_ticks = 0
        self._task_ticket: CommandTicket | None = None
        self._task_settled_ticks = 0
        self._safe_wait = False
        self._fault: str | None = None
        self._sent_ticks = 0
        self._deadline_misses = 0
        self._status = ControllerStatus("idle", None, False, None, 0, 0)

    def submit(self, request: Request) -> CommandTicket:
        with self._lock:
            cached = self._pending.get(request.id) or self._recent.get(request.id)
            if cached is not None:
                if request.id in self._recent:
                    self._recent.move_to_end(request.id)
                return cached
            ticket = CommandTicket(request.id)
            self._remember(request.id, ticket)
            self._pending[request.id] = ticket
            if self._fault:
                self._finish(ticket, "failed", "fault", self._fault)
            elif self._stop_event.is_set():
                self._finish(ticket, "failed", "stopped", self._stop_reason)
            elif request.type in {"track.point", "track.bearing"}:
                slot = "_latest_point" if request.type == "track.point" else "_latest_bearing"
                previous = getattr(self, slot)
                setattr(self, slot, (request, ticket))
                if previous:
                    self._finish(previous[1], "cancelled", "superseded")
            else:
                try:
                    self._commands.put_nowait((request, ticket))
                except queue.Full:
                    self._finish(ticket, "failed", "queue_full")
            return ticket

    def _remember(self, request_id: str, ticket: CommandTicket) -> None:
        self._recent[request_id] = ticket
        self._recent.move_to_end(request_id)
        while len(self._recent) > 256:
            self._recent.popitem(last=False)

    @staticmethod
    def _resolve(future: Future[CommandResult], result: CommandResult) -> None:
        if not future.done():
            try:
                future.set_result(result)
            except InvalidStateError:
                # A transport may cancel its own wait concurrently.
                pass

    def _accept(self, ticket: CommandTicket) -> None:
        self._resolve(ticket.accepted, CommandResult(ticket.request_id, "accepted", "accepted"))

    def _finish(self, ticket: CommandTicket, state: str, code: str,
                message: str = "", data: dict[str, object] | None = None) -> None:
        result = CommandResult(ticket.request_id, state, code, message, data or {})
        with self._lock:
            if self._pending.pop(ticket.request_id, None) is not None:
                self._remember(ticket.request_id, ticket)
            self._resolve(ticket.accepted, result)
            self._resolve(ticket.completed, result)

    def snapshot(self) -> ControllerStatus:
        with self._lock:
            return self._status

    def remote_disconnected(self) -> None:
        """Request safe wait without depending on space in the discrete queue."""
        with self._lock:
            self._disconnect = True

    def stop(self, reason: str) -> None:
        with self._lock:
            if not self._stop_event.is_set():
                self._stop_reason = reason
                self._stop_event.set()

    def _claim_owner(self) -> None:
        with self._lock:
            current = threading.current_thread()
            if self._owner is None:
                self._owner = current
            elif self._owner is not current:
                raise RuntimeError("motion runtime already has another owner thread")

    def _cancel_motion(self, code: str) -> None:
        if self._active is not None:
            self._finish(self._active[1], "cancelled", code)
            self._active = None
        self._repeats_left = 0
        self._settled_ticks = 0

    def _cancel_task(self, code: str) -> None:
        if self._task_ticket is not None:
            self._finish(self._task_ticket, "cancelled", code)
            self._task_ticket = None
        self._task_settled_ticks = 0

    def _clear_mailbox(self, state: str, code: str, message: str = "") -> None:
        with self._lock:
            self._latest_point = self._latest_bearing = None
            while True:
                try:
                    self._commands.get_nowait()
                except queue.Empty:
                    break
            for ticket in list(self._pending.values()):
                self._finish(ticket, state, code, message)

    def _safe_disconnect(self) -> bool:
        with self._lock:
            if not self._disconnect:
                return False
            self._disconnect = False
            self._clear_mailbox("cancelled", "disconnected")
        self.runtime.clear_tracking()
        self.runtime.barge_in()
        self._tracking_expires = None
        self._cancel_motion("disconnected")
        self._cancel_task("disconnected")
        self._safe_wait = True
        return True

    def _apply_latest_tracking(self, now: float) -> None:
        with self._lock:
            latest = (self._latest_point, self._latest_bearing)
            self._latest_point = self._latest_bearing = None
        if self._tracking_expires is not None and now >= self._tracking_expires:
            self.runtime.clear_tracking()
            self._tracking_expires = None
        for entry in latest:
            if entry is None:
                continue
            request, ticket = entry
            if now >= request.expires_at:
                self._finish(ticket, "failed", "expired")
                continue
            if request.type == "track.point":
                self.runtime.observe_point(request.payload["point"])
            else:
                self.runtime.observe_bearing(request.payload["direction"])
            self._tracking_expires = max(self._tracking_expires or 0., request.expires_at)
            self._safe_wait = False
            self._accept(ticket)
            self._finish(ticket, "completed", "completed")

    def _start_clip(self, request: Request) -> None:
        self.runtime.play_primitive(
            request.payload["name"],
            scale=DEFAULT_SCALE * request.payload["intensity"],
            loop=False,  # A remote action is finite, including the idle recording.
        )

    def _apply_one_discrete_command(self, now: float) -> None:
        try:
            request, ticket = self._commands.get_nowait()
        except queue.Empty:
            return
        if now >= request.expires_at:
            self._finish(ticket, "failed", "expired")
            return
        kind, payload = request.type, request.payload
        if kind == "motion.play":
            if payload["name"] not in self.catalog.names():
                self._finish(ticket, "failed", "unknown_motion")
                return
            if (self._active is not None or self.runtime.primitive.busy) and not payload["replace_current"]:
                self._finish(ticket, "failed", "busy")
                return
            self._start_clip(request)
            self._cancel_motion("replaced")
            self._active = (request, ticket)
            self._repeats_left = payload["repeat"] - 1
            self._safe_wait = False
            self._accept(ticket)
            return
        if kind == "motion.cancel":
            if self._active is None or self._active[0].id != payload["request_id"]:
                self._finish(ticket, "failed", "not_found")
                return
            self.runtime.barge_in()
            self._cancel_motion("cancelled")
            self._cancel_task("interrupted")
        elif kind == "motion.interrupt":
            self.runtime.barge_in()
            self._cancel_motion("interrupted")
            self._cancel_task("interrupted")
            self._safe_wait = True
        elif kind == "task_light.place":
            self.runtime.place_task_light(payload["point"])
            self._cancel_task("replaced")
            self._task_ticket = ticket
            self._safe_wait = False
            self._accept(ticket)
            return
        elif kind in {"task_light.cancel", "task_light.clear"}:
            if kind == "task_light.cancel" and (
                self._task_ticket is None or self._task_ticket.request_id != payload["request_id"]
            ):
                self._finish(ticket, "failed", "not_found")
                return
            self.runtime.clear_task_light()
            self._cancel_task("cancelled")
        elif kind == "track.clear":
            self.runtime.clear_tracking()
            self._tracking_expires = None
        elif kind not in {"motion.list", "motion.status", "system.heartbeat"}:
            self._finish(ticket, "failed", "unknown_type")
            return
        data = {}
        if kind == "motion.list":
            data = {"motions": self.catalog.names()}
        elif kind == "motion.status":
            data = asdict(self.snapshot())
        self._accept(ticket)
        self._finish(ticket, "completed", "completed", data=data)

    def _update_active_ticket(self, step: StepState) -> None:
        slow = np.max(np.abs(step.vel)) < self.settling_tolerance
        if self._active is not None:
            if not self.runtime.primitive.busy and self._repeats_left:
                self._start_clip(self._active[0])
                self._repeats_left -= 1
                self._settled_ticks = 0
            else:
                self._settled_ticks = self._settled_ticks + 1 if (
                    not self.runtime.primitive.busy and slow
                ) else 0
                if self._settled_ticks >= 5:
                    self._finish(self._active[1], "completed", "completed")
                    self._active = None
        if self._task_ticket is not None and not self._task_ticket.completed.done():
            self._task_settled_ticks = self._task_settled_ticks + 1 if (
                self.runtime.task_light.env.level >= 1.0 and slow
            ) else 0
            if self._task_settled_ticks >= 5:
                self._finish(self._task_ticket, "completed", "completed")

    def _publish_status(self) -> None:
        busy = self._active is not None or self.runtime.primitive.busy or self.runtime.task_light.busy
        state = "fault" if self._fault else (
            "stopped" if self._stop_event.is_set() else (
                "safe_wait" if self._safe_wait else ("busy" if busy else "idle")
            )
        )
        progress = self.runtime.primitive.progress(self.runtime.t)
        if self._active:
            repeat = self._active[0].payload["repeat"]
            progress = (repeat - self._repeats_left - 1 + (
                progress if self.runtime.primitive.busy else 1.0
            )) / repeat
        with self._lock:
            self._status = ControllerStatus(
                state, self.runtime.primitive.active_name, busy, self._fault,
                self._sent_ticks, self._deadline_misses, progress,
            )

    def _shutdown(self) -> None:
        self._clear_mailbox("failed" if self._fault else "cancelled",
                            "fault" if self._fault else "stopped", self._fault or self._stop_reason)
        self._active = None
        self._task_ticket = None
        self.runtime.clear_tracking()
        self.runtime.barge_in()
        self._publish_status()

    def tick_once(self, *, now: float) -> None:
        self._claim_owner()
        if self._ticking:
            raise RuntimeError("motion owner cannot re-enter a tick")
        self._ticking = True
        try:
            if self._stop_event.is_set():
                self._shutdown()
                return
            if not self._safe_disconnect():
                self._apply_latest_tracking(now)
                self._apply_one_discrete_command(now)
            step = self.runtime.step()
            self._sent_ticks += 1
            self._update_active_ticket(step)
            self._publish_status()
        except Exception as exc:
            with self._lock:
                self._fault = str(exc)
                self._stop_event.set()
            self._shutdown()
        finally:
            self._ticking = False

    def run(self) -> None:
        self._claim_owner()
        try:
            start = self.clock()
            slot = 0
            while not self._stop_event.is_set():
                deadline = start + slot * self.runtime.dt
                remaining = deadline - self.clock()
                while remaining > 0 and not self._stop_event.is_set():
                    self.sleep(remaining)
                    remaining = deadline - self.clock()
                if self._stop_event.is_set():
                    break
                now = self.clock()
                next_slot = math.ceil(
                    (now - start - DEADLINE_JITTER_SECONDS) / self.runtime.dt - 1e-9
                )
                if next_slot > slot:
                    self._deadline_misses += next_slot - slot
                    slot = next_slot
                    self._publish_status()
                    continue
                self.tick_once(now=now)
                slot += 1
        except KeyboardInterrupt:
            self.stop("interrupted")
        except Exception as exc:
            with self._lock:
                self._fault = str(exc)
                self._stop_event.set()
        finally:
            self._shutdown()
