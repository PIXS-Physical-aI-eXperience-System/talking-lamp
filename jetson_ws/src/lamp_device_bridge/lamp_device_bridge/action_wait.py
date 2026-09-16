"""Wait for a transport terminal while forwarding ROS action cancellation."""

from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass
import math
import time
from typing import Callable


@dataclass(frozen=True)
class CancelableResult:
    terminal: dict[str, object]
    cancelled: bool


def wait_cancelable(
    future: Future,
    *,
    cancel_requested: Callable[[], bool],
    cancel: Callable[[], object],
    timeout: float,
    poll_interval: float = 0.05,
) -> CancelableResult:
    if (
        not callable(cancel_requested) or not callable(cancel)
        or not math.isfinite(timeout) or timeout <= 0
        or not math.isfinite(poll_interval) or poll_interval <= 0
    ):
        raise ValueError("cancel wait configuration is invalid")
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("action request timed out")
        try:
            terminal = future.result(timeout=min(poll_interval, remaining))
        except FutureTimeout:
            if not cancel_requested():
                continue
            cancel()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("cancel terminal timed out")
            terminal = future.result(timeout=remaining)
            return CancelableResult(terminal, True)
        return CancelableResult(terminal, False)
