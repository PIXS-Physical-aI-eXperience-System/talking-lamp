from concurrent.futures import Future
import importlib


def test_cancelable_wait_sends_correlated_cancel_before_returning_cancelled():
    action_wait = importlib.import_module("lamp_device_bridge.action_wait")
    terminal = Future()
    calls = []

    def cancel():
        calls.append("cancel")
        terminal.set_result({"state": "cancelled", "code": "cancelled"})

    result = action_wait.wait_cancelable(
        terminal,
        cancel_requested=lambda: True,
        cancel=cancel,
        timeout=1.0,
        poll_interval=0.001,
    )

    assert calls == ["cancel"]
    assert result.cancelled is True
    assert result.terminal == {"state": "cancelled", "code": "cancelled"}
