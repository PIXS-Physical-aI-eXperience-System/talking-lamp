"""Execute the operator's one-shot Python diagnostics against a persistent peer."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import socket
import subprocess
import sys

import pytest


@pytest.mark.parametrize("kind", ["orientation.status", "system.heartbeat"])
@pytest.mark.parametrize("terminal", [
    {"state": "completed", "code": "completed"},
    {"state": "failed", "code": "expired"},
    {"state": "failed", "code": "fault"},
])
def test_one_shot_diagnostic_exits_at_first_terminal_event(tmp_path, kind, terminal):
    """Waiting for EOF after the terminal reply hangs on the persistent Unix server."""
    guide = (Path(__file__).resolve().parents[1] / "docs/base-yaw-orientation.md").read_text()
    example = next(block for block in re.findall(r"```python\n(.*?)```", guide, re.S)
                   if f'"type": "{kind}"' in block)
    path = str(tmp_path / "diagnostic.sock")
    example = example.replace("/run/talking-lamp/motion-control.sock", path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(path)
        listener.listen(1)
        listener.settimeout(3.)

        def serve():
            with listener.accept()[0] as peer:
                peer.settimeout(3.)
                with peer.makefile("rb") as reader:
                    request = json.loads(reader.readline())
                    events = []
                    if terminal["state"] == "completed":
                        events.append(dict(id=request["id"], state="accepted", code="accepted"))
                    events.append(dict(id=request["id"], **terminal))
                    peer.sendall(b"".join(json.dumps(event).encode() + b"\n" for event in events))
                    # The real server waits for the next request here. The
                    # one-shot diagnostic must close first, after its result.
                    closed = reader.readline() == b""
                    return request, events, closed

        with ThreadPoolExecutor(max_workers=1) as pool:
            served = pool.submit(serve)
            result = subprocess.run([sys.executable, "-"], input=example, text=True,
                                    capture_output=True, timeout=1.)
            request, events, closed = served.result(timeout=3.)
    assert request["type"] == kind
    assert result.returncode == 0, result.stderr
    # Python print(dict) is intentionally human-readable rather than NDJSON.
    import ast
    assert [ast.literal_eval(line) for line in result.stdout.splitlines()] == events
    assert closed
