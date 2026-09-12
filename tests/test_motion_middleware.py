"""Real loopback TCP and controller coverage without an asyncio pytest plugin."""
import asyncio
from contextlib import asynccontextmanager, suppress
import json
import time
from uuid import uuid4

import pytest

from motion.catalog import MotionCatalog
from motion.config import RECORDINGS_DIR
from motion.controller import MotionController
from motion.idle import IdleConfig
from motion.middleware_server import MotionTcpServer
from motion.protocol import MAX_LINE_BYTES, encode_message
from motion.remote_client import MotionClient, build_parser, main
from motion.runtime import MotionRuntime

TOKEN = "test-secret"
PLAY = dict(name="nod", replace_current=True, intensity=0., repeat=1)


def envelope(kind="motion.status", *, ident=None, token=TOKEN, payload=None):
    return encode_message(dict(version=1, id=ident or str(uuid4()), type=kind,
                               ttl_ms=1000, token=token, payload=payload or {}))


async def eventually(predicate, timeout=2):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.005)


async def receive(reader):
    return json.loads(await asyncio.wait_for(reader.readline(), 2))


@asynccontextmanager
async def running_server(**options):
    catalog = MotionCatalog.load(RECORDINGS_DIR / "catalog.toml")
    controller = MotionController(MotionRuntime(primitives=catalog.library(),
                                  idle_cfg=IdleConfig(enabled=False)), catalog)
    disconnects = []
    disconnect = controller.remote_disconnected
    def record_disconnect():
        disconnects.append(True)
        disconnect()
    controller.remote_disconnected = record_disconnect
    server = MotionTcpServer(controller, token=TOKEN, host="127.0.0.1", port=0, **options)
    await server.start()
    async def tick():
        while True:
            for _ in range(20):
                controller.tick_once(now=time.monotonic())
            await asyncio.sleep(.001)
    ticker = asyncio.create_task(tick())
    try:
        yield server, server.sockets[0].getsockname()[1], disconnects
    finally:
        await server.close()
        ticker.cancel()
        with suppress(asyncio.CancelledError):
            await ticker


def test_play_ack_terminal_and_concurrent_status():
    async def scenario():
        async with running_server() as (server, port, _):
            async with MotionClient("127.0.0.1", port, token=TOKEN) as client:
                playing = asyncio.create_task(client.request("motion.play", PLAY))
                await eventually(lambda: server.controller.snapshot().busy)
                status = await client.request("motion.status", {})
                assert status[0]["state"] == "accepted"
                assert status[-1]["data"]["busy"] is True
                events = await playing
                assert [e["state"] for e in events] == ["accepted", "completed"]
                assert events[0]["id"] == events[1]["id"]
    asyncio.run(scenario())


def test_disconnect_invokes_safe_wait_exactly_once():
    async def scenario():
        async with running_server() as (server, port, disconnects):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(envelope("motion.play", payload=PLAY))
            await writer.drain()
            assert (await receive(reader))["state"] == "accepted"
            writer.close()
            await writer.wait_closed()
            await eventually(lambda: server.controller.snapshot().state == "safe_wait")
            await server.close()
            assert len(disconnects) == 1
    asyncio.run(scenario())


def test_duplicate_request_id_reuses_completed_controller_ticket():
    async def scenario():
        async with running_server() as (server, port, _):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            line = envelope("motion.play", payload=PLAY)
            writer.write(line)
            first = [await receive(reader), await receive(reader)]
            started_at = server.controller.runtime.primitive.started_at
            writer.write(line)
            assert [await receive(reader), await receive(reader)] == first
            assert server.controller.runtime.primitive.started_at == started_at
            writer.close()
            await writer.wait_closed()
    asyncio.run(scenario())


@pytest.mark.parametrize("line, code", [(envelope(token="wrong"), "unauthorized"),
    (b"x" * (MAX_LINE_BYTES + 2) + b"\n", "line_too_long"),
    (b"x" * (MAX_LINE_BYTES + 2), "line_too_long")])
def test_invalid_connections_are_rejected_without_claiming_controller(line, code):
    async def scenario():
        async with running_server() as (_, port, disconnects):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(line)
            await writer.drain()
            assert (await receive(reader))["code"] == code
            assert await asyncio.wait_for(reader.read(), 1) == b""
            writer.close()
            await writer.wait_closed()
            assert not disconnects
    asyncio.run(scenario())


def test_second_authenticated_connection_cannot_disrupt_first():
    async def scenario():
        async with running_server() as (_, port, disconnects):
            async with MotionClient("127.0.0.1", port, token=TOKEN) as first:
                await first.request("motion.status", {})
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(envelope())
                assert (await receive(reader))["code"] == "controller_connected"
                assert await asyncio.wait_for(reader.read(), 1) == b""
                writer.close()
                await writer.wait_closed()
                assert not disconnects
                assert (await first.request("motion.list", {}))[-1]["data"]["motions"]
    asyncio.run(scenario())


def test_heartbeat_timeout_and_client_keepalive():
    async def scenario():
        async with running_server(heartbeat_timeout=.12) as (server, port, disconnects):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(envelope())
            await receive(reader)
            await receive(reader)
            await eventually(lambda: len(disconnects) == 1)
            assert await reader.read() == b""
            writer.close()
            await writer.wait_closed()
            async with MotionClient("127.0.0.1", port, token=TOKEN, heartbeat_interval=.025) as client:
                await client.request("motion.status", {})
                await asyncio.sleep(.25)
                assert len(disconnects) == 1
                assert (await client.request("motion.status", {}))[-1]["state"] == "completed"
    asyncio.run(scenario())


def test_disconnect_fails_inflight_request_reconnects_without_replay():
    async def scenario():
        plays = []
        connections = []
        async def peer(reader, writer):
            connections.append(writer)
            try:
                while line := await reader.readline():
                    req = json.loads(line)
                    if req["type"] == "motion.play":
                        plays.append(req["id"])
                        writer.close()
                        return
                    for state in ("accepted", "completed"):
                        writer.write(encode_message(dict(id=req["id"], state=state, code=state)))
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
        server = await asyncio.start_server(peer, "127.0.0.1", 0)
        try:
            async with MotionClient("127.0.0.1", server.sockets[0].getsockname()[1], token=TOKEN) as client:
                with pytest.raises(ConnectionError):
                    await client.request("motion.play", PLAY)
                await eventually(lambda: len(connections) >= 2)
                assert (await client.request("motion.status", {}))[-1]["state"] == "completed"
                assert len(plays) == 1
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(scenario())


@pytest.mark.parametrize("args", [["play", "nod", "--intensity", "nan"],
    ["play", "nod", "--repeat", "4"], ["track-point", "3", "0", "0"],
    ["--token", "secret", "list"], ["--port", "0", "status"]])
def test_cli_rejects_invalid_arguments(args):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(args)
    assert exc.value.code == 2


def test_cli_requires_environment_token(monkeypatch, capsys):
    monkeypatch.delenv("TALKING_LAMP_TOKEN", raising=False)
    assert main(["list"]) == 2
    assert "TALKING_LAMP_TOKEN" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["list"], ["status"], ["play", "nod"], ["interrupt"],
                                  ["track-point", ".2", "0", ".3"], ["task-light", ".2", "0", ".3"]])
def test_cli_accepts_subcommands(args):
    assert build_parser().parse_args(args).command == args[0]


def test_cli_imports_without_third_party_dependencies():
    from pathlib import Path
    import subprocess
    import sys
    source = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([sys.executable, "-S", "-c",
        f"import sys; sys.path.insert(0, {source!r}); "
        "from motion.remote_client import build_parser; build_parser().parse_args(['status'])"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_malformed_response_fails_request_and_reconnects():
    async def scenario():
        connections = []
        async def peer(reader, writer):
            connections.append(True)
            try:
                request = json.loads(await reader.readline())
                writer.write(encode_message(dict(id=request["id"], state=[])))
                await writer.drain()
                await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()
        server = await asyncio.start_server(peer, "127.0.0.1", 0)
        try:
            async with MotionClient("127.0.0.1", server.sockets[0].getsockname()[1], token=TOKEN) as client:
                with pytest.raises(ConnectionError):
                    await client.request("motion.status", {})
                await eventually(lambda: len(connections) >= 2)
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(scenario())


def test_allowlist_rejects_peer_before_controller_submission():
    async def scenario():
        async with running_server(allowed_hosts={"192.0.2.1"}) as (_, port, disconnects):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            assert (await receive(reader))["code"] == "host_not_allowed"
            assert await reader.read() == b""
            assert not disconnects
            writer.close()
            await writer.wait_closed()
    asyncio.run(scenario())


def test_repeated_parse_errors_close_authenticated_connection():
    async def scenario():
        async with running_server() as (_, port, disconnects):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(envelope())
            await receive(reader)
            await receive(reader)
            writer.write(b"{\n" * 3)
            for _ in range(3):
                assert (await receive(reader))["code"] == "invalid_json"
            assert await reader.read() == b""
            assert len(disconnects) == 1
            writer.close()
            await writer.wait_closed()
    asyncio.run(scenario())


def test_transport_close_does_not_cancel_shared_controller_futures():
    from motion.controller import CommandTicket
    async def scenario():
        tickets = []
        class Mailbox:
            def submit(self, request):
                ticket = CommandTicket(request.id)
                tickets.append(ticket)
                return ticket
            def remote_disconnected(self):
                pass
        server = MotionTcpServer(Mailbox(), token=TOKEN, port=0)
        await server.start()
        reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
        try:
            writer.write(envelope())
            await eventually(lambda: len(tickets) == 1)
            # An unresolved accepted future cannot block subsequent reads.
            writer.write(envelope("motion.interrupt"))
            await eventually(lambda: len(tickets) == 2)
            await server.close()
            assert all(not t.accepted.cancelled() and not t.completed.cancelled() for t in tickets)
        finally:
            writer.close()
            await writer.wait_closed()
            await server.close()
    asyncio.run(scenario())


def test_lazy_package_exports_preserve_public_api():
    import motion
    from importlib import import_module
    modules = ("config", "blender", "idle", "ik", "kalman", "kinematics", "layers",
               "primitives", "runtime", "trajectory")
    for name in motion.__all__:
        expected = next(getattr(import_module(f"motion.{module}"), name) for module in modules
                        if hasattr(import_module(f"motion.{module}"), name))
        assert getattr(motion, name) is expected
    assert set(motion.__all__) <= set(dir(motion))
    with pytest.raises(AttributeError):
        motion.missing_export


def test_reconnect_backoff_is_capped(monkeypatch):
    from motion import remote_client
    delays = []
    async def unavailable(*args, **kwargs):
        raise ConnectionRefusedError("test unavailable")
    async def record_delay(delay):
        delays.append(delay)
        if len(delays) == 7:
            raise asyncio.CancelledError
    monkeypatch.setattr(remote_client.asyncio, "open_connection", unavailable)
    monkeypatch.setattr(remote_client.asyncio, "sleep", record_delay)
    async def scenario():
        client = MotionClient("127.0.0.1", token=TOKEN)
        with pytest.raises(asyncio.CancelledError):
            await client._run()
    asyncio.run(scenario())
    assert delays == [.25, .5, 1., 2., 5., 5., 5.]


def test_cli_list_runs_over_tcp(monkeypatch, capsys):
    monkeypatch.setenv("TALKING_LAMP_TOKEN", TOKEN)
    async def scenario():
        async with running_server() as (_, port, _):
            return await asyncio.to_thread(main, ["--host", "127.0.0.1", "--port", str(port), "list"])
    assert asyncio.run(scenario()) == 0
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [message["state"] for message in messages] == ["accepted", "completed"]
    assert "nod" in messages[-1]["data"]["motions"]


def test_controller_rejection_is_single_terminal_event():
    async def scenario():
        async with running_server() as (_, port, _):
            async with MotionClient("127.0.0.1", port, token=TOKEN) as client:
                events = await client.request("motion.play", {**PLAY, "name": "missing_motion"})
                assert len(events) == 1
                assert events[0]["state"] == "failed"
                assert events[0]["code"] == "unknown_motion"
    asyncio.run(scenario())


def test_invalid_catalog_prevents_backend_construction(monkeypatch, tmp_path, capsys):
    from motion import hardware_backend, middleware_server as server
    opened = []
    monkeypatch.setenv("TALKING_LAMP_TOKEN", TOKEN)
    monkeypatch.setattr(hardware_backend, "FeetechBackend", lambda **kw: opened.append(kw))
    catalog = tmp_path / "catalog.toml"
    catalog.write_text('[motions.bad]\nfile="missing.csv"\nenabled=true\n')
    assert server.main(["--catalog", str(catalog), "--port", "/dev/ttyACM0",
                        "--lamp-id", "lelamp"]) == 2
    assert not opened
    assert "missing.csv" in capsys.readouterr().err


def test_daemon_requires_token_before_catalog_or_hardware(monkeypatch, capsys):
    from motion import middleware_server as server
    monkeypatch.delenv("TALKING_LAMP_TOKEN", raising=False)
    assert server.main(["--catalog", "/missing/catalog.toml", "--null-backend"]) == 2
    assert "TALKING_LAMP_TOKEN" in capsys.readouterr().err


def test_daemon_requires_ruckig_before_hardware(monkeypatch, capsys):
    from motion import hardware_backend, middleware_server as server, trajectory
    opened = []
    monkeypatch.setenv("TALKING_LAMP_TOKEN", TOKEN)
    monkeypatch.setattr(trajectory, "_HAVE_RUCKIG", False)
    monkeypatch.setattr(hardware_backend, "FeetechBackend", lambda **kw: opened.append(kw))
    assert server.main(["--port", "/dev/ttyACM0", "--lamp-id", "lelamp"]) == 2
    assert not opened
    assert "Ruckig" in capsys.readouterr().err


@pytest.mark.parametrize("args", [[], ["--null-backend", "--port", "/dev/ttyACM0"],
    ["--null-backend", "--lamp-id", "lelamp"], ["--port", "/dev/ttyACM0"],
    ["--null-backend", "--tcp-port", "0"], ["--null-backend", "--tcp-port", "65536"],
    ["--null-backend", "--heartbeat-timeout", "nan"],
    ["--port", "/dev/ttyACM0", "--lamp-id", "lelamp", "--feedback-hz", "101"]])
def test_daemon_rejects_ambiguous_or_invalid_config(args):
    from motion import middleware_server as server
    with pytest.raises(SystemExit) as exc:
        server.main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("failure", ["runtime", "bind", "controller", "normal", "park"])
def test_daemon_closes_backend_after_owner_stops(monkeypatch, failure, capsys):
    from motion import hardware_backend, middleware_server as server
    from motion.runtime import NullBackend
    events = []
    class Backend(NullBackend):
        def __init__(self, **kwargs):
            super().__init__()
            assert kwargs == dict(port="/dev/ttyACM0", lamp_id="lelamp", feedback_hz=20.)
        def close(self):
            events.append("park")
            if failure == "park":
                raise RuntimeError("park failed")
    runtime_type = MotionRuntime
    def runtime(**kwargs):
        if failure == "runtime":
            raise RuntimeError("runtime failed")
        return runtime_type(**kwargs)
    def run(controller):
        events.append("owner started")
        if failure == "controller":
            controller.runtime.step = lambda: (_ for _ in ()).throw(RuntimeError("motor failed"))
            controller.tick_once(now=time.monotonic())
        else:
            controller._stop_event.wait(2)
        events.append("owner stopped")
    async def serve(tcp):
        await eventually(lambda: "owner started" in events)
        if failure == "bind":
            raise OSError("bind failed")
        if failure == "controller":
            await asyncio.Future()
    async def close(tcp):
        events.append("transport closed")
    monkeypatch.setenv("TALKING_LAMP_TOKEN", TOKEN)
    monkeypatch.setattr(hardware_backend, "FeetechBackend", Backend)
    monkeypatch.setattr(server, "MotionRuntime", runtime, raising=False)
    monkeypatch.setattr(MotionController, "run", run)
    monkeypatch.setattr(MotionTcpServer, "serve_forever", serve)
    monkeypatch.setattr(MotionTcpServer, "close", close)
    assert server.main(["--port", "/dev/ttyACM0", "--lamp-id", "lelamp"]) == (0 if failure == "normal" else 1)
    assert events[-1] == "park"
    assert events.count("park") == 1
    if failure != "runtime":
        assert events.index("owner stopped") < events.index("park")
        assert events.index("transport closed") < events.index("park")
    if failure != "normal":
        assert "failed" in capsys.readouterr().err


def test_null_backend_close_is_idempotent():
    from motion.runtime import NullBackend
    backend = NullBackend()
    backend.close()
    backend.close()


def test_daemon_reports_fault_that_occurs_while_stopping(monkeypatch, capsys):
    from motion import middleware_server as server
    monkeypatch.setenv("TALKING_LAMP_TOKEN", TOKEN)
    def run(controller):
        controller._stop_event.wait(2)
        controller.runtime.step = lambda: (_ for _ in ()).throw(RuntimeError("late motor failed"))
        controller._stop_event.clear()
        controller.tick_once(now=time.monotonic())
    async def serve(tcp):
        return
    monkeypatch.setattr(MotionController, "run", run)
    monkeypatch.setattr(MotionTcpServer, "serve_forever", serve)
    assert server.main(["--null-backend"]) == 1
    assert "late motor failed" in capsys.readouterr().err


@pytest.mark.parametrize("signum", ["SIGINT", "SIGTERM"])
def test_signal_during_runtime_startup_parks_without_sending(monkeypatch, signum):
    import signal
    from motion import hardware_backend, middleware_server as server
    from motion.runtime import NullBackend
    events = []
    shutdown_signal = getattr(signal, signum)
    previous = signal.getsignal(shutdown_signal)
    class Backend(NullBackend):
        def __init__(self, **kwargs):
            super().__init__()
        def send(self, command):
            events.append("sent")
            super().send(command)
        def close(self):
            events.append("parked")
            # Repeated signals cannot interrupt parking or cause a second close.
            signal.raise_signal(shutdown_signal)
    def runtime(**kwargs):
        result = MotionRuntime(**kwargs)
        signal.raise_signal(shutdown_signal)
        return result
    monkeypatch.setenv("TALKING_LAMP_TOKEN", TOKEN)
    monkeypatch.setattr(hardware_backend, "FeetechBackend", Backend)
    monkeypatch.setattr(server, "MotionRuntime", runtime)
    assert server.main(["--port", "/dev/ttyACM0", "--lamp-id", "lelamp"]) == 0
    assert events == ["parked"]
    assert signal.getsignal(shutdown_signal) == previous


@pytest.mark.parametrize("signum", ["SIGINT", "SIGTERM"])
def test_null_daemon_roundtrip_signal_exit_and_no_hardware_import(tmp_path, signum):
    import os
    from pathlib import Path
    import signal
    import socket
    import subprocess
    import sys
    # A fresh process forbids even indirect imports of the physical backend.
    guard = tmp_path / "sitecustomize.py"
    guard.write_text('''import sys
class Guard:
    def find_spec(self, fullname, *args):
        if fullname in {"motion.hardware_backend", "motion.hardware_run", "serial"} or fullname.startswith("lelamp."):
            raise AssertionError("hardware import in null mode: " + fullname)
sys.meta_path.insert(0, Guard())
def audit(event, args):
    if event == "open" and args[0] == "/dev/ttyACM0":
        raise AssertionError("serial device opened in null mode")
sys.addaudithook(audit)
''')
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "TALKING_LAMP_TOKEN": TOKEN,
           "PYTHONPATH": f"{tmp_path}:{Path(__file__).resolve().parents[1] / 'src'}"}
    process = subprocess.Popen([sys.executable, "-m", "motion.middleware_server", "--null-backend",
        "--bind", "127.0.0.1", "--tcp-port", str(port)], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.2) as connection:
                    connection.settimeout(2)
                    connection.sendall(envelope("motion.list"))
                    with connection.makefile("rb") as stream:
                        assert json.loads(stream.readline())["state"] == "accepted"
                        assert "nod" in json.loads(stream.readline())["data"]["motions"]
                process.send_signal(getattr(signal, signum))
                break
            except ConnectionRefusedError:
                time.sleep(.02)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)
        assert "hardware import" not in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
