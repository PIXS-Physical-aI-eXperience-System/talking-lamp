"""The S2 launcher waits for the model server and cleans up both processes."""

from pathlib import Path
import os
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "deploy/jetson/run-s2-voice.sh"


def test_s2_voice_runner_starts_node_after_agent_and_stops_agent(tmp_path):
    repo = tmp_path / "repo"
    overlay = repo / "jetson_ws/install/setup.bash"
    overlay.parent.mkdir(parents=True)
    overlay.write_text("")
    ros_setup = tmp_path / "ros-setup.bash"
    ros_setup.write_text("")
    voice = tmp_path / "voice"
    (voice / "bench").mkdir(parents=True)
    (voice / "ros").mkdir()
    (voice / "bench/voice_agent.py").write_text(
        "import argparse, os, pathlib, socket\n"
        "p=argparse.ArgumentParser(); p.add_argument('--port',type=int); "
        "p.add_argument('--llm'); a=p.parse_args()\n"
        "s=socket.socket(); s.bind(('127.0.0.1',a.port)); s.listen()\n"
        "pathlib.Path(os.environ['S2_TEST_DIR'],'agent.pid').write_text(str(os.getpid()))\n"
        "while True: s.accept()\n")
    (voice / "ros/lamp_voice_node.py").write_text(
        "import os, pathlib\n"
        "pathlib.Path(os.environ['S2_TEST_DIR'],'node.started').write_text('yes')\n")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    result = subprocess.run(
        ["bash", str(RUNNER)],
        env={**os.environ, "TALKING_LAMP_REPO": str(repo),
             "TALKING_LAMP_VOICE_ROOT": str(voice),
             "TALKING_LAMP_VOICE_PYTHON": sys.executable,
             "TALKING_LAMP_VOICE_PORT": str(port),
             "ROS_SETUP_FILE": str(ros_setup), "S2_TEST_DIR": str(tmp_path)},
        capture_output=True, text=True, timeout=8,
    )
    assert result.returncode == 1, result.stderr
    assert (tmp_path / "node.started").read_text() == "yes"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def test_s2_voice_service_is_manual_only():
    service = (ROOT / "deploy/jetson/talking-lamp-s2-voice.service").read_text()
    assert "Requires=talking-lamp-bridges.service" in service
    assert "[Install]" not in service
