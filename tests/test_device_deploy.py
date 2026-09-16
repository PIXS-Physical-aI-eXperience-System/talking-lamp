import configparser
import os
from pathlib import Path
import shlex
import stat
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "deploy/pi/install-device-service.sh"
UNIT = ROOT / "deploy/pi/talking-lamp-device.service"


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    for name in (
        "src/device/daemon.py",
        "voice-bench/out/doa/calibration.json",
    ):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    python = repo / "lelamp_runtime/.venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to("/usr/bin/python3")
    return repo


def install(repo, dest, *args):
    commands = dest.parent / "commands"
    commands.mkdir(exist_ok=True)
    forbidden = commands / "systemctl"
    forbidden.write_text("#!/bin/sh\necho unexpected-systemctl >&2\nexit 99\n")
    forbidden.chmod(0o755)
    return subprocess.run(
        ["bash", str(INSTALLER), "--repo", str(repo), "--destdir", str(dest), *args],
        env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        capture_output=True, text=True,
    )


def parse_unit(path):
    unit = configparser.ConfigParser(interpolation=None, strict=False)
    unit.read(path)
    return unit


def test_device_unit_is_independent_safe_and_has_no_led_hardware_flag():
    unit = parse_unit(UNIT)
    assert unit["Service"]["User"] == "pixs"
    assert unit["Service"]["Group"] == "talking-lamp"
    assert unit["Service"]["UMask"] == "0007"
    assert unit["Service"]["Restart"] == "on-failure"
    assert "talking-lamp-motion.service" in unit["Unit"]["After"]
    assert "Requires" not in unit["Unit"]
    argv = shlex.split(unit["Service"]["ExecStart"])
    assert argv[:3] == [
        "/home/pixs/talking-lamp/lelamp_runtime/.venv/bin/python", "-m", "device.daemon"]
    assert "--enable-led-hardware" not in argv
    assert dict(zip(argv[3::2], argv[4::2])) == {
        "--bind": "192.168.100.2", "--port": "8766",
        "--allow-host": "192.168.100.1",
        "--motion-socket": "/run/talking-lamp/motion-control.sock",
        "--calibration": "/home/pixs/talking-lamp/voice-bench/out/doa/calibration.json",
        "--sample-rate": "20", "--gpio-pin": "12",
        "--xvf-vid": "0x2886", "--xvf-pid": "0x0022",
        "--max-brightness": "0.10",
    }


def test_staged_installer_preserves_secret_and_defaults_disabled(repository, tmp_path):
    destination = tmp_path / "stage"
    first = install(repository, destination)
    assert first.returncode == 0, first.stderr
    env = destination / "etc/talking-lamp/device.env"
    original = env.read_text()
    assert original.startswith("TALKING_LAMP_TOKEN=")
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    env.write_text("TALKING_LAMP_TOKEN=existing-device-token\n")
    second = install(repository, destination)
    assert second.returncode == 0, second.stderr
    assert env.read_text() == "TALKING_LAMP_TOKEN=existing-device-token\n"
    assert not (destination / "etc/systemd/system/multi-user.target.wants").exists()


def test_staged_enable_is_explicit_and_dry_run_writes_nothing(repository, tmp_path):
    destination = tmp_path / "stage"
    dry = install(repository, destination, "--dry-run")
    assert dry.returncode == 0
    assert "disabled" in dry.stdout
    assert not destination.exists()

    enabled = install(repository, destination, "--enable")
    assert enabled.returncode == 0, enabled.stderr
    link = destination / "etc/systemd/system/multi-user.target.wants/talking-lamp-device.service"
    assert link.is_symlink()
    assert link.resolve() == destination / "etc/systemd/system/talking-lamp-device.service"
