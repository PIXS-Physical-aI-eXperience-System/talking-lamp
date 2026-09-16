import configparser
import os
from pathlib import Path
import shlex
import stat
import subprocess
import tomllib

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


def test_device_unit_enables_commissioned_pi5_led_rotated_at_eight_percent():
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
    assert argv[-1] == "--enable-led-hardware"
    assert dict(zip(argv[3:-1:2], argv[4:-1:2])) == {
        "--bind": "192.168.100.2", "--port": "8766",
        "--allow-host": "192.168.100.1",
        "--motion-socket": "/run/talking-lamp/motion-control.sock",
        "--calibration": "/home/pixs/talking-lamp/voice-bench/out/doa/calibration.json",
        "--sample-rate": "20", "--gpio-pin": "12",
        "--led-rotation": "180",
        "--xvf-vid": "0x2886", "--xvf-pid": "0x0022",
        "--max-brightness": "0.08",
        "--alsa-card": "L16K6Ch", "--capture-port": "5004",
        "--playback-port": "5006", "--jitter-ms": "40",
    }


def test_audio_diagnostic_is_read_only_and_checks_required_elements():
    script = (ROOT / "deploy/pi/check-audio.sh").read_text()
    assert "gst-inspect-1.0" in script
    assert "L16K6Ch" in script
    assert "2886:0022" in script
    assert "dfu-util" not in script
    assert "flash" not in script.lower()


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
    pio_rule = destination / "etc/udev/rules.d/99-talking-lamp-pio.rules"
    assert pio_rule.read_text() == 'SUBSYSTEM=="*-pio", GROUP="gpio", MODE="0660"\n'


def test_pi5_extra_does_not_install_the_unsupported_legacy_ws281x_backend():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["optional-dependencies"]["pi-device"] == ["pyusb"]


def test_live_installer_preflights_pi5_pio_as_the_service_user():
    script = INSTALLER.read_text()
    assert 'usermod -a -G "$service_group,gpio" "$service_user"' in script
    assert 'test -r /dev/pio0 -a -w /dev/pio0' in script
    assert 'import adafruit_raspberry_pi5_neopixel_write' in script


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
