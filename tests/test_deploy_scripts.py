"""Installer behavior in a staged filesystem; never mutate the host systemd."""
import configparser
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import stat
import subprocess
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "deploy/pi/install-motion-service.sh"
UNIT = ROOT / "deploy/pi/talking-lamp-motion.service"


@dataclass(frozen=True)
class RosSections:
    goal: list[str]
    result: list[str]
    feedback: list[str]


def read_ros_sections(relative_path):
    """Read a ROS action definition into its goal, result, and feedback sections."""
    sections = (ROOT / relative_path).read_text().split("---")
    assert len(sections) == 3
    return RosSections(*[
        [line.strip() for line in section.splitlines()
         if line.strip() and not line.lstrip().startswith("#")]
        for section in sections
    ])


def read_ros_fields(relative_path):
    return [line.strip() for line in (ROOT / relative_path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def read_ros_service_sections(relative_path):
    request, response = (ROOT / relative_path).read_text().split("---")
    return (
        [line.strip() for line in request.splitlines() if line.strip() and not line.lstrip().startswith("#")],
        [line.strip() for line in response.splitlines() if line.strip() and not line.lstrip().startswith("#")],
    )


def test_play_motion_action_is_generic_and_cancellable():
    sections = read_ros_sections("jetson_ws/src/lamp_interfaces/action/PlayMotion.action")
    assert sections.goal == ["string name", "bool replace_current", "float32 intensity", "uint32 repeat"]
    assert sections.result == ["bool success", "string code", "string message"]
    assert sections.feedback == ["string state", "float32 progress"]


def test_task_light_action_reports_target_progress_and_result():
    sections = read_ros_sections("jetson_ws/src/lamp_interfaces/action/PlaceTaskLight.action")
    assert sections.goal == ["geometry_msgs/PointStamped target"]
    assert sections.result == ["bool success", "string code", "string message"]
    assert sections.feedback == ["string state", "float32 progress"]


def test_motion_status_exposes_bridge_health_and_control_counters():
    assert read_ros_fields("jetson_ws/src/lamp_interfaces/msg/MotionStatus.msg") == [
        "builtin_interfaces/Time stamp", "bool connected", "string state", "string active_motion",
        "bool busy", "string fault", "uint64 sent_ticks", "uint64 deadline_misses",
    ]


def test_motion_services_list_catalog_and_interrupt_active_work():
    assert read_ros_service_sections("jetson_ws/src/lamp_interfaces/srv/ListMotions.srv") == (
        [], ["string[] motions"])
    assert read_ros_service_sections("jetson_ws/src/lamp_interfaces/srv/InterruptMotion.srv") == (
        [], ["bool success", "string code", "string message"])


def test_interface_package_exports_ament_rosidl_metadata():
    package = ET.parse(ROOT / "jetson_ws/src/lamp_interfaces/package.xml").getroot()
    assert package.findtext("export/build_type") == "ament_cmake"
    assert {element.text for element in package.findall("buildtool_depend")} >= {
        "ament_cmake", "rosidl_default_generators"}
    assert {element.text for element in package.findall("depend")} >= {
        "builtin_interfaces", "geometry_msgs"}
    assert package.findtext("exec_depend") == "rosidl_default_runtime"
    assert package.findtext("member_of_group") == "rosidl_interface_packages"


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    for name in ("src/motion/middleware_server.py", "lelamp_runtime/lelamp/recordings/catalog.toml"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    python = repo / "lelamp_runtime/.venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to("/usr/bin/python3")
    return repo


def install(repo, dest, *args):
    # Any accidental live-system command is visible and fails immediately.
    commands = dest.parent / "commands"
    commands.mkdir(exist_ok=True)
    forbidden = commands / "systemctl"
    forbidden.write_text("#!/bin/sh\necho 'unexpected live systemctl' >&2\nexit 99\n")
    forbidden.chmod(0o755)
    return subprocess.run(["bash", str(INSTALLER), "--repo", str(repo), "--destdir", str(dest), *args],
        env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"}, capture_output=True, text=True)


def parse_unit(path):
    unit = configparser.ConfigParser(interpolation=None, strict=False)
    unit.read(path)
    return unit


def test_pi_unit_has_safe_runtime_and_shutdown_contract():
    unit = parse_unit(UNIT)
    service = unit["Service"]
    assert service["User"] == "pixs"
    assert service["WorkingDirectory"] == "/home/pixs/talking-lamp"
    assert service["EnvironmentFile"] == "/etc/talking-lamp/motion.env"
    argv = shlex.split(service["ExecStart"])
    assert argv[:3] == ["/home/pixs/talking-lamp/lelamp_runtime/.venv/bin/python", "-m", "motion.middleware_server"]
    assert dict(zip(argv[3::2], argv[4::2])) == {
        "--bind": "192.168.100.2", "--tcp-port": "8765", "--port": "/dev/ttyACM0",
        "--lamp-id": "lelamp", "--allow-host": "192.168.100.1"}
    assert service["KillSignal"] == "SIGTERM"
    assert int(service["TimeoutStopSec"]) == 20
    assert service["Restart"] == "on-failure"
    assert int(service["RestartSec"]) == 3
    assert int(unit["Unit"]["StartLimitIntervalSec"]) == 60
    assert int(unit["Unit"]["StartLimitBurst"]) == 3


def test_installer_dry_run_is_repeatable_without_writes(repository, tmp_path):
    destination = tmp_path / "stage"
    first = install(repository, destination, "--dry-run")
    second = install(repository, destination, "--dry-run")
    assert first.returncode == second.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    assert "disabled" in first.stdout
    assert not destination.exists()


def test_staged_installer_preserves_token_and_defaults_disabled(repository, tmp_path):
    destination = tmp_path / "stage"
    first = install(repository, destination)
    assert first.returncode == 0, first.stderr
    token_file = destination / "etc/talking-lamp/motion.env"
    token = token_file.read_text()
    assert token.startswith("TALKING_LAMP_TOKEN=")
    assert len(token.strip().split("=", 1)[1]) == 64
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    token_file.write_text("TALKING_LAMP_TOKEN=operator-existing-token\n")
    second = install(repository, destination)
    assert second.returncode == 0, second.stderr
    assert token_file.read_text() == "TALKING_LAMP_TOKEN=operator-existing-token\n"
    unit_path = destination / "etc/systemd/system/talking-lamp-motion.service"
    assert parse_unit(unit_path)["Service"]["WorkingDirectory"] == str(repository)
    assert stat.S_IMODE(unit_path.stat().st_mode) == 0o644
    assert not (destination / "etc/systemd/system/multi-user.target.wants").exists()


def test_staged_enable_requires_explicit_flag(repository, tmp_path):
    destination = tmp_path / "stage"
    result = install(repository, destination, "--enable")
    assert result.returncode == 0, result.stderr
    enabled = destination / "etc/systemd/system/multi-user.target.wants/talking-lamp-motion.service"
    assert enabled.is_symlink()
    assert enabled.resolve() == destination / "etc/systemd/system/talking-lamp-motion.service"


def test_installer_rejects_missing_interpreter_before_writes(repository, tmp_path):
    (repository / "lelamp_runtime/.venv/bin/python").unlink()
    destination = tmp_path / "stage"
    result = install(repository, destination)
    assert result.returncode != 0
    assert "python" in result.stderr
    assert not destination.exists()


def test_installer_preserves_invalid_existing_env_and_fails(repository, tmp_path):
    destination = tmp_path / "stage"
    token_file = destination / "etc/talking-lamp/motion.env"
    token_file.parent.mkdir(parents=True)
    token_file.write_text("TALKING_LAMP_TOKEN=\n")
    result = install(repository, destination)
    assert result.returncode != 0
    assert "non-empty TALKING_LAMP_TOKEN" in result.stderr
    assert token_file.read_text() == "TALKING_LAMP_TOKEN=\n"
    assert not (destination / "etc/systemd/system/talking-lamp-motion.service").exists()
