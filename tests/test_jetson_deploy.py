import configparser
import os
from pathlib import Path
import stat
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DETECT = ROOT / "deploy/jetson/detect-platform.sh"
INSTALL = ROOT / "deploy/jetson/install-ros-bridges.sh"
NORMALIZE_APT = ROOT / "deploy/jetson/normalize-apt-sources.sh"
UNIT = ROOT / "deploy/jetson/talking-lamp-bridges.service"
RUN_BRIDGES = ROOT / "deploy/jetson/run-bridges.sh"


def test_platform_detection_maps_actual_l4t39_ubuntu24_to_jazzy(tmp_path):
    root = tmp_path / "root"
    (root / "etc").mkdir(parents=True)
    (root / "etc/os-release").write_text('ID=ubuntu\nVERSION_ID="24.04"\n')
    (root / "etc/nv_tegra_release").write_text("# R39 (release), REVISION: 2.0\n")
    result = subprocess.run(
        ["bash", str(DETECT)], env={**os.environ, "PLATFORM_ROOT": str(root)},
        capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == "jazzy"

    (root / "etc/nv_tegra_release").write_text("# R36 (release), REVISION: 4.0\n")
    rejected = subprocess.run(
        ["bash", str(DETECT)], env={**os.environ, "PLATFORM_ROOT": str(root)},
        capture_output=True, text=True)
    assert rejected.returncode != 0


def test_bridge_unit_uses_ros_jazzy_wired_config_and_two_protected_env_files():
    unit = configparser.ConfigParser(interpolation=None, strict=False)
    unit.read(UNIT)
    service = unit["Service"]
    assert service["User"] == "asdf"
    environment_files = [
        line.removeprefix("EnvironmentFile=")
        for line in UNIT.read_text().splitlines()
        if line.startswith("EnvironmentFile=")
    ]
    assert environment_files == [
        "/etc/talking-lamp/motion-bridge.env",
        "/etc/talking-lamp/device-bridge.env",
    ]
    verified = subprocess.run(
        ["systemd-analyze", "verify", str(UNIT)], capture_output=True, text=True)
    assert "Missing '='" not in verified.stderr
    assert "/opt/ros/jazzy/setup.bash" in service["ExecStart"]
    assert "192.168.100.2" in service["ExecStart"]
    assert service["Restart"] == "on-failure"


def test_staged_installer_preserves_tokens_and_defaults_disabled(tmp_path):
    repo = tmp_path / "repo"
    (repo / "jetson_ws/src/lamp_interfaces").mkdir(parents=True)
    destination = tmp_path / "stage"
    result = subprocess.run([
        "bash", str(INSTALL), "--repo", str(repo), "--destdir", str(destination)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for name, variable in (
        ("motion-bridge.env", "TALKING_LAMP_MOTION_TOKEN"),
        ("device-bridge.env", "TALKING_LAMP_DEVICE_TOKEN"),
    ):
        path = destination / "etc/talking-lamp" / name
        assert path.read_text() == f"{variable}=\n"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        path.write_text(f"{variable}=operator-secret\n")
    second = subprocess.run([
        "bash", str(INSTALL), "--repo", str(repo), "--destdir", str(destination)],
        capture_output=True, text=True)
    assert second.returncode == 0
    assert "operator-secret" in (
        destination / "etc/talking-lamp/motion-bridge.env").read_text()
    assert not (destination / "etc/systemd/system/multi-user.target.wants").exists()


def test_apt_source_normalizer_uses_tls_valid_ubuntu_and_ros_mirrors(tmp_path):
    ubuntu = tmp_path / "sources.list"
    ubuntu.write_text(
        "deb http://ports.ubuntu.com/ubuntu-ports/ noble main\n")
    ros = tmp_path / "ros2.sources"
    ros.write_text(
        "Types: deb\n"
        "URIs: https://packages.ros.org/ros2/ubuntu\n"
        "Suites: noble\n")

    result = subprocess.run(
        ["bash", str(NORMALIZE_APT), str(ubuntu), str(ros)],
        capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert ubuntu.read_text() == (
        "deb https://ports.ubuntu.com/ubuntu-ports/ noble main\n")
    assert ros.read_text() == (
        "Types: deb\n"
        "URIs: https://ftp.osuosl.org/pub/ros2/\n"
        "Suites: noble\n")


def test_bridge_runner_sources_ros_environment_before_enabling_nounset(tmp_path):
    ros_setup = tmp_path / "ros-setup.bash"
    ros_setup.write_text(
        ': "$AMENT_TRACE_SETUP_FILES"\n'
        'export AMENT_TRACE_SETUP_FILES=ready\n')
    repo = tmp_path / "repo"
    workspace_setup = repo / "jetson_ws/install/setup.bash"
    workspace_setup.parent.mkdir(parents=True)
    workspace_setup.write_text(
        'test "$AMENT_TRACE_SETUP_FILES" = ready\n')
    commands = tmp_path / "commands"
    commands.mkdir()
    ros2 = commands / "ros2"
    ros2.write_text(
        '#!/bin/sh\n'
        'echo "$*" >> "$BRIDGE_CALLS_FILE"\n'
        'sleep 0.1\n')
    ros2.chmod(0o755)
    calls = tmp_path / "calls"

    result = subprocess.run(
        ["bash", str(RUN_BRIDGES)],
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "ROS_SETUP_FILE": str(ros_setup),
            "TALKING_LAMP_REPO": str(repo),
            "BRIDGE_CALLS_FILE": str(calls),
        },
        capture_output=True, text=True)

    assert result.returncode == 1
    assert "unbound variable" not in result.stderr
    assert sorted(calls.read_text().splitlines()) == [
        "run lamp_device_bridge lamp_device_bridge --ros-args -p pi_host:=192.168.100.2 "
        "-p device_port:=8766 -p capture_rtp_port:=5004 -p playback_rtp_port:=5006 "
        "-p token_env:=TALKING_LAMP_DEVICE_TOKEN -p audio_frame_ms:=20 "
        "-p jitter_buffer_ms:=40",
        "run lamp_motion_bridge lamp_motion_bridge --ros-args -p pi_host:=192.168.100.2 "
        "-p motion_port:=8765 -p token_env:=TALKING_LAMP_MOTION_TOKEN",
    ]
