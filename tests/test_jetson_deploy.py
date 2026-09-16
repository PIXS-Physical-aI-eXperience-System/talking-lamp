import configparser
import os
from pathlib import Path
import stat
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DETECT = ROOT / "deploy/jetson/detect-platform.sh"
INSTALL = ROOT / "deploy/jetson/install-ros-bridges.sh"
UNIT = ROOT / "deploy/jetson/talking-lamp-bridges.service"


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
    assert service["EnvironmentFile"].splitlines() == [
        "/etc/talking-lamp/motion-bridge.env", "/etc/talking-lamp/device-bridge.env"]
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
