import ast
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def source(relative):
    return (ROOT / relative).read_text()


def test_motion_bridge_exposes_exact_ros_names_and_parameters():
    text = source("jetson_ws/src/lamp_motion_bridge/lamp_motion_bridge/node.py")
    for name in (
        "/lamp/play_motion", "/lamp/place_task_light", "/lamp/interrupt_motion",
        "/lamp/list_motions", "/lamp/track_point", "/lamp/track_bearing",
        "/lamp/motion_status",
    ):
        assert name in text
    for parameter in ("pi_host", "motion_port", "token_env"):
        assert f'declare_parameter("{parameter}"' in text
    assert "QoSProfile(depth=1)" in text
    assert "from motion" not in text


def test_motion_bridge_gives_terminal_motion_results_the_action_timeout():
    text = source("jetson_ws/src/lamp_motion_bridge/lamp_motion_bridge/node.py")
    assert "response_timeout=timeout" in text


def test_motion_bridge_keeps_commands_responsive_during_looping_idle():
    text = source("jetson_ws/src/lamp_motion_bridge/lamp_motion_bridge/node.py")
    assert "ReentrantCallbackGroup()" in text
    assert text.count("callback_group=self.command_group") == 7
    assert "MultiThreadedExecutor(num_threads=4)" in text
    assert "if rclpy.ok():" in text


def test_device_bridge_exposes_exact_ros_names_parameters_and_bounded_qos():
    text = source("jetson_ws/src/lamp_device_bridge/lamp_device_bridge/node.py")
    for name in (
        "/lamp/audio/capture", "/lamp/audio/playback_frames", "/lamp/play_audio",
        "/lamp/audio_status", "/lamp/orientation_status", "/lamp/return_center",
        "/lamp/led/frame", "/lamp/led/set_solid", "/lamp/led/clear", "/lamp/led/status",
    ):
        assert name in text
    for parameter in (
        "pi_host", "device_port", "capture_rtp_port", "playback_rtp_port",
        "token_env", "audio_frame_ms", "jitter_buffer_ms",
    ):
        assert f'declare_parameter("{parameter}"' in text
    assert "QoSProfile(depth=1)" in text
    assert "QoSProfile(depth=10)" in text
    assert "from device" not in text


def test_device_bridge_streaming_action_uses_reentrant_multithreaded_callbacks():
    text = source("jetson_ws/src/lamp_device_bridge/lamp_device_bridge/node.py")
    assert "ReentrantCallbackGroup()" in text
    assert text.count("callback_group=self.command_group") == 6
    assert "MultiThreadedExecutor(num_threads=4)" in text
    assert "if rclpy.ok():" in text


def test_bridge_packages_install_nodes_launch_files_and_dependencies():
    for package, executable in (
        ("lamp_motion_bridge", "lamp_motion_bridge"),
        ("lamp_device_bridge", "lamp_device_bridge"),
    ):
        base = ROOT / "jetson_ws/src" / package
        setup = ast.parse((base / "setup.py").read_text())
        assert setup is not None
        setup_text = (base / "setup.py").read_text()
        assert f'"{executable} = {package}.node:main"' in setup_text
        assert (base / "launch" / ("motion_bridge.launch.py" if "motion" in package else "device_bridge.launch.py")).is_file()
        package_xml = (base / "package.xml").read_text()
        assert "rclpy" in package_xml
        assert "lamp_interfaces" in package_xml


def test_python_package_manifests_export_ament_python_without_rosdep_key():
    for package in (
        "lamp_motion_bridge", "lamp_device_bridge", "lamp_interaction"):
        manifest = ET.parse(
            ROOT / "jetson_ws/src" / package / "package.xml").getroot()
        assert manifest.findtext("export/build_type") == "ament_python"
        dependency_names = {
            element.text
            for tag in ("buildtool_depend", "depend", "exec_depend")
            for element in manifest.findall(tag)
        }
        assert "ament_python" not in dependency_names
