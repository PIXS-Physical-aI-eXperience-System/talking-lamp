from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([Node(
        package="lamp_device_bridge", executable="lamp_device_bridge",
        name="lamp_device_bridge", output="screen",
        parameters=[{
            "pi_host": "192.168.100.2", "device_port": 8766,
            "capture_rtp_port": 5004, "playback_rtp_port": 5006,
            "token_env": "TALKING_LAMP_DEVICE_TOKEN",
            "audio_frame_ms": 20, "jitter_buffer_ms": 40,
        }],
    )])
