from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([Node(
        package="lamp_motion_bridge", executable="lamp_motion_bridge",
        name="lamp_motion_bridge", output="screen",
        parameters=[{
            "pi_host": "192.168.100.2", "motion_port": 8765,
            "token_env": "TALKING_LAMP_MOTION_TOKEN",
        }],
    )])
