from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([Node(
        package="lamp_vision", executable="lamp_vision",
        name="lamp_vision", output="screen",
        parameters=[{
            "camera_index": 0, "width": 1920, "height": 1080, "fps": 10.0,
            "model_dir": "/home/asdf/vision-bench/models",
            "object_conf": 0.3, "face_conf": 0.6,
            "center_on_start": True,
        }],
    )])
