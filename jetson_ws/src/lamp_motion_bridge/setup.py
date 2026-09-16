from glob import glob
from setuptools import find_packages, setup

package_name = "lamp_motion_bridge"
setup(
    name=package_name, version="0.1.0", packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="Talking Lamp maintainers", maintainer_email="maintainers@example.com",
    description="ROS bridge for Talking Lamp motion control.", license="Apache-2.0",
    entry_points={"console_scripts": [
        "lamp_motion_bridge = lamp_motion_bridge.node:main",
    ]},
)
