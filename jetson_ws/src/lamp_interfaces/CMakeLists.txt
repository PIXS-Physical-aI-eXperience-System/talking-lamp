cmake_minimum_required(VERSION 3.8)
project(lamp_interfaces)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/PlayMotion.action"
  "action/PlaceTaskLight.action"
  "action/PlayAudio.action"
  "action/ReturnCenter.action"
  "msg/AudioFrame.msg"
  "msg/AudioStatus.msg"
  "msg/OrientationStatus.msg"
  "msg/LedStatus.msg"
  "msg/MotionStatus.msg"
  "srv/ListMotions.srv"
  "srv/SetLedSolid.srv"
  "srv/InterruptMotion.srv"
  DEPENDENCIES builtin_interfaces geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
