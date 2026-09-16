#!/usr/bin/env bash
set -euo pipefail

repo=${TALKING_LAMP_REPO:-/home/asdf/talking-lamp}
source /opt/ros/jazzy/setup.bash
source "$repo/jetson_ws/install/setup.bash"

ros2 run lamp_motion_bridge lamp_motion_bridge --ros-args \
    -p pi_host:=192.168.100.2 -p motion_port:=8765 \
    -p token_env:=TALKING_LAMP_MOTION_TOKEN &
motion_pid=$!
ros2 run lamp_device_bridge lamp_device_bridge --ros-args \
    -p pi_host:=192.168.100.2 -p device_port:=8766 \
    -p capture_rtp_port:=5004 -p playback_rtp_port:=5006 \
    -p token_env:=TALKING_LAMP_DEVICE_TOKEN \
    -p audio_frame_ms:=20 -p jitter_buffer_ms:=40 &
device_pid=$!

cleanup() {
    kill -TERM "$motion_pid" "$device_pid" 2>/dev/null || true
    wait "$motion_pid" "$device_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait -n "$motion_pid" "$device_pid"
exit 1
