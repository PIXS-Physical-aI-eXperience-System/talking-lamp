#!/usr/bin/env bash
# Wait until this host has the Pi link address before ROS nodes start.
#
# At boot the bridges started at 53 s, the Pi link came up at 56 s and Wi-Fi
# at 58 s (2026-11-01). Nodes created before the interfaces existed never
# discovered the voice node started later: /lamp/audio/capture had no
# publisher for it. network-online.target does not wait here because no
# wait-online service is enabled. Gives up after the timeout and lets the
# service start anyway, so a missing link never keeps the lamp down.
set -u

addr=${1:-192.168.100.1}
timeout_s=${2:-120}
for ((waited=0; waited<timeout_s; waited++)); do
    if ip -4 -o addr show | grep -q " ${addr}/"; then
        # Let the remaining interfaces (Wi-Fi came 2 s later) come up too.
        sleep 3
        exit 0
    fi
    sleep 1
done
echo "link address $addr did not appear within ${timeout_s}s; starting anyway" >&2
exit 0
