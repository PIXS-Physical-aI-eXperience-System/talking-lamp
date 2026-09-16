#!/usr/bin/env bash
set -euo pipefail

(($# > 0)) || {
    echo "Usage: $0 SOURCE_FILE [...]" >&2
    exit 2
}

for source_file in "$@"; do
    [[ -f "$source_file" ]] || continue
    sed -i \
        -e 's|http://ports\.ubuntu\.com/ubuntu-ports|https://ports.ubuntu.com/ubuntu-ports|g' \
        -e 's|http://packages\.ros\.org/ros2/ubuntu|https://packages.ros.org/ros2/ubuntu|g' \
        "$source_file"
done
