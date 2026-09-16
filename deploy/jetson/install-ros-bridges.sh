#!/usr/bin/env bash
# Live install requires root. Staging never runs apt, rosdep, colcon or systemd.
set -euo pipefail

repo=/home/asdf/talking-lamp
destdir=
enable=false
usage() { echo "Usage: $0 [--repo PATH] [--destdir ROOT] [--enable]"; }
fail() { echo "Jetson bridge install failed: $*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --repo|--destdir)
            (($# >= 2)) || fail "$1 requires a path"
            if [[ "$1" == --repo ]]; then repo=$2; else destdir=$2; fi
            shift 2 ;;
        --enable) enable=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; fail "unknown option: $1" ;;
    esac
done
[[ "$repo" =~ ^/[a-zA-Z0-9_./-]+$ ]] || fail "--repo must be an absolute path without spaces"
repo=${repo%/}
[[ -d "$repo/jetson_ws/src/lamp_interfaces" ]] || fail "missing Jetson workspace under $repo"
if [[ -n "$destdir" ]]; then
    [[ "$destdir" == /* && "$destdir" != / ]] || fail "invalid --destdir"
    destdir=$(realpath -m -- "$destdir")
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
unit=talking-lamp-bridges.service

write_files() {
    local root=${destdir:-/}
    install -d -m 0755 "$root/etc/talking-lamp" "$root/etc/systemd/system"
    for entry in \
        "motion-bridge.env:TALKING_LAMP_MOTION_TOKEN" \
        "device-bridge.env:TALKING_LAMP_DEVICE_TOKEN"; do
        name=${entry%%:*}
        variable=${entry#*:}
        path=$root/etc/talking-lamp/$name
        if [[ ! -e "$path" ]]; then
            umask 077
            printf '%s=\n' "$variable" >"$path"
        fi
        chmod 0600 "$path"
    done
    sed "s#/home/asdf/talking-lamp#$repo#g" "$script_dir/$unit" \
        >"$root/etc/systemd/system/$unit"
    chmod 0644 "$root/etc/systemd/system/$unit"
}

if [[ -n "$destdir" ]]; then
    write_files
    if $enable; then
        wants=$destdir/etc/systemd/system/multi-user.target.wants
        mkdir -p "$wants"
        ln -sfn "../$unit" "$wants/$unit"
    fi
    echo "$unit staged and disabled by default; host systemd was not contacted."
    exit 0
fi

((EUID == 0)) || fail "run as root for live installation or use --destdir"
[[ $("$script_dir/detect-platform.sh") == jazzy ]] || fail "ROS 2 Jazzy platform check failed"
getent passwd asdf >/dev/null || fail "required user asdf does not exist"

"$script_dir/normalize-apt-sources.sh" \
    /etc/apt/sources.list \
    /etc/apt/sources.list.d/ros2.sources \
    /etc/apt/sources.list.d/ros2.list
apt-get update
env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common curl ca-certificates
add-apt-repository -y universe
apt-get update
ros_source_version=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
    | sed -n 's/.*"tag_name": "\([^"]*\)".*/\1/p')
[[ -n "$ros_source_version" ]] || fail "cannot resolve ros-apt-source release"
curl -fsSL -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/$ros_source_version/ros2-apt-source_${ros_source_version}.noble_all.deb"
dpkg -i /tmp/ros2-apt-source.deb
"$script_dir/normalize-apt-sources.sh" \
    /etc/apt/sources.list.d/ros2.sources \
    /etc/apt/sources.list.d/ros2.list
apt-get update
env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ros-jazzy-ros-base python3-colcon-common-extensions python3-rosdep python3-gi \
    gir1.2-gstreamer-1.0 gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then rosdep init; fi
sudo -u asdf rosdep update --rosdistro jazzy
sudo -u asdf bash -lc "source /opt/ros/jazzy/setup.bash && cd '$repo/jetson_ws' && rosdep install --from-paths src --ignore-src --rosdistro jazzy -y && colcon build --symlink-install"

write_files
systemctl daemon-reload
if $enable; then systemctl enable "$unit"; else systemctl disable "$unit"; fi
echo "$unit installed and disabled unless --enable was supplied; it was not started."
