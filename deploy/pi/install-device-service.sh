#!/usr/bin/env bash
# Install disabled by default. --enable enables boot startup without starting it.
set -euo pipefail

repo=/home/pixs/talking-lamp
destdir=
enable=false
dry_run=false
usage() {
    echo "Usage: $0 [--repo /absolute/repo] [--dry-run] [--enable] [--destdir /staging/root]"
}
fail() { echo "Device service install failed: $*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --repo|--destdir)
            (($# >= 2)) || fail "$1 requires a path"
            if [[ "$1" == --repo ]]; then repo=$2; else destdir=$2; fi
            shift 2 ;;
        --enable) enable=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; fail "unknown option: $1" ;;
    esac
done

[[ "$repo" =~ ^/[a-zA-Z0-9_./-]+$ ]] || fail "--repo must be an absolute path without spaces or specifiers"
repo=${repo%/}
if [[ -n "$destdir" ]]; then
    [[ "$destdir" == /* && "$destdir" != / ]] || fail "--destdir must be an absolute staging path other than /"
    destdir=$(realpath -m -- "$destdir")
    [[ "$destdir" != / ]] || fail "--destdir must not resolve to /"
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
unit_name=talking-lamp-device.service
service_user=pixs
service_group=talking-lamp
unit_source=$script_dir/$unit_name
calibration=$repo/voice-bench/out/doa/calibration.json
[[ -f "$unit_source" ]] || fail "missing unit: $unit_source"
[[ -f "$repo/src/device/daemon.py" ]] || fail "missing device daemon under $repo"
[[ -f "$calibration" ]] || fail "missing commissioned calibration: $calibration"
[[ -x "$repo/lelamp_runtime/.venv/bin/python" ]] || fail "missing executable python: $repo/lelamp_runtime/.venv/bin/python"

if $dry_run; then
    echo "Would validate gpio access and add $service_user to $service_group and gpio"
    echo "Would install the persistent /dev/pio0 udev permission rule"
    echo "Would verify the Pi 5 PIO backend import as $service_user"
    echo "Would validate/preserve existing token or create mode-0600 $destdir/etc/talking-lamp/device.env"
    echo "Would install $destdir/etc/systemd/system/$unit_name for $repo"
    if [[ -z "$destdir" ]]; then echo "Would run systemctl daemon-reload"; fi
    if $enable; then echo "Would enable $unit_name (without starting)"; else echo "Would leave $unit_name disabled"; fi
    exit 0
fi

if [[ -z "$destdir" ]]; then
    ((EUID == 0)) || fail "run as root for live installation, or use --dry-run/--destdir"
    getent passwd "$service_user" >/dev/null || fail "required service user $service_user does not exist"
    command -v systemctl >/dev/null || fail "systemctl is required for live installation"
    command -v runuser >/dev/null || fail "runuser is required for live installation"
    command -v udevadm >/dev/null || fail "udevadm is required for live installation"
    if ! getent group "$service_group" >/dev/null; then
        groupadd --system "$service_group"
    fi
    getent group gpio >/dev/null || fail "required gpio group does not exist"
    usermod -a -G "$service_group,gpio" "$service_user"
fi

python3 - "$repo" "$destdir" "$unit_source" <<'PY'
import os
from pathlib import Path
import secrets
import sys

repo, destination, source = sys.argv[1:]
root = Path(destination or "/")
env_path = root / "etc/talking-lamp/device.env"
env_path.parent.mkdir(parents=True, exist_ok=True)
try:
    descriptor = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    if env_path.is_symlink() or not env_path.is_file():
        sys.exit("device.env must be a regular file, not a symlink")
    values = [line.split("=", 1)[1].strip().strip("\"'")
              for line in env_path.read_text().splitlines()
              if line.startswith("TALKING_LAMP_TOKEN=")]
    if not values or not values[-1].strip():
        sys.exit("Existing device.env must contain a non-empty TALKING_LAMP_TOKEN; left unchanged")
    env_path.chmod(0o600)
else:
    with os.fdopen(descriptor, "w") as stream:
        stream.write(f"TALKING_LAMP_TOKEN={secrets.token_hex(32)}\n")

unit = root / "etc/systemd/system/talking-lamp-device.service"
unit.parent.mkdir(parents=True, exist_ok=True)
unit.write_text(Path(source).read_text().replace("/home/pixs/talking-lamp", repo))
unit.chmod(0o644)

pio_rule = root / "etc/udev/rules.d/99-talking-lamp-pio.rules"
pio_rule.parent.mkdir(parents=True, exist_ok=True)
pio_rule.write_text('SUBSYSTEM=="*-pio", GROUP="gpio", MODE="0660"\n')
pio_rule.chmod(0o644)
PY

if [[ -n "$destdir" ]]; then
    wants=$destdir/etc/systemd/system/multi-user.target.wants
    if $enable; then
        mkdir -p -- "$wants"
        ln -sfn -- "../$unit_name" "$wants/$unit_name"
    else
        rm -f -- "$wants/$unit_name"
    fi
    echo "Staged $unit_name; host systemd was not contacted."
else
    [[ -e /dev/pio0 ]] || fail "/dev/pio0 is required for the commissioned Pi 5 LED backend"
    udevadm control --reload-rules
    chgrp gpio /dev/pio0
    chmod 0660 /dev/pio0
    runuser -u "$service_user" -- test -r /dev/pio0 -a -w /dev/pio0 \
        || fail "$service_user cannot read and write /dev/pio0"
    runuser -u "$service_user" -- "$repo/lelamp_runtime/.venv/bin/python" -c \
        'import adafruit_raspberry_pi5_neopixel_write' \
        || fail "Adafruit Raspberry Pi 5 NeoPixel backend is not importable"
    systemctl daemon-reload
    if $enable; then systemctl enable "$unit_name"; else systemctl disable "$unit_name"; fi
fi
if $enable; then
    echo "$unit_name enabled; start it explicitly after commissioning."
else
    echo "$unit_name installed and disabled; pass --enable after commissioning."
fi
