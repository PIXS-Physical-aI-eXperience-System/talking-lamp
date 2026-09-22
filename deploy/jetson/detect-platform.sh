#!/usr/bin/env bash
set -euo pipefail

root=${PLATFORM_ROOT:-/}
[[ -f "$root/etc/os-release" ]] || { echo "missing os-release" >&2; exit 2; }
[[ -f "$root/etc/nv_tegra_release" ]] || { echo "missing nv_tegra_release" >&2; exit 2; }
# shellcheck disable=SC1090
source "$root/etc/os-release"
l4t=$(head -n 1 "$root/etc/nv_tegra_release")
if [[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 24.04 && "$l4t" =~ R39 ]]; then
    echo jazzy
    exit 0
fi
echo "unsupported Jetson platform: ${ID:-unknown} ${VERSION_ID:-unknown}; $l4t" >&2
exit 2
