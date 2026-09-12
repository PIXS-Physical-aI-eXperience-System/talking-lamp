#!/usr/bin/env bash
set -euo pipefail
# Run inside the prepared CUDA environment. Models must already be cached.
# Usage: bash tools/run_vlm_ram_profiles.sh IMAGE OUTPUT_DIRECTORY
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image=$(realpath -- "${1:?image required}")
output=${2:?output directory required}
mkdir -p -- "$output"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
for profile in baseline one_tile short_ko short_en; do
    options=()
    case "$profile" in
        one_tile) options=(--one-tile) ;;
        short_ko) options=(--one-tile --max-new-tokens 32) ;;
        short_en) options=(--one-tile --max-new-tokens 32 --prompt 'Describe this image in one short English sentence.') ;;
    esac
    python "$script_dir/ram_probe.py" --out "$output/$profile-system.json" \
        --min-available-mb 800 --max-swap-pages 0 -- \
        python "$script_dir/bench_vlm_ram.py" --image "$image" \
        --out "$output/$profile.json" "${options[@]}" \
        > "$output/$profile.log" 2>&1
done
