#!/usr/bin/env bash
#
# Regenerates the hash-pinned lock files in requirements/ with uv.
# Usage: scripts/lock_requirements.sh [--upgrade]
#
set -euo pipefail

cd "$(dirname "$0")/.."

UPGRADE=()
if [ "${1:-}" = "--upgrade" ]; then
    UPGRADE=(--upgrade)
fi

compile() {
    local input=$1 output=$2 platform=$3
    shift 3
    uv pip compile "$input" \
        --output-file "$output" \
        --generate-hashes \
        --python-version 3.11 \
        --python-platform "$platform" \
        --no-header \
        "${UPGRADE[@]}" "$@"
}

TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

compile requirements/edge_server.in requirements/edge_server.lock x86_64-manylinux_2_28 \
    --extra-index-url "$TORCH_CPU_INDEX" --index-strategy unsafe-best-match
compile requirements/edge_device.in requirements/edge_device.lock aarch64-manylinux_2_28
compile requirements/edge_device_llm.in requirements/edge_device_llm.lock aarch64-manylinux_2_28
compile requirements/dev.in requirements/dev.lock x86_64-manylinux_2_28
compile requirements/docs.in requirements/docs.lock x86_64-manylinux_2_28

echo "Lock files regenerated."
