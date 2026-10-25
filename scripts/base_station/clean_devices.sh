#!/usr/bin/env bash
#
# Removes everything a testbed run may have left on the devices: tagged
# processes, tc shaping on the data-plane interface, and testbed cgroups; then
# undoes the pending journal entries of unfinished runs.
#
# Usage: clean_devices.sh [--local [ROOT]] [--nodes-file FILE] [--config FILE]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${TESTBED_PYTHON:-$CODE_DIR/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

cd "$CODE_DIR"
exec "$PYTHON" -m src.orchestration.experiment teardown "$@"
