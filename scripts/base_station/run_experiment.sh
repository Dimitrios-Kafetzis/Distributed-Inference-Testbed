#!/usr/bin/env bash
#
# Runs a full experiment scenario (apply load and shaping, profile, collect,
# aggregate, optimise, teardown) from the edge server.
#
# Usage: run_experiment.sh <scenario.yaml | --name NAME [--version N]> [--local [ROOT]] [options]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${TESTBED_PYTHON:-$CODE_DIR/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

if [ "$#" -eq 0 ]; then
    sed -n '3,7p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 1
fi

ARGS=()
if [ "${1#-}" = "$1" ]; then
    ARGS+=(--scenario "$1")
    shift
fi

cd "$CODE_DIR"
exec "$PYTHON" -m src.orchestration.experiment run "${ARGS[@]}" "$@"
