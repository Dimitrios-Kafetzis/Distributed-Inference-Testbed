#!/usr/bin/env bash
#
# Applies, varies or clears network delay on an interface with tc-netem.
# Thin wrapper over src/testbed/latency_injector.py that keeps the historical
# command line and runs:  tc qdisc add dev <iface> root netem delay <latency>ms
#
# Usage:
#   apply_latency.sh <interface> <min_ms> <max_ms> [--duration S] [--once|--continuous|--periodic]
#                    [--jitter MS] [--loss PERCENT] [--rate MBPS] [--dst IP ...] [--seed N] [--dry-run]
#   apply_latency.sh <interface> --clear
#
# Modes: --once (constant: one value drawn from [min, max], kept), --continuous
# (a new value every --duration seconds), --periodic (default: on for --duration
# seconds, off for --duration seconds, repeated).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${TESTBED_PYTHON:-python3}"

if [ "$#" -lt 2 ]; then
    sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

IFACE=$1
shift

if [ "${1:-}" = "--clear" ]; then
    exec "$PYTHON" -m src.testbed.latency_injector "$IFACE" --clear "${@:2}"
fi

if [ "$#" -lt 2 ]; then
    echo "error: <min_ms> <max_ms> required (or --clear)" >&2
    exit 1
fi

MIN=$1
MAX=$2
shift 2

MODE="periodic"
DURATION=""
EXTRA=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --once) MODE="constant"; shift ;;
        --continuous) MODE="uniform"; shift ;;
        --periodic) MODE="square"; shift ;;
        --duration) DURATION=$2; shift 2 ;;
        --jitter|--loss|--rate|--dst|--seed) EXTRA+=("$1" "$2"); shift 2 ;;
        --dry-run|--no-sudo) EXTRA+=("$1"); shift ;;
        --tag) shift 2 ;;
        *) echo "error: unknown option $1" >&2; exit 1 ;;
    esac
done
[ "$MODE" = "periodic" ] && MODE="square"

ARGS=("$IFACE" --delay "$MIN" "$MAX" --mode "$MODE")
if [ "$MODE" = "constant" ]; then
    # --once: apply one drawn value and keep it (a duration removes it afterwards).
    [ -n "$DURATION" ] && ARGS+=(--duration "$DURATION")
else
    if [ -z "$DURATION" ]; then
        echo "error: --continuous and --periodic need --duration (the period in seconds)" >&2
        exit 1
    fi
    ARGS+=(--period "$DURATION")
fi

cd "$CODE_DIR"
exec "$PYTHON" -m src.testbed.latency_injector "${ARGS[@]}" "${EXTRA[@]}"
