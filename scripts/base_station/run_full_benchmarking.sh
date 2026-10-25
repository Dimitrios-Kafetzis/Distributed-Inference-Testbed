#!/usr/bin/env bash
#
# Master script of the four-stage benchmarking pipeline, run on the edge server:
#   1. dispatch train_and_run_benchmark.sh both to all registered end devices
#      concurrently through the SSH executor and wait for all of them;
#   2. collect the result files (hostname-prefixed) into the run directory;
#   3. post-processing: latency measurement on every device, result
#      aggregation, partition-feasibility calculation;
#   4. outputs: total_averages_of_benchmark_results.json,
#      calculated_max_partition_points.json, final_latencies_averages.json
#      in results/<run id>/, ready for the optimiser (NNPM or delay_optimizer.py).
#
# Usage: run_full_benchmarking.sh [scenario.yaml] [--local [ROOT]] [extra experiment options]
# Default scenario: examples/scenarios/baseline.yaml
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${TESTBED_PYTHON:-$CODE_DIR/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

SCENARIO="$CODE_DIR/examples/scenarios/baseline.yaml"
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
    SCENARIO=$1
    shift
fi

cd "$CODE_DIR"
exec "$PYTHON" -m src.orchestration.experiment benchmark --scenario "$SCENARIO" "$@"
