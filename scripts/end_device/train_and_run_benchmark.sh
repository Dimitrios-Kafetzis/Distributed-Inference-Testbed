#!/usr/bin/env bash
#
# Local operations on an edge device: prepare the model and the test input,
# run the partial-inference benchmark across all partition points, or both.
#
# Usage: train_and_run_benchmark.sh <mode> <model_dir> <model_name> <output_dir> [options]
#   mode        train | benchmark | both
#   model_dir   directory that holds <model_name>.npz (created by train)
#   model_name  registered model name (alexnet, vgg16, resnet50, gru_qoe, fake)
#   output_dir  where the benchmark JSON files are written
# Options (after the four arguments):
#   --run-id ID --node-id ID --repetitions N --warmup N --seed N --image PATH
#   --start-at T0 | --origin-wall T --origin-perf NS   (start signal, from the orchestrator)
#   --model-args JSON   adapter parameters (the scenario's model_args)
#   --input-source JSON input source (the scenario's input block)
#
# train mode prepares the model weights with the given seed and generates the
# test input image on the device; benchmark mode invokes src/models/inference.py
# to run the benchmarking loop across all partition points and write the JSON
# results; both runs the two in sequence.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${TESTBED_PYTHON:-python3}"
DATA_DIR="${TESTBED_ED_DATA_DIR:-$CODE_DIR/data}"

if [ "$#" -lt 4 ]; then
    sed -n '3,19p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 1
fi

MODE=$1
MODEL_DIR=$2
MODEL_NAME=$3
OUTPUT_DIR=$4
shift 4

RUN_ID="manual-$(date -u +%Y%m%d-%H%M%S)"
NODE_ID="${TESTBED_NODE_ID:-$(hostname)}"
REPETITIONS=1
WARMUP=1
SEED=0
IMAGE="$DATA_DIR/test_image_${MODEL_NAME}.png"
ORIGIN=()
MODEL_ARGS=()
INPUT_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --run-id) RUN_ID=$2; shift 2 ;;
        --node-id) NODE_ID=$2; shift 2 ;;
        --repetitions) REPETITIONS=$2; shift 2 ;;
        --warmup) WARMUP=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --image) IMAGE=$2; shift 2 ;;
        --origin-wall) ORIGIN+=(--origin-wall "$2"); shift 2 ;;
        --origin-perf) ORIGIN+=(--origin-perf "$2"); shift 2 ;;
        --start-at) ORIGIN+=(--start-at "$2"); shift 2 ;;
        --model-args) MODEL_ARGS=(--model-args "$2"); shift 2 ;;
        --input-source) INPUT_ARGS=(--input-source "$2"); shift 2 ;;
        --tag) shift 2 ;;
        *) echo "error: unknown option $1" >&2; exit 1 ;;
    esac
done

WEIGHTS="$MODEL_DIR/$MODEL_NAME.npz"
cd "$CODE_DIR"

prepare_model() {
    echo "Preparing model $MODEL_NAME (seed $SEED) and test image..."
    mkdir -p "$MODEL_DIR" "$(dirname "$IMAGE")"
    "$PYTHON" -m src.models.inference prepare --model "$MODEL_NAME" --weights "$WEIGHTS" \
        --image "$IMAGE" --seed "$SEED" "${MODEL_ARGS[@]}"
    [ -f "$WEIGHTS" ] || { echo "error: weights not written: $WEIGHTS" >&2; exit 1; }
    [ -f "$IMAGE" ] || { echo "error: test image not written: $IMAGE" >&2; exit 1; }
    echo "Model ready: $WEIGHTS; test image: $IMAGE"
}

run_benchmarking() {
    echo "Benchmarking $MODEL_NAME across all partition points ($REPETITIONS repetition(s))..."
    [ -f "$WEIGHTS" ] || { echo "error: weights missing: $WEIGHTS (run train first)" >&2; exit 1; }
    mkdir -p "$OUTPUT_DIR"
    "$PYTHON" -m src.models.inference benchmark --model "$MODEL_NAME" --weights "$WEIGHTS" \
        --image "$IMAGE" --output-dir "$OUTPUT_DIR" --run-id "$RUN_ID" --node-id "$NODE_ID" \
        --repetitions "$REPETITIONS" --warmup "$WARMUP" --seed "$SEED" "${ORIGIN[@]}" \
        "${MODEL_ARGS[@]}" "${INPUT_ARGS[@]}"
    echo "Benchmarking completed. Results in $OUTPUT_DIR"
}

case "$MODE" in
    train) prepare_model ;;
    benchmark) run_benchmarking ;;
    both) prepare_model; run_benchmarking ;;
    *) echo "error: invalid mode '$MODE' (train, benchmark, both)" >&2; exit 1 ;;
esac
