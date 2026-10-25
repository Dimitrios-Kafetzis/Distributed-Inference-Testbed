"""Device-side inference entry point: prepare, benchmark, partial and complete.

Modes:

* ``prepare``: build (or check) the model weights with a fixed seed and write
  the test input image, so that the device is ready to benchmark;
* ``benchmark``: run the model in partial-inference mode across all partition
  points, measuring each stage's execution time, peak CPU utilisation and
  peak memory, and write one ``BenchmarkResults`` JSON file per repetition;
* ``partial``: run stages 1..s on an input and write the intermediate tensor
  (the cut) to a file, serialised with its shape and dtype;
* ``complete``: read an intermediate tensor and run stages s+1..S;
* ``capacity``: report the device's average FLOP/s over the stages.

Usage::

    python -m src.models.inference prepare --model alexnet --weights W.npz --image IMG --seed 1
    python -m src.models.inference benchmark --model alexnet --weights W.npz --image IMG \\
        --output-dir RESULTS --run-id R --node-id device1 --repetitions 5 --warmup 2
    python -m src.models.inference partial --model alexnet --weights W.npz --image IMG --s 4 \\
        --output cut.npy
    python -m src.models.inference complete --model alexnet --weights W.npz --s 4 --input cut.npy
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmarking.benchmark_results import Origin
from src.common.seeds import seed_everything
from src.common.start_signal import origin_from_arguments
from src.models import registry
from src.models.profiler import capacity_flops_per_second, profile_once, warm_up
from src.models.registry import Array, ModelAdapter, deserialize_array, serialize_array


def load_adapter(name: str, weights: Path | None, seed: int, **kwargs: Any) -> ModelAdapter:
    adapter = registry.get(name, **kwargs)
    if weights is not None and weights.exists():
        adapter.load_weights(weights)
    else:
        adapter.build(seed)
    return adapter


def apply_thread_count() -> int | None:
    value = os.environ.get("TESTBED_THREADS")
    if not value:
        return None
    from src.testbed.heterogeneity import apply_thread_count as apply

    count = int(value)
    apply(count)
    return count


def load_input(adapter: ModelAdapter, image: Path | None, seed: int) -> Array:
    if image is None or not image.exists():
        return adapter.make_input(seed)
    from src.models.images import is_numpy_file, load_image

    if image.suffix == ".npy" or is_numpy_file(image):
        array: Array = np.load(image, allow_pickle=False)
        if array.ndim == len(adapter.input_shape) + 1:
            return array
        batched: Array = array[np.newaxis, ...]
        return batched
    return load_image(image, adapter.input_shape)


def origin_from_args(args: argparse.Namespace) -> Origin:
    """The run origin: given explicitly, or captured when the start signal T0 is reached."""
    return origin_from_arguments(args)


def cmd_prepare(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    adapter = load_adapter(args.model, None, args.seed)
    if args.weights:
        if args.weights.exists() and not args.force:
            print(f"weights present: {args.weights}", file=sys.stderr)
        else:
            adapter.save_weights(args.weights)
            print(f"weights written: {args.weights}", file=sys.stderr)
    if args.image:
        from src.models.images import write_synthetic_image

        if args.image.exists() and not args.force:
            print(f"image present: {args.image}", file=sys.stderr)
        else:
            write_synthetic_image(args.image, adapter.input_shape, args.seed)
            print(f"image written: {args.image}", file=sys.stderr)


def cmd_benchmark(args: argparse.Namespace) -> list[Path]:
    seed_everything(args.seed)
    threads = apply_thread_count()
    adapter = load_adapter(args.model, args.weights, args.seed)
    x = load_input(adapter, args.image, args.seed)
    hostname = socket.gethostname()
    import psutil

    baseline = psutil.Process().memory_info().rss / 2**20
    warm_up(adapter, x, args.warmup)
    origin = origin_from_args(args)  # waits for the start signal when --start-at is given
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for repetition in range(args.repetitions):
        record = profile_once(
            adapter,
            x,
            run_id=args.run_id,
            node_id=args.node_id,
            repetition=repetition,
            warmup_runs=args.warmup,
            origin=origin,
            hostname=hostname,
            sampler_interval_s=args.sampler_interval,
            baseline_rss_mb=baseline,
        )
        record.metadata["threads"] = threads
        record.metadata["flops_per_second"] = capacity_flops_per_second(record)
        path = args.output_dir / f"benchmark_{args.run_id}_rep{repetition}.json"
        record.save(path)
        written.append(path)
        total_ms = record.partition_points[-1].cumulative_latency_ns / 1e6
        print(
            f"repetition {repetition}: {len(record.stages)} stages, {total_ms:.1f} ms -> {path}",
            file=sys.stderr,
        )
    return written


def cmd_partial(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    apply_thread_count()
    adapter = load_adapter(args.model, args.weights, args.seed)
    x = load_input(adapter, args.image, args.seed)
    started = time.perf_counter_ns()
    cut = adapter.run_stages(x, 1, args.s) if args.s > 0 else x
    elapsed_ns = time.perf_counter_ns() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(serialize_array(cut))
    print(
        json.dumps(
            {
                "s": args.s,
                "local_ns": elapsed_ns,
                "bytes": args.output.stat().st_size,
                "shape": list(cut.shape),
            }
        )
    )


def cmd_complete(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    apply_thread_count()
    adapter = load_adapter(args.model, args.weights, args.seed)
    cut = deserialize_array(args.input.read_bytes())
    stages = len(adapter.stages)
    started = time.perf_counter_ns()
    out = adapter.run_stages(cut, args.s + 1, stages) if args.s < stages else cut
    elapsed_ns = time.perf_counter_ns() - started
    flat = out.reshape(out.shape[0], -1)
    result = {
        "s": args.s,
        "remote_ns": elapsed_ns,
        "shape": list(out.shape),
        "argmax": [int(i) for i in flat.argmax(axis=1)],
        "max": [float(v) for v in flat.max(axis=1)],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(serialize_array(out))
        result["output"] = str(args.output)
    print(json.dumps(result))


def cmd_capacity(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    apply_thread_count()
    adapter = load_adapter(args.model, args.weights, args.seed)
    x = load_input(adapter, args.image, args.seed)
    warm_up(adapter, x, args.warmup)
    rates = []
    for repetition in range(args.repetitions):
        record = profile_once(
            adapter,
            x,
            run_id="capacity",
            node_id=args.node_id,
            repetition=repetition,
            warmup_runs=args.warmup,
            origin=Origin(time.time(), time.perf_counter_ns()),
        )
        rates.append(capacity_flops_per_second(record))
    print(
        json.dumps(
            {
                "model": adapter.name,
                "node_id": args.node_id,
                "flops_per_second": sum(rates) / len(rates),
                "repetitions": len(rates),
                "total_flops": sum(s.flops for s in adapter.stages),
            }
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Device-side inference: prepare, benchmark, partial, complete."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    def common(p: argparse.ArgumentParser, weights_required: bool = False) -> None:
        p.add_argument("--model", required=True, help="registered model name")
        p.add_argument(
            "--weights",
            type=Path,
            default=None,
            help="weights file (.npz); built from --seed when absent",
        )
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--tag", help="process tag (ignored; present for pkill -f)")

    prep = sub.add_parser("prepare")
    common(prep)
    prep.add_argument("--image", type=Path, default=None, help="test input image to write")
    prep.add_argument("--force", action="store_true")

    bench = sub.add_parser("benchmark")
    common(bench)
    bench.add_argument("--image", type=Path, default=None)
    bench.add_argument("--output-dir", type=Path, required=True)
    bench.add_argument("--run-id", required=True)
    bench.add_argument("--node-id", required=True)
    bench.add_argument("--repetitions", type=int, default=1)
    bench.add_argument("--warmup", type=int, default=1)
    bench.add_argument("--sampler-interval", type=float, default=0.005)
    bench.add_argument("--origin-wall", type=float)
    bench.add_argument("--origin-perf", type=int)
    bench.add_argument("--start-at", type=float, default=None, help="wall-clock T0 to wait for")

    partial = sub.add_parser("partial")
    common(partial)
    partial.add_argument("--image", type=Path, default=None)
    partial.add_argument("--s", type=int, required=True)
    partial.add_argument("--output", type=Path, required=True)

    complete = sub.add_parser("complete")
    common(complete)
    complete.add_argument("--s", type=int, required=True)
    complete.add_argument("--input", type=Path, required=True)
    complete.add_argument("--output", type=Path, default=None)

    capacity = sub.add_parser("capacity")
    common(capacity)
    capacity.add_argument("--image", type=Path, default=None)
    capacity.add_argument("--node-id", default="edge_server")
    capacity.add_argument("--repetitions", type=int, default=3)
    capacity.add_argument("--warmup", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    {
        "prepare": cmd_prepare,
        "benchmark": cmd_benchmark,
        "partial": cmd_partial,
        "complete": cmd_complete,
        "capacity": cmd_capacity,
    }[args.mode](args)


if __name__ == "__main__":
    main()
