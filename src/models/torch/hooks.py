"""Per-block profiling of LLMs with PyTorch forward hooks.

Forward hooks (callbacks PyTorch invokes immediately before and after a
module's forward computation) wrap each attention head, output projection
and feed-forward block of the head-wise decomposition with
``time.perf_counter_ns()`` timers, capturing per-block computation times for
a prompt of T tokens. The hooks are removed after profiling.

Usage::

    python -m src.models.torch.hooks --model gpt2 --seq-len 32 --repetitions 3 --output blocks.json
        [--weights DIR_OR_NPZ] [--config '{"n_layer": 2, "n_embd": 32, "n_head": 4}']
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from src.benchmarking.benchmark_results import Origin
from src.common.runs import write_json_atomic
from src.models.torch.decoder import DecoderStageAdapter
from src.models.torch.headwise import HeadwiseLayer, headwise_layers


@dataclass
class BlockTimer:
    """Registers pre- and post-forward hooks on named modules and records durations."""

    timings_ns: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    _handles: list[RemovableHandle] = field(default_factory=list)
    _started: dict[str, int] = field(default_factory=dict)

    def attach(self, modules: dict[str, nn.Module]) -> BlockTimer:
        for name, module in modules.items():

            def pre(_module: nn.Module, _inputs: Any, name: str = name) -> None:
                self._started[name] = time.perf_counter_ns()

            def post(_module: nn.Module, _inputs: Any, _output: Any, name: str = name) -> None:
                self.timings_ns[name].append(time.perf_counter_ns() - self._started.pop(name))

            self._handles.append(module.register_forward_pre_hook(pre))
            self._handles.append(module.register_forward_hook(post))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> BlockTimer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.detach()


def block_modules(
    adapter: DecoderStageAdapter,
) -> tuple[list[HeadwiseLayer], dict[str, nn.Module]]:
    """The head-wise layers and the name -> module map of every block."""
    assert adapter.model is not None
    layers = headwise_layers(adapter)
    modules: dict[str, nn.Module] = {}
    modules["embed"] = adapter.embed_module()
    for index, layer in enumerate(layers, start=1):
        for head, unit in enumerate(layer.heads):
            modules[f"layer{index}.head{head}"] = unit
        modules[f"layer{index}.projection"] = layer.projection
        modules[f"layer{index}.ffn"] = layer.ffn
    modules["lm_head"] = adapter.model.lm_head
    return layers, modules


@torch.no_grad()
def forward_headwise(
    adapter: DecoderStageAdapter, layers: Sequence[nn.Module], ids: torch.Tensor
) -> torch.Tensor:
    hidden = adapter.embed(ids)
    for layer in layers:
        hidden = layer(hidden)
    return adapter.head(hidden)


def profile_blocks(
    adapter: DecoderStageAdapter,
    x: np.ndarray,  # type: ignore[type-arg]
    *,
    repetitions: int = 1,
    warmup: int = 1,
) -> dict[str, Any]:
    """Per-block timings (ns) over ``repetitions`` prefill passes, hooks removed afterwards."""
    adapter._ensure_built()
    layers, modules = block_modules(adapter)
    ids = torch.from_numpy(np.ascontiguousarray(x)).to(torch.int64)
    for _ in range(warmup):
        forward_headwise(adapter, layers, ids)
    with BlockTimer() as timer:
        timer.attach(modules)
        for _ in range(repetitions):
            forward_headwise(adapter, layers, ids)
    for module in modules.values():
        assert not module._forward_hooks and not module._forward_pre_hooks
    per_layer: dict[str, dict[str, float]] = {}
    for index in range(1, adapter.num_layers() + 1):
        heads = [
            np.mean(timer.timings_ns[f"layer{index}.head{h}"]) for h in range(adapter.num_heads())
        ]
        per_layer[f"layer{index}"] = {
            "attention_heads_ns": float(sum(heads)),
            "head_mean_ns": float(np.mean(heads)),
            "head_max_ns": float(max(heads)),
            "projection_ns": float(np.mean(timer.timings_ns[f"layer{index}.projection"])),
            "ffn_ns": float(np.mean(timer.timings_ns[f"layer{index}.ffn"])),
        }
    return {
        "model": adapter.name,
        "family": adapter.family,
        "seq_len": adapter.seq_len,
        "num_layers": adapter.num_layers(),
        "num_heads": adapter.num_heads(),
        "repetitions": repetitions,
        "warmup": warmup,
        "blocks_ns": {name: [int(v) for v in values] for name, values in timer.timings_ns.items()},
        "per_layer": per_layer,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Profile an LLM per attention head, projection and FFN block."
    )
    parser.add_argument("--model", required=True, choices=["gpt2", "tinyllama"])
    parser.add_argument("--weights", type=Path, default=None, help="checkpoint directory or .npz")
    parser.add_argument("--config", default=None, help="JSON overrides of the model configuration")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="blocks")
    parser.add_argument("--node-id", default=socket.gethostname())
    parser.add_argument("--tag", help="process tag (ignored; present for pkill -f)")
    args = parser.parse_args(argv)

    from src.common.seeds import seed_everything
    from src.models import registry
    from src.models.inference import apply_thread_count

    seed_everything(args.seed)
    apply_thread_count()
    overrides = json.loads(args.config) if args.config else None
    adapter = registry.get(args.model, seq_len=args.seq_len, config=overrides)
    assert isinstance(adapter, DecoderStageAdapter)
    if args.weights is not None and args.weights.exists():
        adapter.load_weights(args.weights)
    else:
        adapter.build(args.seed)
    x = adapter.make_input(args.seed)
    result = profile_blocks(adapter, x, repetitions=args.repetitions, warmup=args.warmup)
    origin = Origin(time.time(), time.perf_counter_ns())
    result.update({"run_id": args.run_id, "node_id": args.node_id, "origin": origin.__dict__})
    write_json_atomic(args.output, result)
    for name, values in result["per_layer"].items():
        print(
            f"{name}: heads {values['attention_heads_ns'] / 1e6:.2f} ms "
            f"(max head {values['head_max_ns'] / 1e6:.2f}), "
            f"projection {values['projection_ns'] / 1e6:.2f} ms, "
            f"ffn {values['ffn_ns'] / 1e6:.2f} ms",
            file=sys.stderr,
        )
    print(f"written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
