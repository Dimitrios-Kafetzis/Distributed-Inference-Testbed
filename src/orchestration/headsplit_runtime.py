"""HeadSplit as an optional external LLM runtime.

HeadSplit (the author's separate repository) is a PyTorch RPC runtime for
head-level partitioning and inter-layer speculation of decoder-only LLMs on
the five-device cluster: one controller and four workers, all Raspberry Pis.
The testbed does not import it; provisioning installs a pinned commit with its
own virtual environment on the devices (``headsplit_commit``,
``headsplit_dir`` in ``config.yaml``), and this module starts its scripts as
tagged device-side processes under the testbed's monitors, shaping,
heterogeneity and metadata:

* on every worker: ``scripts/run_worker.py --name workerN --rank N
  --world-size W --master-addr <controller> --master-port P --model M``;
* on the controller: a per-run HeadSplit ``config.yaml`` listing the workers,
  then ``scripts/run_controller.py --config ... --profile ... --tokens N
  --mode <mode>``; its summary lines ("<mode> total ... ms prefill ... ms
  decode ... ms/token payload ... MB") are parsed into
  ``results/<run_id>/headsplit/results.json``.

Workers exit when the controller shuts the RPC group down, so each repetition
starts them afresh; teardown stops anything left.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from typing import TYPE_CHECKING, Any

import yaml

from src.common.nodes import Node
from src.common.runs import write_json_atomic
from src.orchestration import local_mode

if TYPE_CHECKING:
    from src.orchestration.experiment import ExperimentContext

SUMMARY_RE = re.compile(
    r"^(?P<mode>headsplit|head_only|layer_based)\s+total\s+(?P<total_ms>[\d.]+) ms\s+"
    r"prefill\s+(?P<prefill_ms>[\d.]+) ms\s+decode\s+(?P<decode_ms>[\d.]+) ms/token\s+"
    r"payload\s+(?P<payload_mb>[\d.]+) MB"
    r"(?:\s+speculation (?P<accepted>\d+)/(?P<attempts>\d+) accepted)?"
    r"(?:\s+re-placements (?P<replacements>\d+))?"
)
SPEEDUP_RE = re.compile(r"^\s+(?P<mode>head_only|headsplit)\s+(?P<speedup>[\d.]+) times")


class HeadSplitError(RuntimeError):
    pass


def parse_controller_output(text: str) -> dict[str, Any]:
    """The per-mode summaries and speed-ups printed by run_controller.py."""
    modes: dict[str, dict[str, float | int | None]] = {}
    speedups: dict[str, float] = {}
    for line in text.splitlines():
        match = SUMMARY_RE.match(line.strip())
        if match:
            fields = match.groupdict()
            modes[fields["mode"]] = {
                "total_ms": float(fields["total_ms"]),
                "prefill_ms": float(fields["prefill_ms"]),
                "decode_ms_per_token": float(fields["decode_ms"]),
                "payload_mb": float(fields["payload_mb"]),
                "speculation_accepted": int(fields["accepted"]) if fields["accepted"] else None,
                "speculation_attempts": int(fields["attempts"]) if fields["attempts"] else None,
                "replacements": int(fields["replacements"]) if fields["replacements"] else 0,
            }
            continue
        match = SPEEDUP_RE.match(line)
        if match:
            speedups[match.group("mode")] = float(match.group("speedup"))
    return {"modes": modes, "speedup_vs_layer_based": speedups}


def headsplit_dir(ctx: ExperimentContext, node: Node) -> str:
    return ctx.config.headsplit_path


def headsplit_python(ctx: ExperimentContext, node: Node) -> str:
    if local_mode.is_local(node):
        return sys.executable
    return f"{ctx.config.headsplit_path}/venv/bin/python"


def worker_command(
    ctx: ExperimentContext,
    node: Node,
    name: str,
    rank: int,
    world_size: int,
    master: str,
    port: int,
    model: str,
) -> str:
    directory = headsplit_dir(ctx, node)
    python = headsplit_python(ctx, node)
    args = [
        "--name",
        name,
        "--rank",
        str(rank),
        "--world-size",
        str(world_size),
        "--master-addr",
        master,
        "--master-port",
        str(port),
        "--model",
        model,
    ]
    return (
        f"cd {shlex.quote(directory)} && {shlex.quote(python)} "
        f"scripts/run_worker.py {shlex.join(args)}"
    )


def controller_command(
    ctx: ExperimentContext,
    node: Node,
    config_path: str,
    profile: str,
    master: str,
    port: int,
    tokens: int,
    mode: str,
    prompt: str | None,
    extra: list[str],
) -> str:
    directory = headsplit_dir(ctx, node)
    python = headsplit_python(ctx, node)
    args = [
        "--config",
        config_path,
        "--profile",
        profile,
        "--master-addr",
        master,
        "--master-port",
        str(port),
        "--tokens",
        str(tokens),
        "--mode",
        mode,
    ]
    if prompt:
        args += ["--prompt", prompt]
    args += extra
    return (
        f"cd {shlex.quote(directory)} && {shlex.quote(python)} "
        f"scripts/run_controller.py {shlex.join(args)}"
    )


def headsplit_config(ctx: ExperimentContext, workers: list[Node], model: str, profile: str) -> str:
    """A HeadSplit config.yaml naming the workers by their data-plane addresses."""
    settings = ctx.scenario.get("headsplit", {})
    data: dict[str, Any] = {
        "model_name": model,
        "dtype": "float32",
        "devices": [
            {
                "name": f"worker{i}",
                "host": local_mode.LOOPBACK if ctx.local else w.data_address,
                "memory_bytes": int(settings.get("worker_memory_bytes", 3 * 1024**3)),
            }
            for i, w in enumerate(workers, start=1)
        ],
        "speculation": {
            "epsilon": float(settings.get("epsilon", 0.05)),
            "alpha_min": float(settings.get("alpha_min", 0.70)),
        },
        "memory_replace_threshold": float(settings.get("memory_replace_threshold", 0.85)),
        "rtt_margin": float(settings.get("rtt_margin", 1.15)),
        "profile_path": profile,
    }
    return yaml.safe_dump(data, sort_keys=False)


def run_headsplit(ctx: ExperimentContext) -> None:
    settings = ctx.scenario.get("headsplit") or {}
    if ctx.config.headsplit_commit is None and not ctx.local:
        raise HeadSplitError(
            "HeadSplit is not configured: set headsplit_commit in config.yaml "
            "and provision the devices"
        )
    try:
        controller = ctx.topology.get(settings["controller"])
        workers = [ctx.topology.get(w) for w in settings["workers"]]
    except KeyError as exc:
        raise HeadSplitError(
            f"headsplit.controller and headsplit.workers must name nodes of nodes.json ({exc})"
        ) from exc
    model = settings.get("model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokens = int(settings.get("tokens", 50))
    mode = settings.get("mode", "headsplit")
    profile = settings.get("profile", "artifacts/profile.json")
    prompt = settings.get("prompt")
    extra = list(settings.get("extra_args", []))
    port = int(ctx.config.headsplit_master_port)
    master = local_mode.LOOPBACK if ctx.local else controller.data_address
    world_size = len(workers) + 1
    repetitions = int(ctx.scenario.get("repetitions", 1))
    out_dir = ctx.paths.root / "headsplit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The per-run HeadSplit configuration on the controller.
    config_text = headsplit_config(ctx, workers, model, profile)
    cdp = ctx.device_paths(controller)
    config_path = f"{cdp.results_dir}/headsplit_config.yaml"
    ctx.executor.run_on_node(
        controller,
        f"mkdir -p {shlex.quote(cdp.results_dir)} && "
        f"cat > {shlex.quote(config_path)} <<'EOF'\n{config_text}EOF",
    )
    (out_dir / "headsplit_config.yaml").write_text(config_text, encoding="utf-8")

    results: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        ctx.emit(
            f"HeadSplit repetition {repetition + 1}/{repetitions}: "
            f"{len(workers)} worker(s), mode {mode}"
        )
        for rank, worker in enumerate(workers, start=1):
            command = worker_command(
                ctx, worker, f"worker{rank}", rank, world_size, master, port, model
            )
            ctx.start_device_process(worker, f"hs-worker{rank}-r{repetition}", command)
        if ctx.plan_only:
            command = controller_command(
                ctx, controller, config_path, profile, master, port, tokens, mode, prompt, extra
            )
            ctx.emit(f"controller: {command}")
            continue
        command = controller_command(
            ctx, controller, config_path, profile, master, port, tokens, mode, prompt, extra
        )
        outcome = ctx.executor.run_on_node(
            controller, command, timeout=float(settings.get("timeout_s", 3600))
        )
        parsed = parse_controller_output(outcome.stdout)
        if not parsed["modes"]:
            raise HeadSplitError(
                "the HeadSplit controller printed no summary line; see logs/merged.log"
            )
        parsed["repetition"] = repetition
        results.append(parsed)
        (out_dir / f"controller_rep{repetition}.log").write_text(outcome.stdout, encoding="utf-8")
        for rank, worker in enumerate(workers, start=1):
            ctx.executor.stop_tagged(
                worker, ctx.tagged(worker, f"hs-worker{rank}-r{repetition}"), grace_s=0.5
            )
    if not ctx.plan_only:
        summary: dict[str, Any] = {
            "model": model,
            "tokens": tokens,
            "mode": mode,
            "controller": controller.id,
            "workers": [w.id for w in workers],
            "repetitions": results,
        }
        modes = {m for r in results for m in r["modes"]}
        summary["mean_total_ms"] = {
            m: sum(r["modes"][m]["total_ms"] for r in results if m in r["modes"])
            / sum(1 for r in results if m in r["modes"])
            for m in modes
        }
        write_json_atomic(out_dir / "results.json", summary)
        ctx.outputs["headsplit"] = summary["mean_total_ms"]
        ctx.emit(
            "HeadSplit results: "
            + ", ".join(f"{m}: {v:.1f} ms" for m, v in summary["mean_total_ms"].items())
        )
    ctx.outputs["headsplit_results"] = str(out_dir / "results.json")


def load_results(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data
