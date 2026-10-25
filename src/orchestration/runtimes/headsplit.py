"""HeadSplit as an optional external LLM runtime.

HeadSplit (the author's separate repository) is a PyTorch RPC runtime for
head-level partitioning and inter-layer speculation of decoder-only LLMs on
the five-device cluster: one controller and four workers, all Raspberry Pis.
The testbed does not import it; provisioning installs a pinned commit with its
own virtual environment (``<headsplit_dir>/.venv``) on the devices
(``headsplit_commit``, ``headsplit_dir`` in ``config.yaml``), and this runtime
starts its scripts as tagged device-side processes under the testbed's
monitors, shaping, heterogeneity and metadata:

* on every worker: ``scripts/run_worker.py --name workerN --rank N
  --world-size W --master-addr <controller> --master-port P --model M``;
* on the controller: a per-run HeadSplit ``config.yaml`` listing the workers,
  then ``scripts/run_controller.py --config ... --profile ... --tokens N
  --mode <mode>``; its summary lines ("<mode> total ... ms prefill ... ms
  decode ... ms/token payload ... MB") are parsed into
  ``results/<run_id>/headsplit/results.json``. A controller that accepts
  ``--json`` (HeadSplit v0.1.0 and later) writes the same figures, plus the
  per-token timeline, as JSON, which is used instead of the summary lines.

Workers exit when the controller shuts the RPC group down, so each repetition
starts them afresh; teardown stops anything left.

The runtime also exposes to NNPM its scenario block (the ``headsplit`` block of
the scenario schema), per-device readiness checks (checkout, environment,
profile, checkpoint) and three maintenance actions: install or update
HeadSplit from the mirror, build the speculation profile on the edge server
and copy it to the controller, and distribute a checkpoint to the devices.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.common.nodes import Node
from src.common.runs import write_json_atomic
from src.orchestration import local_mode
from src.orchestration.runtimes import (
    STATUS_FAIL,
    STATUS_OK,
    STATUS_WARN,
    ActionSession,
    ActionSpec,
    Check,
    RuntimeActionError,
)

if TYPE_CHECKING:
    from src.orchestration.experiment import ExperimentContext

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROFILE = "artifacts/profile.json"
MODES = ("headsplit", "head_only", "layer_based", "all")

SUMMARY_RE = re.compile(
    r"^(?P<mode>headsplit|head_only|layer_based)\s+total\s+(?P<total_ms>[\d.]+) ms\s+"
    r"prefill\s+(?P<prefill_ms>[\d.]+) ms\s+decode\s+(?P<decode_ms>[\d.]+) ms/token\s+"
    r"payload\s+(?P<payload_mb>[\d.]+) MB"
    r"(?:\s+speculation (?P<accepted>\d+)/(?P<attempts>\d+) accepted)?"
    r"(?:\s+re-placements (?P<replacements>\d+))?"
)
SPEEDUP_RE = re.compile(r"^\s+(?P<mode>head_only|headsplit)\s+(?P<speedup>[\d.]+) times")


class HeadSplitError(RuntimeActionError):
    pass


# --- parsing -------------------------------------------------------------------------------------


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


def parse_controller_json(data: dict[str, Any]) -> dict[str, Any]:
    """The JSON results file of ``run_controller.py --json``: the same per-mode figures
    as the summary lines, plus whatever the controller adds (per-token times, the
    speculation decisions, the placement, the measured round-trip times)."""
    modes: dict[str, dict[str, Any]] = {}
    for mode, entry in (data.get("modes") or {}).items():
        modes[mode] = {
            "total_ms": float(entry["total_ms"]),
            "prefill_ms": float(entry.get("prefill_ms", 0.0)),
            "decode_ms_per_token": float(entry.get("decode_ms_per_token", 0.0)),
            "payload_mb": float(entry.get("payload_mb", 0.0)),
            "speculation_accepted": entry.get("speculation_accepted"),
            "speculation_attempts": entry.get("speculation_attempts"),
            "replacements": int(entry.get("replacements", 0) or 0),
            "per_token_ms": entry.get("per_token_ms"),
            "speculation": entry.get("speculation"),
            "placement": entry.get("placement"),
        }
    return {
        "modes": modes,
        "speedup_vs_layer_based": {
            k: float(v) for k, v in (data.get("speedup_vs_layer_based") or {}).items()
        },
        "rtt_ms": data.get("rtt_ms"),
        "origin": data.get("origin"),
    }


# --- commands ------------------------------------------------------------------------------------


def headsplit_dir(ctx: ExperimentContext, node: Node) -> str:
    return ctx.config.headsplit_path


def headsplit_python(ctx: ExperimentContext, node: Node) -> str:
    """The interpreter of HeadSplit's own environment, ``<headsplit_dir>/.venv`` as
    provisioning creates it (local mode: the testbed interpreter)."""
    if local_mode.is_local(node):
        return sys.executable
    return f"{ctx.config.headsplit_path}/.venv/bin/python"


def checkpoint_dirname(model: str) -> str:
    """The directory a Hugging Face model name maps to under the models directory."""
    return model.strip("/").replace("/", "__")


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
    json_path: str | None = None,
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
    if json_path:
        args += ["--json", json_path]
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


def controller_accepts_json(ctx: ExperimentContext, controller: Node) -> bool:
    """Whether the checked-out run_controller.py has the ``--json`` option."""
    if ctx.plan_only:
        return False
    directory = headsplit_dir(ctx, controller)
    outcome = ctx.executor.run_on_node(
        controller,
        f"grep -q -- '--json' {shlex.quote(directory + '/scripts/run_controller.py')} "
        "&& echo yes || echo no",
        check=False,
    )
    return outcome.stdout.strip().endswith("yes")


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
    model = settings.get("model", DEFAULT_MODEL)
    tokens = int(settings.get("tokens", 50))
    mode = settings.get("mode", "headsplit")
    profile = settings.get("profile", DEFAULT_PROFILE)
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
    json_results = controller_accepts_json(ctx, controller)
    if json_results:
        ctx.emit("the HeadSplit controller writes JSON results (--json)")

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
        json_path = f"{cdp.results_dir}/headsplit_rep{repetition}.json" if json_results else None
        command = controller_command(
            ctx,
            controller,
            config_path,
            profile,
            master,
            port,
            tokens,
            mode,
            prompt,
            extra,
            json_path,
        )
        if ctx.plan_only:
            ctx.emit(f"controller: {command}")
            continue
        outcome = ctx.executor.run_on_node(
            controller, command, timeout=float(settings.get("timeout_s", 3600))
        )
        parsed: dict[str, Any] | None = None
        if json_path is not None:
            local_json = out_dir / f"controller_rep{repetition}.json"
            try:
                ctx.executor.fetch(controller, json_path, local_json)
                parsed = parse_controller_json(json.loads(local_json.read_text(encoding="utf-8")))
            except (FileNotFoundError, ValueError, KeyError) as exc:
                ctx.emit(f"JSON results unusable ({exc}); parsing the summary lines instead")
                parsed = None
        if parsed is None or not parsed["modes"]:
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
            "json_results": json_results,
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


def summarise_results(data: dict[str, Any]) -> dict[str, Any]:
    """Per-mode means over the repetitions, for the run summary page."""
    repetitions = data.get("repetitions") or []
    modes: dict[str, dict[str, list[float]]] = {}
    for rep in repetitions:
        for mode, entry in rep.get("modes", {}).items():
            bucket = modes.setdefault(mode, {})
            for key in ("total_ms", "prefill_ms", "decode_ms_per_token", "payload_mb"):
                if entry.get(key) is not None:
                    bucket.setdefault(key, []).append(float(entry[key]))
            if entry.get("speculation_attempts"):
                bucket.setdefault("acceptance", []).append(
                    float(entry["speculation_accepted"]) / float(entry["speculation_attempts"])
                )
            bucket.setdefault("replacements", []).append(float(entry.get("replacements") or 0))
    speedups: dict[str, list[float]] = {}
    for rep in repetitions:
        for mode, value in rep.get("speedup_vs_layer_based", {}).items():
            speedups.setdefault(mode, []).append(float(value))
    rows = []
    for mode in ("layer_based", "head_only", "headsplit"):
        if mode not in modes:
            continue
        bucket = modes[mode]
        mean = {k: sum(v) / len(v) for k, v in bucket.items() if v}
        rows.append(
            {
                "mode": mode,
                **mean,
                "speedup_vs_layer_based": (
                    sum(speedups[mode]) / len(speedups[mode]) if mode in speedups else None
                ),
            }
        )
    return {
        "kind": "headsplit",
        "model": data.get("model"),
        "tokens": data.get("tokens"),
        "mode": data.get("mode"),
        "controller": data.get("controller"),
        "workers": data.get("workers", []),
        "repetitions": len(repetitions),
        "json_results": bool(data.get("json_results")),
        "modes": rows,
        "mean_total_ms": data.get("mean_total_ms", {}),
    }


# --- readiness checks and maintenance actions ------------------------------------------------


def _models_dir(session: ActionSession, node: Node) -> str:
    if local_mode.is_local(node):
        return f"{node.workdir or '.'}/models"
    return session.config.ed_models_path


def _edge_headsplit_dir(session: ActionSession) -> Path:
    """HeadSplit's checkout on the edge server (where the profile is built)."""
    if session.local:
        return Path(session.config.headsplit_path)
    return session.config.bs_root_path / "headsplit"


class HeadSplitRuntime:
    name = "headsplit"
    label = "HeadSplit (head-level LLM partitioning, external)"
    external = True
    description = (
        "Runs the HeadSplit controller and workers on the devices: head-level partitioning "
        "and inter-layer speculation of a decoder-only LLM under the testbed's shaping, load, "
        "heterogeneity and monitors. No per-layer profile: aggregation and optimisation are "
        "skipped."
    )

    @property
    def schema(self) -> dict[str, Any]:
        from src.orchestration import scenario as scenario_module

        block: dict[str, Any] = scenario_module.load_file_schema()["properties"]["headsplit"]
        return block

    def run(self, ctx: ExperimentContext) -> None:
        run_headsplit(ctx)

    # --- checks --------------------------------------------------------------------------------

    def check_node(self, session: ActionSession, node: Node, params: dict[str, Any]) -> list[Check]:
        if node.is_edge_server:
            return []
        config = session.config
        directory = config.headsplit_path
        q = shlex.quote
        checks: list[Check] = []

        def run(command: str) -> tuple[int, str]:
            outcome = session.executor.run_on_node(node, command, check=False, timeout=120)
            return outcome.exit_status, (outcome.stdout.strip() or outcome.stderr.strip())

        status, head = run(f"git -C {q(directory)} rev-parse HEAD 2>/dev/null || echo missing")
        head = head.splitlines()[-1] if head else "missing"
        wanted = config.headsplit_commit
        if head == "missing":
            if session.local and Path(directory, "scripts", "run_controller.py").exists():
                checks.append(Check("checkout", STATUS_OK, f"{directory} (local mode)"))
            else:
                checks.append(Check("checkout", STATUS_FAIL, f"no checkout at {directory}"))
        elif wanted and not head.startswith(wanted) and not wanted.startswith(head):
            checks.append(Check("checkout", STATUS_WARN, f"{head[:12]} (pinned {wanted[:12]})"))
        else:
            checks.append(Check("checkout", STATUS_OK, head[:12]))

        if session.local:
            python = sys.executable
            probe = "import sys; print('local mode, interpreter', sys.version.split()[0])"
        else:
            python = f"{directory}/.venv/bin/python"
            probe = (
                "import headsplit, torch, transformers; "
                "print('torch', torch.__version__, 'transformers', transformers.__version__)"
            )
        status, output = run(f"{q(python)} -c {q(probe)}")
        checks.append(
            Check(
                "environment",
                STATUS_OK if status == 0 else STATUS_FAIL,
                output.splitlines()[-1] if output else f"exit status {status}",
            )
        )

        profile = params.get("profile") or DEFAULT_PROFILE
        profile_path = profile if profile.startswith("/") else f"{directory}/{profile}"
        status, _ = run(f"test -f {q(profile_path)}")
        checks.append(Check("profile", STATUS_OK if status == 0 else STATUS_WARN, profile_path))

        model = params.get("model") or DEFAULT_MODEL
        checkpoint = f"{_models_dir(session, node)}/{checkpoint_dirname(model)}"
        status, _ = run(f"test -f {q(checkpoint + '/config.json')}")
        checks.append(
            Check(
                "checkpoint",
                STATUS_OK if status == 0 else STATUS_WARN,
                checkpoint
                if status == 0
                else f"{checkpoint} absent (Hugging Face cache or distribute the checkpoint)",
            )
        )
        return checks

    # --- actions -------------------------------------------------------------------------------

    def actions(self) -> list[ActionSpec]:
        nodes_schema = {
            "type": "array",
            "items": {"type": "string"},
            "description": "devices (all end devices when empty)",
            "x-nnpm": {"widget": "nodes"},
        }
        return [
            ActionSpec(
                "install",
                "Install or update HeadSplit",
                "Clone or update HeadSplit from the mirror at the pinned commit and build its "
                "environment on the selected devices (device_setup.sh headsplit).",
                {"type": "object", "properties": {"nodes": nodes_schema}},
            ),
            ActionSpec(
                "profile",
                "Build the speculation profile",
                "Run HeadSplit's profile_offline.py on the edge server (needs upstream access "
                "for the calibration data) and copy the profile to the controller.",
                {
                    "type": "object",
                    "properties": {
                        "controller": {"type": "string", "x-nnpm": {"widget": "node"}},
                        "model": {"type": "string", "default": DEFAULT_MODEL},
                        "chunks": {"type": "integer", "minimum": 1, "default": 64},
                        "chunk_len": {"type": "integer", "minimum": 1, "default": 256},
                        "epsilon": {"type": "number", "exclusiveMinimum": 0, "default": 0.05},
                        "allow_upstream": {
                            "type": "boolean",
                            "default": False,
                            "description": "the edge server may reach the internet",
                        },
                    },
                    "required": ["controller"],
                },
            ),
            ActionSpec(
                "checkpoint",
                "Distribute a checkpoint",
                "Copy <bs_models_dir>/<model directory> to the selected devices and verify "
                "SHA256SUMS there.",
                {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string", "default": DEFAULT_MODEL},
                        "nodes": nodes_schema,
                    },
                },
            ),
        ]

    def perform(self, action: str, params: dict[str, Any], session: ActionSession) -> None:
        if action == "install":
            self._install(params, session)
        elif action == "profile":
            self._profile(params, session)
        elif action == "checkpoint":
            self._checkpoint(params, session)
        else:
            raise KeyError(f"HeadSplit has no action {action!r}")

    def _install(self, params: dict[str, Any], session: ActionSession) -> None:
        config = session.config
        if config.headsplit_commit is None:
            raise HeadSplitError("set headsplit_commit in config.yaml first")
        from src.provisioning.render import mirror_url

        mirror = mirror_url(config, session.topology)
        for node in session.devices(params.get("nodes")):
            if session.stopped():
                return
            if local_mode.is_local(node):
                code_dir = str(session.repo_root)
                root = node.workdir or "."
                venv = f"{root}/venv"
            else:
                code_dir = config.ed_code_dir
                root = config.ed_root.rstrip("/")
                venv = config.ed_venv_dir
            argv = [
                "bash",
                f"{code_dir}/scripts/provision/device_setup.sh",
                "--rendered",
                f"{root}/provision/rendered",
                "--node",
                node.id,
                "--code",
                code_dir,
                "--venv",
                venv,
                "--locks",
                "edge_device",
                "--mirror",
                mirror,
                "--headsplit-dir",
                config.headsplit_path,
                "--headsplit-commit",
                config.headsplit_commit,
            ]
            if session.local:
                argv.append("--dry-run")  # local mode: the plan, nothing is installed
            argv.append("headsplit")
            session.log(f"[task][info] {node.id}: install HeadSplit {config.headsplit_commit[:12]}")
            session.executor.run_on_node(node, shlex.join(argv), timeout=3600)

    def _profile(self, params: dict[str, Any], session: ActionSession) -> None:
        config = session.config
        controller = session.topology.get(params["controller"])
        model = params.get("model") or DEFAULT_MODEL
        directory = _edge_headsplit_dir(session)
        if not session.local:
            if not params.get("allow_upstream"):
                raise HeadSplitError(
                    "profile_offline.py downloads its calibration data: tick allow_upstream "
                    "if the edge server may reach the internet"
                )
            if config.headsplit_commit is None:
                raise HeadSplitError("set headsplit_commit in config.yaml first")
            bare = config.mirror_path / "src" / "headsplit.git"
            if not bare.exists():
                raise HeadSplitError(f"{bare} is missing: run build_mirror.sh headsplit first")
            if (directory / ".git").exists():
                session.run_local(["git", "-C", str(directory), "fetch", "--quiet"])
            else:
                session.run_local(["git", "clone", "--quiet", str(bare), str(directory)])
            session.run_local(
                ["git", "-C", str(directory), "checkout", "--quiet", config.headsplit_commit]
            )
            python = directory / ".venv" / "bin" / "python"
            if not python.exists():
                session.run_local([sys.executable, "-m", "venv", str(directory / ".venv")])
                session.run_local(
                    [
                        str(directory / ".venv" / "bin" / "pip"),
                        "install",
                        "--no-index",
                        "--find-links",
                        str(config.mirror_path / "wheels" / "x86_64-headsplit"),
                        "-r",
                        str(directory / "requirements.txt"),
                    ]
                )
        else:
            python = Path(sys.executable)
        script = directory / "scripts" / "profile_offline.py"
        if not script.exists():
            raise HeadSplitError(f"{script} is missing")
        output = directory / "artifacts" / "profile.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        session.run_local(
            [
                str(python),
                "scripts/profile_offline.py",
                "--model",
                model,
                "--chunks",
                str(int(params.get("chunks", 64))),
                "--chunk-len",
                str(int(params.get("chunk_len", 256))),
                "--epsilon",
                f"{float(params.get('epsilon', 0.05)):g}",
                "--out",
                str(output),
            ],
            cwd=directory,
        )
        if not output.exists():
            raise HeadSplitError(f"profile_offline.py wrote no {output}")
        remote = f"{config.headsplit_path}/{DEFAULT_PROFILE}"
        session.log(f"[task][info] copying the profile to {controller.id}:{remote}")
        session.executor.put(controller, output, remote)

    def _checkpoint(self, params: dict[str, Any], session: ActionSession) -> None:
        config = session.config
        model = params.get("model") or DEFAULT_MODEL
        dirname = checkpoint_dirname(model)
        source = config.bs_models_path / dirname
        if not (source / "SHA256SUMS").exists():
            raise HeadSplitError(
                f"{source} has no SHA256SUMS: fetch the checkpoint into bs_models_dir first "
                "(python -m src.models.weights fetch)"
            )
        for node in session.devices(params.get("nodes")):
            if session.stopped():
                return
            target = f"{_models_dir(session, node)}/{dirname}"
            session.log(f"[task][info] {node.id}: {model} -> {target}")
            session.copy_to_node(node, source, target)
            session.executor.run_on_node(
                node, f"cd {shlex.quote(target)} && sha256sum -c --quiet SHA256SUMS", timeout=3600
            )
            session.log(f"[task][info] {node.id}: checksums verified")

    def results(self, run_dir: Path) -> dict[str, Any] | None:
        path = run_dir / "headsplit" / "results.json"
        if not path.exists():
            return None
        return summarise_results(load_results(str(path)))


RUNTIME = HeadSplitRuntime()
