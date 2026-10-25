"""The experiment pipeline: one scenario, from preflight to teardown.

Stages (each registers every change it makes in the run journal, so that
teardown can undo it in reverse order, also after a failure):

1. **preflight**: metadata (scenario, Git commit, framework and kernel
   versions, NTP status of every node) written before any change;
2. **apply**: edge-server frequency policy, heterogeneity, network shaping,
   background traffic, compute load, monitoring agents; then a settling time;
3. **profile**: a start signal T0; per-layer profiling on all devices
   concurrently; the edge server's capacity; optional partitioned-inference
   runs against the edge-server service; (or the HeadSplit runtime);
4. **collect**: latency and system-metric capture on every device, then the
   result files (hostname-prefixed) into the run directory;
5. **aggregate**: node statistics, per-device averages and trace merging,
   partition feasibility: the three thesis output files;
6. **optimise**: the partition vector for the scenario's objective;
7. **teardown**: always.

``benchmarking`` is the four-stage pipeline of run_full_benchmarking.sh:
dispatch, collection, post-processing (latency measurement, aggregation,
feasibility), outputs; no load or shaping is applied.

Usage::

    python -m src.orchestration.experiment run --scenario examples/scenarios/local_demo.yaml --local
    python -m src.orchestration.experiment benchmark --scenario baseline.yaml
    python -m src.orchestration.experiment plan --scenario S.yaml [--local]
    python -m src.orchestration.experiment teardown [--run-id ID]
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.benchmarking import partition_calculator, result_aggregation
from src.benchmarking.benchmark_results import BenchmarkResults
from src.common import metadata as metadata_module
from src.common.clock import Clock, SystemClock
from src.common.nodes import Node, Topology, load_topology, save_topology
from src.common.process_tags import TaggedProcess, stop_all_command
from src.common.runs import (
    JournalEntry,
    RunJournal,
    RunPaths,
    new_run_id,
    open_journals,
    write_json_atomic,
)
from src.config import REPO_ROOT, Config, get_ssh_password, load_config
from src.network import collection, node_statistics
from src.network.ssh_executor import Executor, ExperimentAbortedError, make_transport
from src.network.transport import FakeTransport, LocalTransport
from src.orchestration import local_mode, runtimes
from src.orchestration import scenario as scenario_module
from src.orchestration.geo import calculate_distance, calculate_latency
from src.partitioning import delay_optimizer
from src.partitioning.algorithms import SearchLog
from src.partitioning.delay_model import DelayModel, DelayModelConfig, profiles_from_files

NNPM = "nnpm"


class ExperimentError(RuntimeError):
    pass


@dataclass
class DevicePaths:
    python: str
    code_dir: str
    models_dir: str
    data_dir: str
    results_dir: str  # this run's result files on the device
    run_dir: str  # PID files and logs of tagged processes
    sysfs_root: str

    def module(self, name: str, *args: str) -> str:
        return (
            f"cd {shlex.quote(self.code_dir)} && {shlex.quote(self.python)} -m {name} "
            + shlex.join(args)
        )


@dataclass
class ExperimentContext:
    config: Config
    topology: Topology
    scenario: dict[str, Any]
    run_id: str
    paths: RunPaths
    executor: Executor
    journal: RunJournal
    local: bool = False
    allow_dirty: bool = False
    clock: Clock = field(default_factory=SystemClock)
    nodes_path: Path | None = None
    scenario_version: dict[str, Any] = field(default_factory=dict)
    t0: float | None = None
    local_processes: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict)
    plan_only: bool = False
    outputs: dict[str, Any] = field(default_factory=dict)
    # Set by NNPM's Stop button: the pipeline stops before its next stage and tears down.
    cancel: threading.Event = field(default_factory=threading.Event)

    # --- nodes -----------------------------------------------------------------------------------

    @property
    def devices(self) -> list[Node]:
        return self.topology.select(self.scenario.get("devices"))

    @property
    def edge(self) -> Node | None:
        return self.topology.edge_server()

    @property
    def edge_data_address(self) -> str:
        edge = self.edge
        if edge is None:
            raise ExperimentError("nodes.json has no 'Base Station' entry")
        return edge.data_address

    def device_paths(self, node: Node) -> DevicePaths:
        if local_mode.is_local(node):
            root = node.workdir or "."
            return DevicePaths(
                python=sys.executable,
                code_dir=str(REPO_ROOT),
                models_dir=f"{root}/models",
                data_dir=f"{root}/data",
                results_dir=f"{root}/results/{self.run_id}",
                run_dir=f"{root}/run/{self.run_id}",
                sysfs_root=f"{root}/sysfs",
            )
        c = self.config
        return DevicePaths(
            python=f"{c.ed_venv_dir}/bin/python",
            code_dir=c.ed_code_dir,
            models_dir=c.ed_models_path,
            data_dir=c.ed_data_path,
            results_dir=f"{c.ed_results_path}/{self.run_id}",
            run_dir=f"{c.ed_root.rstrip('/')}/run/{self.run_id}",
            sysfs_root="/",
        )

    def sudo(self, node: Node, command: str) -> str:
        """Run ``command`` as root on a real device; local mode runs it unprivileged.

        Module commands start with ``cd <code dir> && ``; the ``sudo -n`` goes in front
        of the interpreter, not in front of ``cd`` (a shell builtin sudo cannot run).
        The sudoers rule installed by provisioning allows exactly this interpreter
        with the heterogeneity module, and tc.
        """
        if local_mode.is_local(node):
            return command
        head, sep, tail = command.partition(" && ")
        if sep and head.startswith("cd "):
            return f"{head} && sudo -n {tail}"
        return f"sudo -n {command}"

    # --- settings per node -----------------------------------------------------------------------

    def network_for(self, node: Node) -> dict[str, Any]:
        return scenario_module.for_node(self.scenario.get("network"), node.id)

    def load_for(self, node: Node) -> dict[str, Any]:
        return scenario_module.for_node(self.scenario.get("load"), node.id)

    def heterogeneity_for(self, node: Node) -> dict[str, Any]:
        return scenario_module.for_node(self.scenario.get("heterogeneity"), node.id)

    # --- logging and journal -------------------------------------------------------------------

    def emit(self, message: str) -> None:
        self.executor.log(NNPM, "info", message)

    def record(
        self, node_id: str, kind: str, description: str, undo: dict[str, Any]
    ) -> JournalEntry:
        return self.journal.record(node_id, kind, description, undo, self.clock)

    # --- processes -----------------------------------------------------------------------------

    def tagged(self, node: Node, component: str) -> TaggedProcess:
        return TaggedProcess(self.run_id, component, self.device_paths(node).run_dir)

    def start_device_process(
        self, node: Node, component: str, command: str, env: dict[str, str] | None = None
    ) -> None:
        process = self.tagged(node, component)
        self.record(
            node.id,
            "process",
            f"{component} on {node.id}",
            {"kind": "stop_process", "node_id": node.id, "component": component},
        )
        self.executor.start_tagged(node, process, command, env)

    def start_local_process(
        self, tag: str, argv: list[str], log_name: str
    ) -> subprocess.Popen[bytes]:
        log_path = self.paths.root / "logs" / f"{log_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.emit(f"edge server: {shlex.join(argv)}")
        self.record(NNPM, "local_process", tag, {"kind": "stop_local", "tag": tag})
        if self.plan_only:
            return subprocess.Popen(["true"])
        with log_path.open("ab") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        self.local_processes[tag] = process
        return process

    def stop_local_process(self, tag: str) -> None:
        process = self.local_processes.pop(tag, None)
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


# --- helpers -----------------------------------------------------------------------------------


def shaping_arguments(
    ctx: ExperimentContext, node: Node, net: dict[str, Any]
) -> tuple[list[str], str | None]:
    """latency_injector arguments for a node; the second item is the variation mode or None."""
    delay = net.get("delay_ms")
    if net.get("from_map") and ctx.edge is not None:
        distance = calculate_distance(ctx.edge.lat, ctx.edge.lon, node.lat, node.lon)
        delay = calculate_latency(distance)
    if net.get("added_rtt_ms"):
        delay = (delay or 0.0) + float(net["added_rtt_ms"])
    variation = net.get("variation")
    args: list[str] = []
    mode = None
    if variation and variation.get("mode") != "constant":
        low = variation.get("min_ms", delay or 0.0)
        high = variation.get("max_ms", delay or 0.0)
        args += [
            "--delay",
            f"{low:g}",
            f"{high:g}",
            "--mode",
            variation["mode"],
            "--period",
            f"{variation.get('period_s', 10):g}",
        ]
        mode = variation["mode"]
    elif delay is not None:
        args += ["--delay", f"{delay:g}"]
    if net.get("jitter_ms") is not None:
        args += ["--jitter", f"{net['jitter_ms']:g}"]
    if net.get("loss_percent") is not None:
        args += ["--loss", f"{net['loss_percent']:g}"]
    if net.get("rate_mbps") is not None:
        args += ["--rate", f"{net['rate_mbps']:g}"]
    for destination in net.get("destinations", []) or []:
        args += ["--dst", ctx.topology.get(destination).data_address]
    if ctx.local:
        args.append("--dry-run")
    return args, mode


def load_arguments(load: dict[str, Any]) -> list[str]:
    mode = load.get("mode", "fixed")
    args: list[str] = []
    if mode == "regime":
        args += ["--regime", load["regime"]]
    elif mode == "processes":
        args += ["--processes", str(load["processes"])]
    elif mode == "continuous":
        args += ["--mode", "continuous"]
        if load.get("cpu_range"):
            args += ["--cpu-range", *(f"{v:g}" for v in load["cpu_range"])]
        if load.get("memory_range"):
            args += ["--memory-range", *(f"{v:g}" for v in load["memory_range"])]
    else:
        if load.get("cpu_percent") is not None:
            args += ["--cpu", f"{load['cpu_percent']:g}"]
        if load.get("memory_percent") is not None:
            args += ["--memory", f"{load['memory_percent']:g}"]
    args += [
        "--cores",
        str(load.get("cores", 1)),
        "--cycle-s",
        f"{load.get('cycle_s', 10):g}",
        "--period-ms",
        f"{load.get('period_ms', 100):g}",
    ]
    return args


def model_args_option(ctx: ExperimentContext) -> list[str]:
    """``--model-args JSON`` for the device and edge-server tools when the scenario has them."""
    args = ctx.scenario.get("model_args") or {}
    return ["--model-args", json.dumps(args, sort_keys=True)] if args else []


def with_device_env(command: str, env: str) -> str:
    """Prefix a ``cd <dir> && <interpreter> ...`` command's interpreter with variables."""
    head, sep, tail = command.partition(" && ")
    return f"{head} && {env} {tail}" if sep else f"{env} {command}"


def input_option(ctx: ExperimentContext) -> list[str]:
    """``--input-source JSON`` when the scenario names a source other than synthetic."""
    spec = ctx.scenario.get("input") or {}
    if spec.get("source", "synthetic") == "synthetic":
        return []
    return [
        "--input-source",
        json.dumps({k: v for k, v in spec.items() if v is not None}, sort_keys=True),
    ]


def remote_list(ctx: ExperimentContext, node: Node, pattern: str) -> list[str]:
    outcome = ctx.executor.run_on_node(node, f"ls -1 {pattern} 2>/dev/null || true", check=False)
    return [line.strip() for line in outcome.stdout.splitlines() if line.strip()]


# --- stages ------------------------------------------------------------------------------------


def stage_preflight(ctx: ExperimentContext) -> None:
    ctx.emit(
        f"run {ctx.run_id}: scenario {ctx.scenario['name']} (pipeline {ctx.scenario['pipeline']})"
    )
    ctx.paths.create()
    write_json_atomic(ctx.paths.scenario, {**ctx.scenario, "_version": ctx.scenario_version})
    devices = ctx.devices
    if not devices:
        raise ExperimentError("the scenario selects no device")
    if ctx.plan_only:
        return
    meta = metadata_module.collect_run_metadata(
        run_id=ctx.run_id,
        scenario={**ctx.scenario, "_version": ctx.scenario_version},
        config=ctx.config,
        executor=ctx.executor,
        nodes=devices,
        seeds={"scenario": int(ctx.scenario.get("seed", 0))},
        allow_dirty=ctx.allow_dirty or ctx.local,
        frameworks=not ctx.local,
        local=ctx.local,
    )
    meta.save(ctx.paths.metadata)
    threshold = ctx.config.ntp_offset_threshold_ms
    for node_id, facts in meta.nodes.items():
        ntp = facts.get("ntp") or {}
        offset = ntp.get("last_offset_ms")
        if offset is not None and abs(offset) > threshold:
            raise ExperimentError(f"{node_id}: NTP offset {offset:.3f} ms exceeds {threshold} ms")
    ctx.emit("metadata recorded")


def stage_apply(ctx: ExperimentContext, full: bool = True) -> None:
    edge_cfg = ctx.scenario.get("edge_server", {})
    if edge_cfg.get("nominal_frequency", True):
        state = ctx.paths.root / "edge_frequency_state.json"
        sysfs_root = str(ctx.paths.root / "fake_sysfs") if ctx.local else "/"
        if ctx.local:
            local_mode.fake_sysfs(Path(sysfs_root), cores=2)
        argv = [
            sys.executable,
            "-m",
            "src.testbed.cpu_frequency",
            "apply",
            "--state",
            str(state),
            "--sysfs-root",
            sysfs_root,
        ]
        if not ctx.local:
            argv = ["sudo", "-n", *argv]
        ctx.record(
            NNPM,
            "edge_frequency",
            "nominal frequency policy",
            {"kind": "edge_freq_restore", "state": str(state), "sysfs_root": sysfs_root},
        )
        ctx.emit(f"edge server: {shlex.join(argv)}")
        if not ctx.plan_only:
            result = subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT)
            if result.returncode != 0:
                ctx.emit(f"frequency policy not applied: {result.stderr.strip()[-200:]}")
    for node in ctx.devices:
        dp = ctx.device_paths(node)
        if full:
            het = ctx.heterogeneity_for(node)
            if het.get("max_freq_mhz"):
                state_path = f"{dp.run_dir}/freq_state.json"
                ctx.executor.run_on_node(node, f"mkdir -p {shlex.quote(dp.run_dir)}")
                ctx.record(
                    node.id,
                    "cpufreq",
                    f"max {het['max_freq_mhz']} MHz",
                    {"kind": "cpufreq_restore", "node_id": node.id, "state": state_path},
                )
                ctx.executor.run_on_node(
                    node,
                    ctx.sudo(
                        node,
                        dp.module(
                            "src.testbed.heterogeneity",
                            "--sysfs-root",
                            dp.sysfs_root,
                            "freq",
                            "--max-mhz",
                            f"{het['max_freq_mhz']:g}",
                            "--state",
                            state_path,
                        ),
                    ),
                )
            if het.get("memory_limit_mb"):
                name = f"testbed-{ctx.run_id}"
                ctx.record(
                    node.id,
                    "cgroup",
                    f"memory {het['memory_limit_mb']} MB",
                    {"kind": "cgroup_remove", "node_id": node.id, "name": name},
                )
                ctx.executor.run_on_node(
                    node,
                    ctx.sudo(
                        node,
                        dp.module(
                            "src.testbed.heterogeneity",
                            "--sysfs-root",
                            dp.sysfs_root,
                            "cgroup",
                            "create",
                            "--name",
                            name,
                            "--memory-mb",
                            f"{het['memory_limit_mb']:g}",
                        ),
                    ),
                )
            net = ctx.network_for(node)
            args, mode = shaping_arguments(ctx, node, net)
            if args and any(a in ("--delay", "--loss", "--rate") for a in args):
                iface = ctx.config.network_interface
                ctx.record(
                    node.id,
                    "shaping",
                    f"tc on {iface}",
                    {"kind": "tc_clear", "node_id": node.id, "interface": iface},
                )
                command = dp.module("src.testbed.latency_injector", iface, *args)
                if mode:
                    ctx.start_device_process(node, "shaping", command)
                else:
                    ctx.executor.run_on_node(node, command)
            if net.get("background_traffic_mbps"):
                from src.testbed.background_traffic import BackgroundTraffic

                traffic = BackgroundTraffic(
                    ctx.edge_data_address,
                    float(net["background_traffic_mbps"]),
                    24 * 3600,
                    bind=node.data_address if not ctx.local else None,
                )
                if "iperf-server" not in ctx.local_processes and not ctx.plan_only:
                    ctx.start_local_process("iperf-server", traffic.server_argv(), "iperf_server")
                ctx.start_device_process(node, "traffic", shlex.join(traffic.client_argv()))
            load = ctx.load_for(node)
            if load:
                ctx.start_device_process(
                    node,
                    "load",
                    dp.module(
                        "src.testbed.load_generator",
                        *load_arguments(load),
                        "--seed",
                        str(ctx.scenario.get("seed", 0)),
                        "--log",
                        f"{dp.results_dir}/load.json",
                    ),
                )
        monitoring = ctx.scenario.get("monitoring", {})
        if monitoring.get("enabled", True):
            ctx.executor.run_on_node(node, f"mkdir -p {shlex.quote(dp.results_dir)}")
            ctx.start_device_process(
                node,
                "monitor",
                dp.module(
                    "src.testbed.resource_monitor",
                    "--output",
                    f"{dp.results_dir}/monitor.json",
                    "--run-id",
                    ctx.run_id,
                    "--node-id",
                    node.id,
                    "--interval",
                    f"{monitoring.get('interval_s', 1.0):g}",
                    "--sysfs-root",
                    dp.sysfs_root,
                ),
            )
            if monitoring.get("power"):
                power_args = [
                    "--output",
                    f"{dp.results_dir}/power.json",
                    "--run-id",
                    ctx.run_id,
                    "--node-id",
                    node.id,
                    "--interval",
                    f"{monitoring.get('power_interval_s', 0.5):g}",
                ]
                if ctx.local:
                    power_args.append("--fake")
                ctx.start_device_process(
                    node, "power", dp.module("src.network.power_measurement", *power_args)
                )
    settle = float(ctx.scenario.get("settle_s", 0)) if full else 0.0
    if settle > 0 and not ctx.plan_only:
        ctx.emit(f"settling for {settle:g} s")
        ctx.clock.sleep(settle)


def stage_profile(ctx: ExperimentContext) -> None:
    """The profile stage belongs to the scenario's runtime (``testbed`` by default)."""
    runtimes.get_runtime(ctx.scenario.get("runtime", runtimes.DEFAULT_RUNTIME)).run(ctx)


def profile_with_testbed(ctx: ExperimentContext) -> None:
    """The built-in runtime: a start signal, per-layer profiling on every device, the
    edge server's capacity and, optionally, partitioned inference."""
    scenario = ctx.scenario
    ctx.t0 = 0.0 if ctx.plan_only else ctx.clock.time() + float(scenario.get("start_lead_s", 3))
    ctx.emit(f"start signal T0 = {ctx.t0:.3f}")
    model = scenario["model"]
    seed = str(scenario.get("seed", 0))

    def profile_command(node: Node) -> str:
        dp = ctx.device_paths(node)
        het = ctx.heterogeneity_for(node)
        env = (
            f"TESTBED_PYTHON={shlex.quote(dp.python)} "
            f"TESTBED_ED_DATA_DIR={shlex.quote(dp.data_dir)} TESTBED_NODE_ID={node.id}"
        )
        if het.get("threads"):
            env += f" TESTBED_THREADS={het['threads']} OMP_NUM_THREADS={het['threads']}"
        script = f"{dp.code_dir}/scripts/end_device/train_and_run_benchmark.sh"
        args = [
            "both",
            dp.models_dir,
            model,
            dp.results_dir,
            "--run-id",
            ctx.run_id,
            "--node-id",
            node.id,
            "--repetitions",
            str(scenario.get("repetitions", 1)),
            "--warmup",
            str(scenario.get("warmup", 1)),
            "--seed",
            seed,
            "--start-at",
            f"{ctx.t0:.6f}",
            *model_args_option(ctx),
            *input_option(ctx),
        ]
        return (
            f"cd {shlex.quote(dp.code_dir)} && {env} bash {shlex.quote(script)} {shlex.join(args)}"
        )

    ctx.executor.run_on_nodes(ctx.devices, "", per_node_command=profile_command, timeout=None)
    checks = scenario.get("checks") or {}
    if checks.get("partition_equivalence"):

        def verify_command(node: Node) -> str:
            dp = ctx.device_paths(node)
            env = f"TESTBED_ED_DATA_DIR={shlex.quote(dp.data_dir)}"
            command = dp.module(
                "src.models.inference",
                "verify",
                "--model",
                model,
                "--weights",
                f"{dp.models_dir}/{model}.npz",
                "--seed",
                seed,
                "--tolerance",
                f"{checks.get('tolerance', 1e-4):g}",
                "--output",
                f"{dp.results_dir}/verification.json",
                "--node-id",
                node.id,
                *model_args_option(ctx),
                *input_option(ctx),
            )
            return with_device_env(command, env)

        ctx.emit("partition equivalence check on every device")
        ctx.executor.run_on_nodes(ctx.devices, "", per_node_command=verify_command, timeout=None)
    edge_cfg = scenario.get("edge_server", {})
    if edge_cfg.get("profile_capacity", True):
        weights = ctx.config.bs_models_path / f"{model}.npz"
        ctx.paths.root.mkdir(parents=True, exist_ok=True)
        prep = [
            sys.executable,
            "-m",
            "src.models.inference",
            "prepare",
            "--model",
            model,
            "--weights",
            str(weights),
            "--seed",
            seed,
            *model_args_option(ctx),
        ]
        cap = [
            sys.executable,
            "-m",
            "src.models.inference",
            "capacity",
            "--model",
            model,
            "--weights",
            str(weights),
            "--seed",
            seed,
            "--node-id",
            "edge_server",
            "--repetitions",
            "2",
            *model_args_option(ctx),
        ]
        ctx.emit(f"edge server: {shlex.join(cap)}")
        if not ctx.plan_only:
            subprocess.run(prep, check=True, capture_output=True, cwd=REPO_ROOT)
            result = subprocess.run(cap, check=True, capture_output=True, text=True, cwd=REPO_ROOT)
            capacity = json.loads(result.stdout.strip().splitlines()[-1])
            write_json_atomic(ctx.paths.root / "edge_capacity.json", capacity)
            ctx.outputs["edge_capacity_flops"] = capacity["flops_per_second"]
            ctx.emit(f"edge-server capacity {capacity['flops_per_second'] / 1e9:.2f} GFLOP/s")
    inference = scenario.get("inference", {})
    if inference.get("enabled"):
        run_partitioned_inference(ctx, inference)


def run_partitioned_inference(ctx: ExperimentContext, inference: dict[str, Any]) -> None:
    model = ctx.scenario["model"]
    seed = str(ctx.scenario.get("seed", 0))
    policy = ctx.scenario.get("edge_server", {}).get("policy", "SJF")
    port = int(ctx.scenario.get("edge_server", {}).get("port", 5300))
    weights = ctx.config.bs_models_path / f"{model}.npz"
    host = local_mode.LOOPBACK if ctx.local else ctx.edge_data_address
    argv = [
        sys.executable,
        "-m",
        "src.inference.server",
        "--model",
        model,
        "--weights",
        str(weights),
        "--seed",
        seed,
        "--policy",
        policy,
        "--host",
        host,
        "--port",
        str(port),
        "--nodes",
        ",".join(d.id for d in ctx.devices),
        *model_args_option(ctx),
    ]
    if not ctx.plan_only:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.models.inference",
                "prepare",
                "--model",
                model,
                "--weights",
                str(weights),
                "--seed",
                seed,
                *model_args_option(ctx),
            ],
            check=True,
            capture_output=True,
            cwd=REPO_ROOT,
        )
    process = ctx.start_local_process("inference-server", argv, "inference_server")
    if not ctx.plan_only:
        log_path = ctx.paths.root / "logs" / "inference_server.log"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ExperimentError(
                    "the inference server exited before listening; see logs/inference_server.log"
                )
            if log_path.exists() and '"listening"' in log_path.read_text(
                encoding="utf-8", errors="replace"
            ):
                break
            time.sleep(0.1)
        else:
            raise ExperimentError("the inference server did not start within 120 s")
    points = [str(s) for s in inference.get("partition_points", [0])]

    def client_command(node: Node) -> str:
        dp = ctx.device_paths(node)
        args = [
            "--model",
            model,
            "--weights",
            f"{dp.models_dir}/{model}.npz",
            "--seed",
            seed,
            "--server",
            f"{host}:{port}",
            "--repetitions",
            str(inference.get("repetitions", 1)),
            "--output",
            f"{dp.results_dir}/phases.json",
            "--run-id",
            ctx.run_id,
            "--node-id",
            node.id,
            *model_args_option(ctx),
            *input_option(ctx),
        ]
        for s in points:
            args += ["--s", s]
        # The input source resolves relative paths against the device's data directory.
        return with_device_env(
            dp.module("src.inference.client", *args),
            f"TESTBED_ED_DATA_DIR={shlex.quote(dp.data_dir)}",
        )

    try:
        ctx.executor.run_on_nodes(ctx.devices, "", per_node_command=client_command, timeout=None)
    finally:
        ctx.stop_local_process("inference-server")


def stage_collect(ctx: ExperimentContext) -> None:
    monitoring = ctx.scenario.get("monitoring", {})
    edge_ip = local_mode.LOOPBACK if ctx.local else ctx.edge_data_address

    def latency_command(node: Node) -> str:
        dp = ctx.device_paths(node)
        args = ["--target", f"{node_statistics.EDGE_SERVER_TARGET}={edge_ip}"]
        for other in ctx.devices:
            if other.id != node.id:
                args += [
                    "--target",
                    f"{other.id}={local_mode.LOOPBACK if ctx.local else other.data_address}",
                ]
        args += [
            "--count",
            str(monitoring.get("latency_pings", 10)),
            "--output",
            f"{dp.results_dir}/latency.json",
            "--run-id",
            ctx.run_id,
            "--node-id",
            node.id,
        ]
        if not ctx.local:
            args += ["--interface", ctx.config.network_interface]
        return dp.module("src.network.latency_measurement", *args)

    ctx.executor.run_on_nodes(ctx.devices, "", per_node_command=latency_command)
    if ctx.plan_only:
        return
    # Monitors flush on SIGTERM: stop them before collecting their files.
    for node in ctx.devices:
        for component in ("monitor", "power"):
            process = ctx.tagged(node, component)
            if any(
                e.undo.get("component") == component
                and e.undo.get("node_id") == node.id
                and not e.undone
                for e in ctx.journal.entries
            ):
                ctx.executor.stop_tagged(node, process, grace_s=0.5)
    hostnames = {}
    if ctx.paths.metadata.exists():
        meta = json.loads(ctx.paths.metadata.read_text(encoding="utf-8"))
        hostnames = {n: f.get("hostname", n) for n, f in meta.get("nodes", {}).items()}
    if len(set(hostnames.values())) < len(hostnames):
        # Several devices report the same hostname (local mode): prefix by node id instead.
        hostnames = {n: n for n in hostnames}
    for node in ctx.devices:
        dp = ctx.device_paths(node)
        files = remote_list(ctx, node, f"{dp.results_dir}/*.json")
        for remote in files:
            name = Path(remote).name
            if name.startswith("benchmark_") or name == "phases.json":
                target = ctx.paths.raw
            else:
                target = {
                    "monitor.json": ctx.paths.monitor,
                    "power.json": ctx.paths.power,
                    "latency.json": ctx.paths.latency,
                    "load.json": ctx.paths.root / "load",
                    "verification.json": ctx.paths.root / "verification",
                }.get(name, ctx.paths.root / "other")
            collection.collect(
                ctx.executor,
                [node],
                [remote],
                target,
                config=ctx.config,
                edge_server_ip=edge_ip,
                mode="pull",
                hostnames={node.id: hostnames.get(node.id, node.id)},
            )
    ctx.emit("result files collected")


def stage_aggregate(ctx: ExperimentContext) -> None:
    if ctx.plan_only:
        return
    records = node_statistics.load_records(ctx.paths.latency)
    if records:
        averages = node_statistics.compute_averages(records)
        node_statistics.write_averages(averages, ctx.paths.final_latencies_averages)
        node_statistics.update_topology_statistics(ctx.topology, averages, ctx.run_id)
        if ctx.nodes_path is not None:
            save_topology(ctx.topology, ctx.nodes_path)
    all_records = result_aggregation.load_records(ctx.paths.raw)
    benchmarks = [r for r in all_records if isinstance(r, BenchmarkResults)]
    if not any(r.stages for r in benchmarks):
        runtime = runtimes.get_runtime(ctx.scenario.get("runtime", runtimes.DEFAULT_RUNTIME))
        if runtime.external:
            ctx.emit(f"external runtime {runtime.name}: no per-layer profile to aggregate")
            ctx.paths.link_latest()
            return
        raise ExperimentError("no benchmark records were collected")
    totals = result_aggregation.aggregate(benchmarks)
    write_json_atomic(ctx.paths.total_averages, totals)
    for extra in (ctx.paths.monitor, ctx.paths.power):
        all_records.extend(result_aggregation.load_records(extra))
    merged_dir = ctx.paths.root / "merged"
    for key, entry in result_aggregation.merge_traces(all_records).items():
        write_json_atomic(merged_dir / f"{key}.json", entry)
    state = (
        json.loads(ctx.paths.final_latencies_averages.read_text(encoding="utf-8"))
        if ctx.paths.final_latencies_averages.exists()
        else {}
    )
    for node_id in totals:
        state.setdefault(node_id, {"available_memory_mb": 1e9, "cpu_usage_percent": 0.0})
    results, details = partition_calculator.calculate(state, totals)
    write_json_atomic(ctx.paths.max_partition_points, results)
    write_json_atomic(
        partition_calculator.details_path(ctx.paths.max_partition_points),
        [d.__dict__ for d in details],
    )
    ctx.paths.link_latest()
    ctx.emit(f"aggregated: {', '.join(f'{k}: s<={v}' for k, v in results.items())}")


def stage_optimise(ctx: ExperimentContext) -> None:
    opt = ctx.scenario.get("optimisation", {})
    if not opt.get("enabled", True) or ctx.plan_only or not ctx.paths.total_averages.exists():
        return
    max_points = json.loads(ctx.paths.max_partition_points.read_text(encoding="utf-8"))
    totals = json.loads(ctx.paths.total_averages.read_text(encoding="utf-8"))
    latencies = (
        json.loads(ctx.paths.final_latencies_averages.read_text(encoding="utf-8"))
        if ctx.paths.final_latencies_averages.exists()
        else {}
    )
    devices = profiles_from_files(
        totals, max_points, latencies, None, [d.id for d in ctx.devices if d.id in totals]
    )
    capacity = float(ctx.outputs.get("edge_capacity_flops") or ctx.config.bs_computation_capacity)
    model = DelayModel(
        devices,
        DelayModelConfig(
            ctx.scenario.get("edge_server", {}).get("policy", "SJF"),
            capacity,
            transmission_mode=opt.get("transmission_mode", "rtt_only"),
        ),
    )
    initial = delay_optimizer.initial_vector(
        devices,
        opt.get("initial_scheme"),
        "percentage" if str(opt.get("initial_scheme", "")).endswith("%") else "explicit",
        model.num_stages,
    )
    limits_raw = opt.get("max_total_delays_ms")
    limits = (
        delay_optimizer.max_delays_seconds(json.dumps(limits_raw), devices) if limits_raw else None
    )
    out_dir = ctx.paths.optimisation / time.strftime("%Y%m%d-%H%M%S")
    log = SearchLog(out_dir / "search_log.txt")
    result = delay_optimizer.optimise(
        model,
        opt.get("algorithm", "greedy"),
        opt.get("objective", "total_delay"),
        initial,
        limits,
        log,
        num_solutions=opt.get("num_solutions", 5),
        max_time_s=opt.get("max_time_s", 300),
    )
    inputs = {
        "run_id": ctx.run_id,
        "scenario": ctx.scenario_version,
        "edge_capacity_flops": capacity,
    }
    write_json_atomic(out_dir / "scenario.json", {**ctx.scenario, "_version": ctx.scenario_version})
    path = delay_optimizer.write_result(
        out_dir, model, result, inputs, log.path or out_dir / "search_log.txt"
    )
    ctx.outputs["optimised_vector"] = result.vector
    ctx.outputs["optimised_objective"] = {
        "objective": result.objective,
        "value": result.value,
        "algorithm": opt.get("algorithm", "greedy"),
        "policy": ctx.scenario.get("edge_server", {}).get("policy", "SJF"),
    }
    ctx.emit(
        f"optimised partition vector {result.vector} "
        f"({result.objective}={result.value:.6f}) -> {path}"
    )


# --- teardown ----------------------------------------------------------------------------------


def undo_entry(ctx: ExperimentContext, entry: JournalEntry) -> None:
    undo = entry.undo
    kind = undo.get("kind")
    if kind == "stop_process":
        node = ctx.topology.get(undo["node_id"])
        stopped = ctx.executor.stop_tagged(node, ctx.tagged(node, undo["component"]), grace_s=0.5)
        if not stopped:
            raise ExperimentError(f"{node.id}: {undo['component']} still running after stop")
    elif kind == "tc_clear":
        node = ctx.topology.get(undo["node_id"])
        dp = ctx.device_paths(node)
        args = [undo["interface"], "--clear"] + (["--dry-run"] if ctx.local else [])
        ctx.executor.run_on_node(
            node, dp.module("src.testbed.latency_injector", *args), check=False
        )
    elif kind == "cpufreq_restore":
        node = ctx.topology.get(undo["node_id"])
        dp = ctx.device_paths(node)
        ctx.executor.run_on_node(
            node,
            ctx.sudo(
                node,
                dp.module(
                    "src.testbed.heterogeneity",
                    "--sysfs-root",
                    dp.sysfs_root,
                    "freq",
                    "--restore",
                    undo["state"],
                ),
            ),
        )
    elif kind == "cgroup_remove":
        node = ctx.topology.get(undo["node_id"])
        dp = ctx.device_paths(node)
        ctx.executor.run_on_node(
            node,
            ctx.sudo(
                node,
                dp.module(
                    "src.testbed.heterogeneity",
                    "--sysfs-root",
                    dp.sysfs_root,
                    "cgroup",
                    "remove",
                    "--name",
                    undo["name"],
                ),
            ),
            check=False,
        )
    elif kind == "edge_freq_restore":
        argv = [
            sys.executable,
            "-m",
            "src.testbed.cpu_frequency",
            "restore",
            "--state",
            undo["state"],
            "--sysfs-root",
            undo["sysfs_root"],
        ]
        if not ctx.local:
            argv = ["sudo", "-n", *argv]
        if Path(undo["state"]).exists():
            subprocess.run(argv, check=False, capture_output=True, cwd=REPO_ROOT)
    elif kind == "stop_local":
        ctx.stop_local_process(undo["tag"])
    else:
        raise ExperimentError(f"unknown undo kind {kind!r}")


def teardown(ctx: ExperimentContext) -> list[JournalEntry]:
    ctx.emit("teardown")
    if ctx.plan_only:
        for entry in ctx.journal.entries:
            entry.undone = True
        return []
    failures = ctx.journal.undo_all(lambda entry: undo_entry(ctx, entry))
    for node in ctx.devices:
        ctx.executor.run_on_node(node, stop_all_command(ctx.run_id, grace_s=0.5), check=False)
    for tag in list(ctx.local_processes):
        ctx.stop_local_process(tag)
    if failures:
        ctx.emit(
            "teardown incomplete: "
            + "; ".join(f"{f.node_id}: {f.description} ({f.undo_error})" for f in failures)
        )
    else:
        ctx.emit("teardown complete: every change undone")
    return failures


# --- pipelines ---------------------------------------------------------------------------------


PIPELINES: dict[str, list[Callable[[ExperimentContext], None]]] = {
    "experiment": [
        stage_preflight,
        stage_apply,
        stage_profile,
        stage_collect,
        stage_aggregate,
        stage_optimise,
    ],
    "benchmarking": [
        stage_preflight,
        lambda ctx: stage_apply(ctx, full=False),
        stage_profile,
        stage_collect,
        stage_aggregate,
        stage_optimise,
    ],
}


def run_pipeline(ctx: ExperimentContext) -> dict[str, Any]:
    stages = PIPELINES[ctx.scenario.get("pipeline", "experiment")]
    status = "completed"
    error: str | None = None
    try:
        for stage in stages:
            if ctx.cancel.is_set():
                raise ExperimentError("stopped by the user")
            stage(ctx)
    except (ExperimentAbortedError, ExperimentError, subprocess.CalledProcessError) as exc:
        status = "aborted"
        error = str(exc)
        ctx.emit(f"aborted: {error}")
        raise
    finally:
        failures = teardown(ctx)
        summary = {
            "run_id": ctx.run_id,
            "status": status,
            "error": error,
            "teardown_failures": len(failures),
            "outputs": ctx.outputs,
            "t0": ctx.t0,
        }
        if not ctx.plan_only:
            write_json_atomic(ctx.paths.root / "summary.json", summary)
    return summary


def build_context(
    scenario: dict[str, Any],
    *,
    config: Config,
    topology: Topology,
    local: bool,
    results_dir: Path | None = None,
    run_id: str | None = None,
    allow_dirty: bool = False,
    nodes_path: Path | None = None,
    scenario_version: dict[str, Any] | None = None,
    subscribers: list[Callable[[str], None]] | None = None,
    plan_only: bool = False,
    clock: Clock | None = None,
) -> ExperimentContext:
    run_id = run_id or new_run_id()
    runtime_name = scenario.get("runtime", runtimes.DEFAULT_RUNTIME)
    if runtime_name not in runtimes.names():
        raise ExperimentError(
            f"unknown runtime {runtime_name!r}; registered: {', '.join(runtimes.names())}"
        )
    if scenario.get("model_args"):
        from src.models import registry

        try:
            registry.validate_model_args(scenario["model"], scenario["model_args"])
        except (ValueError, KeyError) as exc:
            raise ExperimentError(str(exc)) from exc
    paths = RunPaths(results_dir or config.bs_results_path, run_id)
    if plan_only:
        transport: Any = FakeTransport(
            default=__import__(
                "src.network.transport", fromlist=["ScriptedOutcome"]
            ).ScriptedOutcome(0, ["0"])
        )
    else:
        transport = make_transport(config, topology, local=local, password=get_ssh_password(config))
    executor = Executor(
        transport,
        log_path=None if plan_only else paths.merged_log,
        subscribers=list(subscribers or []),
    )
    journal = RunJournal(paths.journal)
    ctx = ExperimentContext(
        config,
        topology,
        scenario,
        run_id,
        paths,
        executor,
        journal,
        local=local or isinstance(transport, LocalTransport),
        allow_dirty=allow_dirty,
        nodes_path=nodes_path,
        scenario_version=dict(scenario_version or {}),
        plan_only=plan_only,
        clock=clock or SystemClock(),
    )
    return ctx


def plan(ctx: ExperimentContext) -> list[str]:
    """The commands the apply, profile and collect stages would issue, without running them.

    The context must have been built with ``plan_only=True`` (a recording transport)."""
    if not ctx.plan_only:
        raise ExperimentError("plan() needs a context built with plan_only=True")
    ctx.t0 = 0.0
    lines: list[str] = []
    ctx.executor.subscribers.append(lines.append)
    stages = PIPELINES[ctx.scenario.get("pipeline", "experiment")]
    for stage in stages:
        stage(ctx)
    teardown(ctx)
    return [line.split(" ", 1)[1] for line in lines if "[exec]" in line or "[info]" in line]


def teardown_open_runs(
    config: Config, topology: Topology, local: bool, results_dir: Path | None = None
) -> list[str]:
    """Undo the pending journal entries of every run that did not finish (after a restart)."""
    handled = []
    for journal in open_journals(results_dir or config.bs_results_path):
        run_id = journal.path.parent.name
        scenario_path = journal.path.parent / "scenario.json"
        scenario = (
            json.loads(scenario_path.read_text(encoding="utf-8"))
            if scenario_path.exists()
            else {"name": run_id, "model": "fake", "devices": None}
        )
        ctx = build_context(
            scenario,
            config=config,
            topology=topology,
            local=local,
            results_dir=results_dir,
            run_id=run_id,
        )
        ctx.journal = journal
        teardown(ctx)
        handled.append(run_id)
    return handled


# --- command line ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an experiment scenario on the testbed.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "benchmark", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--scenario", type=Path, help="scenario file (YAML or JSON)")
        p.add_argument("--name", help="saved scenario name (scenarios/<name>)")
        p.add_argument("--version", type=int, default=None)
        p.add_argument(
            "--local",
            nargs="?",
            const=".local_testbed",
            default=None,
            metavar="ROOT",
            help="local mode (simulated nodes under ROOT)",
        )
        p.add_argument(
            "--devices", type=int, default=2, help="local mode: number of simulated devices"
        )
        p.add_argument("--config", default=None)
        p.add_argument("--nodes-file", default=None)
        p.add_argument("--results-dir", type=Path, default=None)
        p.add_argument("--allow-dirty", action="store_true")
        p.add_argument("--run-id", default=None)
    t = sub.add_parser("teardown")
    t.add_argument("--local", nargs="?", const=".local_testbed", default=None)
    t.add_argument("--config", default=None)
    t.add_argument("--nodes-file", default=None)
    t.add_argument("--results-dir", type=Path, default=None)
    return parser


def load_environment(args: argparse.Namespace) -> tuple[Config, Topology, Path | None]:
    if args.local is not None:
        root = Path(args.local).resolve()
        config, topology = local_mode.create_local_testbed(
            root, devices=getattr(args, "devices", 2)
        )
        return config, topology, root / "nodes.json"
    config = load_config(args.config)
    nodes_path = Path(args.nodes_file) if args.nodes_file else None
    return config, load_topology(nodes_path), nodes_path


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config, topology, nodes_path = load_environment(args)
    local = args.local is not None
    if args.command == "teardown":
        handled = teardown_open_runs(config, topology, local, args.results_dir)
        print(f"teardown of {len(handled)} run(s): {handled}")
        return
    if args.scenario:
        scenario = scenario_module.load_scenario_file(args.scenario)
        version: dict[str, Any] = {
            "source": str(args.scenario),
            "sha256": scenario_module.content_hash(scenario),
        }
    elif args.name:
        store = scenario_module.ScenarioStore(config.bs_root_path / "scenarios")
        scenario, saved = store.load(args.name, args.version)
        version = {"name": saved.name, "version": saved.version, "sha256": saved.sha256}
    else:
        raise SystemExit("give --scenario FILE or --name NAME")
    if args.command == "benchmark":
        scenario["pipeline"] = "benchmarking"
    variants = scenario_module.expand_sweep(scenario)
    for variant in variants:
        ctx = build_context(
            variant,
            config=config,
            topology=topology,
            local=local,
            results_dir=args.results_dir,
            run_id=args.run_id if len(variants) == 1 else None,
            allow_dirty=args.allow_dirty,
            nodes_path=nodes_path,
            scenario_version=version,
            subscribers=[lambda line: print(line, flush=True)],
            plan_only=args.command == "plan",
        )
        if args.command == "plan":
            for line in plan(ctx):
                print(line)
            continue
        try:
            summary = run_pipeline(ctx)
        except (ExperimentAbortedError, ExperimentError, subprocess.CalledProcessError) as exc:
            print(f"run {ctx.run_id} aborted: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
