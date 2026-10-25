# Module reference

One entry per module: purpose, command line where the module has one, inputs and outputs. Every command line prints its full help with `--help`. Device-side tools accept `--tag` (the process tag that `clean_devices.sh` uses to find survivors) and, where they produce records, `--run-id`, `--node-id`, `--origin-wall` and `--origin-perf` (the time origin from the start signal).

## Configuration

### `src/config.py`
`config.yaml` with environment-variable overrides (`TESTBED_<KEY>`), validated against `docs/schemas/config.schema.json`; secrets only from the environment. `python -m src.config [--config FILE] [--env]` prints the effective configuration as JSON, or as `TESTBED_*` assignments for scripts.

### `src/common/nodes.py`
`nodes.json`: `Node` (id, role `Base Station` or `End Device`, data-plane and management addresses, coordinates, statistics), `Topology` (at most one edge server), load, validate and save atomically; `TESTBED_NODES` selects another file. Imports the earlier release's format.

## Common infrastructure (`src/common/`)

| Module | Purpose |
|---|---|
| `runner.py` | `CommandRunner` protocol with `SubprocessRunner`, `FakeRunner` (scripted replies, records calls) and `DryRunRunner` (prints the commands) |
| `clock.py` | injectable clocks; `perf_counter_ns` for intervals, the wall clock only for the start-signal origin |
| `sysfs.py` | rooted access to `/sys`, `/proc` and cgroupfs (a temporary tree in tests) |
| `process_tags.py` | detached tagged processes: `setsid` process groups, PID files, `TESTBED_TAG`, stop by group and by tag |
| `runs.py` | run ids, the results layout (`RunPaths`), atomic JSON writes, the run journal (`RunJournal`: record, undo in reverse order, pending entries after a restart) |
| `seeds.py` | `seed_everything(seed)` for `random`, NumPy, TensorFlow, PyTorch and `PYTHONHASHSEED` |
| `start_signal.py` | `wait_until(T0)` and the captured origin; `python -m src.common.start_signal --start-at T0` |
| `metadata.py` | `metadata.json`: scenario, Git commit, framework versions, per-node facts, before any device command |
| `probe.py` | `python -m src.common.probe [--no-frameworks]`: the node's hostname, kernel, OS, Python, framework versions, CPU model, cores, maximum frequency, memory, NTP status, as JSON |

## Network layer (`src/network/`)

### `transport.py`
`ParamikoTransport` (SSH), `LocalTransport` (local subprocesses in per-node directories), `FakeTransport` (tests). Line streaming, exit status, failure classes (authentication, unreachable, command), file fetch.

### `ssh_executor.py`
Concurrent remote commands with a merged `[node][stream]` log streamed to subscribers; aborts the experiment on the first failure unless told otherwise. `python -m src.network.ssh_executor "<command>" <node id or last octet>... [--log FILE] [--no-abort] [--timeout S] [--nodes-file FILE] [--local]`.

### `collection.py`
Result files from the devices into the run directory: `push` through `scripts/end_device/transfer_files.sh` (scp with the hostname prefixed), or `pull` through the transport.

### `latency_measurement.py` (device)
Ping latency to the edge server and other nodes plus a psutil snapshot of memory and CPU, written as a latency record. `python -m src.network.latency_measurement --target ID=IP ... --count N --output FILE --run-id R --node-id D [--interface wlan0] [--interval S] [--timeout S]`.

### `node_statistics.py` (edge server)
Per-node averages of the latency records into `final_latencies_averages.json`, and the statistics in `nodes.json`. `python -m src.network.node_statistics <latency dir> <output> [--nodes nodes.json] [--run-id ID]`.

### `power_measurement.py` (device)
INA260 current, voltage and power at an interval into a `PowerTrace`; energy functions (total energy, energy per inference or per token, energy-delay product). `python -m src.network.power_measurement --interval S --duration S --output FILE --run-id R --node-id D [--address 0x40] [--fake]`.

### `throughput_measurement.py` (device)
iperf3 throughput to the edge server, as a `ThroughputResult`. `python -m src.network.throughput_measurement --server IP --duration S --output FILE [--parallel N] [--reverse] [--udp --bitrate 50M] [--port P] [--bind IP]`.

### `records.py`
The latency-and-system record type.

## Device-side control (`src/testbed/`)

### `load_generator.py` (device)
Deterministic CPU and memory contention: a duty-cycle loop per worker for a CPU percentage, 1 KB blocks held for a memory percentage; modes `fixed`, `continuous` (seeded redraw every cycle), `--regime light|moderate|heavy` (the Chapter 3 ranges) and `--processes N` (the Chapter 4 model). `python -m src.testbed.load_generator [--cpu % --memory %] [--cores N] [--mode fixed|continuous --cpu-range MIN MAX --memory-range MIN MAX] [--regime R] [--processes N] [--period-ms MS] [--cycle-s S] [--duration S] [--seed N] [--log FILE]`. Releases everything on SIGTERM.

### `latency_injector.py` (device)
tc-netem shaping on an interface: delay with variation modes `constant`, `uniform`, `square`, jitter, loss, a token-bucket rate cap, per-destination leaves under an HTB root. `python -m src.testbed.latency_injector <iface> [--delay MS | --delay MIN MAX --mode M --period S] [--jitter MS] [--loss %] [--rate MBPS] [--dst IP ...] [--duration S] [--seed N] [--clear] [--no-sudo] [--dry-run]`. `scripts/end_device/apply_latency.sh` keeps the historical command line.

### `background_traffic.py`
Builds the iperf3 stream command for a device (UDP or TCP at a target rate); the orchestrator starts it.

### `heterogeneity.py` (device, sudo)
`freq --max-mhz N` (cpufreq cap, with `--state FILE` to record and `--restore STATE` to revert), `cgroup create --name N --memory-mb M` and `cgroup remove --name N` (cgroups v2), `threads --count N` (environment for the inference runtime). `--sysfs-root` for tests.

### `resource_monitor.py` (device)
The 1 Hz monitoring agent: CPU, memory, temperature, per-core frequency, throttling flags, as a `ResourceTrace`. `python -m src.testbed.resource_monitor --output FILE --run-id R --node-id D [--interval S] [--duration S] [--nominal-khz K]`.

### `cpu_frequency.py` (edge server, sudo)
Nominal frequency policy: `performance` governor, turbo off, maximum frequency at the base frequency; `apply --state FILE`, `verify`, `restore --state FILE`.

## Models and profiling (`src/models/`)

### `registry.py`
The adapter interface (a model as S stages with FLOPs, MACs and output size per stage) and the name registry; plug-ins through `TESTBED_MODEL_MODULES` or the `testbed.models` entry-point group. An adapter's `PARAMETERS` (a JSON Schema) describes its constructor arguments: `parameter_schema(name)`, `validate_model_args(name, args)` (the scenario's `model_args`), `parse_model_args(text)` (the tools' `--model-args`). See [adding_a_model.md](adding_a_model.md).

### `tf/` (TensorFlow, CPU)
`alexnet.py` (the 11 stages of Chapter 4, 227 × 227 × 3 input), `vgg16.py` (eight stages), `resnet50.py` (six stages cut at the residual-stage boundaries), `gru_qoe.py` (the Chapter 4 QoE predictor: GRU, attention, output), on `keras_stages.py` (sequential and functional-cut adapters, analytical operation counts, `.npz` weights; `select_cuts(model)` for adapters that compute their cuts). `keras_applications.py`: any `tf.keras.applications` classifier (`keras_app` with `application`, and the registered names `mobilenetv2`, `efficientnetb0`, `densenet121`, `resnet50v2`, `inceptionv3`, `nasnetmobile`, `xception`), cut where a single tensor crosses the graph (`find_single_tensor_cuts`) and thinned to `max_stages` stages of comparable work (`thin_cuts`).

### `torch/` (PyTorch, CPU)
`decoder.py` (decoder-only LLMs as stages: embedding, layers, LM head; the cut is the hidden state; the `FAMILIES` table gives the attribute names and structural facts of GPT-2, Llama, Mistral, Qwen2 and Gemma), `gpt2.py`, `tinyllama.py`, `hf_decoder.py` (`hf_decoder`: any checkpoint or configuration of a known family; `llama32_1b`, `qwen25_05b`), `headwise.py` (per-head decomposition of a decoder layer: head units, projection, FFN, equal to the stock layer; follows the family table), `hooks.py` (`python -m src.models.torch.hooks --model NAME --seq-len T --repetitions N --output FILE [--weights DIR] [--config JSON]`: per-block timing with forward hooks).

### `inputs.py`
Input sources (`synthetic`, `numpy_file`, `image_file`, `image_folder`, `text_prompt`): `load_sample(spec, adapter, seed, data_dir)` returns the array, a label when the source has one, and a description that the measurement record keeps. The tools take the spec as `--input-source JSON`.

### `weights.py`
The model catalogue `<bs_models_dir>/models.yaml` (example: `examples/models.yaml`): `python -m src.models.weights list|verify|checksums|fetch [--models-dir DIR] [NAME ...] [--all] [--allow-upstream]`. Kinds: `npz` (seeded weights), `keras_imagenet` (the ImageNet weights of a Keras application), `hf_checkpoint` (a Hugging Face snapshot). `SHA256SUMS` files make every copy verifiable; `fetch` is the only command that reaches the internet.

### `fake.py`
A model with configurable stage costs for tests and local mode.

### `flops.py`
Analytical operation counts (FLOPs = 2 × MACs) from layer configuration and shapes.

### `profiler.py`
Stage-by-stage execution with `perf_counter_ns`, a sampler for peak CPU (from CPU-time deltas over windows of at least 20 ms, clipped to 100 % × cores) and peak resident memory, cumulative values per partition point.

### `inference.py` (device)
`python -m src.models.inference {prepare,benchmark,partial,complete,capacity,verify} ...`: prepare the weights and the test image with a seed; benchmark across all partition points into `BenchmarkResults` files; run stages 1..s (`partial`) or s+1..S (`complete`) on a tensor file; report FLOP/s; `verify` compares every partition point with the unpartitioned model (maximum deviation, top-1 agreement, accuracy when the input has a label) and exits with status 1 above `--tolerance`. Every mode takes `--model-args JSON` and `--input-source JSON`. `scripts/end_device/train_and_run_benchmark.sh <train|benchmark|both> <model dir> <model> <output dir> [options]` wraps it.

## Partitioned inference (`src/inference/`)

### `server.py` (edge server)
The inference service: receives offloaded requests, orders them under RR or SJF, runs stages s+1..S, acknowledges every request when its last byte arrives, returns the output with the server-side timestamps. `python -m src.inference.server --model M [--weights W] --policy RR|SJF --port P [--host H] [--nodes d1,d2,...] [--max-requests N]`.

### `client.py` (device)
Partitioned inference with per-phase timing (local, transmission, remote, queue wait) into a phases record. `python -m src.inference.client --model M [--weights W] --server HOST:PORT --s S [--s S ...] --repetitions N --output FILE --run-id R --node-id D`.

### `scheduler.py`, `protocol.py`
The pure RR and SJF schedulers; the length-prefixed JSON-plus-payload wire protocol (`request`, `ack`, `result`, `error`).

## Measurement records and aggregation (`src/benchmarking/`)

### `benchmark_results.py`
`BenchmarkResults`, `ResourceTrace`, `PowerTrace` and their JSON Schema; `python -m src.benchmarking.benchmark_results FILE` validates a record.

### `result_aggregation.py`
Per-device averages over repetitions and trace merging. `python -m src.benchmarking.result_aggregation <raw dir> <output> [--monitor-dir D] [--power-dir D] [--merged-dir D]`.

### `partition_calculator.py`
Maximum feasible partition point per device (memory and CPU constraints). `python -m src.benchmarking.partition_calculator <final_latencies_averages.json> <total_averages_of_benchmark_results.json> <output>`.

### `stats.py`
Mean, standard deviation, minimum, maximum, 95th percentile, count.

## Partition optimisation (`src/partitioning/`)

### `delay_model.py`
Local (measured or analytical), transmission (`size_over_rate` or `rtt_only`) and remote (RR or SJF) delay per device; total delay and variance of a vector.

### `algorithms/`
`greedy.py` (single-step moves, Chapter 4), `exhaustive.py` (enumeration with the optimality gap), `trend_based.py` (linear trends over `num_solutions` neighbours); a registry with plug-ins.

### `delay_optimizer.py`
`python -m src.partitioning.delay_optimizer <max points> <total averages> <latencies> --scheduling_algorithm SJF|RR --mode default|optimize|exhaustive_search|trend_based_optimize [--algorithm NAME] --objective total_delay|variance [--partition_scheme '[3,5,8,1]'|'{"device1": 3}'|50% --partition_scheme_mode explicit|percentage] [--max_total_delays '[150,150]'] [--output-dir DIR] [--log_file FILE] [--edge-capacity FLOPS] [--rr-quantum-ms MS] [--transmission-mode M] [--throughput-file FILE] [--nodes d1,d2] [--scenario FILE]`. Writes `optimized_partition_scheme.json` and `search_log.txt`.

## Orchestration (`src/orchestration/`)

### `experiment.py`
The pipeline (preflight, apply, profile, collect, aggregate, optimise, teardown) and the benchmarking subset. `python -m src.orchestration.experiment run|benchmark|plan --scenario FILE | --name NAME [--version N] [--local [ROOT] --devices N] [--results-dir DIR] [--run-id ID] [--allow-dirty]`; `teardown [--local [ROOT]]` undoes the pending entries of unfinished runs. `plan` prints the command plan without running anything. The scripts `scripts/base_station/run_experiment.sh`, `run_full_benchmarking.sh` and `clean_devices.sh` wrap these.

### `scenario.py`
Schema validation with defaults, the versioned `ScenarioStore`, sweep expansion, per-node overlays, content hashes.

### `local_mode.py`
`create_local_testbed(root, devices)`: the simulated testbed on one machine.

### `geo.py`
`calculate_distance` (Haversine) and `calculate_latency` (5 ms + 1 ms/km, clipped to [5, 50] ms), re-exported by `gui/app.py`.

### `runtimes/`
The runtime plug-in interface (`Runtime`: scenario block schema, `run`, `check_node`, `actions` and `perform`, `results`; `ActionSession`; the registry with `TESTBED_RUNTIME_MODULES` and the `testbed.runtimes` entry-point group), `testbed.py` (the built-in profile stage) and `headsplit.py` (HeadSplit: commands, readiness checks, the install, profile and checkpoint actions, the JSON and summary-line parsers). `headsplit_runtime.py` keeps the earlier import path. See [adding_a_runtime.md](adding_a_runtime.md) and [headsplit_integration.md](headsplit_integration.md).

## Provisioning support (`src/provisioning/`)

`render.py` (`python -m src.provisioning.render --role edge-server|device [--node ID] [--out DIR] [--vars] [--var K=V]`), `networks.py` (prefix to network, DHCP pool, 802.11ac channel helpers), `mirror.py` (`python -m src.provisioning.mirror manifest|verify [--mirror DIR]`). The scripts are described in [`../scripts/provision/README.md`](../scripts/provision/README.md).

## NNPM (`gui/`)

`app.py` (`python -m gui.app [--host H] [--port P] [--config FILE] [--nodes-file FILE] [--local [ROOT]] [--devices N] [--allow-dirty]`; the `nnpm` console script), `state.py` (the prober, the run manager with sweeps, the task manager, the application state), `blueprints/` (topology, status, scenarios, runs with the summary page, optimise, catalog, runtimes), `templates/run.html` (the run summary page). See [nnpm_user_guide.md](nnpm_user_guide.md).

## Shell scripts

| Script | Where | Purpose |
|---|---|---|
| `scripts/base_station/run_full_benchmarking.sh [scenario] [--local [ROOT]]` | edge server | the four-stage benchmarking pipeline (default scenario `examples/scenarios/baseline.yaml`) |
| `scripts/base_station/run_experiment.sh <scenario.yaml \| --name NAME [--version N]> [--local [ROOT]]` | edge server | a full experiment |
| `scripts/base_station/clean_devices.sh [--local [ROOT]]` | edge server | stop every tagged process, clear shaping and cgroups, undo pending journals |
| `scripts/end_device/train_and_run_benchmark.sh <train\|benchmark\|both> <model dir> <model> <output dir> [options]` | device | prepare the weights and the test input, run the benchmark, or both |
| `scripts/end_device/transfer_files.sh <user> <edge server> <directory> <files...>` | device | scp the result files with the hostname prefixed |
| `scripts/end_device/apply_latency.sh <iface> <min> <max> [--once\|--continuous\|--periodic] ... \| --clear` | device | the historical tc-netem wrapper |
| `scripts/provision/*.sh` | edge server, device | provisioning (see its README) |
| `scripts/lock_requirements.sh [--upgrade]` | development | regenerate the lock files |
