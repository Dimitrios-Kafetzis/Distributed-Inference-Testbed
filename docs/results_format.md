# Results format

Every run writes one directory, `results/<run id>/`, with `results/latest` pointing at the most recent one. The run id is `YYYYMMDD-HHMMSS-<6 hex>` in UTC. All files are JSON written atomically (a temporary file renamed into place), so a reader never sees a partial file. Synthetic examples of the main files are in [`../data/sample/`](../data/sample/) (labelled as synthetic; regenerate with `python -m tests.make_sample_data`).

## The run directory

```
results/<run id>/
  metadata.json                           scenario, software and hardware facts, written before any device command
  scenario.json                           the scenario version that was run (with defaults filled in)
  journal.json                            every applied change and its undo action, with the outcome of the undo
  summary.json                            status, error, number of teardown failures, main outputs, T0
  logs/merged.log                         time-stamped [node][stream] lines of every remote command
  logs/inference_server.log               the edge-server inference service (when partitioned inference ran)
  raw/<host>_benchmark_<run id>_rep<k>.json   one BenchmarkResults record per device and repetition
  raw/<host>_phases.json                  per-phase timing of the partitioned-inference runs (per device)
  monitor/<host>_monitor.json             ResourceTrace of each device
  power/<host>_power.json                 PowerTrace of each device (when monitoring.power is true)
  latency/<host>_latency.json             latency-and-system record of each device
  load/<host>_load.json                   the load generator's log of applied targets
  verification/<host>_verification.json   the partition-equivalence report of each device (checks.partition_equivalence)
  merged/<node id>__<run id>.json         the device's records and traces aligned on a common origin
  edge_capacity.json                      the edge server's measured FLOP/s on the model
  edge_frequency_state.json               the edge server's previous cpufreq settings (restored at teardown)
  total_averages_of_benchmark_results.json
  calculated_max_partition_points.json  (+ .details.json)
  final_latencies_averages.json
  optimisation/<timestamp>/optimized_partition_scheme.json, scenario.json, search_log.txt
  headsplit/                              HeadSplit runs only: results.json, headsplit_config.yaml, the controller logs
                                          and, with a controller that writes JSON, controller_rep<k>.json
```

`<host>` is the device's hostname as `transfer_files.sh` prefixes it (the node id when hostnames collide, as in local mode).

## Conventions

- Intervals are integers in nanoseconds from `time.perf_counter_ns()`; wall-clock instants are floats in seconds since the epoch (`time.time()`, NTP-synchronised).
- Every record carries `origin`: `{wall_time_s, perf_counter_ns}` captured together at the start signal. A sample's instant on the wall clock is `origin.wall_time_s + (sample.perf_counter_ns - origin.perf_counter_ns) / 1e9`; this is how traces of different devices are aligned.
- Memory in mebibytes, CPU in percent, sizes in bytes, FLOPs as 2 × multiply-accumulates.

## Measurement records (`docs/schemas/benchmark_results.schema.json`)

All device-side records share a header: `schema_version`, `kind` (`benchmark`, `resource_trace` or `power_trace`), `run_id`, `hostname`, `node_id`, `origin`, `metadata` (framework, input shape, baseline resident memory, thread count, the device's FLOP/s).

**BenchmarkResults** (`kind: benchmark`): `model`, `repetition`, `warmup_runs`, and

- `stages[]`: `index` (1-based), `name`, `latency_ns`, `cpu_peak_percent`, `memory_peak_mb`, `flops`, `macs`, `output_bytes`;
- `partition_points[]`: for each s in 0..S, `cumulative_latency_ns` and `cumulative_cpu_percent` (sums over stages 1..s), `cumulative_memory_mb` (the largest stage peak among stages 1..s, the memory needed to run them) and `transmitted_bytes` (the size of the cut tensor; 0 at s = S);
- `phases[]` (partitioned-inference runs): `s`, `local_ns`, `transmission_ns` (first byte sent to the server's acknowledgement of the last byte), `remote_ns`, `queue_wait_ns`, and the server's wall-clock instants `server_arrival_wall_s`, `server_service_start_wall_s`, `server_service_end_wall_s`.

**ResourceTrace** (`kind: resource_trace`): `interval_s`, `nominal_frequency_khz`, `samples[]` with `perf_counter_ns`, `cpu_percent`, `memory_used_mb`, `memory_available_mb`, `temperature_c`, `frequencies_khz` (per core), `throttled` (the firmware flags).

**PowerTrace** (`kind: power_trace`): `interval_s`, `samples[]` with `perf_counter_ns`, `current_ma`, `voltage_v`, `power_mw`.

## The three thesis output files

`total_averages_of_benchmark_results.json` (`src/benchmarking/result_aggregation.py`): per node id, `hostname`, `model`, `repetitions`, `run_ids`, then `stages` and `partition_points` keyed by index with, for every measured quantity, `{mean, std, min, max, p95, count}` over the repetitions, and the per-phase statistics when partitioned inference ran.

`final_latencies_averages.json` (`src/network/node_statistics.py`): per node id, `samples`, `measured_at`, `average_latency_ms`, `latency_std_ms`, `latency_p95_ms`, `loss_percent` (to the edge server), `total_memory_mb`, `available_memory_mb`, `available_memory_percent`, `cpu_usage_percent`, and `targets` with the same latency figures per pinged node. The same averages are written into `nodes.json` as the node's `statistics`.

`calculated_max_partition_points.json` (`src/benchmarking/partition_calculator.py`): `{node id: s}`, the largest feasible partition point under the memory constraint (cumulative memory up to s within the available memory) and the CPU constraint (baseline usage plus the stage's increment below 100 %). The sibling `.details.json` lists per node `available_memory_mb`, `baseline_cpu_percent`, `memory_limited_s`, `cpu_limited_s`, `max_partition_point` and `binding` (`memory` or `cpu`).

## Optimisation result

`optimisation/<timestamp>/optimized_partition_scheme.json` (`src/partitioning/delay_optimizer.py`): `optimized_partition_scheme` (the vector in device order), `by_node`, `objective`, `value` and `value_unit` (seconds), `algorithm`, `policy`, `evaluations`, `iterations`, `per_device[]` with `s`, `local_ms`, `transmission_ms`, `remote_ms`, `total_ms`, the `all_local_baseline`, the `inputs` (run id, the scenario's source and SHA-256, the edge capacity, the hashes of the input files) and the path of the `search_log.txt`, which has one line per evaluated vector. The scenario version is copied next to it.

## Metadata and journal

`metadata.json`: `schema_version`, `run_id`, `created_at`, the full `scenario`, `software` (the Git commit of the testbed and whether the tree was clean, Python, TensorFlow and PyTorch versions), `nodes` (per node: hostname, kernel, OS release, CPU model, cores, maximum frequency, memory, NTP status), `config` (the effective configuration without secrets).

`journal.json`: `entries[]` with `sequence`, `node_id`, `kind` (`edge_frequency`, `shaping`, `load`, `monitor`, `cpufreq`, `cgroup`, `process`, ...), `description`, `undo` (the action teardown runs: `stop_process`, `tc_clear`, `cpufreq_restore`, `cgroup_remove`, `edge_freq_restore`, `stop_local`), `applied_at`, `undone`, `undo_error`. A completed run has every entry `undone: true` with `undo_error: null`; `python -m src.orchestration.experiment teardown` undoes the pending entries of unfinished runs.

`summary.json`: `run_id`, `status` (`completed` or `aborted`), `error`, `teardown_failures`, `outputs` (edge capacity, optimised vector, ...) and `t0`.
