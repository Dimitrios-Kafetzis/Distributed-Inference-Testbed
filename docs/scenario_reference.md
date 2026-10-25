# Scenario reference

A scenario is one JSON (or YAML) object that holds every parameter of an experiment. It is validated against [`schemas/scenario.schema.json`](schemas/scenario.schema.json); the defaults below are the schema's. NNPM stores each saved scenario as a new version under `scenarios/<name>/v<N>.json` with its content hash (identical content reuses the version), and every run records the version it used, so a run can be replayed with `run_experiment.sh --name <name> --version <N>`.

Only `name` and `model` are required. Example files are in [`../examples/scenarios/`](../examples/scenarios/).

## Top level

| Field | Type, default | Meaning |
|---|---|---|
| `name` | string | scenario name (`[A-Za-z0-9][A-Za-z0-9_.-]*`) |
| `description` | string | free text |
| `pipeline` | `experiment` (default) or `benchmarking` | the seven-stage pipeline, or the four-stage benchmarking pipeline of Appendix B (no load, no shaping) |
| `runtime` | string, `testbed` | a registered runtime: `testbed` (per-layer profiling and partitioned inference), `headsplit` (the external HeadSplit runtime, see [headsplit_integration.md](headsplit_integration.md)) or a plug-in (see [adding_a_runtime.md](adding_a_runtime.md)); the runtime's settings go in the block of the same name |
| `model` | string | a registered model name: `alexnet`, `vgg16`, `resnet50`, `gru_qoe`, `mobilenetv2`, `efficientnetb0`, `densenet121`, `keras_app`, `gpt2`, `tinyllama`, `llama32_1b`, `qwen25_05b`, `hf_decoder`, `fake`, or a plug-in (see [adding_a_model.md](adding_a_model.md)) |
| `model_args` | object | the model's parameters (for example `num_classes`, `seq_len`, `application`, `checkpoint`), validated against the adapter's parameter schema and passed to every device tool |
| `devices` | list of node ids, or `null` (default) | the end devices that take part; `null` means every `End Device` in `nodes.json` |
| `repetitions` | integer, 1 | measured profiling repetitions per device |
| `warmup` | integer, 1 | warm-up passes before the measured repetitions |
| `seed` | integer, 0 | the random seed set on every device (`random`, NumPy, TensorFlow, PyTorch) |
| `settle_s` | number, 5 | seconds between applying load and shaping and the start of profiling |
| `start_lead_s` | number, 3 | how far in the future the start instant T0 is placed |
| `input` | object | what the devices feed to the model: `source` (`synthetic`, default; `numpy_file`; `image_file`; `image_folder`; `text_prompt`), `path` (on the device, relative to its data directory), `index` (image_folder: which image of the sorted listing; class sub-directories give a label), `text` and `tokenizer` (text_prompt) |
| `checks` | object | `partition_equivalence` (false): after profiling, every device compares each partition point with the unpartitioned model (`python -m src.models.inference verify`) and a relative deviation above `tolerance` (0.0001) aborts the run; the reports are collected under `verification/` |
| `sweep` | object | `path` (dotted field, for example `network.default.added_rtt_ms`) and `values`; expands into one run per value, each carrying `sweep_origin` (name, path, value) so that NNPM groups them |

## `edge_server`

| Field | Default | Meaning |
|---|---|---|
| `policy` | `SJF` | scheduling policy of the inference service: `RR` (one stage per request per round) or `SJF` (ascending offloaded work, ties by device index) |
| `nominal_frequency` | `true` | pin the edge server's cores to their base frequency with turbo off for the run, restored at teardown |
| `profile_capacity` | `true` | measure the edge server's FLOP/s on the model, used as C^R by the delay model |
| `port` | 5300 | port of the inference service |

## `inference`

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | run partitioned inference against the edge-server service after profiling |
| `partition_points` | list | the values of s to run (0 sends the raw input, S runs everything locally) |
| `repetitions` | 1 | inferences per partition point |

## `network`

A mapping from node id to a shaping block; the key `default` applies to every selected device. Shaping is applied with tc-netem on the device's data-plane interface (`network_interface`) at launch and removed at teardown.

| Field | Meaning |
|---|---|
| `delay_ms` | added egress delay in milliseconds |
| `added_rtt_ms` | extra round-trip time to the edge server, applied as egress delay; the Chapter 6 sweep variable |
| `from_map` | `true` derives `delay_ms` from the marker positions on the map (5 ms + 1 ms/km, clipped to [5, 50] ms) |
| `variation` | `mode` (`constant`; `uniform`: a new delay drawn from [`min_ms`, `max_ms`] every `period_s`; `square`: alternation between the delay and none every `period_s`) |
| `jitter_ms` | uniformly distributed random component of the delay |
| `loss_percent` | packet drop probability |
| `rate_mbps` | egress bandwidth cap (token-bucket filter) |
| `destinations` | node ids to shape separately (one netem leaf per destination); empty means all traffic of the interface |
| `background_traffic_mbps` | an iperf3 UDP stream from the device to the edge server at this rate for the whole run |

## `load`

A mapping from node id to a load block (`default` applies to every selected device). The load generator runs on the device for the whole run; leave the node out of `load` for no injected load.

| Field | Default | Meaning |
|---|---|---|
| `mode` | `fixed` | `fixed` (one CPU and memory target); `continuous` (a new target drawn every `cycle_s` within `cpu_range` and `memory_range`); `regime` (the Chapter 3 regimes: `light` 10-20 % CPU and 10-20 % memory, `moderate` 30-50 % and 30-40 %, `heavy` 60-80 % and 50-60 %); `processes` (`processes` fully busy processes, the Chapter 4 background-process model) |
| `cpu_percent`, `memory_percent` | | targets, in percent of one core (per worker) and of the device's memory |
| `cpu_range`, `memory_range` | | `[min, max]` for the continuous mode |
| `regime` | | `light`, `moderate` or `heavy` |
| `processes` | | number of fully busy processes |
| `cores` | 1 | number of CPU workers |
| `cycle_s` | 10 | resampling cycle of the continuous modes |
| `period_ms` | 100 | duty-cycle period of a CPU worker |

## `heterogeneity`

A mapping from node id to a block (`default` applies to every selected device).

| Field | Meaning |
|---|---|
| `max_freq_mhz` | cap the CPU frequency with cpufreq (for example 1200 or 600 on a Raspberry Pi 4 at 1800) |
| `memory_limit_mb` | run the experiment processes in a cgroups v2 group with this memory limit |
| `threads` | number of threads the inference runtime may use |

Every setting records the previous state and is restored at teardown.

## `monitoring`

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | run the resource-monitoring agent on every device |
| `interval_s` | 1.0 | sampling interval (CPU, memory, temperature, per-core frequency, throttling flags) |
| `power` | `false` | run the INA260 power monitor |
| `power_interval_s` | 0.5 | its sampling interval |
| `latency_pings` | 10 | pings per target in the latency capture of the collect stage |
| `throughput` | `false` | measure iperf3 throughput to the edge server |
| `throughput_duration_s` | 5 | duration of that measurement |

## `optimisation`

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | run the partition optimiser after aggregation |
| `algorithm` | `greedy` | a registered algorithm: `greedy`, `exhaustive`, `trend_based`, or a plug-in |
| `objective` | `total_delay` | `total_delay` (sum of the per-device delays) or `variance` (population variance of the per-device delays) |
| `initial_scheme` | `null` | starting vector: a JSON list, a `{node: s}` mapping or a percentage of the model depth (`"50%"`) |
| `max_total_delays_ms` | `null` | per-device limit on the total delay, a list or a `{node: ms}` mapping |
| `transmission_mode` | `rtt_only` | `rtt_only` (the ping round-trip time) or `size_over_rate` (cut size over the measured rate plus the one-way latency) |
| `num_solutions`, `max_time_s` | 5, 300 | parameters of the trend-based search |

## `headsplit`

Used when `runtime` is `headsplit`; the Create scenario dialog renders it from the runtime's schema. `controller` and `workers` (required) name nodes of `nodes.json`; `model` (the Hugging Face model name, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` by default), `tokens` (50), `prompt_length` (256), `mode` (`headsplit`, `head_only`, `layer_based`, `all`), `profile` (path of the speculation-accuracy profile, relative to the HeadSplit directory), `prompt`, `worker_memory_bytes`, `epsilon`, `alpha_min`, `memory_replace_threshold`, `rtt_margin`, `timeout_s`, `extra_args`. See [headsplit_integration.md](headsplit_integration.md). A plug-in runtime contributes its own block under its name.

## Examples

`examples/scenarios/baseline.yaml`: per-layer profiling of AlexNet on every device with no injected load (the `benchmarking` pipeline).

```yaml
name: baseline
pipeline: benchmarking
model: alexnet
devices: null
repetitions: 3
warmup: 1
seed: 42
edge_server: {policy: SJF, nominal_frequency: true, profile_capacity: true}
monitoring: {enabled: true, interval_s: 1.0, latency_pings: 10}
optimisation: {enabled: true, algorithm: greedy, objective: total_delay, initial_scheme: "50%"}
```

`examples/scenarios/local_demo.yaml`: the fake model on two simulated devices with shaping, load and heterogeneity, runnable on any machine:

```bash
python -m src.orchestration.experiment run --scenario examples/scenarios/local_demo.yaml --local
```

The other examples: `heavy_load.yaml` (the heavy regime on every device), `ch4_validation.yaml` (the Chapter 4 procedure: AlexNet, four devices, partitioned inference at the table's partition points, RR and SJF), `ch6_rtt_sweep.yaml` (HeadSplit under a sweep of added round-trip times) and `ch6_heterogeneity.yaml` (HeadSplit with frequency caps and memory limits on the workers).
