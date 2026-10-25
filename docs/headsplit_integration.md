# HeadSplit integration

HeadSplit is the author's separate runtime for head-level partitioning and inter-layer speculation of decoder-only Large Language Models (LLMs) on the five-device cluster (Chapter 6): one controller and four workers, all Raspberry Pis, communicating through PyTorch RPC. The testbed treats it as an **optional external runtime**: nothing of HeadSplit is imported by the testbed, its code is not vendored, and the integration is command-line only. What the testbed adds is what HeadSplit alone does not have: provisioning of a pinned version on every device, the scenario's network shaping, load and heterogeneity, the monitors and power traces, the run metadata and the results layout.

## Configuration

In `config.yaml`:

```yaml
headsplit_repo_url: "https://github.com/Dimitrios-Kafetzis/headsplit.git"
headsplit_commit: <commit SHA>     # null disables HeadSplit
headsplit_dir: null                # directory on the devices; null means <ed_root>/headsplit
headsplit_master_port: 29500
```

`scripts/provision/build_mirror.sh headsplit` makes a bare clone at the pinned commit in the mirror and downloads the wheels of HeadSplit's `requirements.txt` for aarch64 (its Transformers requirement is newer than the testbed's pin, which is why HeadSplit gets its own virtual environment). `provision_device.sh` then clones from the mirror, checks out the commit and creates `<headsplit_dir>/.venv` on every device. The speculation-accuracy profile that the controller needs (`artifacts/profile.json`) is produced once on the controller device with HeadSplit's own `scripts/profile_offline.py`.

## Scenario

A scenario with `runtime: headsplit` replaces the profiling stage with a HeadSplit run; everything else (apply, collect, teardown, metadata, monitors) is the testbed's:

```yaml
name: ch6_rtt_sweep
runtime: headsplit
model: tinyllama
devices: [device1, device2, device3, device4, device5]
headsplit:
  controller: device1
  workers: [device2, device3, device4, device5]
  model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  tokens: 50
  prompt_length: 256
  mode: all                 # headsplit, head_only, layer_based or all
  profile: artifacts/profile.json
network:
  default: {added_rtt_ms: 0}
sweep:
  path: network.default.added_rtt_ms
  values: [0, 5, 10, 20, 40]
```

Fields: `controller` and `workers` name nodes of `nodes.json`; `model` is the Hugging Face model name HeadSplit expects; `tokens`, `prompt_length`, `prompt`; `mode`; `profile` (relative to the HeadSplit directory); `worker_memory_bytes` (3 GiB by default), `epsilon`, `alpha_min`, `memory_replace_threshold`, `rtt_margin` (HeadSplit's placement and speculation parameters, written into the per-run HeadSplit `config.yaml`); `timeout_s`; `extra_args` passed to the controller.

## What a run does

1. The testbed applies the scenario (shaping, load, heterogeneity, monitors, power) on every named node and writes the metadata, as for any run.
2. On every worker it starts, as a tagged detached process, `scripts/run_worker.py --name workerN --rank N --world-size W --master-addr <controller data-plane address> --master-port <headsplit_master_port> --model <model>` from `<headsplit_dir>/.venv`.
3. On the controller it writes a per-run HeadSplit `config.yaml` (the workers by data-plane address and memory, the speculation parameters, the profile path) and runs `scripts/run_controller.py --config ... --profile ... --master-addr ... --master-port ... --tokens N --mode <mode>`, waiting up to `timeout_s`.
4. The controller's summary lines (`<mode> total ... ms prefill ... ms decode ... ms/token payload ... MB`, with the speculation acceptance and re-placement counts when present, and the speed-up lines) are parsed into `results/<run id>/headsplit/results.json`; the controller and worker logs are kept next to it.
5. Teardown stops any worker still running and undoes the scenario. Workers exit when the controller shuts the RPC group down, so each repetition starts them afresh.

The aggregation stage does not compute the DNN output files for a HeadSplit run (there is no per-layer profile), and the optimiser is skipped. The monitor and power traces of every device are collected as usual, so energy per token and the energy-delay product can be computed from `power/` with `src/network/power_measurement.py`'s energy functions.

## Examples

`examples/scenarios/ch6_rtt_sweep.yaml` (one run per added round-trip time) and `examples/scenarios/ch6_heterogeneity.yaml` (frequency caps and memory limits on the workers) reproduce the Chapter 6 procedures. Without hardware, `runtime: headsplit` runs in local mode against stub scripts in the tests (`tests/integration/test_headsplit_runtime.py`), which is how the command lines and the parser are verified; the real runtime is a hardware-checklist item.
