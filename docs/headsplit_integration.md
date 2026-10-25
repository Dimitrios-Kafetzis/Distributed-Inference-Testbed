# HeadSplit integration

HeadSplit is the author's separate runtime for head-level partitioning and inter-layer speculation of decoder-only Large Language Models (LLMs) on the five-device cluster (Chapter 6): one controller and four workers, all Raspberry Pis, communicating through PyTorch RPC. The testbed treats it as an **optional external runtime**: nothing of HeadSplit is imported by the testbed, its code is not vendored, and the integration is command-line only. What the testbed adds is what HeadSplit alone does not have: provisioning of a pinned version on every device, the scenario's network shaping, load and heterogeneity, the monitors and power traces, the run metadata and the results layout. HeadSplit is the first implementation of the runtime plug-in interface ([adding_a_runtime.md](adding_a_runtime.md), `src/orchestration/runtimes/headsplit.py`), so NNPM configures it in the Create scenario dialog, checks and prepares the devices in the Runtimes dialog and shows its results on the run summary page ([nnpm_user_guide.md](nnpm_user_guide.md)).

HeadSplit supports Llama-family models (TinyLlama, Llama 3.2 and other models with the Llama attention layout). GPT-2 has a different layout (fused query, key and value projection, learned positions) and is not supported by HeadSplit; the GPT-2 figures of the thesis come from the testbed's own layer-based path (`model: gpt2` with the `testbed` runtime).

## Configuration

In `config.yaml`:

```yaml
headsplit_repo_url: "https://github.com/Dimitrios-Kafetzis/headsplit.git"
headsplit_commit: "4864a22e5bdc2c712ffe071d7f7dba0c3656fb33"   # HeadSplit v0.1.0; null disables HeadSplit
headsplit_dir: null                # directory on the devices; null means <ed_root>/headsplit
headsplit_master_port: 29500
```

`scripts/provision/build_mirror.sh headsplit` makes a bare clone at the pinned commit in the mirror and downloads the wheels of HeadSplit's `requirements.txt` for aarch64 (the devices) and x86_64 (the edge server, for the profile step). HeadSplit keeps its own virtual environment, `<headsplit_dir>/.venv`, so that its requirements never constrain the testbed's locks. `provision_device.sh` clones from the mirror, checks out the commit and creates the environment on every device; the same can be done later, device by device, with the **Install or update HeadSplit** action of NNPM's Runtimes dialog.

The speculation-accuracy profile that the controller needs (`artifacts/profile.json`) is built once with HeadSplit's `scripts/profile_offline.py`, which downloads its calibration data (WikiText-103): NNPM's **Build the speculation profile** action runs it on the edge server (which may reach the internet when `allow_upstream` is ticked; the devices never do) and copies the result to the controller. The checkpoint (`TinyLlama/TinyLlama-1.1B-Chat-v1.0` by default) must be on every device: the model catalogue fetches it into `bs_models_dir` (`python -m src.models.weights fetch tinyllama --allow-upstream`, see `examples/models.yaml`), provisioning copies `bs_models_dir` to the devices, and the **Distribute a checkpoint** action does the same on demand with checksum verification. The **Check the devices** button shows all four facts per device: checkout and commit, environment, profile, checkpoint.

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
3. On the controller it writes a per-run HeadSplit `config.yaml` (the workers by data-plane address and memory, the speculation parameters, the profile path) and runs `scripts/run_controller.py --config ... --profile ... --master-addr ... --master-port ... --tokens N --mode <mode>`, waiting up to `timeout_s`. A controller that accepts `--json` (HeadSplit v0.1.0 and later) also receives `--json <results dir>/headsplit_rep<k>.json`.
4. The results come from the JSON file when there is one (the per-mode figures plus the per-token times, the speculation decisions per layer, the placement and the measured round-trip times), otherwise from the controller's summary lines (`<mode> total ... ms prefill ... ms decode ... ms/token payload ... MB`, with the speculation acceptance and re-placement counts when present, and the speed-up lines). They are written to `results/<run id>/headsplit/results.json`; the controller and worker logs are kept next to it, and the run summary page shows the per-mode table.
5. Teardown stops any worker still running and undoes the scenario. Workers exit when the controller shuts the RPC group down, so each repetition starts them afresh.

The aggregation stage does not compute the DNN output files for a HeadSplit run (there is no per-layer profile), and the optimiser is skipped. The monitor and power traces of every device are collected as usual, so energy per token and the energy-delay product can be computed from `power/` with `src/network/power_measurement.py`'s energy functions.

## Examples

`examples/scenarios/ch6_rtt_sweep.yaml` (one run per added round-trip time) and `examples/scenarios/ch6_heterogeneity.yaml` (frequency caps and memory limits on the workers) reproduce the Chapter 6 procedures. Without hardware, `runtime: headsplit` runs in local mode against stub scripts in the tests (`tests/integration/test_headsplit_runtime.py`), which is how the command lines and the parser are verified; the real runtime is a hardware-checklist item.
