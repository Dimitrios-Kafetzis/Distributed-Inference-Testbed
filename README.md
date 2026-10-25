# Distributed Inference Testbed

An experimental testbed for partitioned Artificial Intelligence (AI) inference across resource-constrained edge devices: a cluster of five Raspberry Pi 4 single-board computers connected over a dedicated wireless network to an edge server that hosts the access point. The Neural Network Architecture Partitioning (NNPM) manager, a web application served from the edge server, configures the testbed and orchestrates experiments: it dispatches commands to the devices over Secure Shell (SSH), profiles the models layer by layer under controlled compute and network load, collects the measurements and computes partition assignments.

The testbed supports Deep Neural Networks (DNNs; AlexNet, VGG-16, ResNet-50) in TensorFlow and Transformer-based Large Language Models (LLMs; GPT-2, TinyLlama-1.1B) in PyTorch, and runs the author's HeadSplit runtime as an optional external component. It requires no daemon on the devices beyond an SSH server: every device-side process is started for the duration of an experiment and stopped at its end. Every component runs without hardware in a local mode, which is also how the test suite exercises the whole pipeline.

This repository implements the testbed described in Chapter 3 ("Experimental Testbed Description") and Appendix B of the author's PhD thesis (Athens University of Economics and Business, 2026). The thesis is the specification: the file, script and output names it cites are the names used here.

## What is in the repository

| Component | Location |
|---|---|
| Configuration: `config.yaml` with `TESTBED_*` overrides, `nodes.json` topology, JSON Schemas | `config.example.yaml`, `nodes.example.json`, `src/config.py`, `src/common/nodes.py`, `docs/schemas/` |
| Device-side tools: load generator (Chapter 3 regimes), tc-netem shaping (delay, variation, jitter, loss, rate cap, per destination), heterogeneity (cpufreq, cgroups v2, threads), 1 Hz resource monitor, INA260 power monitor with energy metrics, ping latency, iperf3 throughput | `src/testbed/`, `src/network/` |
| Models and per-layer profiling: AlexNet (the 11 stages of Chapter 4), VGG-16, ResNet-50, the GRU QoE predictor (TensorFlow); GPT-2 and TinyLlama at layer granularity with per-head, projection and FFN timing through forward hooks (PyTorch); a fake model for tests | `src/models/`, `scripts/end_device/train_and_run_benchmark.sh` |
| Partitioned inference: the edge-server service with Round-Robin and Shortest-Job-First scheduling, the device client with local, transmission and remote phase timing | `src/inference/` |
| Measurement records, aggregation over repetitions, trace merging on the NTP origin, partition feasibility | `src/benchmarking/` |
| Partition optimisation: the Chapter 4 delay model (RR and SJF), greedy, exhaustive and trend-based search, `delay_optimizer.py` | `src/partitioning/` |
| Orchestration: versioned, replayable scenarios; the experiment pipeline with a journal that undoes every change at teardown, also on failure; coordinated start; SSH dispatch with a merged live log; local mode | `src/orchestration/`, `src/network/ssh_executor.py`, `scripts/base_station/` |
| NNPM web application: Leaflet map with three-ping reachability probes every two seconds, click-to-add nodes, per-pair links with the distance-to-latency rule, Create Scenario, Benchmarking and Optimise dialogs, live terminal, results browser | `gui/` |
| HeadSplit as an optional external LLM runtime (pinned commit, own environment, started as tagged processes) | `src/orchestration/headsplit_runtime.py`, `docs/headsplit_integration.md` |
| Provisioning: first-boot image, edge server (access point, DHCP and DNS, NTP server, mirror and NNPM services), devices over SSH from a versioned package mirror | `scripts/provision/`, `src/provisioning/` |
| Documentation | `docs/` (see below) |

## Quick start without hardware

```bash
git clone https://github.com/Dimitrios-Kafetzis/Distributed-Inference-Testbed.git
cd Distributed-Inference-Testbed
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python --require-hashes -r requirements/dev.lock
.venv/bin/pytest -m "not slow"                       # the fast suite: fakes, dry runs, local mode
.venv/bin/python -m src.orchestration.experiment run \
    --scenario examples/scenarios/local_demo.yaml --local  # a whole experiment on simulated nodes
ls .local_testbed/results/latest/                    # the run directory with the three output files
.venv/bin/python -m gui.app --local                  # NNPM at http://127.0.0.1:8001/ with simulated nodes
```

Without `uv`: `python3.11 -m venv .venv && .venv/bin/pip install --require-hashes -r requirements/dev.lock`.

## Quick start on the hardware

1. Assemble the hardware and networks as in [docs/setup_guide.md](docs/setup_guide.md): five Raspberry Pi 4 boards on a rack, an edge server with a USB IEEE 802.11ac adapter for the isolated access point, an Ethernet switch for the management network.
2. On the edge server: copy `config.example.yaml` to `config.yaml` and `nodes.example.json` to `nodes.json`, edit both, then `scripts/provision/provision_edge_server.sh --allow-upstream` and `scripts/provision/build_mirror.sh wheels`.
3. Per device: flash Raspberry Pi OS Bookworm 64-bit, `scripts/provision/prepare_image.sh <node id> <boot partition>`, boot it on the switch, then `scripts/provision/provision_device.sh <node id>` (after `build_mirror.sh apt <node id>` once).
4. Open NNPM at `http://<edge server management address>:8001/`; every marker should be green. Run `scripts/base_station/run_full_benchmarking.sh examples/scenarios/baseline.yaml` and find the three output files (`total_averages_of_benchmark_results.json`, `calculated_max_partition_points.json`, `final_latencies_averages.json`) in `results/latest/`.
5. Work through [docs/hardware_validation_checklist.md](docs/hardware_validation_checklist.md): everything the test suite cannot verify without the physical testbed, with the expected values.

## Configuration

`config.yaml` holds the SSH settings, the network prefixes and interfaces, the paths on the edge server and on the devices, the NNPM, access-point and mirror settings. Every key can be overridden by an environment variable `TESTBED_<KEY>`; secrets (the SSH password, the access-point passphrase) come only from environment variables named in the file. `nodes.json` holds the topology: one entry per node with its identifier, role (`Base Station` is the name of the edge server in the software, `End Device` a Raspberry Pi), data-plane and management addresses, map coordinates and the statistics of the last benchmarking run. Both files live at the repository root and are validated against the schemas in `docs/schemas/`.

## Documentation

| Document | Content |
|---|---|
| [docs/architecture.md](docs/architecture.md) | the three layers, the repository layout, an experiment stage by stage, daemonless operation, local mode |
| [docs/installation.md](docs/installation.md) | installing from the lock files on the edge server, the devices and a development machine; the version table |
| [docs/setup_guide.md](docs/setup_guide.md) | hardware and bill of materials, physical layout, networks and addressing, operating systems, access point, NTP, power sensors, mirror, provisioning order |
| [docs/nnpm_user_guide.md](docs/nnpm_user_guide.md) | the web application: map, probes, topology editing, the three dialogs, runs, terminal, local mode, the HTTP interface |
| [docs/scenario_reference.md](docs/scenario_reference.md) | every scenario field with its default |
| [docs/results_format.md](docs/results_format.md) | the run directory and every record format |
| [docs/module_reference.md](docs/module_reference.md) | every module and script with its command line |
| [docs/headsplit_integration.md](docs/headsplit_integration.md) | the optional HeadSplit runtime |
| [docs/hardware_validation_checklist.md](docs/hardware_validation_checklist.md) | what to verify on the physical testbed |
| [scripts/provision/README.md](scripts/provision/README.md), [requirements/README.md](requirements/README.md), [data/sample/README.md](data/sample/README.md) | provisioning scripts, lock files, synthetic sample data |

## Development

Python 3.11; `ruff` (formatting and lint), `mypy --strict` on `src/` and `gui/`, `shellcheck` on every script, `pytest` with the markers `slow` (real TensorFlow and PyTorch models), `browser` and `hardware` (never run in continuous integration). Every external effect goes through an injectable runner, transport, clock or rooted `/sys`, so the suite runs on any machine; the continuous-integration workflow (`.github/workflows/ci.yml`) runs lint, types, shellcheck, the fast suite with the local-mode pipeline, and the slow model tests.

## Provenance

This testbed continues the author's earlier repository [DNN-Partitioning-and-Offloading-Testbed](https://github.com/Dimitrios-Kafetzis/DNN-Partitioning-and-Offloading-Testbed), first presented as a demonstration at ACM MobiHoc 2024 (D. Kafetzis and I. Koutsopoulos, "Demo: An Experimental Platform for AI Model Partitioning on Resource-constrained Devices", pp. 375-376). The code was rewritten so that it implements the testbed as the thesis describes it; the earlier repository is reference material. Citation details are in [CITATION.cff](CITATION.cff).

## Licence

MIT, see [LICENSE](LICENSE). Vendored third-party software is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
