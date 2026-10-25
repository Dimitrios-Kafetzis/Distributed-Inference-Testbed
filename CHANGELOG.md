# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project documentation for contributors: `CONTRIBUTING.md`, `SECURITY.md`, this changelog, issue and pull-request templates.
- A screenshot of the NNPM web application in local mode in the README.

### Changed

- The README was rewritten: highlights, architecture diagram, supported models, getting started with and without hardware, running experiments, quality assurance, requirements and citation.
- TensorFlow runtime messages are suppressed on the devices (`TF_CPP_MIN_LOG_LEVEL=3`): on machines without a GPU, TensorFlow 2.15 prints spurious error-level messages at import that filled the merged experiment log.
- The continuous-integration workflow and the provisioning tests were made independent of the machine they run on (pre-created virtual environments, the test interpreter in the golden plans, the PyTorch CPU index for the edge-server lock).

### Fixed

- The Runs panel of NNPM listed the newest run twice, once under its own name and once through the `results/latest` link.

## [1.0.0] - 2026-10-25

First release of the testbed described in Chapter 3 and Appendix B of the author's PhD thesis (Athens University of Economics and Business, 2026). It continues the earlier repository DNN-Partitioning-and-Offloading-Testbed (ACM MobiHoc 2024 demonstration), rewritten so that the code matches the thesis.

### Added

- Configuration: `config.yaml` with `TESTBED_*` overrides, `nodes.json` topology with management and data-plane addresses, JSON Schemas.
- Device-side tools: load generator (Chapter 3 regimes, Chapter 4 process model), tc-netem shaping (delay, variation, jitter, loss, rate cap, per destination), cpufreq and cgroups v2 heterogeneity, 1 Hz resource monitor, INA260 power monitor with energy metrics, ping latency, iperf3 throughput.
- Models and profiling: AlexNet (the 11 stages of Chapter 4), VGG-16, ResNet-50, the GRU QoE predictor (TensorFlow); GPT-2 and TinyLlama with per-head, projection and feed-forward timing through forward hooks (PyTorch).
- Partitioned inference: edge-server service with Round-Robin and Shortest-Job-First scheduling; device client with local, transmission and remote phase timing.
- Aggregation, partition feasibility, the Chapter 4 delay model and the greedy, exhaustive and trend-based search (`delay_optimizer.py`).
- Orchestration: versioned, replayable scenarios with sweeps; the experiment pipeline with a journal that undoes every change at teardown; coordinated start; run metadata; `run_full_benchmarking.sh`; local mode.
- The Neural Network Architecture Partitioning (NNPM) manager: Leaflet map with three-ping reachability probes, click-to-add nodes, per-pair links with the distance-to-latency rule, Create Scenario, Benchmarking and Optimise dialogs, live terminal, results browser.
- HeadSplit as an optional external LLM runtime at a pinned commit.
- Provisioning: first-boot image, edge server (access point, DHCP and DNS, NTP, mirror and NNPM services), devices over SSH from a versioned package mirror.
- Documentation: architecture, installation, setup guide, NNPM user guide, scenario and results references, module reference, HeadSplit integration, hardware validation checklist.

[Unreleased]: https://github.com/Dimitrios-Kafetzis/Distributed-Inference-Testbed/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Dimitrios-Kafetzis/Distributed-Inference-Testbed/releases/tag/v1.0.0
