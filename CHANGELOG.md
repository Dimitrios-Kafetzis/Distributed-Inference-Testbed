# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- An NNPM tutorial (`docs/nnpm_tutorial.md`) with screenshots of every view, taken in local mode by `scripts/docs/nnpm_screenshots.py` (Playwright).
- A documentation website built with MkDocs (Material) from `docs/` (`mkdocs.yml`, `requirements/docs.lock`), published to GitHub Pages on every push and built strictly in CI.

### Fixed

- The runtime drop-down of the Create scenario dialog listed HeadSplit first and selected it by default; `testbed` is now first and selected.
- Form fields rendered from a schema appeared in alphabetical order because Flask 3 sorts JSON keys; the schema's order is kept.
- The run summary page printed the scenario version as a raw mapping.
- Documentation: the edge server is an Intel Core i7-5820K (setup guide); the version tables name the thesis software table as the list of pinned versions; the hardware checklist expects Transformers 4.46.3; the collection and aggregation docstrings describe the pull collection and the per-record time origins.

## [1.1.0] - 2026-10-25

Extensions beyond the thesis: runtime plug-ins with HeadSplit managed from NNPM, generic model adapters, a model catalogue, input sources and a partition check. The thesis requirements keep their status; the hardware checklist gains a section for these extensions.

### Added

- **Runtime plug-ins** (`src/orchestration/runtimes/`): an inference runtime exposes its scenario block as a JSON Schema, per-device readiness checks, maintenance actions and a results parser; runtimes register like models and algorithms. The built-in `testbed` runtime and HeadSplit are the two implementations; `docs/adding_a_runtime.md` explains the interface.
- **HeadSplit through NNPM**: the Create scenario dialog has a runtime selector and renders the HeadSplit block (controller, workers, model, tokens, mode, speculation parameters); the Runtimes dialog checks every device (checkout and commit, environment, profile, checkpoint) and offers three actions (install or update from the mirror, build the speculation profile on the edge server and copy it to the controller, distribute a checkpoint with checksum verification), run as tasks that log to the terminal; the run summary page shows the per-mode results. A controller that accepts `--json` (HeadSplit v0.1.0 and later) is used in preference to the summary lines.
- **Sweeps from NNPM**: a field path and values in the Create scenario dialog; the variants run one after the other, each carrying `sweep_origin`; the run summary page (`/runs/<id>`) groups them with their main figures.
- **Model parameters** (`model_args`): adapters declare a `PARAMETERS` schema (classes, sequence length, configuration overrides, application, checkpoint); scenarios pass the values, validated when the run starts, to every device tool; the dialog renders the fields of the chosen model.
- **Generic adapters**: any `tf.keras.applications` classifier with automatic single-tensor cut points thinned to stages of comparable work (MobileNetV2, EfficientNet-B0, DenseNet-121 and others by name); any Hugging Face decoder through a family table (GPT-2, Llama, Mistral, Qwen2, Gemma), with Llama 3.2 1B and Qwen2.5 0.5B registered. `docs/adding_a_model.md` explains the three ways to add a model.
- **Model catalogue** (`bs_models_dir/models.yaml`, `python -m src.models.weights list|verify|checksums|fetch`, `build_mirror.sh models`): seeded weights, Keras ImageNet weights and Hugging Face checkpoints with `SHA256SUMS`.
- **Input sources** (`input` block: `synthetic`, `numpy_file`, `image_file`, `image_folder` with class labels, `text_prompt`) recorded in every measurement, and the **partition-equivalence check** (`checks.partition_equivalence`, `python -m src.models.inference verify`: deviation per partition point, top-1 agreement, accuracy with a label).
- Project documentation for contributors: `CONTRIBUTING.md`, `SECURITY.md`, this changelog, issue and pull-request templates.
- A screenshot of the NNPM web application in local mode in the README.

### Changed

- Transformers is pinned to 4.46.3 in the LLM locks (Qwen2 needs 4.37, Llama 3.2 needs 4.43, HeadSplit 4.40); the head-wise decomposition accepts both the 4.36 and the 4.38+ rotary interfaces. The HeadSplit wheels are mirrored for x86_64 as well, for the profile step on the edge server.
- The README was rewritten: highlights, architecture diagram, supported models, getting started with and without hardware, running experiments, quality assurance, requirements and citation.
- TensorFlow runtime messages are suppressed on the devices (`TF_CPP_MIN_LOG_LEVEL=3`): on machines without a GPU, TensorFlow 2.15 prints spurious error-level messages at import that filled the merged experiment log.
- The continuous-integration workflow and the provisioning tests were made independent of the machine they run on (pre-created virtual environments, the test interpreter in the golden plans, the PyTorch CPU index for the edge-server lock).

### Fixed

- HeadSplit's workers and controller were started from `<headsplit_dir>/venv/bin/python` while provisioning creates `<headsplit_dir>/.venv`; on the hardware every worker would have failed to start.
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

[Unreleased]: https://github.com/Dimitrios-Kafetzis/Distributed-Inference-Testbed/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Dimitrios-Kafetzis/Distributed-Inference-Testbed/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Dimitrios-Kafetzis/Distributed-Inference-Testbed/releases/tag/v1.0.0
