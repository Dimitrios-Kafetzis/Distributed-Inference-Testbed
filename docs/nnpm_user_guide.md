# NNPM user guide

The Neural Network Architecture Partitioning (NNPM) manager is the web application that configures the testbed and runs experiments. It is served from the edge server and used from any browser on the management network.

## Starting NNPM

On the edge server (after [installation](installation.md)):

```bash
.venv/bin/python -m gui.app                # nnpm_host:nnpm_port from config.yaml (default port 8001)
.venv/bin/python -m gui.app --local        # a demonstration with simulated nodes, on this machine
```

Provisioning installs `nnpm.service`, so NNPM is normally already running at `http://<edge server management address>:8001/`. Flask's debug mode is off; `--debug` exists for development on a laptop only. To require a login, set `nnpm_basic_auth_env` in `config.yaml` to the name of an environment variable holding `user:password`; NNPM then asks for HTTP basic authentication.

## The main view

The page has three parts: the toolbar, the map with the side panel, and the terminal.

**Map.** One marker per node of `nodes.json`, the edge server included (its marker is drawn in a different colour). The label under a marker shows the node id, its data-plane address and a coloured dot: green when the last reachability probe received three replies out of three pings, red otherwise, grey before the first probe. A background prober pings every node concurrently and starts a new round every two seconds (`probe_period_s`); a slow or unreachable node keeps its last colour until its probe answers. Red lines join every pair of markers; hovering a line shows the distance between the two nodes and the synthetic latency of the link (5 ms plus 1 ms per kilometre, clipped to 5 to 50 ms). The "Draw links" box hides them.

The map tiles come from `map_tiles_url`; on an isolated network set `map_offline: true` and the map draws a plain grid instead of fetching tiles. Leaflet is served by NNPM itself, so the application never needs internet access.

**Node panel.** Clicking a marker shows the node's role, addresses, reachability, and the statistics of the last benchmarking run: average ping latency, available memory and CPU usage with the time of the measurement and the run that produced them. The panel also has Edit and Delete buttons.

**Runs panel.** The runs found in the results directory, newest first, with their scenario name and status. The run id links to the run's summary, from which every result file can be downloaded (`/api/runs/<id>/files/<path>`).

**Terminal.** The merged log of the running experiment: every line is tagged `[node][stream]` and appears as it is produced on the device. The filter box keeps the lines of one node; Clear empties the panel. Lines that NNPM writes itself are tagged `[nnpm]`.

## Editing the topology

- **Add node by clicking the map**: press the button (it stays highlighted), click the map at the desired position; the add dialog opens with the clicked coordinates. Enter the id, the role, the data-plane address and, with a separate management network, the management address. Saving writes `nodes.json`; the marker appears at once and the prober includes the node in its next round.
- **Edit network**: a table of all nodes with Edit and Delete buttons, and the same form for a new node.
- **Dragging a marker** moves the node and saves the new coordinates. The links and the derived latencies follow.

`nodes.json` is the single source of truth: the command-line scripts read the same file.

## Create scenario

The dialog captures every parameter of an experiment and saves it as a scenario version (`scenarios/<name>/v<N>.json`, see [scenario_reference.md](scenario_reference.md)).

Top part: name, **runtime**, model, repetitions, warm-up passes, seed, the edge server's scheduling policy (SJF or RR), the settling time, and optionally partitioned inference at a list of partition points.

- **Runtime.** `testbed` is the per-layer profiling and partitioned-inference path. An external runtime such as `headsplit` adds its own block below the top part, rendered from the runtime's schema: for HeadSplit the controller (a device drop-down), the workers (check boxes), the Hugging Face model, tokens, prompt length, mode, the speculation parameters and the time-out. The per-node shaping, load and heterogeneity blocks stay available, since the Chapter 6 studies combine them with HeadSplit. See [headsplit_integration.md](headsplit_integration.md).
- **Model parameters.** A model with parameters (the number of classes of a Keras model, the sequence length of a decoder, the application name of `keras_app`, the checkpoint of `hf_decoder`) shows them under the model drop-down; they are saved as `model_args`.
- **Sweep.** A field path (for example `network.default.added_rtt_ms` or `edge_server.policy`) and comma-separated values make one run per value. NNPM runs them one after the other; the Runs panel shows every run with its value, and each run's summary page has the sweep table.

Per node (one block per end device, each with a selection box):

- link delay in milliseconds, typed or filled by **Derive latencies from the map** (the rule's value for the link to the edge server, constant mode); variation (constant, uniform redraw within a range every period, square wave); jitter; loss; bandwidth cap. These are applied with tc-netem on the device's data-plane interface at launch and removed at the end;
- load mode: none, fixed (CPU percentage of one core and memory percentage), regime (light, moderate, heavy: the Chapter 3 ranges) or backlogged processes (the Chapter 4 model); the load generator runs on the node for the whole run;
- heterogeneity: maximum CPU frequency, memory limit, thread count.

**Save scenario** stores the version; **Save and run** also starts the run and returns at once with a run id, which appears in the toolbar indicator. All per-node actions are issued concurrently through one SSH session per node.

## Run experiment

Choose a saved scenario version and run it. The same scenario can be run again later, identically, which is what versioning is for.

## Run summary page

The run id in the Runs panel opens `/runs/<run id>`: the scenario (name and version, runtime, model, devices, sweep value), the outputs (edge-server capacity, optimised vector and objective, HeadSplit totals), the three output files of the thesis with download links and every other file, the runtime's results (for HeadSplit a table per mode: total, prefill, decode time per token, payload, speculation acceptance, re-placements, speed-up versus layer-based), the other runs of the same sweep (one row per value with the main figure), and the last lines of the merged log. `GET /api/runs/<run id>` returns the same as JSON.

## Runtimes

The Runtimes dialog shows, for a chosen runtime, whether the devices are ready for it and lets you prepare them. **Check the devices** runs the runtime's readiness checks on every node through the same connections as experiments and shows them as a table (green, amber, red; hover a cell for the detail). For HeadSplit: the checkout and its commit against `headsplit_commit`, the environment (`torch` and `transformers` import from `<headsplit_dir>/.venv`), the speculation profile on the controller, and the checkpoint under the devices' models directory (type a model name to look for another checkpoint).

Below the table, one form per maintenance action. For HeadSplit: **Install or update HeadSplit** (clone or update from the mirror at the pinned commit and build the environment on the selected devices), **Build the speculation profile** (HeadSplit's `profile_offline.py` on the edge server, then a copy to the controller; the calibration data is downloaded, so `allow_upstream` must be ticked and the edge server must reach the internet), **Distribute a checkpoint** (from `bs_models_dir/<model directory>` to the selected devices, verified with `SHA256SUMS` there; the catalogue, `python -m src.models.weights`, fetches checkpoints into `bs_models_dir`). Actions run as tasks: one at a time, never while an experiment runs, with their log in the terminal tagged `[task]`; Stop ends a running task after its current step. See [adding_a_runtime.md](adding_a_runtime.md) for the interface behind the dialog.

## Benchmarking

The Benchmarking dialog (model, repetitions, warm-up, seed) runs the four-stage profiling pipeline across the active devices: dispatch of `train_and_run_benchmark.sh both`, result collection, post-processing (latency measurement, aggregation, feasibility) and the three output files `total_averages_of_benchmark_results.json`, `calculated_max_partition_points.json` and `final_latencies_averages.json` in `results/<run id>/`. The devices whose last probe was green are preselected; before the first probe, every registered device takes part. No load or shaping is applied.

## Optimise

The Optimise dialog loads a profiling run and runs the partition optimiser on it: algorithm (greedy, exhaustive, trend-based, or a plug-in), edge-server scheduling policy (RR or SJF), objective (total delay or variance), initial partition (`50%` of the model depth, an explicit list, or a `{node: s}` mapping), optional maximum total delay per device, and the transmission model (round-trip time only, or cut size over the measured rate plus latency). The result is shown as a table (device, s, local, transmission, remote and total delay) with the all-local baseline, and written to `results/<run id>/optimisation/<timestamp>-nnpm/` together with the scenario version, the hashes of the input files and the search log.

## Stop

Stop aborts the running experiment: the pipeline stops at the next stage boundary, every journal entry is undone in reverse order (processes stopped, shaping cleared, frequencies restored, cgroups removed) and the run is marked aborted. The run directory keeps everything produced so far. `scripts/base_station/clean_devices.sh` does the same from the command line, also for runs interrupted by a crash or a restart.

## Local mode

`python -m gui.app --local [ROOT] [--devices N]` creates a simulated testbed under `ROOT` (default `.local_testbed`): an edge server and N devices as working directories on this machine, with the fake model, dry-run shaping and a fake `/sys` tree. Every dialog works; the terminal shows the same log a real run produces. The page shows a "local mode" badge. The Runtimes dialog works too: the HeadSplit checks run against `headsplit_dir` on this machine, the install action prints the plan it would run (`--dry-run`), the profile action runs the profile script found there, and a checkpoint is copied between local directories.

## Interface summary (for scripts)

| Method and path | Purpose |
|---|---|
| `GET /api/nodes`, `POST /api/nodes`, `GET/PUT/DELETE /api/nodes/<id>` | topology |
| `GET /api/links`, `POST /api/latency-from-map` | per-pair distances and latencies |
| `GET /api/status`, `POST /api/status/probe`, `GET /events/status` | reachability (the last is a server-sent event stream) |
| `GET /api/scenarios`, `GET /api/scenarios/<name>[/<version>]`, `POST /api/scenarios`, `POST /api/scenarios/validate` | scenario versions |
| `GET /api/runs`, `POST /api/runs`, `GET /api/runs/<id>`, `POST /api/runs/<id>/stop`, `GET /api/runs/<id>/files/<path>`, `GET /events/runs/<id>` | runs and their live log |
| `POST /api/optimise` | the Optimise dialog's back end |
| `GET /api/catalog` | registered models with their parameter schemas, runtimes with their block schemas and actions, algorithms, policies, regimes, example scenarios, the scenario schema |
| `GET /api/runtimes`, `GET /api/runtimes/<name>/status[?model=...]`, `POST /api/runtimes/<name>/actions/<action>` | the runtimes, their readiness checks per node, and their maintenance actions (started as tasks) |
| `GET /api/tasks`, `GET /api/tasks/<id>`, `POST /api/tasks/<id>/stop`, `GET /events/tasks/<id>` | the tasks and their live log |
| `GET /runs/<id>` | the run summary page |
| `GET /healthz` | liveness |
