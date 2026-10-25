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

Top part: name, model, repetitions, warm-up passes, seed, the edge server's scheduling policy (SJF or RR), the settling time, and optionally partitioned inference at a list of partition points.

Per node (one block per end device, each with a selection box):

- link delay in milliseconds, typed or filled by **Derive latencies from the map** (the rule's value for the link to the edge server, constant mode); variation (constant, uniform redraw within a range every period, square wave); jitter; loss; bandwidth cap. These are applied with tc-netem on the device's data-plane interface at launch and removed at the end;
- load mode: none, fixed (CPU percentage of one core and memory percentage), regime (light, moderate, heavy: the Chapter 3 ranges) or backlogged processes (the Chapter 4 model); the load generator runs on the node for the whole run;
- heterogeneity: maximum CPU frequency, memory limit, thread count.

**Save scenario** stores the version; **Save and run** also starts the run and returns at once with a run id, which appears in the toolbar indicator. All per-node actions are issued concurrently through one SSH session per node.

## Run experiment

Choose a saved scenario version and run it. The same scenario can be run again later, identically, which is what versioning is for.

## Benchmarking

The Benchmarking dialog (model, repetitions, warm-up, seed) runs the four-stage profiling pipeline across the active devices: dispatch of `train_and_run_benchmark.sh both`, result collection, post-processing (latency measurement, aggregation, feasibility) and the three output files `total_averages_of_benchmark_results.json`, `calculated_max_partition_points.json` and `final_latencies_averages.json` in `results/<run id>/`. The devices whose last probe was green are preselected; before the first probe, every registered device takes part. No load or shaping is applied.

## Optimise

The Optimise dialog loads a profiling run and runs the partition optimiser on it: algorithm (greedy, exhaustive, trend-based, or a plug-in), edge-server scheduling policy (RR or SJF), objective (total delay or variance), initial partition (`50%` of the model depth, an explicit list, or a `{node: s}` mapping), optional maximum total delay per device, and the transmission model (round-trip time only, or cut size over the measured rate plus latency). The result is shown as a table (device, s, local, transmission, remote and total delay) with the all-local baseline, and written to `results/<run id>/optimisation/<timestamp>-nnpm/` together with the scenario version, the hashes of the input files and the search log.

## Stop

Stop aborts the running experiment: the pipeline stops at the next stage boundary, every journal entry is undone in reverse order (processes stopped, shaping cleared, frequencies restored, cgroups removed) and the run is marked aborted. The run directory keeps everything produced so far. `scripts/base_station/clean_devices.sh` does the same from the command line, also for runs interrupted by a crash or a restart.

## Local mode

`python -m gui.app --local [ROOT] [--devices N]` creates a simulated testbed under `ROOT` (default `.local_testbed`): an edge server and N devices as working directories on this machine, with the fake model, dry-run shaping and a fake `/sys` tree. Every dialog works; the terminal shows the same log a real run produces. The page shows a "local mode" badge.

## Interface summary (for scripts)

| Method and path | Purpose |
|---|---|
| `GET /api/nodes`, `POST /api/nodes`, `GET/PUT/DELETE /api/nodes/<id>` | topology |
| `GET /api/links`, `POST /api/latency-from-map` | per-pair distances and latencies |
| `GET /api/status`, `POST /api/status/probe`, `GET /events/status` | reachability (the last is a server-sent event stream) |
| `GET /api/scenarios`, `GET /api/scenarios/<name>[/<version>]`, `POST /api/scenarios`, `POST /api/scenarios/validate` | scenario versions |
| `GET /api/runs`, `POST /api/runs`, `GET /api/runs/<id>`, `POST /api/runs/<id>/stop`, `GET /api/runs/<id>/files/<path>`, `GET /events/runs/<id>` | runs and their live log |
| `POST /api/optimise` | the Optimise dialog's back end |
| `GET /api/catalog` | registered models, algorithms, policies, regimes, example scenarios, the scenario schema |
| `GET /healthz` | liveness |
