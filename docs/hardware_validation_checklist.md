# Hardware validation checklist

Everything in this repository is tested without hardware: with fake command runners, fake clocks, a fake sysfs tree and a local mode that simulates all nodes on one machine. The items below are the physical checks that only the real testbed can answer. Each item names the requirement it closes (IDs from the thesis requirement catalogue), the steps, and the expected outcome. Record the outcome and the date next to each item.

## Software stack (A-07, J-19)

- [ ] On one Raspberry Pi 4 (Raspberry Pi OS Bookworm 64-bit, Python 3.11): create a virtual environment and install `requirements/edge_device.lock` with `pip install --require-hashes -r requirements/edge_device.lock`. Expected: every wheel is found for `aarch64` and `cp311`; `python -c "import tensorflow; print(tensorflow.__version__)"` prints `2.15.1`.
- [ ] Same device: install `requirements/edge_device_llm.lock`. Expected: `torch` 2.1.2 and `transformers` 4.36.2 import; `torch.get_num_threads()` returns 4.
- [ ] Note the wall-clock time of each installation (J-15; the thesis states about 25 minutes per device for the whole deployment).

## Process control (C-02)

- [ ] On a device, run `python -m src.config --env` through SSH and confirm the printed `TESTBED_*` values match `config.yaml`.
- [ ] Start a tagged process over SSH with the start command of `src/common/process_tags.py` (for example a `sleep 60`), close the SSH session, reconnect and run the check command. Expected: `running`. Run the stop command. Expected: `stopped`, no process with the tag in `ps`, PID file removed.

## Compute load (G-01, J-03)

- [ ] On a device, run `python -m src.testbed.load_generator --cpu 50 --duration 60` and watch `top` or `mpstat -P ALL 1`. Expected: one core at about 50 % (within 5 points), the others idle.
- [ ] Run `--regime heavy --duration 120 --log /tmp/load.json`. Expected: total CPU use between 60 and 80 % of one core, memory in use between 50 and 60 % (`free -m`), both changing every 10 s; `/tmp/load.json` lists the drawn targets.
- [ ] Run `--processes 2 --duration 30`. Expected: two cores fully busy.

## Network shaping (G-02, G-03, C-07)

- [ ] On a device, `scripts/end_device/apply_latency.sh wlan0 20 20 --once`, then `ping -c 10 <edge server>` from the device. Expected: RTT about 20 ms above the baseline. `apply_latency.sh wlan0 --clear` restores the baseline.
- [ ] `python -m src.testbed.latency_injector wlan0 --delay 10 --jitter 5 --loss 1 --rate 10`, then `iperf3 -c <edge server> -t 10` and `ping -c 50`. Expected: about 10 Mbit/s, RTT 10 ms +- 5 ms, about 1 % loss.
- [ ] Per destination: `--dst <edge server IP> --delay 15 --dst <another device IP> --delay 30`. Expected: the two pings differ by about 15 ms.
- [ ] Note any driver that rejects the HTB tree on `wlan0`; the single-root fallback (`apply_all`) is the alternative.

## Heterogeneity and frequency (G-04, B-04, F-01)

- [ ] Device: `sudo python -m src.testbed.heterogeneity freq --max-mhz 600 --state /tmp/f.json`, then `python -m src.testbed.resource_monitor --output /tmp/m.json --run-id t --node-id d --duration 10`. Expected: `frequencies_khz` at or below 600000 in every sample; `freq --restore /tmp/f.json` brings them back to 1800000.
- [ ] Device: `sudo python -m src.testbed.heterogeneity cgroup create --name testbed-check --memory-mb 512`, then run `sh -c "$(python -c 'from src.testbed.heterogeneity import MemoryLimit; print(MemoryLimit().attach_command("testbed-check", "python3 -c \"x=bytearray(700*2**20)\""))')"`. Expected: the allocation is killed by the OOM killer of the group; `cgroup remove` cleans up.
- [ ] Edge server: `sudo python -m src.testbed.cpu_frequency apply --state /tmp/s.json`, then `verify`. Expected: governor `performance`, turbo disabled, every core near the base frequency; `restore` returns the previous settings.
- [ ] Device under a long heavy load: the monitor trace shows the temperature and whether `throttled` ever leaves `0x0`; the monitor's own CPU use (`top -p <pid>`) stays below 2 %.

## Power and throughput (F-04, F-08, J-07)

- [ ] Device with INA260: `pip install -r requirements/power.in`, then `python -m src.network.power_measurement --interval 0.2 --duration 30 --output /tmp/p.json --run-id t --node-id d`. Expected: about 150 samples, idle power 2.5-3.5 W, `energy_j` consistent with power x 30 s.
- [ ] Device: `python -m src.network.throughput_measurement --server <edge server> --duration 10 --output /tmp/tp.json` with `iperf3 -s -1` on the edge server, once alone and once with all five devices at the same time. Record the figures (the thesis states 80-120 Mbit/s per device and, in Chapter 5, a measured 19.6 Mbit/s).

## Partitioned inference and per-phase timing (B-05, D-05)

- [ ] Edge server (data-plane address): `python -m src.inference.server --model alexnet --weights models/alexnet.npz --policy SJF --port 5300 --nodes device1,device2,device3,device4`. Device: `python -m src.inference.client --model alexnet --weights <ed_root>/models/alexnet.npz --server <edge server>:5300 --s 0 --s 4 --s 8 --s 11 --repetitions 10 --output /tmp/phases.json --run-id t --node-id device1`. Expected: `transmission_ms` grows with the transmitted size (s = 1 is the largest cut, s = 8 is 36 KB), `remote_ms` shrinks with s, s = 11 has no transmission.
- [ ] Four devices at once with `--s 4` and `--policy SJF`, then `--policy RR`: the per-device `queue_ms` values follow the policy (SJF: ascending offloaded work; RR: interleaved), as in the loopback tests.
- [ ] Compare the measured `transmission_ms` of the s = 1 cut (1.1 MB) with size / throughput from the iperf3 figure plus the ping RTT.

## Whole pipeline on the testbed (C-01, C-04, C-13, D-10, D-13)

- [ ] Edge server: `scripts/base_station/run_full_benchmarking.sh examples/scenarios/baseline.yaml`. Expected: every device runs `train_and_run_benchmark.sh both`, the three output files appear in `results/<run id>/`, `nodes.json` gains statistics, and `journal.json` shows every entry undone.
- [ ] Edge server: `scripts/base_station/run_experiment.sh examples/scenarios/heavy_load.yaml`. During the run, on a device: `tc qdisc show dev wlan0` lists the netem (and tbf) queue, `top` shows the load generator and the monitor; after the run both are gone and `tc qdisc show` is back to the default. The monitor trace in `results/<run id>/monitor/` shows the heavy regime's CPU and memory levels.
- [ ] Start skew (C-04): in `results/<run id>/raw/`, compare the `origin.wall_time_s` of the devices' records. Expected: within the NTP offset (sub-millisecond on the wired management network).
- [ ] Interrupt a run with Ctrl-C during profiling, then run `scripts/base_station/clean_devices.sh`. Expected: no `testbed-` process, no netem queue and no `testbed-` cgroup remains on any device.
- [ ] NTP guard: stop chrony on one device and start a run. Expected: the preflight refuses the run with the offending node named.

## LLM profiling and HeadSplit (A-02, D-04, I-03, I-04)

- [ ] Device with the LLM environment and a TinyLlama checkpoint in the mirror: `python -m src.models.torch.hooks --model tinyllama --weights <checkpoint dir> --seq-len 256 --repetitions 3 --output /tmp/blocks.json`. Expected: per layer, FFN time about nine times the projection and the attention heads a few milliseconds each (Chapter 5, Table small-real-blocks gives 2.4-6.7 ms per head, 27-64 ms projection, 0.23-0.62 s FFN at 1.8 GHz and four threads); `TESTBED_THREADS=1` roughly doubles them.
- [ ] Device: `scripts/end_device/train_and_run_benchmark.sh both <models dir> gpt2 /tmp/r --repetitions 2` with `TESTBED_MODEL_WEIGHTS` pointing at a GPT-2 checkpoint directory (or random weights): per-layer stage times and the logits stage are recorded; `python -m src.inference.client --model gpt2 ...` against the edge-server service moves the hidden-state cut (T x 768 floats) over the wireless link.
- [ ] HeadSplit: provision the devices with `headsplit_commit` set; on the controller device run `scripts/profile_offline.py` once (needs the calibration data) to produce `artifacts/profile.json`; then `scripts/base_station/run_experiment.sh examples/scenarios/ch6_rtt_sweep.yaml`. Expected: one run per added RTT value, `results/<run id>/headsplit/results.json` with the layer_based, head_only and headsplit totals, and the monitor and power traces of every worker. Compare the slopes with Chapter 6 (0.15 s and 1.1 s per millisecond of added RTT).

## NNPM web application (C-05 to C-10, J-14)

- [ ] Edge server: `nnpm` (or `python -m gui.app`), then open `http://<management address>:<nnpm_port>/` from a laptop on the management network. Expected: the page title reads "Neural Network Architecture Partitioning (NNPM) manager", every node of `nodes.json` has a marker, the edge server included, and the map tiles load from the configured source (or the grid appears with `map_offline: true`).
- [ ] Power off one device. Expected: its marker turns red within two probe rounds (about 4 s) while the others stay green; power it on again and it turns green once three of three pings succeed.
- [ ] Enable "Add node by clicking the map", click a position, type an id and the device's addresses, save. Expected: `nodes.json` contains the clicked coordinates; the marker appears at the clicked position; dragging it updates the coordinates in the file.
- [ ] Click a device marker after a benchmarking run. Expected: the panel shows the last-measured latency, available memory and CPU usage with the time of the measurement.
- [ ] Create Scenario, "Derive latencies from the map", "Save and run". During the run, on the device: `tc qdisc show dev <interface>` shows `netem delay <derived value>ms`; the panel's ping value is close to the injected one. After the run the queue is gone.
- [ ] Benchmarking dialog with the default values. Expected: the terminal shows the `[node][stream]` lines of every device as they are produced; the three output files appear under `results/<run id>/`.
- [ ] Optimise dialog on that run: greedy, SJF, `50%`. Expected: the table and the JSON file under `results/<run id>/optimisation/`.
- [ ] Press Stop during a run. Expected: the run is reported as aborted, and on every device no `testbed-` process, no netem queue and no `testbed-` cgroup remains.

## Provisioning, access point, time synchronisation (H-01, H-02, B-06, F-06, A-08, J-15)

- [ ] Edge server (Ubuntu 22.04, USB 802.11ac adapter as `ap_interface`): `scripts/provision/provision_edge_server.sh --allow-upstream`, then `scripts/provision/build_mirror.sh wheels` and `build_mirror.sh headsplit` (with `headsplit_commit` set). Expected: `hostapd`, `dnsmasq`, `chrony`, `testbed-mirror` and `nnpm` are active; a laptop sees the SSID and, when joined, gets an address from the data-plane pool and no default route; `curl http://<management address>:<mirror_port>/wheels/aarch64/` lists the wheels. Record the elapsed time from `results/provisioning/timing.log`.
- [ ] Flash the 64-bit Raspberry Pi OS Bookworm image, mount its boot partition and run `scripts/provision/prepare_image.sh device1 /media/.../bootfs`. Boot the device on the management switch. Expected: within a minute `journalctl -u dnsmasq` on the edge server shows a lease for `device1` at its `nodes.json` management address, and `ssh pi@device1` works with the edge server's key only.
- [ ] `scripts/provision/build_mirror.sh apt device1` once (the reference device downloads the Debian packages), then `build_mirror.sh manifest`. Expected: `mirror/apt/Packages.gz` exists; `python -m src.provisioning.mirror verify` prints "mirror matches its manifest".
- [ ] `scripts/provision/provision_device.sh device1 --llm` with the edge server's internet access disabled. Expected: the device ends with both static addresses, `apt-get update` on it contacts only the mirror, `pip` shows `no-index`, `sudo -n tc qdisc show` works for the device user and `sudo -n ls` does not; `chronyc tracking` shows the edge server as the reference. Record the elapsed time (the thesis quotes about 25 minutes per device; the measured figure replaces the "from source" wording, J-15). Run the script a second time: expected "unchanged" for every file and no reinstallation.
- [ ] Flash a 32-bit image on a spare card and run `prepare_image.sh`. Expected: refused with "32-bit image". Boot a device from a 32-bit card and run `provision_device.sh`. Expected: refused with the kernel architecture named.
- [ ] Repeat the device steps for the remaining four devices; record the total time from the blank cards to the five provisioned devices (the thesis quotes two hours for the whole testbed).
- [ ] NTP offset (F-06): after ten minutes, on every device `chronyc tracking` and `chronyc sources -v`. Expected: the only source is the edge server and the system-time offset is below 1 ms (sub-millisecond, Chapter 3); record the values. Start a run: the preflight accepts all devices with `ntp_offset_threshold_ms: 1.0`.
- [ ] Access point (B-06): `iw dev <ap_interface> info` on the edge server shows the configured channel at 80 MHz; `iw dev wlan0 link` on a device shows the 5 GHz frequency and a VHT rate. Measure the single-device throughput with `src.network.throughput_measurement` (F-08 item above).

(Items are added with every work package.)

## Extensions of version 1.1 (HeadSplit through NNPM, generic adapters, catalogue)

- [ ] Runtimes dialog, HeadSplit, **Check the devices** before provisioning HeadSplit: every device red on checkout and environment. After **Install or update HeadSplit** on all devices: checkout green with the pinned commit, environment green with the `torch` and `transformers` versions.
- [ ] **Build the speculation profile** with `allow_upstream` on the edge server (internet reachable): the task completes, `artifacts/profile.json` appears on the controller and its check turns green; note the duration.
- [ ] `python -m src.models.weights fetch tinyllama --allow-upstream`, then **Distribute a checkpoint**: the checksums verify on every device and the checkpoint check turns green.
- [ ] A HeadSplit scenario created in the dialog (controller, four workers, mode `all`, 50 tokens) runs to completion; the summary page shows the three modes with totals of the order reported in Chapter 6, and the sweep table for an `added_rtt_ms` sweep created in the dialog.
- [ ] `model: mobilenetv2` and `model: efficientnetb0` on one device (`train_and_run_benchmark.sh both` with `--model-args '{"input_size": 224}'`): the stages build on the Pi within its memory, the benchmark completes, and `python -m src.models.inference verify` reports `ok` for every partition point.
- [ ] `model: hf_decoder` with `model_args: {checkpoint: <models dir>/Qwen__Qwen2.5-0.5B, seq_len: 32}` on one device: the stages run; note the per-layer times.
- [ ] A scenario with `input: {source: image_folder, path: <folder with class sub-directories>}` and `checks: {partition_equivalence: true}` on the devices: the verification files are collected and report top-1 accuracy.
