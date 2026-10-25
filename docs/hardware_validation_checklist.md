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

(Items are added with every work package.)
