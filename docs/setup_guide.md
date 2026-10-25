# Setup guide

How to assemble the testbed described in Chapter 3 of the thesis and bring it to the point where NNPM can run an experiment: hardware, networks, operating systems, time synchronisation, the package mirror and provisioning. The software installation itself is in [installation.md](installation.md); the step-by-step provisioning commands are in [`../scripts/provision/README.md`](../scripts/provision/README.md).

## Hardware

| Node | Hardware |
|---|---|
| Edge devices (five) | Raspberry Pi 4 Model B: Broadcom BCM2711, four Cortex-A72 cores at 1.8 GHz, LPDDR4-3200 memory (the thesis topology figure shows two boards with 8 GB and three with 4 GB), 32 GB microSD card, on-board Cypress CYW43455 wireless chipset (one antenna chain, IEEE 802.11ac, 433 Mbit/s physical rate), Gigabit Ethernet. No accelerator. |
| Edge server (one) | Intel Core i7-9700K (8 cores, 3.6 GHz base, 4.9 GHz turbo), 32 GB DDR4-3200, 512 GB NVMe SSD, Gigabit Ethernet, a USB IEEE 802.11ac adapter that hosts the access point. No accelerator. |
| Power sensors (optional) | one INA260 current-and-power sensor per device on the I2C bus. |
| Network | a small Ethernet switch for the management network; Ethernet cables. |

The RAM split between the devices (which boards have 8 GB) is to be confirmed by the author; `nodes.example.json` carries five devices and the run metadata records each node's real CPU model, core count, maximum frequency and memory on every run, so the facts are checked automatically.

Storage: the microSD card is the default; its read throughput (about 80 to 100 MB/s) can bottleneck model loading, which is why profiling records the model-load time separately from the inference time. A USB 3.0 SSD is an optional replacement.

### Bill of materials (indicative)

Prices are indicative, from 2024 to 2026 price lists, and vary by country and supplier.

| Item | Quantity | Indicative unit price |
|---|---|---|
| Raspberry Pi 4 Model B, 8 GB | 2 | 85 EUR |
| Raspberry Pi 4 Model B, 4 GB | 3 | 60 EUR |
| 32 GB microSD card (A1 class) | 5 | 8 EUR |
| USB-C power supply, 5 V 3 A | 5 | 10 EUR |
| Open rack or stackable cases, heat sinks | 1 set | 25 EUR |
| Ethernet switch, 8 ports, unmanaged | 1 | 20 EUR |
| Ethernet cables | 6 | 3 EUR |
| USB IEEE 802.11ac adapter for the access point (a chipset with access-point mode support in Linux) | 1 | 30 EUR |
| INA260 breakout board (optional) | 5 | 12 EUR |

The five boards with cards and power supplies come to roughly 450 to 500 EUR; the rack, the switch, the cables and the WiFi adapter add about 100 EUR, the power sensors another 60 EUR. The statement in the thesis that an identical set-up can be assembled for under 500 EUR refers to the edge devices and their accessories; the edge server is not included.

## Physical layout

The devices sit on an open rack 2 to 3 m from the edge server, in line of sight of the access-point antenna. Each device has two network connections:

- the **data plane**: the on-board WiFi joined to the edge server's access point; this is where inference traffic flows, where latency is measured and where tc-netem shaping is applied (`network_interface`, `wlan0`);
- the **management network**: an Ethernet cable to the switch, which is connected to the edge server's wired interface; SSH, result collection, NTP and the package mirror use it (`management_interface`, `eth0`), so management traffic does not cross the wireless data plane during experiments.

`nodes.json` holds both addresses of every node (`ip` on the data plane, `management_ip` on the management network); `config.yaml` holds the two prefixes (`base_ip_prefix`, `management_ip_prefix`) and interface names. A single-network set-up (no switch) is possible: leave `management_ip_prefix` as `null` and omit `management_ip`; every subsystem then uses the wireless address.

## Addressing

The example files use `192.168.0.0/24` for the data plane (edge server `.1`, devices `.11` to `.15`) and `10.0.0.0/24` for the management network (same last octets). The edge server's dnsmasq serves both networks: a pool (`.100` to `.199`) for devices that are not provisioned yet and for a laptop plugged into the switch, and a reservation per device name on the management network, so a freshly flashed device is reachable under its `nodes.json` address before it is provisioned. Provisioning then installs static addresses on both interfaces. Neither network hands out a default route or an upstream resolver: both are isolated, which keeps the measured delays free of foreign traffic.

## Operating systems

- Devices: Raspberry Pi OS **64-bit** (Bookworm), which ships Python 3.11. The 64-bit variant is required: the LLM experiments exceed the 3 GB per-process limit of 32-bit address spaces, and `provision_device.sh` refuses a device whose kernel is not `aarch64` (`prepare_image.sh` refuses a 32-bit image). The thesis table names Bullseye; the release targets Bookworm because of Python 3.11.
- Edge server: Ubuntu 22.04 LTS with Python 3.11 (`python3.11`, `python3.11-venv`) and NetworkManager (the Desktop default; on Ubuntu Server the management address is configured with netplan instead of the rendered NetworkManager profile).

## Access point

`provision_edge_server.sh` renders `/etc/hostapd/hostapd.conf` from `config.yaml`: 5 GHz band (`hw_mode=a`), IEEE 802.11n/ac with an 80 MHz channel whose centre is derived from `ap_channel`, WPA2-PSK with the passphrase taken from the environment variable named by `ap_passphrase_env` (never written to `config.yaml`), the regulatory domain from `ap_country`. NetworkManager is told to leave the adapter alone and a small systemd unit gives it the edge server's data-plane address before hostapd starts. The achievable application-layer throughput per device is measured with `src/network/throughput_measurement.py` (the thesis quotes 80 to 120 Mbit/s for a single device; record the figure on your hardware, see the checklist).

## Time synchronisation

The edge server runs chrony as the NTP server for both networks (with its own clock as a stratum-8 fallback when it has no internet access); each device runs chrony as a client with the edge server as its only source, polling every 4 to 16 s. Timestamps on different nodes are correlated through the NTP-synchronised wall clock captured at the start signal of every run; the preflight stage refuses a run when a device's offset exceeds `ntp_offset_threshold_ms` (1 ms by default). On the wired management network the offset is sub-millisecond; measure it with `chronyc tracking` (checklist).

## Power sensors (optional)

An INA260 measures the current through its internal shunt. Insert it in the 5 V supply line of each device (the supply's 5 V to `IN+`, `IN-` to the board's 5 V input; a common ground), power the sensor from the board's 3.3 V pin and connect `SDA` and `SCL` to GPIO 2 and GPIO 3. Enable I2C on the device (`raspi-config` or `dtparam=i2c_arm=on` in `/boot/firmware/config.txt`); the default I2C address is `0x40`. Install the optional packages of `requirements/power.in` on the device and set `monitoring.power: true` in the scenario. The sensor reports current in steps of 1.25 mA, bus voltage in steps of 1.25 mV and power in steps of 10 mW.

## Package mirror and reproducibility

All Python and Debian packages are installed from a versioned local mirror on the edge server (`mirror_dir`, served by `testbed-mirror.service`): wheels per lock file for x86_64 and aarch64, Debian packages fetched once through a reference device, a bare clone of HeadSplit at the pinned commit, and `MANIFEST.json` with the SHA-256 of every file. `python -m src.provisioning.mirror verify` detects any change. Re-running the provisioning scripts later therefore installs an identical software stack. The random seeds of every stochastic library are set at the start of every experiment to the scenario's `seed`, which is stored in the run metadata.

## Provisioning in order

1. Edge server: `config.yaml` and `nodes.json` from the examples; export the passphrase; `scripts/provision/provision_edge_server.sh --allow-upstream`.
2. `scripts/provision/build_mirror.sh wheels` (and `headsplit`).
3. Per device: flash the 64-bit image, `scripts/provision/prepare_image.sh <node id> <boot partition>`, boot on the switch.
4. Once: `scripts/provision/build_mirror.sh apt <node id>` and `build_mirror.sh manifest`.
5. Per device: `scripts/provision/provision_device.sh <node id> [--llm]`.

Every script accepts `--dry-run` and prints the commands it would run; every step is idempotent. The thesis quotes about 25 minutes per device and two hours for the whole testbed; the scripts append their elapsed time to `results/provisioning/timing.log` so the figures can be measured on your hardware.

## Model weights

Weights are prepared once with a fixed seed (`train_and_run_benchmark.sh train`, or copied from a checkpoint for the LLMs) and distributed to the devices by `provision_device.sh` from `bs_models_dir`, with a `SHA256SUMS` file checked on arrival. No training happens on the devices.

## Checking the set-up

Open `http://<edge server management address>:<nnpm_port>/`: every node should show a green marker within a few seconds. Run `scripts/base_station/run_full_benchmarking.sh examples/scenarios/baseline.yaml` and check `results/latest/` for the three output files. The complete list of what to verify on the hardware, with expected values, is [hardware_validation_checklist.md](hardware_validation_checklist.md).
