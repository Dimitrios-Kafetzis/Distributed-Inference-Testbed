# Provisioning scripts

Bash scripts that take the testbed from blank microSD cards to provisioned nodes. They read `config.yaml` and `nodes.json` through `src/provisioning/render.py`; no address, user name or path is written in a script. Every script accepts `--dry-run`, which prints the state-changing commands instead of running them (read-only queries such as `uname -m` over SSH still run), and every step is idempotent: a file is installed only when its content differs, an environment is created only when absent. Elapsed times are appended to `results/provisioning/timing.log`.

Order of operations on a new testbed:

1. On the edge server: copy `config.example.yaml` to `config.yaml` and `nodes.example.json` to `nodes.json`, edit both, export the access-point passphrase in the variable named by `ap_passphrase_env`, then `scripts/provision/provision_edge_server.sh --allow-upstream`. This installs the packages, the access point (`hostapd`), DHCP and DNS for both networks (`dnsmasq`), the NTP server (`chrony`), the static management address, the mirror service and the NNPM service, and creates the Python environment from `requirements/edge_server.lock`.
2. `scripts/provision/build_mirror.sh wheels` (and `headsplit` when `headsplit_commit` is set): the wheels of every lock file go to `mirror/wheels/<arch>/`.
3. For each device: flash the 64-bit Raspberry Pi OS Bookworm image, mount its boot partition and run `scripts/provision/prepare_image.sh <node id> <boot mountpoint>`. Boot the device on the management switch; it registers under its `nodes.json` management address.
4. Once, with the first device booted: `scripts/provision/build_mirror.sh apt <node id>` fetches the Debian packages through that device into `mirror/apt/`, then `build_mirror.sh manifest` writes `mirror/MANIFEST.json`.
5. For each device: `scripts/provision/provision_device.sh <node id> [--llm]`. From then on the device installs only from the mirror.

| Script | Runs on | Purpose |
|---|---|---|
| `prepare_image.sh` | edge server, card mounted | first-boot `custom.toml`: hostname, user and SSH key, WiFi, time zone |
| `provision_edge_server.sh` | edge server | packages, SSH key, access point, DHCP/DNS, NTP server, mirror and NNPM services, Python environment |
| `build_mirror.sh` | edge server | the only script that contacts upstream sources; builds the versioned mirror |
| `provision_device.sh` | edge server | provisions one device over SSH; calls `device_setup.sh` on it |
| `device_setup.sh` | device | the device-side steps: hostname, apt, chrony, sudoers, pip, venv, headsplit, verify, network |
| `lib.sh` | both | shared helpers (`run`, `remote`, `install_file`, `load_variables`, `timing_log`) |

Templates are in `templates/` and rendered by `python -m src.provisioning.render`; the rendered files for the example configuration are kept as golden files under `tests/golden/provision/`. The mirror layout and manifest are described in `src/provisioning/mirror.py`.
