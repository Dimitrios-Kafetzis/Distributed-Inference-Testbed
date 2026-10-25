# Installation

Three roles, three lock files. Every dependency is pinned with a hash in `requirements/*.lock` (generated with `scripts/lock_requirements.sh`); installing with `--require-hashes` guarantees the same stack on every machine. The direct dependencies and their minimum versions are in the `.in` files and follow the software table of Appendix B.

| Component | Appendix B minimum | Pinned in the lock files |
|---|---|---|
| Operating system | Raspberry Pi OS 64-bit (devices), Ubuntu 22.04 LTS (edge server) | Bookworm 64-bit on the devices (Python 3.11), Ubuntu 22.04 LTS |
| Python | 3.11 | 3.11 |
| TensorFlow (DNN, Chapter 4) | ≥ 2.8.0, CPU | 2.15.1 |
| PyTorch (LLM, Chapters 5 and 6) | 2.1.2, CPU | 2.1.2 (+cpu on x86_64) |
| Transformers | 4.36 | 4.36.2 |
| scikit-learn | ≥ 1.0.0 | 1.9.1 |
| Paramiko (SSH) | ≥ 2.11, edge server only | 5.1.0 |
| Flask (web application) | ≥ 2.2.0, edge server only | 3.1.3 |
| NTP | client (devices), server (edge server) | chrony, configured by provisioning |
| psutil, PyYAML, jsonschema, NumPy | | 7.2.2, 6.0.3, 4.26.0, 1.26.4 |

## Edge server

Ubuntu 22.04 LTS, x86_64. The provisioning script does all of this (and the network services); by hand:

```bash
sudo apt-get install -y python3.11 python3.11-venv git rsync iperf3
git clone https://github.com/Dimitrios-Kafetzis/Distributed-Inference-Testbed.git
cd Distributed-Inference-Testbed
python3.11 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements/edge_server.lock \
    --extra-index-url https://download.pytorch.org/whl/cpu   # the CPU build of PyTorch lives there
cp config.example.yaml config.yaml      # then edit
cp nodes.example.json nodes.json        # then edit, or manage through NNPM
.venv/bin/python -m src.config          # shows the effective configuration
.venv/bin/python -m gui.app             # NNPM on nnpm_host:nnpm_port
```

`requirements/edge_server.lock` contains TensorFlow and PyTorch (CPU builds): the edge server profiles its own capacity and runs the partitioned-inference service, so it needs the same frameworks as the devices. The x86_64 CPU build of PyTorch (`torch==2.1.2+cpu`) is published on the PyTorch index, not on PyPI, hence the `--extra-index-url` (with `uv`, add `--index-strategy unsafe-best-match`). With a package mirror in place, replace it with `--no-index --find-links mirror/wheels/x86_64`.

## Edge devices

Raspberry Pi OS Bookworm 64-bit (`uname -m` must print `aarch64`). `scripts/provision/provision_device.sh <node id>` installs everything from the mirror; by hand, on the device:

```bash
sudo apt-get install -y python3 python3-venv python3-pip chrony iproute2 iperf3 rsync git
mkdir -p ~/testbed && cd ~/testbed
git clone https://github.com/Dimitrios-Kafetzis/Distributed-Inference-Testbed.git code
python3 -m venv venv
venv/bin/pip install --require-hashes -r code/requirements/edge_device.lock       # TensorFlow
venv/bin/pip install --require-hashes -r code/requirements/edge_device_llm.lock   # PyTorch and Transformers (optional)
```

The directory layout the edge server expects is given by `ed_root` in `config.yaml` (default `/home/pi/testbed`): `code/`, `venv/`, `models/`, `data/`, `results/`, `run/`. The devices need no copy of `config.yaml`: the edge server passes every setting as `TESTBED_*` environment variables and command-line arguments.

Optional, for the INA260 power monitor: `venv/bin/pip install -r code/requirements/power.in` (the Adafruit Blinka and INA260 packages are platform-specific and therefore not in the lock files) and I2C enabled on the board.

Privileges: `tc` and the heterogeneity module (cpufreq and cgroup writes) run with `sudo -n`; provisioning installs a sudoers rule that allows exactly those two commands for the device user. Nothing else on a device needs root.

## Development machine

Any Linux machine with Python 3.11 (this is also what the continuous-integration workflow does):

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python --require-hashes -r requirements/dev.lock
.venv/bin/pytest -m "not slow"            # fast suite: fakes, dry runs, local mode
.venv/bin/pytest -m slow                  # real TensorFlow and PyTorch models on the CPU
.venv/bin/ruff check . && .venv/bin/mypy && .venv/bin/shellcheck -x $(git ls-files '*.sh')
```

`requirements/dev.lock` includes the edge-server stack plus pytest, ruff, mypy and shellcheck. The whole pipeline runs without hardware in local mode:

```bash
.venv/bin/python -m src.orchestration.experiment run --scenario examples/scenarios/local_demo.yaml --local
.venv/bin/python -m gui.app --local        # NNPM with simulated nodes
```

## Updating the pins

Edit the `.in` files, then `scripts/lock_requirements.sh` (needs `uv`), which resolves each lock file for its platform (`x86_64-manylinux_2_28` for the edge server and development, `aarch64-manylinux_2_28` for the devices) with hashes. Rebuild the mirror afterwards (`scripts/provision/build_mirror.sh wheels` and `manifest`).
