# Distributed Inference Testbed

An experimental testbed for partitioned Artificial Intelligence (AI) inference across resource-constrained edge devices: a cluster of Raspberry Pi 4 single-board computers connected over a dedicated wireless network to an edge server that hosts the access point. The Neural Network Architecture Partitioning (NNPM) manager, a web application served from the edge server, configures the testbed and orchestrates experiments: it dispatches commands to the devices over SSH, profiles the models layer by layer under controlled compute and network load, collects the measurements, and computes partition assignments.

The testbed supports Deep Neural Networks (DNNs; AlexNet, VGG-16, ResNet-50) in TensorFlow and Transformer-based Large Language Models (LLMs; GPT-2, TinyLlama-1.1B) in PyTorch. It requires no daemon on the devices beyond an SSH server: every device-side process is started for the duration of an experiment and stopped at its end.

This repository is under construction; the sections below are filled in as the components land.

## Status

| Component | State |
|---|---|
| Configuration (`config.yaml`, `nodes.json`, `src/config.py`) | done |
| Measurement records and post-processing (`src/benchmarking/`, `src/network/node_statistics.py`) | done |
| Device-side tools (`src/testbed/`, `src/network/`) | planned |
| SSH dispatch and collection (`src/network/ssh_executor.py`) | planned |
| Models and profiling (`src/models/`) | planned |
| Partitioned inference service (`src/inference/`) | planned |
| Partition optimisation (`src/partitioning/`) | planned |
| Orchestration and local mode (`src/orchestration/`) | planned |
| NNPM web application (`gui/`) | planned |
| Provisioning (`scripts/provision/`) | planned |

## Quick start (development, no hardware)

```bash
git clone https://github.com/Dimitrios-Kafetzis/Distributed-Inference-Testbed.git
cd Distributed-Inference-Testbed
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python --require-hashes -r requirements/dev.lock
cp config.example.yaml config.yaml
cp nodes.example.json nodes.json
.venv/bin/pytest
.venv/bin/python -m src.config      # show the effective configuration
```

## Configuration

`config.yaml` (from `config.example.yaml`) holds the SSH settings, the network prefixes and interfaces, the paths on the edge server and on the devices, and the NNPM settings. Every key can be overridden by an environment variable `TESTBED_<KEY>`; the SSH password, when used at all, comes only from the environment variable named by `ssh_password_env`. `nodes.json` (from `nodes.example.json`) holds the topology: one entry per node with its identifier, role (`Base Station` is the name of the edge server in the software, `End Device` a Raspberry Pi), addresses, map coordinates and the statistics of the last benchmarking run. Both files live at the repository root and are validated against the schemas in `docs/schemas/`.

## Provenance

This testbed continues the author's earlier repository [DNN-Partitioning-and-Offloading-Testbed](https://github.com/Dimitrios-Kafetzis/DNN-Partitioning-and-Offloading-Testbed), first presented as a demonstration at ACM MobiHoc 2024. The code was rewritten so that it implements the testbed described in the author's PhD thesis (Athens University of Economics and Business, 2026), Chapter 3 and Appendix B.

## Licence

MIT. See `LICENSE`.
