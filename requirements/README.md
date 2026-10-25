# Pinned dependencies

The `.in` files list direct dependencies with the minimum versions documented in the thesis software table (Appendix B). The `.lock` files pin every package to an exact version with hashes, per role and platform, and are what gets installed:

| Lock file | Role | Platform |
|---|---|---|
| `edge_server.lock` | edge server (NNPM, SSH dispatch, aggregation, optimiser, model profiling) | x86_64, Python 3.11 |
| `edge_device.lock` | edge device, DNN experiments (TensorFlow) | aarch64, Python 3.11 |
| `edge_device_llm.lock` | edge device, LLM experiments (PyTorch, Transformers) | aarch64, Python 3.11 |
| `dev.lock` | development and continuous integration | x86_64, Python 3.11 |

Regenerate with `scripts/lock_requirements.sh` (uses `uv pip compile --generate-hashes`). Install with `uv pip install --require-hashes -r requirements/<role>.lock` or `pip install --require-hashes -r ...`.

The optional INA260 power-monitor packages (`power.in`) depend on the Raspberry Pi platform (`adafruit-blinka`) and are installed on the devices only; they are listed without a lock file because their platform wheels cannot be resolved off the device.
