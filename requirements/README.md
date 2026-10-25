# Pinned dependencies

The `.in` files list direct dependencies with the minimum versions documented in the thesis software table (Appendix B). The `.lock` files pin every package to an exact version with hashes, per role and platform, and are what gets installed:

| Lock file | Role | Platform |
|---|---|---|
| `edge_server.lock` | edge server (NNPM, SSH dispatch, aggregation, optimiser, model profiling) | x86_64, Python 3.11 |
| `edge_device.lock` | edge device, DNN experiments (TensorFlow) | aarch64, Python 3.11 |
| `edge_device_llm.lock` | edge device, LLM experiments (PyTorch, Transformers) | aarch64, Python 3.11 |
| `dev.lock` | development and continuous integration | x86_64, Python 3.11 |
| `docs.lock` | the documentation site (MkDocs) and the screenshot tooling (Playwright) | x86_64, Python 3.11 |

Regenerate with `scripts/lock_requirements.sh` (uses `uv pip compile --generate-hashes`). Install with `uv pip install --require-hashes -r requirements/<role>.lock` or `pip install --require-hashes -r ...`. The edge-server lock pins the x86_64 CPU build of PyTorch (`torch==2.1.2+cpu`), which is on the PyTorch index: add `--extra-index-url https://download.pytorch.org/whl/cpu` (and, with `uv`, `--index-strategy unsafe-best-match`), or install from the local mirror with `--no-index --find-links`.

Transformers is pinned to 4.46.3 rather than the 4.36 of the thesis software table because the Qwen2 (4.37), Llama 3.2 (4.43) and HeadSplit (4.40) adapters need it; the testbed's code accepts both the 4.36 and the 4.38+ interfaces.

The optional INA260 power-monitor packages (`power.in`) depend on the Raspberry Pi platform (`adafruit-blinka`) and are installed on the devices only; they are listed without a lock file because their platform wheels cannot be resolved off the device.
