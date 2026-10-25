# Contributing

Thank you for your interest in the Distributed Inference Testbed. This document explains how to set up a development environment, which checks a change must pass, and the conventions the code base follows.

## Development set-up

Any Linux machine with Python 3.11; no testbed hardware is needed.

```bash
git clone https://github.com/Dimitrios-Kafetzis/Distributed-Inference-Testbed.git
cd Distributed-Inference-Testbed
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python --require-hashes -r requirements/dev.lock
```

`requirements/dev.lock` contains the edge-server stack plus pytest, ruff, mypy and shellcheck. The real TensorFlow and PyTorch models (the `slow` tests) need `requirements/edge_server.lock` as well; see [docs/installation.md](docs/installation.md).

## Checks that must pass

Run these before opening a pull request. The continuous-integration workflow runs the same commands.

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/shellcheck -x $(git ls-files '*.sh')
.venv/bin/pytest -m "not slow"
.venv/bin/pytest -m slow            # when a change touches the model adapters or the profiler
```

The documentation site is built with MkDocs from `docs/` (`mkdocs.yml`; `requirements/docs.lock` holds the tooling): `uv pip install --python .venv/bin/python --require-hashes -r requirements/docs.lock && .venv/bin/mkdocs serve` previews it, and `mkdocs build --strict` must pass (it runs in CI). The NNPM screenshots of the tutorial are regenerated with `python scripts/docs/nnpm_screenshots.py --server-python .venv/bin/python` (Playwright, from the same lock, after `python -m playwright install chromium`).

A change to the provisioning scripts or templates usually changes the dry-run command plans; regenerate the golden files with `UPDATE_GOLDEN=1 .venv/bin/pytest tests/test_provisioning.py` and review the diff before committing it.

## Conventions

- **The thesis is the specification.** File, script and output names cited in the thesis (for example `src/config.py`, `gui/app.py`, `run_full_benchmarking.sh`, `total_averages_of_benchmark_results.json`) must keep their names; `tests/test_thesis_names.py` enforces this. The software role string of the edge server is `Base Station`.
- **Everything runs without hardware.** Every external effect (commands, SSH, `/sys`, clocks, sensors) goes through an injectable runner, transport, clock or rooted file system, so that it can be faked or dry-run in tests. A new device-side tool needs a `--dry-run` path and a unit test with a fake runner.
- **Daemonless.** Nothing of the testbed stays resident on a device. A process that runs on a device is started over SSH as a tagged detached process for the duration of an experiment and stopped at its end (`src/common/process_tags.py`).
- **Undo everything.** A change made on a device during a run (shaping, frequency, cgroup, load, process) is recorded in the run journal before the next step, with its undo action, so that teardown can revert it, also on failure.
- **Small modules, strict types.** Python 3.11, `mypy --strict`, ruff with a line length of 100. Prefer plain functions and dataclasses; keep a module to one responsibility.
- **Pinned dependencies.** Direct dependencies live in `requirements/*.in`; the lock files are regenerated with `scripts/lock_requirements.sh` and the mirror rebuilt afterwards. Do not edit a lock file by hand.
- **No secrets, no personal data.** Secrets come only from environment variables named in `config.yaml`. Addresses and home directories do not appear in code (`tests/test_no_hardcoded_literals.py`). Sample data shipped in the repository is synthetic and labelled as such.
- **Documentation and comments** are in British English, in plain wording, and are updated in the same change as the code. Every relative link must resolve (`tests/test_docs_links.py`). Give the full name first, then the acronym in parentheses.

## Commits and pull requests

- One logical change per commit, with a short imperative subject line that states what the change does.
- A pull request describes the motivation, the change, how it was tested, and whether it was verified on the physical testbed. Changes that can only be verified on the hardware should say so and name the item of [docs/hardware_validation_checklist.md](docs/hardware_validation_checklist.md) that covers them.
- Keep the change focused; refactoring that is not needed for the change belongs in a separate pull request.

## Reporting problems

Use the issue templates: a bug report should include the scenario, the run's `metadata.json` and `summary.json` where applicable, and the relevant part of `logs/merged.log`. Please remove addresses and host names you do not want to publish.
