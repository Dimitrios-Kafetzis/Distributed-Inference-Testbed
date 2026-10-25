## Motivation

<!-- What problem does this change solve? Link the issue if there is one. -->

## Change

<!-- What was changed, and why this way. -->

## Verification

- [ ] `ruff check . && ruff format --check .`
- [ ] `mypy`
- [ ] `shellcheck -x $(git ls-files '*.sh')`
- [ ] `pytest -m "not slow"`
- [ ] `pytest -m slow` (model adapters or profiler changed)
- [ ] Golden files regenerated and reviewed (provisioning scripts or templates changed)
- [ ] Verified on the physical testbed, or the relevant item of `docs/hardware_validation_checklist.md` is named below

## Documentation

- [ ] Documentation and `CHANGELOG.md` updated in this pull request
