## Description

<!-- What does this PR change and why? Link the issue(s) it fixes. -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would change existing behavior)
- [ ] Docs / packaging only

## Testing

- [ ] `python -m compileall fast_tts` passes
- [ ] GPU test run against **plain PyPI** `qwen-tts` (`test_v14_pypi.py`) is green — see CONTRIBUTING.md for the PYTHONPATH shadow trick if your env has an editable/patched qwen-tts install
- [ ] No new hard dependencies added (or they are declared in `pyproject.toml`)

## Packaging checklist

- [ ] New root-level files added to `[tool.hatch.build.targets.sdist]` exclude list in `pyproject.toml` (if any)
- [ ] If code changed: version bumped in `pyproject.toml` and a release is planned (PyPI versions are immutable)
