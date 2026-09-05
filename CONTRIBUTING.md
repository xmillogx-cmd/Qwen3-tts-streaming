# Contributing to qwen3-tts-streaming

Thanks for your interest in contributing! This repository contains the
`fast_tts` package (published on PyPI as `qwen3-tts-streaming`) plus dev/test
scripts. Please read this guide before opening a PR.

## Environment setup

- Python ≥ 3.10; an NVIDIA GPU with CUDA is strongly recommended (CPU works but is slow).
- Install a CUDA build of PyTorch matching your driver/CUDA toolkit, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Plain PyPI wheels are CPU-only on some platforms — verify with `torch.cuda.is_available()`.

- Install the package in editable mode for development:

```bash
pip install -e .
```

This pulls `qwen-tts>=0.1.1,<0.2`, `transformers>=4.57,<5`, and friends. The
streaming + CUDA-graph methods are attached to stock `Qwen3TTSModel` at import
time — you do **not** need a patched or editable copy of `qwen-tts`.

> **Attention implementation:** use SDPA (the default). Flash Attention 2 is
> incompatible with CUDA graph capture and crashes during warmup.

## Testing

- `test_v14.py` / `test_v14.bat` — full suite (10 sentences, playback verification) against the dev tree in this repo.
- `test_v14_pypi.py` / `test_v14_pypi.bat` — the same suite run against **plain PyPI** `qwen-tts`. If your environment has an editable/patched `qwen-tts` install, the patch only attaches methods that are missing, so a green run would not exercise this repo's code. Extract a stock wheel into a temp dir and prepend it to `PYTHONPATH`:

```bash
pip install --target .tmp_pypi_check "qwen-tts==0.1.1" --no-deps
set PYTHONPATH=<abs path>\.tmp_pypi_check   # Windows; use export on Linux/macOS
python test_v14_pypi.py
```

- `profile_v14.py` — baseline profiler (TTFA/RTF, load/capture cost).
- `bench_sdpa.py`, `debug_graphs.py` — attention benchmark and CUDA-graph diagnostics.

Model weights: [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice).

## Development workflow

1. Fork the repo and create a topic branch from `master`.
2. Keep changes minimal and focused; follow existing code style (the `_patch/` modules are deliberately self-contained).
3. Verify with `python -m compileall fast_tts` at minimum, plus the relevant GPU test above.
4. Commit messages: concise imperative sentences describing *what* changed and *why*.

### Packaging rules (important)

- The sdist must contain **only** what is needed to build the package. If you add a new file at the repository root or in `docs/`, add it to the `[tool.hatch.build.targets.sdist]` exclude list in `pyproject.toml` — otherwise it will ship inside the PyPI artifact.
- Never commit tokens, keys, or other secrets (see [SECURITY.md](SECURITY.md)).

## Releasing a new version

1. Bump `version` in `pyproject.toml` (PyPI versions are immutable — an already-published version can never be re-uploaded).
2. Build: `python -m build` (or `hatch build`).
3. Check: `twine check dist/*`.
4. Upload with a PyPI API token: `twine upload dist/*`.

## License

Contributions are licensed under MIT, the same as the rest of this project. By submitting a PR you agree to license your contribution under MIT.
