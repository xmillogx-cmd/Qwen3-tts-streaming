# Qwen3-TTS Streaming Engine (`fast_tts`)

True-streaming speech synthesis for **Qwen3-TTS-0.6B**, with **CUDA Graphs** acceleration for real-time playback. Installable as a PyPI package: `pip install qwen3-tts-streaming`.

## What's inside

| Component | Description |
|-----------|-------------|
| `fast_tts/` | The pip-installable package — streaming engine, audio player, CLI, and the CUDA-graph patch for stock `qwen-tts` |
| `fast_tts/engine.py` | `FastTTSv14` — true streaming via native `generate_custom_voice_streaming`, seamless chunk concatenation |
| `fast_tts/player.py` | `StreamingAudioPlayer` — callback-based live playback (sounddevice) with backpressure |
| `fast_tts/_patch/` | CUDA Graph acceleration (`PredictorGraph`, `TalkerGraph`) + streaming methods — our own implementation (provenance and attribution in the [License](#license) section); attached to stock `Qwen3TTSModel` at import time |
| `profile_v14.py` | Baseline profiler: load/capture cost, Mimi decode vs context, TTFA/RTF, token-cap usage |
| `run_native.py` / `run_faster.py` | Quick tests of the native and FasterQwen3TTS streaming APIs |
| `bench_sdpa.py` | SDPA attention performance benchmark |
| `debug_graphs.py` | CUDA graph timing diagnostics |
| `test_v14.py` | Full test suite — 10 sentences with playback verification |
| `*.bat` | Windows launchers (double-click or from terminal) |
| `docs/` | Technical documentation on the optimizations |
| `Qwen3-TTS/` | Official [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) source (Apache-2.0) — development reference only, not shipped; the package itself works against plain `pip install qwen-tts` |

## Results on RTX 5060 Ti (CC 12.0)

| Metric | Before optimization | After | Speedup |
|--------|--------------------|-------|---------|
| ms/step | ~248ms | **31ms** | **7.9x** |
| RTF | 0.336 (slower than realtime) | **2.67** (2.7× faster than realtime) | **8.0x** |
| TTFA (time to first audio) | ~40s | **~355ms** | **113x** |

## Key optimizations

### CUDA Graphs
- `PredictorGraph` — captures the full 15-step code predictor loop as a single CUDA graph (~26ms vs ~190ms)
- `TalkerGraph` — captures single-token talker decode (~12ms vs ~75ms)
- Uses `transformers.StaticCache` instead of DynamicCache for fixed-size KV buffers

### Streaming pipeline (v14)
- Native `generate_custom_voice_streaming()` — no manual token management
- Producer-consumer architecture: generation thread → queue → player
- Automatic long-text segmentation via `split_segments()`
- Seamless chunk concatenation (stateful generation preserves phase continuity)
- 0.3s preroll — audio starts in ~350ms

## Installation

### 1. Install the package

```bash
pip install qwen3-tts-streaming
```

This pulls `qwen-tts`, `transformers`, `accelerate`, `torchaudio`, `soundfile`, `sounddevice` and friends automatically. The streaming + CUDA-graph methods are attached to stock `Qwen3TTSModel` at import time — no patched or editable `qwen_tts` copy needed.

> **CUDA note:** for GPU use, make sure you have a PyTorch build with your CUDA version
> (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu126`).
> Plain PyPI wheels are CPU-only on some platforms — check with `torch.cuda.is_available()`.

### 2. Get the model weights

Download [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) from Hugging Face and point `--model` / `MODEL_PATH` at the local directory.

### Attention implementation

SDPA attention is used by default — it is compatible with CUDA graph capture.
**Flash Attention 2 is NOT compatible with CUDA graph capture** and will crash during warmup
(`CUDA error: operation not permitted when stream is capturing`).

## Quick start

### CLI

```bash
fast-tts --model /path/to/Qwen3-TTS-0.6B --text "Hello world"
# or via env var:
MODEL_PATH=/path/to/model fast-tts --text "Привет мир"
```

Options: `--speaker` (default `Sohee`), `--chunk-size {2,4,8}`, `--min-start-sec`, `--device <audio-device-index>` (interactive menu if omitted).

### Python API

```python
from fast_tts import FastTTSv14

tts = FastTTSv14(
    model_path="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    speaker="Sohee",
)

# Streaming generation with live playback (optionally saves a WAV)
tts.generate_and_play("Hello! This is a streaming TTS test.", save_wav="out.wav")
```

## Generation architecture (v14)

```
Text → split_segments() → generate_custom_voice_streaming() × N segments
                                      ↓
                    Talker prefill + Predictor × 15 (CUDA graphs)
                                      ↓
                    Codec tokens [cb0..cb14] → chunked_decode()
                                      ↓
                    Producer thread → queue → StreamingAudioPlayer
                                      ↓
                    Concatenate chunks (stateful) → Live playback @24kHz
```

## Dependencies

- Python 3.10+
- PyTorch ≥ 2.5.1 with CUDA (for GPU use)
- `qwen-tts>=0.1.1`
- `transformers>=4.57,<5`
- `accelerate`, `torchaudio`, `soundfile`, `sounddevice`, `numpy`

## Repository layout (dev repo)

```
qwen3-tts-streaming/
├── fast_tts/                     # The pip package (built by hatchling, see pyproject.toml)
│   ├── __init__.py               # Exports + applies the qwen_tts patch on import
│   ├── engine.py                 # FastTTSv14 — true streaming, seamless chunk concatenation
│   ├── player.py                 # StreamingAudioPlayer — callback-based live playback
│   ├── cli.py                    # `fast-tts` entry point + dev test suite
│   └── _patch/                   # CUDA-graph modules + streaming methods (our own implementation — see License section)
├── fast_tts_v14.py               # Backward-compat shim → re-exports from the package
├── run_native.py / run_faster.py # Quick API tests
├── bench_sdpa.py                 # SDPA attention benchmark
├── debug_graphs.py               # CUDA graph timing diagnostics
├── test_v14.py                   # Full test suite — 10 sentences + playback
├── profile_v14.py                # Baseline profiler: load/capture, Mimi vs ctx, TTFA/RTF, token caps
├── pyproject.toml / LICENSE      # Packaging metadata (hatchling) + MIT license
├── *.bat                         # Windows launchers (double-click friendly)
├── docs/
│   ├── cuda_graphs_optimization.md  # Detailed CUDA Graphs breakdown
│   └── qwen3-tts-implementation.md  # Architecture and version history
└── Qwen3-TTS/                    # Official Qwen3-TTS source — dev-time reference only, not shipped
```

## License

- This package (`fast_tts`) — **MIT** (see `LICENSE`). Provenance of the `_patch/` modules: the CUDA-graph capture strategy follows the approach published in [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (**MIT**, © andimarafioti); our implementation is written independently against that design. The talker input construction follows the official Qwen3-TTS algorithm (**Apache-2.0**), restructured with hoisted constants and per-sample helper dispatch.
- Base model Qwen3-TTS and the `qwen-tts` dependency — **Apache 2.0**, [Alibaba Group / QwenLM](https://github.com/QwenLM/Qwen3-TTS).
