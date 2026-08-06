# Qwen-TTS Project — Developer Notes

## Environment

**Always use the `.conda` environment for GPU/CUDA work:**

```powershell
# Python with CUDA (PyTorch 2.13.0+cu132)
G:\qwen-tts\.conda\python.exe script.py

# Or activate the Scripts path
$env:PATH = "G:\qwen-tts\.conda\Scripts;G:\qwen-tts\.conda;$env:PATH"
python script.py
```

**System Python (NO CUDA!)**: `C:\Python314\python.exe` — PyTorch CPU-only, do NOT use for TTS.

### Verified Setup
- GPU: NVIDIA GeForce RTX 5060 Ti
- CUDA: 13.2
- Compute Capability: 12.0
- PyTorch: 2.13.0+cu132 (in `.conda`)
- Attention: SDPA (flash-attn is NOT compatible with CUDA graphs)

### Important: Flash Attention 2 vs SDPA
Flash Attention 2 crashes during CUDA graph capture with:
```
CUDA error: operation not permitted when stream is capturing
```
Always use `attn_implementation='sdpa'` for streaming TTS with CUDA graphs.

## Project Structure

| File | Description |
|------|-------------|
| `streaming_tts_v6.py` | Producer-consumer pipeline (fixed underruns) |
| `streaming_tts_v7.py` | Parallel generation via ThreadPoolExecutor |
| `streaming_tts_v8.py` | torch.compile + chunked decode |
| `fast_tts_final.py` | StoppingCriteria patch + auto max_new_tokens |
| `Qwen3-TTS/` | Official Qwen3-TTS source (qwen_tts package) |
| `faster-qwen3-tts/` | CUDA Graphs optimization repo (cloned for reference) |

## Key Architecture Notes

### Qwen3TTSModel Access Pattern
```python
# Correct: model.model.talker (NOT model.talker)
self.model.model.talker  # Talker with 28 layers
self.model.model.speech_tokenizer  # Codec tokenizer/decoder
```

### Generation Flow
1. `model._build_assistant_text(text)` → builds `<|assistant|>\ntext\n<|end|>` format
2. `model._tokenize_texts([...])` → tokenizes with processor
3. `model.model.generate(...)` → generates codec tokens via talker + code_predictor
4. `decoder.chunked_decode(...)` → decodes codec tokens to waveform (24kHz)

### CUDA Graph Optimization (from faster-qwen3-tts)
- Uses `transformers.StaticCache` instead of DynamicCache for fixed-size KV buffers
- Captures full 15-step predictor loop as single CUDA graph (~26ms vs 190ms)
- Captures single-token talker decode as CUDA graph (~12ms vs 75ms)
- Requires PyTorch ≥ 2.5.1 for stable capture

## Memory & Reference

See `C:\Users\Admin\.qwen\projects\g--qwen-tts\memory\` for cross-session knowledge:
- `optimization-results.md` — Previous optimization results (v6-v8)
- `cuda-graph-warmup.md` — CUDA graph warmup behavior in generate_custom_voice_streaming
