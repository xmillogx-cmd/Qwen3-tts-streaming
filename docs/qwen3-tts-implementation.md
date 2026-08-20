# Qwen3-TTS: Streaming Generation with CUDA Graphs Acceleration

Complete technical breakdown of our streaming TTS implementation based on Qwen3-TTS.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Qwen3-TTS Architecture](#2-qwen3-tts-architecture)
3. [Standard Implementation Problems](#3-standard-implementation-problems)
4. [CUDA Graphs Optimization](#4-cuda-graphs-optimization)
5. [Streaming Generation (Streaming Pipeline)](#5-streaming-generation-streaming-pipeline)
6. [Version Evolution (v6 → v14)](#6-version-evolution-v6--v14)
7. [Comparison with FasterQwen3TTS](#7-comparison-with-fasterqwen3tts)
8. [Benchmarks on RTX 5060 Ti](#8-benchmarks-on-rtx-5060-ti)
9. [Installation and Running](#9-installation-and-running)
10. [Key Code Patterns](#10-key-code-patterns)

---

## 1. Introduction

We implemented streaming speech generation based on **Qwen3-TTS-0.6B** with **CUDA Graphs** acceleration. Results:

| Metric | Before optimization | After | Speedup |
|---------|---------------|-------|---------|
| ms/step | 247.7ms | 31.3ms | **7.9x** |
| RTF | 0.336 (slower than realtime) | 2.67 (2.7× faster!) | **8.0x** |
| TTFA | ~40s (all audio at once) | ~355ms | **113x** |

### Key achievements

- **Streaming generation**: audio starts in ~350ms instead of 40 seconds
- **CUDA Graphs**: ~8× decoding speedup by eliminating Python overhead
- **Producer-consumer pipeline**: generation thread → queue → player with no underruns
- **Text segmentation**: automatic long-text splitting into segments

### Implementation files

| File | Description |
|------|-------------|
| `fast_tts_v14.py` | Final implementation — true streaming via native `generate_custom_voice_streaming`, seamless chunk concatenation |
| `profile_v14.py` | Baseline profiler: load/capture cost, Mimi decode vs context, TTFA/RTF, token-cap usage |
| `Qwen3-TTS/qwen_tts/inference/` | Custom PredictorGraph and TalkerGraph (patched in-repo) |
| `docs/cuda_graphs_optimization.md` | Detailed CUDA Graphs optimization breakdown |

> Note: earlier iterations (`streaming_tts_v6/v7/v8.py`, `fast_tts_final.py`, `fast_tts_v10.py`) no longer exist in this repository — v14 is the only maintained implementation (their history lives on in git log / version notes below).

---

## 2. Qwen3-TTS Architecture

Qwen3-TTS is a **two-component model** for speech synthesis:

```
┌──────────────────────────────────────────────────────────────┐
│              Qwen3TTSForConditionalGeneration                │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │                  Talker (28 layers)                   │   │
│  │                                                       │   │
│  │  Input: text + language + speaker                     │   │
│  │  Output: codec token (codebook 0)                     │   │
│  │                                                       │   │
│  │  ┌───────────────────────────────────────────────┐    │   │
│  │  │     Code Predictor (5 layers)                 │    │   │
│  │  │                                               │    │   │
│  │  │  Input: codebook N                            │    │   │
│  │  │  Output: codebook N+1                         │    │   │
│  │  │  (15 codebook groups total)                   │    │   │
│  │  └───────────────────────────────────────────────┘    │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  Speech Tokenizer (HiFi-GGAN decoder)                       │
│  Input: 15 codebook tokens → Output: waveform @24kHz        │
└──────────────────────────────────────────────────────────────┘
```

### Generation flow

```
Text → tokenize → Talker prefill (dynamic KV cache)
                ↓
        Code Predictor × 15 (sequential)
                ↓
        Codec token [cb0, cb1, ..., cb14]
                ↓
        Speech Tokenizer decoder.chunked_decode()
                ↓
        Waveform @24kHz (12Hz codec = ~0.83s per token)
```

### Key parameters

| Parameter | Value | Description |
|----------|-------|-------------|
| Talker layers | 28 | Main autoregressive decoder |
| Code Predictor layers | 5 | Predicts remaining codebook groups |
| Codebook groups | 15 | Total groups per timestep |
| Codec rate | 12 Hz | 12 codec tokens per second of audio |
| Sample rate | 24 kHz | Output waveform sampling rate |
| Hidden size | 1024 | Embedding size (0.6B model) |

### One decode step = ~15 forward passes

```python
# Each decoding step:
for step in range(max_new_tokens):      # ~130 steps per sentence
    cb_0 = talker.forward(...)          # 1 forward (28 layers)

    for group in range(1, 15):          # 14 sequential forwards
        hidden = predictor.forward(cb_{group-1})
        cb_{group} = sample(hidden)     # Code Predictor (5 layers)

    # Total: 1 + 14 = 15 forward per step
    # × 130 steps = ~2080 forward passes!
```

---

## 3. Standard Implementation Problems

Standard `Qwen3TTSModel.generate_custom_voice()` runs at **~250ms/step**. Let's examine 4 bottlenecks:

### Bottleneck #1: Dynamic KV Cache (~40ms/step)

On each step `past_key_values` grows by 1 position. PyTorch is forced to:
- Allocate a new larger tensor
- Copy old values
- Update attention mask

```python
# Each step — dynamic allocation:
step_0: key = [K₀]                    # size 1
step_1: key = concat([K₀, K₁])        # size 2 → allocation!
step_2: key = concat([K₀, K₁, K₂])    # size 3 → allocation!
...
# On 28 layers × 4 heads × 128 dim ≈ 720KB allocations/step
```

### Bottleneck #2: Python → GPU dispatcher (~50ms/step)

Each forward pass goes through the Python runtime:

```
Python call → PyTorch dispatcher → Triton kernel compilation → CUDA launch
     ↑                                    ↑                      ↑
  ~10ms                              ~30ms (first time!)      ~5ms
```

Per step there are **~150+ kernel launches** (28 layers × attention + MLP + norm):

```python
# Each forward = hundreds of individual CUDA kernel launches:
layer_0.attention.q_proj.forward()     # → kernel launch 1
layer_0.attention.k_proj.forward()     # → kernel launch 2
...
# 150 × 4μs = 600μs just on launch overhead
```

### Bottleneck #3: HF `generate()` loop (~20ms/step)

Universal generation loop with lots of checks per step:

```python
# transformers/generation/utils.py — simplified
for i in range(max_new_tokens):
    outputs = self(inputs, past_key_values=past_kv)  # forward

    scores = self.compute_scores(inputs, outputs)     # ~5ms
    scores = logits_processor(inputs, scores)         # top_k, top_p...

    for criteria in stopping_criteria:                # EOS check
        if criteria(outputs, scores):
            break

    next_tokens = self.sample(scores, ...)            # sampling
```

### Bottleneck #4: Code Predictor sub-loop (~30ms/step)

15 codebook groups are generated **sequentially** inside each talker step — cannot be parallelized without changing the architecture.

### Overhead summary

| Component | Time/step | % of total |
|-----------|-----------|-------------|
| Pure GPU computation | ~60ms | 24% |
| Dynamic KV cache (alloc + cat) | ~40ms | 16% |
| Python dispatcher (~150 kernel launches) | ~50ms | 20% |
| HF generate() overhead | ~20ms | 8% |
| Code predictor sub-loop (14× forward) | ~30ms | 12% |
| CUDA kernel compilation (Triton) | ~50ms | 20% |
| **Total** | **~250ms/step** | **100%** |

---

## 4. CUDA Graphs Optimization

### Idea

Instead of executing a forward pass through Python each step, we:
1. Run forward **once** and **record** the entire sequence of CUDA operations
2. On each subsequent step, just **replay** the recorded graph — GPU plays it directly

```
┌─────────────────────────────────────────────────────┐
│  First time (capture):                              │
│                                                     │
│  Python → forward() → [recorder]                    │
│              ↓                                      │
│         CUDA Graph:                                 │
│         ┌───────────────────────────────┐           │
│         │ Kernel 1: layer_0.q_proj      │           │
│         │ Kernel 2: layer_0.k_proj      │           │
│         │ ...                           │           │
│         │ Kernel N: codec_head          │           │
│         └───────────────────────────────┘           │
│                                                     │
│  Each subsequent step (replay):                     │
│                                                     │
│  cudaGraphLaunch(graph, stream)                     │
│    ↓                                                │
│  GPU plays the graph directly                       │
│  (no Python! no dispatcher!)                        │
└─────────────────────────────────────────────────────┘
```

### StaticCache vs DynamicCache

**Dynamic (standard):**
```python
# Each step — new tensor:
past_key = torch.cat([old_key, new_key], dim=1)  # alloc + copy!
```

**Static (CUDA graphs):**
```python
# Allocate fixed buffer once:
static_cache = torch.zeros(1, num_heads, max_seq_len, head_dim, device='cuda')

# Each step — write to fixed position:
step = 50
static_cache[:, :, step, :] = new_key_value  # just a write!
```

### Our CUDA Graphs architecture

```
┌───────────────────────────────────────────────────────┐
│       Qwen3-TTS/qwen_tts/inference/ (patched)         │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │           PredictorGraph                        │  │
│  │                                                 │  │
│  │  CUDA Graph:                                    │  │
│  │    input_embed → predictor.forward() × 15       │  │
│  │    (all 15 codebook groups in one graph!)       │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │           TalkerGraph                           │  │
│  │                                                 │  │
│  │  CUDA Graph:                                    │  │
│  │    input_embed → talker.forward() (1 step)      │  │
│  │    static_kv_cache[2048]                        │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  Decode loop:                                         │
│    predictor_graph.run(input)   → codebook_ids        │
│    talker_graph.run(codebook_id, pos) → hidden        │
│    codec_head(hidden) → next_token                    │
└───────────────────────────────────────────────────────┘
```

### PredictorGraph — capturing the predictor graph

File: `Qwen3-TTS/qwen_tts/inference/predictor_graph.py`

```python
class PredictorGraph:
    def __init__(self, code_predictor, pred_config, talker_hidden_size, ...):
        # Static buffers for input/output
        self.input_buf = torch.zeros(1, 2, talker_hidden_size, ...)
        self.output_tokens = torch.zeros(15, dtype=torch.long, ...)

        # StaticCache for predictor (max_seq=17)
        self.static_cache = StaticCache(config=pred_config, max_cache_len=17)

    def capture(self, num_warmup=3):
        # Warmup — compile Triton kernels once
        for _ in range(num_warmup):
            self._full_loop()  # 15 codebook groups

        # Capture graph
        stream = torch.cuda.Stream()
        with torch.cuda.graph(self.graph):
            self._full_loop()

    def run(self, pred_input: torch.Tensor) -> torch.Tensor:
        self.input_buf.copy_(pred_input)
        self.static_cache.reset()
        self.graph.replay()  # ← GPU replay!
        return self.output_tokens.clone()
```

### TalkerGraph — capturing the decoder

File: `Qwen3-TTS/qwen_tts/inference/talker_graph.py`

```python
class TalkerGraph:
    def __init__(self, talker_model, talker_config, max_seq_len=2048):
        # Static KV cache — allocated once for lifetime
        self.static_cache = StaticCache(config=talker_config, max_cache_len=max_seq_len)

        # Static buffers
        self.input_buf = torch.zeros(1, 1, hidden_size, ...)
        self.output_buf = torch.zeros(1, 1, hidden_size, ...)
        self.cache_position = torch.zeros(1, dtype=torch.long, ...)

    def prefill_kv(self, dynamic_past_kv):
        """Copy prefill KV from DynamicCache to StaticCache."""
        for layer_idx in range(num_layers):
            k, v = dynamic_past_kv[layer_idx]
            cache_pos = torch.arange(k.shape[2], device=self.device)
            self.static_cache.update(k, v, layer_idx,
                                     {"cache_position": cache_pos})
        return k.shape[2]

    def run(self, input_embeds: torch.Tensor, position: int) -> torch.Tensor:
        self.input_buf.copy_(input_embeds)
        self.cache_position[0] = position
        self._set_attention_mask(position)
        self.graph.replay()  # ← GPU replay!
        return self.output_buf
```

### Main generation loop (fast path)

```python
@torch.inference_mode()
def fast_generate(talker, talker_input_embeds, predictor_graph, talker_graph):
    # === PREFILL (once via regular forward) ===
    out = talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,  # dynamic cache only for prefill!
    )

    # Copy prefill KV into static buffer
    prefill_len = talker_graph.prefill_kv(out.past_key_values)

    # First token — regular forward
    logits = out.logits[:, -1]
    token = sample(logits)

    # === DECODE LOOP (CUDA graphs!) ===
    all_codec_ids = []

    for step_idx in range(max_new_tokens):
        if token.item() == eos_id:
            break

        # 1. Predictor — replay graph (~2ms!)
        last_hidden = talker.get_input_embeddings()(token)
        pred_input = torch.cat([past_hidden, last_hidden], dim=1)
        codebook_ids = predictor_graph.run(pred_input)  # ← CUDA graph!

        # 2. Assemble full codec token [cb0, cb1, ..., cb14]
        all_cb = torch.cat([token.view(1), codebook_ids])
        all_codec_ids.append(all_cb.detach())

        # 3. Build input embedding for talker
        inputs_embeds = build_codec_embedding(codebook_ids)

        # 4. Talker decode — replay graph (~29ms!)
        current_pos = prefill_len + step_idx
        hidden = talker_graph.run(inputs_embeds, position=current_pos)

        # 5. Logits and sampling (fast, small tensor)
        logits = talker.codec_head(hidden[:, -1])
        token = sample(logits)

    return torch.stack(all_codec_ids), timing
```

### Warmup behavior

CUDA graph warmup runs **once per model instance**, not per call:

```python
# First call — full warmup (~8.5s)
model.generate_custom_voice_streaming(...)  # ~9.5s (warmup + gen)

# Subsequent calls — no warmup (~80ms prefill overhead)
model.generate_custom_voice_streaming(...)  # ~1.7s
model.generate_custom_voice_streaming(...)  # ~0.96s
```

**Why ~80ms on subsequent calls?**
1. `talker.forward()` initial prefill pass (~50-70ms)
2. Tokenization + `_build_assistant_text` CPU work (~10-20ms)
3. `prefill_kv()` copying DynamicCache → StaticCache

---

## 5. Streaming Generation (Streaming Pipeline)

### Producer-Consumer Architecture

```
┌──────────────┐     queue      ┌──────────────┐
│   PRODUCER   │ ────────────→  │    PLAYER    │
│  (generation)│                │(sounddevice) │
│              │                │              │
│  Segment 1   │                │ Chunk 1      │
│  → Chunk 1   │                │ → play       │
│  → Chunk 2   │                │ Chunk 2      │
│  → Chunk 3   │                │ → play       │
└──────────────┘                └──────────────┘
```

### StreamingAudioPlayer

File: `fast_tts_v14.py` (lines 47-138)

Callback-based player built on `sounddevice.OutputStream`:

```python
class StreamingAudioPlayer:
    def __init__(self, sample_rate=24000, preroll_sec=0.3):
        self._chunks = queue.Queue(maxsize=32)
        self._buffered = 0
        self._preroll = int(sample_rate * preroll_sec)  # 7200 samples

    def start(self):
        self._stream = self.sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            latency="high", callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status):
        # PortAudio callback — continuously feed audio
        with self._lock:
            if not self._started:
                if self._buffered >= self._preroll:
                    self._started = True  # start playback

            while written < frames:
                chunk = self._chunks.get_nowait()
                out[written:written+n] = chunk[offset:offset+n]
```

### Text Segmentation

File: `fast_tts_v14.py` (lines 28-56)

Automatic splitting of long text into segments ≤85 characters:

```python
def split_segments(text, max_chars=85):
    # 1. Split by sentences (.!? )
    sentences = re.findall(r'([^.!?]+[.!?])|([^.!?]+$)', text)

    # 2. Long sentences — by commas/periods with ellipsis
    parts = re.split(r'(?<=[,;:])\s+', sentence)

    # 3. If still too long — by words
    words = buf.split()
```

### Producer-Consumer Pipeline (v14)

```python
def generate_and_play(self, text, language='Russian', save_wav=None):
    segments = split_segments(text, max_chars=85)

    player = StreamingAudioPlayer(sample_rate=24000, preroll_sec=0.3)
    player.start()

    q = queue.Queue(maxsize=32)

    # PRODUCER: generation in separate thread
    def producer():
        for seg in segments:
            gen = self.model.generate_custom_voice_streaming(
                text=seg, chunk_size=8, max_new_tokens=max_tokens
            )
            for audio_chunk, sr, timing in gen:
                chunk = to_pcm_chunk(audio_chunk)
                chunk = safe_normalize(chunk)  # clip protection

                # Backpressure: wait until queue has space
                while q.qsize() >= 20:
                    time.sleep(0.01)
                q.put(chunk)
            gen.close()
        q.put(None)  # sentinel

    gen_thread = threading.Thread(target=producer, daemon=True)
    gen_thread.start()

    # CONSUMER: read from queue → player
    all_wavs = []
    while True:
        chunk = q.get(timeout=0.1)
        if chunk is None:
            break

        # Backpressure: wait until player buffer drains
        while player.buffered_seconds() > 3.5:
            time.sleep(0.01)

        all_wavs.append(chunk)
        player.add_chunk(chunk)

    gen_thread.join(timeout=120)
    player.add_chunk(None)  # signal end
```

### Real-time metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **TTFA** | Time To First Audio — time until first sound | `first_chunk_time - start_time` |
| **RTF** | Real-Time Factor — generation speed | `audio_duration / wall_time` |
| **ms/step** | Time per codec step | `wall_time / num_steps` |
| **Inference Speed** | How much faster than realtime | `audio_duration / compute_time` |

---

## 6. Version Evolution (v6 → v14)

### Evolution table

| Version | Key change | Result |
|---------|-----------|--------|
| **v6** | Producer-consumer pipeline, fix underruns | Streaming without stuttering |
| **v7** | Parallel generation (ThreadPoolExecutor) | — |
| **v8** | `torch.compile` + chunked decode | Didn't work (dynamic sizes) |
| **v9** | StoppingCriteria patch + auto max_new_tokens | 1-3s for short phrases |
| **v10** | CUDA Graphs backend (FasterQwen3TTS) | ~8× speedup, RTF > 1.0 |
| **v14** | True streaming + crossfade + segmentation | Smooth sound without clicks |

### Details of each version

#### v6 — First working pipeline

```python
# Key fixes:
1. Producer thread generates segments, puts into queue
2. Main thread reads from queue and feeds player
3. player.add_chunk(None) ONLY after generation completes
4. trim_silence + apply_fades for smooth boundaries
```

**Bug:** crossfade accumulated prev_audio after each chunk → sound multiplied by N chunks.

#### v9 — StoppingCriteria fix

**Problem:** `max_new_tokens=8192` default → model generates 15+ seconds of audio for a short phrase "Yes".

**Solution:**
```python
def _get_max_new_tokens(self, text):
    word_count = len(re.findall(r'\b\w+\b', text))
    if word_count <= 2: return 20
    elif word_count <= 5: return 50
    elif word_count <= 10: return 100
    else: return 160
```

**Result:** "Yes" → 1.25s (was 15-27s)

#### v10 — CUDA Graphs transition

Transition to `FasterQwen3TTS` wrapper with CUDA graph capture:
- ~8× decoding speedup
- RTF > 1.0 (faster than realtime)
- TTFA ~350ms instead of ~40s

#### v14 — Final version

```python
# Key improvements vs v10:
1. Crossfade between segments (smooth transitions)
2. Per-chunk normalization (consistent loudness)
3. Backpressure control (queue + player buffer)
4. MIN_START_SEC = 1.0s (wait until buffer is full)
5. split_segments with smart sentence-based splitting
```

---

## 7. Comparison with FasterQwen3TTS

### Architectural differences

| Aspect | Our implementation | FasterQwen3TTS |
|--------|-------------------|----------------|
| **Base model** | `Qwen3TTSModel` (native) | `Qwen3TTSModel` + wrapper |
| **CUDA Graphs** | Custom `PredictorGraph`, `TalkerGraph` | Own implementations in `faster_qwen3_tts/` |
| **Streaming API** | `generate_custom_voice_streaming()` | `generate_custom_voice_streaming()` |
| **Voice Cloning** | Not implemented | Full support (ICL + x-vector) |
| **Voice Design** | Not implemented | Supported via `instruct` |
| **CLI** | None | `faster-qwen3-tts` command |
| **Server mode** | None | OpenAI-compatible API server |
| **GGML backend** | None | qwentts.cpp optional |

### Key implementation differences

#### 1. Bugfix: non_streaming_mode forced

In native `Qwen3TTSModel` there was a bug (line 1576 in original):
```python
# Was (bug):
non_streaming_mode=non_streaming_mode or True

# Should be:
non_streaming_mode=non_streaming_mode if non_streaming_mode is not None else False
```

This caused audio repetition during streaming generation.

#### 2. Crossfade between segments

Our implementation applies crossfade at segment boundaries for smooth transitions:
```python
def apply_fades(wav, sr=24000, in_ms=5, out_ms=25):
    n_in = int(sr * in_ms / 1000.0)
    n_out = int(sr * out_ms / 1000.0)
    wav[:n_in] *= np.linspace(0.0, 1.0, n_in)   # fade-in
    wav[-n_out:] *= np.linspace(1.0, 0.0, n_out) # fade-out
```

#### 3. Backpressure control

Our pipeline uses double backpressure:
- **Producer side**: wait until queue has space (`q.qsize() >= 20`)
- **Consumer side**: wait until player buffer drains (`buffered_seconds() > 3.5`)

#### 4. Safe chunk copying

```python
def to_pcm_chunk(x):
    """Always real copy — protection against static buffer reuse."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32, copy=True).reshape(-1)
```

### What FasterQwen3TTS does better

| Feature | Description |
|---------|-------------|
| **Voice Cloning** | ICL mode with reference audio + x-vector only mode |
| **Voice Design** | Instruction-based generation ("Warm, confident narrator") |
| **CLI utility** | `faster-qwen3-tts clone/design/custom` commands |
| **OpenAI API** | Compatible server for client integration |
| **GGML backend** | Optional qwentts.cpp for CPU/low-memory |
| **Speaker caching** | `.spk` / `.rvq` cache for fast cloning |
| **Demo UI** | Web interface with live TTFA/RTF metrics |

### What our implementation does better

| Feature | Description |
|---------|-------------|
| **Crossfade** | Smooth transitions between segments |
| **Per-chunk normalization** | Consistent loudness without pumping |
| **Simplicity** | Single file, no dependency on faster-qwen3-tts |
| **Bug fixes** | Fix for non_streaming_mode forced bug |

---

## 8. Benchmarks on RTX 5060 Ti

### Configuration

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 5060 Ti (16GB) |
| PyTorch | 2.13.0+cu132 |
| CUDA | 13.2 |
| Compute Capability | 12.0 |
| Model | Qwen3-TTS-0.6B @ bfloat16 |
| Speaker | Sohee (Russian) |
| Text | ~130 codec tokens (~11s audio) |

### Comparison results

| Metric | Plain (baseline) | CUDA Graphs | Speedup |
|---------|-----------------|-------------|---------|
| **ms/step** | 247.7ms | 31.3ms | **7.9x** |
| **RTF** | 0.336 | 2.67 | **8.0x** |
| **TTFA** | ~40,382ms | ~355ms | **113x** |
| **Time for 11s audio** | ~33.1s | ~4.2s | **7.9x** |

### Streaming segments (v10)

| Segment | Generation | TTFA | Audio | Chunks | RTF |
|---------|-----------|------|-------|--------|-----|
| My name is Alexander. | 1376ms | 466ms | 2.48s | 4 | 1.80 |
| I am twenty-five years old. | 1170ms | 389ms | 2.48s | 4 | 2.12 |
| I live in Saint Petersburg. | 1420ms | 372ms | 2.96s | 5 | 2.08 |

**Total:** 6.4s audio in 8.0s wall time (RTF ~0.8 including segment prefills).

### Why RTF > 1.0 matters

```
RTF = audio_duration / wall_time

RTF < 1.0  → generation slower than realtime (waiting for audio)
RTF = 1.0  → exactly realtime
RTF > 1.0  → faster than realtime (ahead of playback)
```

Our result **RTF 2.67** means: the model generates audio 2.67× faster than it plays back. This provides headroom for processing subsequent segments and network latency.

---

## 9. Installation and Running

### Requirements

```
Python >= 3.10
PyTorch >= 2.5.1 (cu132)
CUDA >= 12.8
NVIDIA GPU with CUDA Graphs support
```

### Installing flash-attn (Windows + CUDA 13.2)

```cmd
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "FLASH_ATTN_CUDA_ARCHS=120"
set "MAX_JOBS=10"
python flash-attention/setup.py build_ext --inplace
```

**Important:** For CUDA 13.2 + MSVC 14.43, a fix is needed in `setup.py`:
```python
# Add when sys.platform == "win32":
compiler_c17_flag.append("/Zc:preprocessor")
feature_flags.append("-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING")
```

### Quick start

```python
from fast_tts_v14 import FastTTSv14

tts = FastTTSv14(
    model_path=r'G:\Foundation\models\Qwen3-TTS',
    speaker='Sohee'
)

tts.generate_and_play(
    text="Hello! This is a streaming TTS test.",
    language='English',
    save_wav='output.wav'
)
```

### Test suite (10 sentences)

```bash
python fast_tts_v14.py --test
```

Runs 5 Russian + 5 English paragraphs, saves WAV files.

---

## 10. Key Code Patterns

### Pattern 1: Producer-Consumer for streaming

```python
q = queue.Queue(maxsize=32)

def producer():
    for item in generate_items():
        while q.qsize() >= 20:  # backpressure
            time.sleep(0.01)
        q.put(item)
    q.put(None)  # sentinel

threading.Thread(target=producer, daemon=True).start()

for chunk in iter(lambda: q.get(), None):
    play(chunk)
```

### Pattern 2: CUDA Graph capture/replay

```python
# Capture (once)
stream = torch.cuda.Stream()
with torch.cuda.graph(graph, stream=stream):
    output = model.forward(input_buf)

# Replay (each step)
input_buf.copy_(new_input)
graph.replay()  # no Python overhead!
```

### Pattern 3: StaticCache for fixed KV

```python
from transformers import StaticCache

cache = StaticCache(config=config, max_cache_len=2048)

# Prefill — copy dynamic cache into static
for layer_idx in range(num_layers):
    k, v = dynamic_kv[layer_idx]
    pos = torch.arange(k.shape[2], device=device)
    cache.update(k, v, layer_idx, {"cache_position": pos})

# Decode — just update cache_position
cache_position = torch.tensor([current_pos], device=device)
output = model.forward(input_embeds, past_key_values=cache,
                       cache_position=cache_position)
```

### Pattern 4: Safe tensor copying from static buffer reuse

```python
def to_pcm_chunk(x):
    """Always real copy — protection against static buffer overwrite."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32, copy=True).reshape(-1)
```

### Pattern 5: Backpressure control

```python
# Producer side — don't overload the queue
while q.qsize() >= max_queue_size:
    time.sleep(0.01)
q.put(chunk)

# Consumer side — don't overload the player
while player.buffered_seconds() > max_buffer_sec:
    time.sleep(0.01)
player.add_chunk(chunk)
```

---

## Appendices

### A. File references

| File | Description |
|------|-------------|
| `fast_tts_v14.py` | Final streaming implementation |
| `profile_v14.py` | Baseline profiler (load/capture, Mimi vs ctx, TTFA/RTF, token caps) |
| `Qwen3-TTS/qwen_tts/inference/predictor_graph.py` | Predictor CUDA graph |
| `Qwen3-TTS/qwen_tts/inference/talker_graph.py` | Talker CUDA graph |
| `docs/cuda_graphs_optimization.md` | Detailed optimization breakdown |
| `bench_sdpa.py` | SDPA attention benchmark |

### B. Useful commands

```bash
# Run with text from arguments
python fast_tts_v14.py "Hello world"

# Test suite (10 sentences)
python fast_tts_v14.py --test

# Benchmark Faster vs Plain
python bench_faster_custom_voice.py

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### C. Known limitations

1. **max_seq_len = 2048** — limits prefill length (text + reference audio)
2. **Batch size = 1** — CUDA graphs don't support dynamic batch
3. **NVIDIA GPU only** — requires CUDA and CUDAGraph support
4. **PyTorch >= 2.5.1** — for stable capture

---

*Document created based on analysis of the implementation in `fast_tts_v14.py` and comparison with `faster-qwen3-tts`.*
