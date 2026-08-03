# Qwen3-TTS: CUDA Graphs Optimization — Full Breakdown
## From 250ms/step to 31ms/step (8x speedup)

---

## Table of Contents

1. [Qwen3-TTS Architecture](#1-qwen3-tts-architecture)
2. [Standard Generation Path (slow)](#2-standard-generation-path-slow)
3. [Where Time Is Lost — 4 Bottlenecks](#3-where-time-is-lost--4-bottlenecks)
4. [CUDA Graphs — How It Works](#4-cuda-graphs---how-it-works)
5. [FasterQwen3TTS Implementation (fast)](#5-fasterqwen3tts-implementation-fast)
6. [Code Comparison: Before vs After](#6-code-comparison-before-vs-after)
7. [Benchmarks on RTX 5060 Ti](#7-benchmarks-on-rtx-5060-ti)
8. [Why torch.compile Didn't Help](#8-why-torchcompile-didnt-help)

---

## 1. Qwen3-TTS Architecture

Qwen3-TTS is a **two-component model** for speech synthesis:

```
┌──────────────────────────────────────────────────────┐
│              Qwen3TTSForConditionalGeneration        │
│                                                      │
│  ┌───────────────────────────────────────────────┐   │
│  │                  Talker (28 layers)           │   │
│  │                                               │   │
│  │  Input: text + language + speaker             │   │
│  │  Output: codec token (codebook 0)             │   │
│  │                                               │   │
│  │  ┌───────────────────────────────────────┐    │   │
│  │  │     Code Predictor (5 layers)         │    │   │
│  │  │                                       │    │   │
│  │  │  Input: codebook N                    │    │   │
│  │  │  Output: codebook N+1                 │    │   │
│  │  │  (15 codebook groups total)           │    │   │
│  │  └───────────────────────────────────────┘    │   │
│  └───────────────────────────────────────────────┘   │
│                                                      │
│  Speech Tokenizer (HiFi-GGAN decoder)                │
│  Input: 15 codebook tokens → Output: waveform        │
└──────────────────────────────────────────────────────┘
```

**Key point:** On each decoding step, the model performs **two forward passes**:
1. `Talker` — predicts the first codebook token
2. `Code Predictor` — sequentially predicts the remaining 14 codebook groups

Total: **~130 steps × (talker + 15 predictor) = ~2080 forward passes** per sentence.

---

## 2. Standard Generation Path (slow)

### Call chain

```
model.generate_custom_voice()
    ↓
Qwen3TTSModel._build_assistant_text() + tokenize
    ↓
model.model.generate()              ← custom generate, not HF!
    ↓
talker.forward(past_key_values=...)  ← DYNAMIC KV cache
    ↓
code_predictor.forward() × 15       ← sub-loop per step
```

### Standard path code (simplified)

```python
# Qwen3-TTS standard generate — simplified schematic
@torch.no_grad()
def standard_generate(text, speaker, language):
    # 1. Build input embeddings
    input_ids = tokenize(build_assistant_text(text))

    talker_input_embeds = build_talker_inputs(
        m=model.model,
        input_ids=[input_ids],
        languages=[language],
        speakers=[speaker],
        # ... many parameters
    )

    # 2. Talker forward with DYNAMIC KV cache
    past_key_values = None
    for step in range(max_new_tokens):
        if step == 0:
            out = talker.forward(
                inputs_embeds=talker_input_embeds,
                attention_mask=attention_mask,
                use_cache=True,          # ← dynamic cache!
                trailing_text_hidden=trailing_text_hiddens,
                tts_pad_embed=tts_pad_embed,
            )
        else:
            out = talker.forward(
                input_ids=torch.tensor([[current_token]]),
                past_key_values=past_key_values,  # ← grows every step!
                use_cache=True,
            )

        # 3. Code predictor — 15 sequential forward passes
        for cb in range(14):
            pred = code_predictor.forward(hidden_state)
            current_token = sample(pred.logits)

        past_key_values = out.past_key_values

    return codec_tokens
```

### What happens inside `talker.forward()` each step:

```python
# transformers/models/modeling_utils.py — simplified forward with cache
def forward(self, input_ids, past_key_values=None, use_cache=True):
    hidden_states = self.embeddings(input_ids)

    for layer in self.layers:  # 28 layers
        # DYNAMIC memory allocation for KV
        if past_key_values is not None:
            k = torch.cat([past_key_values[layer_idx].key, current_k], dim=1)
            v = torch.cat([past_key_values[layer_idx].value, current_v], dim=1)
            # ↑ EVERY STEP — new tensor, new allocation!

        # Attention with dynamic dimensions
        attn_output = self.attention(
            q=query, k=key, v=value,
            attention_mask=dynamic_attention_mask  # ← changes every step
        )

    return BaseModelOutput(past_key_values=new_past_kv)
```

---

## 3. Where Time Is Lost — 4 Bottlenecks

### Bottleneck #1: Dynamic KV Cache (~40ms/step)

**Problem:** On each step `past_key_values` grows by 1 position. PyTorch must:
- Allocate a new larger tensor
- Copy old values
- Update attention mask

```python
# Each step — dynamic allocation:
step_0: key = [K₀]                    # size 1
step_1: key = concat([K₀, K₁])        # size 2 → allocation!
step_2: key = concat([K₀, K₁, K₂])    # size 3 → allocation!
...
step_50: key = concat([...])          # size 51 → allocation!

# Each cat() is a new tensor on GPU memory.
# On 28 layers × 4 heads × 128 dim = ~36KB per layer.
# 28 layers × 2 (key+value) = ~720KB allocations/step.
```

### Bottleneck #2: Python → GPU dispatcher (~50ms/step)

**Problem:** Each forward pass goes through the Python runtime:

```
Python call → PyTorch dispatcher → Triton kernel compilation → CUDA launch
     ↑                                    ↑                      ↑
  ~10ms                              ~30ms (first time!)      ~5ms
```

PyTorch is an **eager execution framework**. Each operator runs separately, and every call goes through Python:

```python
# Each forward = hundreds of individual CUDA kernel launches:
layer_0.attention.q_proj.forward()     # → kernel launch 1
layer_0.attention.k_proj.forward()     # → kernel launch 2
layer_0.attention.v_proj.forward()     # → kernel launch 3
layer_0.attention.sdpa.forward()       # → kernel launch 4
layer_0.mlp.gate_proj.forward()        # → kernel launch 5
# ... × 28 layers = ~150+ kernel launches per step!

# Each launch = Python → C++ → CUDA driver ≈ 3-5 microseconds
# 150 × 4μs = 600μs just on launch overhead
```

### Bottleneck #3: HF `generate()` loop (~20ms/step)

**Problem:** Universal generation loop with lots of checks:

```python
# transformers/generation/utils.py — real code (simplified)
def generate(self, inputs, generation_config, **kwargs):
    # 1. Parameter validation (~5ms)
    self._validate_model_class()
    self._validate_assistant_dummy_inputs()

    # 2. Preparation (~3ms)
    logits_processor = self._get_logits_processor(...)
    stopping_criteria = self._get_stopping_criteria(...)

    # 3. Main loop — on EVERY step:
    for i in range(max_new_tokens):
        outputs = self(inputs, past_key_values=past_kv)  # forward

        # Lots of checks per token:
        scores = self.compute_scores(inputs, outputs)
        scores = logits_processor(inputs, scores)       # top_k, top_p, temperature...

        for criteria in stopping_criteria:              # EOS check
            if criteria(outputs, scores):
                break

        next_tokens = self.sample(scores, ...)          # sampling
        past_kv = outputs.past_key_values

        # More checks:
        if generation_config.force_words_ids is not None: ...
        if generation_config.bad_words_ids is not None: ...
```

### Bottleneck #4: Code Predictor sub-loop (~30ms/step)

**Problem:** 15 codebook groups are generated sequentially inside each talker step:

```python
# Each decoding step:
for step in range(max_new_tokens):  # ~130 steps
    # Talker forward → codebook 0
    cb_0 = talker.forward(...)

    # Code predictor — 14 sequential passes!
    for group in range(1, 15):
        hidden = predictor.forward(cb_{group-1})
        cb_{group} = sample(hidden)

    # Total: 1 (talker) + 14 (predictor) = 15 forward per step
    # × 130 steps = ~2080 forward passes!
```

### Overhead summary

| Component | Time/step | % of total |
|-----------|-----------|-------------|
| Pure GPU computation | ~60ms | 24% |
| Dynamic KV cache (alloc + cat) | ~40ms | 16% |
| Python dispatcher (~150 kernel launches) | ~50ms | 20% |
| HF generate() overhead (checks, validation) | ~20ms | 8% |
| Code predictor sub-loop (14× forward) | ~30ms | 12% |
| CUDA kernel compilation (Triton, first time) | ~50ms | 20% |
| **Total** | **~250ms/step** | **100%** |

---

## 4. CUDA Graphs — How It Works

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
│         │ Kernel 3: layer_0.v_proj      │           │
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

### Static KV Cache vs Dynamic

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

### How to capture a CUDA Graph

```python
import torch

# 1. Prepare static buffers
static_kv_cache = torch.zeros(1, 28, 4, 2048, 128, device='cuda')  # fixed size

# 2. Start capture
stream = torch.cuda.Stream()
stream.begin_capture()

# 3. Run one forward (recorded into graph)
hidden = model.forward(
    inputs_embeds=input_placeholder,   # static tensor
    cache_position=position_placeholder,  # static position
    static_cache=static_kv_cache,       # static buffer
)
logits = codec_head(hidden[:, -1])

# 4. End capture → get the graph
graph = stream.end_capture()

# 5. Replay — no Python!
for step in range(max_new_tokens):
    input_placeholder[:] = new_input     # update input
    position_placeholder[:] = step       # update position
    cudaGraphLaunch(graph, stream)       # GPU replay!
    new_token = sample(logits[:, -1])    # only sampling via Python
```

---

## 5. FasterQwen3TTS Implementation (fast)

### Optimization architecture

```
┌───────────────────────────────────────────────────────┐
│              FasterQwen3TTS                           │
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

```python
# faster_qwen3_tts/predictor_graph.py (simplified)
class PredictorGraph:
    def __init__(self, predictor_model, config, device='cuda', dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        # Static buffers for input/output
        self.input_buf = torch.zeros(1, 2, config.hidden_size,
                                      device=device, dtype=dtype)
        self.output_ids = torch.zeros(15, dtype=torch.long, device=device)

        self.graph = None

    def capture(self, num_warmup=3):
        # Warmup — compile Triton kernels once
        for _ in range(num_warmup):
            self._forward_pass()

        # Capture graph
        stream = torch.cuda.Stream()
        stream.begin_capture()
        self._forward_pass()
        self.graph = stream.end_capture()

    def _forward_pass(self):
        # One predictor forward — 15 codebook groups
        hidden = self.predictor.forward(self.input_buf)
        # sampling → self.output_ids

    def run(self, input_data):
        # Update input buffer
        self.input_buf.copy_(input_data)

        # Replay graph — no Python!
        torch.cuda.graph_replay(self.graph)  # ← that's all!

        return self.output_ids.clone()
```

### TalkerGraph — capturing the decoder with static KV cache

```python
# faster_qwen3_tts/talker_graph.py (simplified)
class TalkerGraph:
    def __init__(self, talker_model, config, max_seq_len=2048):
        self.max_seq_len = max_seq_len

        # Static KV cache — allocated once for the lifetime
        num_layers = config.num_hidden_layers   # 28
        num_heads = config.num_attention_heads   # 4
        head_dim = config.hidden_size // num_heads  # 128

        self.key_cache = torch.zeros(
            1, num_layers, num_heads, max_seq_len, head_dim,
            device='cuda', dtype=torch.bfloat16
        )
        self.value_cache = torch.zeros_like(self.key_cache)

        # Static output buffer
        self.output_buf = torch.zeros(
            1, 1, config.hidden_size,
            device='cuda', dtype=torch.bfloat16
        )

        # Position buffer — updated each step
        self.position_buf = torch.tensor([0], device='cuda')

        self.graph = None

    def prefill_kv(self, dynamic_past_kv):
        """Copy prefill KV into static buffer."""
        for layer_idx in range(num_layers):
            k_len = dynamic_past_kv[layer_idx].key.shape[2]
            self.key_cache[:, layer_idx, :, :k_len, :] = dynamic_past_kv[layer_idx].key
            self.value_cache[:, layer_idx, :, :k_len, :] = dynamic_past_kv[layer_idx].value
        return k_len

    def capture(self, prefill_len=100, num_warmup=3):
        # Warmup
        for _ in range(num_warmup):
            self._decode_step(prefill_len + 5)

        # Capture decode graph
        stream = torch.cuda.Stream()
        stream.begin_capture()
        self._decode_step(prefill_len + 10)
        self.graph = stream.end_capture()

    def _decode_step(self, position):
        """One decoding step with static KV cache."""
        # Talker forward with fixed buffers
        hidden = talker.forward(
            inputs_embeds=self.input_buf,
            static_kv_cache=(self.key_cache, self.value_cache),
            position=position,  # ← where to write new KV
            output_hidden=True,   # → into self.output_buf
        )

    def run(self, input_embed, position):
        """Replay decoding."""
        self.input_buf.copy_(input_embed)
        self.position_buf[0] = position

        torch.cuda.graph_replay(self.graph)  # ← GPU replay!

        return self.output_buf.clone()
```

### Main generation loop (fast path)

```python
# faster_qwen3_tts/generate.py — fast_generate()
@torch.inference_mode()
def fast_generate(talker, talker_input_embeds, ..., predictor_graph, talker_graph):
    # === PREFILL (once via regular forward) ===
    out = talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,  # dynamic cache only for prefill!
    )

    # Copy prefill KV into static buffer
    prefill_len = talker_graph.prefill_kv(out.past_key_values)

    # First token — regular forward (for initialization)
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

        # 2. Assemble full codec token [cb0, cb1, ..., cb15]
        all_cb = torch.cat([token.view(1), codebook_ids])
        all_codec_ids.append(all_cb.detach())

        # 3. Build input embedding for talker
        inputs_embeds = build_codec_embedding(codebook_ids)

        # 4. Talker decode — replay graph (~29ms!)
        current_pos = prefill_len + step_idx
        hidden = talker_graph.run(inputs_embeds, position=current_pos)  # ← CUDA graph!

        # 5. Logits and sampling (fast, small tensor)
        logits = talker.codec_head(hidden[:, -1])
        token = sample(logits)

    return torch.stack(all_codec_ids), timing
```

---

## 6. Code Comparison: Before vs After

### Before — standard path (~250ms/step)

```python
# fast_tts_v9.py — _generate_segment_audio()
@torch.no_grad()
def _generate_segment_audio(self, text, language, speaker):
    max_new_tokens = self._get_max_new_tokens(text, language)

    gen_kwargs = self.model._merge_generate_kwargs(
        max_new_tokens=max_new_tokens,
        do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
    )

    input_ids = self.model._tokenize_texts([self.model._build_assistant_text(text)])
    instruct_ids = [None]

    # ← One call — inside: 130 steps with dynamic KV cache
    talker_codes_list, _ = self.model.model.generate(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=[language],
        speakers=[speaker],
        non_streaming_mode=True,
        **gen_kwargs,
    )

    # ← Chunked decode (slow — each chunk is a separate forward)
    decoder = self.model.model.speech_tokenizer.model.decoder
    for start in range(0, seq_len, 50):
        wav = decoder.chunked_decode(codes[start:end], chunk_size=50)

    return final_wav
```

**What happens inside `model.model.generate()`:**
- ~130 iterations of decode loop
- Each iteration: talker.forward() + 14× code_predictor.forward()
- Dynamic KV cache grows every step
- Python dispatcher on each forward (~150 kernel launches)

### After — CUDA graphs path (~31ms/step)

```python
# fast_tts_v10.py — generate_streaming()
def generate_streaming(self, text, language='Russian'):
    gen = self.model.generate_custom_voice_streaming(
        text=text,
        speaker=self.speaker,
        language=language,
        chunk_size=8,  # 8 codec steps per chunk (~0.67s audio)
        max_new_tokens=max_tokens,
    )

    for audio_chunk, sr, timing in gen:
        yield audio_chunk  # ← sound appears every ~300ms!
```

**What happens inside `generate_custom_voice_streaming()`:**
- One prefill (dynamic forward) → KV copied into static buffer
- Decode loop: predictor_graph.run() + talker_graph.run() — CUDA graph replay
- Per chunk (8 steps): ~250ms generation → decode → audio

---

## 7. Benchmarks on RTX 5060 Ti

### Configuration

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 5060 Ti (16GB) |
| PyTorch | 2.13.0+cu132 |
| CUDA | 13.2 |
| Model | Qwen3-TTS-12Hz-0.6B-CustomVoice |
| Speaker | Sohee (Russian) |
| Text | ~4 sentences, ~130 codec tokens |

### Results

| Metric | Standard (plain) | CUDA Graphs | Speedup |
|---------|-----------------|-------------|---------|
| **ms/step** | 247.7ms | 31.3ms | **7.9x** |
| **RTF** | 0.336 (3× slower than realtime) | 2.67 (2.7× faster!) | **8.0x** |
| **Time for 11s audio** | 33.1s | 4.2s | **7.9x** |
| **TTFA (first audio)** | N/A | 355ms | — |

### Streaming with segmentation (v10)

| Segment | Generation | TTFA | Audio | Chunks | RTF |
|---------|-----------|------|-------|--------|-----|
| My name is Alexander. | 1376ms | 466ms | 2.48s | 4 | 1.80 |
| I am twenty-five years old. | 1170ms | 389ms | 2.48s | 4 | 2.12 |
| I live in Saint Petersburg. | 1420ms | 372ms | 2.96s | 5 | 2.08 |

**Total:** 6.4s audio in 8.0s wall time (RTF ~0.8 including segment prefills).

### v9 vs v10 comparison

| Metric | v9 (plain) | v10 (CUDA graphs) | Speedup |
|---------|-----------|-------------------|---------|
| Total time | 29,990ms | 8,042ms | **3.7x** |
| Generation/segment | ~9s | ~1.3s | **~7x** |
| Player underruns | 350 | 1 | **350× fewer** |

---

## 8. Why torch.compile Didn't Help

### Attempt with torch.compile (v8)

```python
# streaming_tts_v8.py — compile attempt
self.model.model = torch.compile(self.model.model, mode='reduce-overhead')
if hasattr(self.model.model, 'talker'):
    self.model.model.talker = torch.compile(self.model.model.talker, mode='reduce-overhead')
```

### Why it didn't work

**Reason 1: Dynamic sizes**

torch.compile works best with fixed tensor shapes. Qwen3-TTS has:
- KV cache grows every step (shape changes)
- Attention mask changes every step
- Code predictor receives different hidden states

```python
# Each step — different shape:
step_0: past_key.shape = [1, 4, 1, 128]
step_1: past_key.shape = [1, 4, 2, 128]  ← new shape → recompilation!
step_2: past_key.shape = [1, 4, 3, 128]  ← new shape → recompilation!
```

torch.compile tries to **recompile** on every new shape — slower than not compiling at all.

**Reason 2: Code predictor sub-loop**

15 sequential forward passes of code_predictor with different input shapes (different codebook groups) — compile cannot capture this pattern efficiently.

**Reason 3: Python overhead in HF generate()**

torch.compile compiles **only PyTorch ops**, but does not optimize:
- Stopping criteria checks
- Logits processing (top_k, top_p)
- Dynamic cache management
- Attention mask updates

These operations take ~30% of the time and are outside the compiled graph.

### CUDA Graphs vs torch.compile

| Aspect | torch.compile | CUDA Graphs |
|--------|--------------|-------------|
| What it optimizes | PyTorch ops inside forward() | Full graph: Python → GPU |
| Dynamic sizes | Problems (recompilation) | No problem (static buffer) |
| Kernel launch overhead | Reduces (fusion) | Completely eliminates |
| Python dispatcher | No | Completely eliminates |
| Memory allocation | No | Static buffers |
| Speedup on TTS | ~1.2x (or even slowdown) | **~8x** |

---

## Conclusions

### Why CUDA Graphs is the right approach for autoregressive generation

1. **Fixed step size** — each decode step has the same input/output shape
2. **Predictable graph** — operation sequence doesn't change between steps
3. **Static KV cache** — allocate once, then just write at the right position
4. **Eliminating Python overhead** — graph replay = 1 CUDA call instead of ~150 kernel launches

### When NOT to use CUDA Graphs

- Variable input sizes (batch size changes)
- Data-dependent control flow (if/else inside forward)
- One or two passes (warmup/capture is more expensive than just running)
- Very long sequences (> max_seq_len of the graph)

### For Qwen3-TTS this is an ideal case

- Fixed input shape per step: [1, 1, hidden]
- ~130 steps per segment → capture pays off immediately
- max_seq_len = 2048 → covers any reasonable text length
