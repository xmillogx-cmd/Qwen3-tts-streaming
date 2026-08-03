# Qwen3-TTS: Оптимизация через CUDA Graphs — Полный разбор
## От 250ms/step до 31ms/step (8x ускорение)

---

## Содержание

1. [Архитектура Qwen3-TTS](#1-архитектура-qwen3-tts)
2. [Стандартный путь генерации (медленно)](#2-стандартный-путь-генерации-медленно)
3. [Где теряется время — 4 узких места](#3-где-теряется-время---4-узких-места)
4. [CUDA Graphs — как это работает](#4-cuda-graphs---как-это-работает)
5. [Реализация FasterQwen3TTS (быстро)](#5-реализация-fasterqwen3tts-быстро)
6. [Сравнение кода: до и после](#6-сравнение-кода-до-и-после)
7. [Бенчмарк на RTX 5060 Ti](#7-бенчмарк-на-rtx-5060-ti)
8. [Почему torch.compile не помог](#8-почему-torchcompile-не-помог)

---

## 1. Архитектура Qwen3-TTS

Qwen3-TTS — это **двухкомпонентная модель** для синтеза речи:

```
┌──────────────────────────────────────────────────────┐
│              Qwen3TTSForConditionalGeneration        │
│                                                      │
│  ┌───────────────────────────────────────────────┐   │
│  │                  Talker (28 слоёв)            │   │
│  │                                               │   │
│  │  Input: текст + язык + спикер                 │   │
│  │  Output: codec token (codebook 0)             │   │
│  │                                               │   │
│  │  ┌───────────────────────────────────────┐    │   │
│  │  │     Code Predictor (5 слоёв)          │    │   │
│  │  │                                       │    │   │
│  │  │  Input: codebook N                     │    │   │
│  │  │  Output: codebook N+1                  │    │   │
│  │  │  (15 codebook-групп всего)             │    │   │
│  │  └───────────────────────────────────────┘    │   │
│  └───────────────────────────────────────────────┘   │
│                                                      │
│  Speech Tokenizer (HiFi-GGAN decoder)                │
│  Input: 15 codebook-токенов → Output: waveform       │
└──────────────────────────────────────────────────────┘
```

**Ключевой момент:** на каждом шаге декодирования модель делает **два forward-прохода**:
1. `Talker` — предсказывает первый codebook-токен
2. `Code Predictor` — последовательно предсказывает оставшиеся 14 codebook-групп

Итого: **~130 шагов × (talker + 15 predictor) = ~2080 forward-проходов** на одно предложение.

---

## 2. Стандартный путь генерации (медленно)

### Цепочка вызовов

```
model.generate_custom_voice()
    ↓
Qwen3TTSModel._build_assistant_text() + tokenize
    ↓
model.model.generate()              ← кастомный generate, не HF!
    ↓
talker.forward(past_key_values=...)  ← ДИНАМИЧЕСКИЙ KV cache
    ↓
code_predictor.forward() × 15       ← подцикл на каждый шаг
```

### Код стандартного пути (упрощённо)

```python
# Qwen3-TTS стандартный generate — упрощённая схема
@torch.no_grad()
def standard_generate(text, speaker, language):
    # 1. Построение входных эмбеддингов
    input_ids = tokenize(build_assistant_text(text))
    
    talker_input_embeds = build_talker_inputs(
        m=model.model,
        input_ids=[input_ids],
        languages=[language],
        speakers=[speaker],
        # ... куча параметров
    )
    
    # 2. Talker forward с ДИНАМИЧЕСКИМ KV cache
    past_key_values = None
    for step in range(max_new_tokens):
        if step == 0:
            out = talker.forward(
                inputs_embeds=talker_input_embeds,
                attention_mask=attention_mask,
                use_cache=True,          # ← динамический cache!
                trailing_text_hidden=trailing_text_hiddens,
                tts_pad_embed=tts_pad_embed,
            )
        else:
            out = talker.forward(
                input_ids=torch.tensor([[current_token]]),
                past_key_values=past_key_values,  # ← растёт каждый шаг!
                use_cache=True,
            )
        
        # 3. Code predictor — 15 последовательных forward-проходов
        for cb in range(14):
            pred = code_predictor.forward(hidden_state)
            current_token = sample(pred.logits)
        
        past_key_values = out.past_key_values
    
    return codec_tokens
```

### Что происходит внутри `talker.forward()` каждый шаг:

```python
# transformers/models/modeling_utils.py — упрощённая схема forward с cache
def forward(self, input_ids, past_key_values=None, use_cache=True):
    hidden_states = self.embeddings(input_ids)
    
    for layer in self.layers:  # 28 слоёв
        # ДИНАМИЧЕСКОЕ выделение памяти для KV
        if past_key_values is not None:
            k = torch.cat([past_key_values[layer_idx].key, current_k], dim=1)
            v = torch.cat([past_key_values[layer_idx].value, current_v], dim=1)
            # ↑ КАЖДЫЙ ШАГ — новый тензор, новая аллокация!
        
        # Attention с динамическими размерами
        attn_output = self.attention(
            q=query, k=key, v=value,
            attention_mask=dynamic_attention_mask  # ← меняется каждый шаг
        )
    
    return BaseModelOutput(past_key_values=new_past_kv)
```

---

## 3. Где теряется время — 4 узких места

### Узкое место #1: Динамический KV Cache (~40ms/step)

**Проблема:** на каждом шаге `past_key_values` растёт на 1 позицию. PyTorch должен:
- Выделить новый тензор большего размера
- Скопировать старые значения
- Обновить attention mask

```python
# Каждый шаг — динамическое выделение:
step_0: key = [K₀]                    # размер 1
step_1: key = concat([K₀, K₁])        # размер 2 → аллокация!
step_2: key = concat([K₀, K₁, K₂])    # размер 3 → аллокация!
...
step_50: key = concat([...])          # размер 51 → аллокация!

# Каждый cat() — это новый тензор на GPU памяти.
# На 28 слоях × 4 heads × 128 dim = ~36KB на слой.
# 28 слоёв × 2 (key+value) = ~720KB аллокации/шаг.
```

### Узкое место #2: Python → GPU dispatcher (~50ms/step)

**Проблема:** каждый forward-проход проходит через Python runtime:

```
Python call → PyTorch dispatcher → Triton kernel compilation → CUDA launch
     ↑                                    ↑                      ↑
  ~10ms                              ~30ms (первый раз!)      ~5ms
```

PyTorch — **eager execution framework**. Каждый оператор выполняется отдельно, и каждый вызов проходит через Python:

```python
# Каждый forward = сотни отдельных CUDA kernel launches:
layer_0.attention.q_proj.forward()     # → kernel launch 1
layer_0.attention.k_proj.forward()     # → kernel launch 2
layer_0.attention.v_proj.forward()     # → kernel launch 3
layer_0.attention.sdpa.forward()       # → kernel launch 4
layer_0.mlp.gate_proj.forward()        # → kernel launch 5
# ... × 28 слоёв = ~150+ kernel launches на шаг!

# Каждый launch = Python → C++ → CUDA driver ≈ 3-5 микросекунд
# 150 × 4μs = 600μs только на launch overhead
```

### Узкое место #3: HF `generate()` цикл (~20ms/step)

**Проблема:** универсальный цикл генерации с кучей проверок:

```python
# transformers/generation/utils.py — реальный код (упрощённо)
def generate(self, inputs, generation_config, **kwargs):
    # 1. Валидация параметров (~5ms)
    self._validate_model_class()
    self._validate_assistant_dummy_inputs()
    
    # 2. Подготовка (~3ms)
    logits_processor = self._get_logits_processor(...)
    stopping_criteria = self._get_stopping_criteria(...)
    
    # 3. Основной цикл — на КАЖДЫЙ шаг:
    for i in range(max_new_tokens):
        outputs = self(inputs, past_key_values=past_kv)  # forward
        
        # Куча проверок на каждый токен:
        scores = self.compute_scores(inputs, outputs)
        scores = logits_processor(inputs, scores)       # top_k, top_p, temperature...
        
        for criteria in stopping_criteria:              # проверка EOS
            if criteria(outputs, scores):
                break
        
        next_tokens = self.sample(scores, ...)          # сэмплинг
        past_kv = outputs.past_key_values
        
        # Ещё проверки:
        if generation_config.force_words_ids is not None: ...
        if generation_config.bad_words_ids is not None: ...
```

### Узкое место #4: Code Predictor подцикл (~30ms/step)

**Проблема:** 15 codebook-групп генерируются последовательно внутри каждого шага talker'а:

```python
# Каждый шаг декодирования:
for step in range(max_new_tokens):  # ~130 шагов
    # Talker forward → codebook 0
    cb_0 = talker.forward(...)
    
    # Code predictor — 14 последовательных проходов!
    for group in range(1, 15):
        hidden = predictor.forward(cb_{group-1})
        cb_{group} = sample(hidden)
    
    # Итого: 1 (talker) + 14 (predictor) = 15 forward на шаг
    # × 130 шагов = ~2080 forward-проходов!
```

### Сводка по оверхеду

| Компонент | Время/шаг | % от общего |
|-----------|-----------|-------------|
| Чистое GPU вычисление | ~60ms | 24% |
| Динамический KV cache (alloc + cat) | ~40ms | 16% |
| Python dispatcher (kernel launches × 150+) | ~50ms | 20% |
| HF generate() overhead (проверки, валидация) | ~20ms | 8% |
| Code predictor подцикл (14× forward) | ~30ms | 12% |
| CUDA kernel compilation (Triton, первый раз) | ~50ms | 20% |
| **Итого** | **~250ms/step** | **100%** |

---

## 4. CUDA Graphs — как это работает

### Идея

Вместо того чтобы каждый шаг выполнять forward-проход через Python, мы:
1. **Один раз** выполняем forward и **записываем** всю последовательность CUDA-операций
2. На каждом шаге просто **replay** записанного графа — GPU проигрывает его напрямую

```
┌─────────────────────────────────────────────────────┐
│  Первый раз (capture):                              │
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
│  Каждый последующий шаг (replay):                   │
│                                                     │
│  cudaGraphLaunch(graph, stream)                     │
│    ↓                                                │
│  GPU проигрывает граф напрямую                      │
│  (без Python! без dispatcher'а!)                    │
└─────────────────────────────────────────────────────┘
```

### Статический KV Cache vs Динамический

**Динамический (стандарт):**
```python
# Каждый шаг — новый тензор:
past_key = torch.cat([old_key, new_key], dim=1)  # alloc + copy!
```

**Статический (CUDA graphs):**
```python
# Один раз выделили фиксированный буфер:
static_cache = torch.zeros(1, num_heads, max_seq_len, head_dim, device='cuda')

# Каждый шаг — запись в фиксированную позицию:
step = 50
static_cache[:, :, step, :] = new_key_value  # просто write!
```

### Как захватить CUDA Graph

```python
import torch

# 1. Подготавливаем статические буферы
static_kv_cache = torch.zeros(1, 28, 4, 2048, 128, device='cuda')  # фиксированный размер

# 2. Начинаем захват
stream = torch.cuda.Stream()
stream.begin_capture()

# 3. Выполняем один forward (записывается в граф)
hidden = model.forward(
    inputs_embeds=input_placeholder,   # статический тензор
    cache_position=position_placeholder,  # статическая позиция
    static_cache=static_kv_cache,       # статический буфер
)
logits = codec_head(hidden[:, -1])

# 4. Завершаем захват → получаем граф
graph = stream.end_capture()

# 5. Replay — без Python!
for step in range(max_new_tokens):
    input_placeholder[:] = new_input     # обновляем вход
    position_placeholder[:] = step       # обновляем позицию
    cudaGraphLaunch(graph, stream)       # GPU replay!
    new_token = sample(logits[:, -1])    # только сэмплинг через Python
```

---

## 5. Реализация FasterQwen3TTS (быстро)

### Архитектура оптимизации

```
┌───────────────────────────────────────────────────────┐
│              FasterQwen3TTS                           │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │           PredictorGraph                        │  │
│  │                                                 │  │
│  │  CUDA Graph:                                    │  │
│  │    input_embed → predictor.forward() × 15       │  │
│  │    (все 15 codebook-групп в одном графе!)       │  │
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

### PredictorGraph — захват кода предиктора

```python
# faster_qwen3_tts/predictor_graph.py (упрощённо)
class PredictorGraph:
    def __init__(self, predictor_model, config, device='cuda', dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        
        # Статические буферы для ввода/вывода
        self.input_buf = torch.zeros(1, 2, config.hidden_size, 
                                      device=device, dtype=dtype)
        self.output_ids = torch.zeros(15, dtype=torch.long, device=device)
        
        self.graph = None
    
    def capture(self, num_warmup=3):
        # Warmup — компилируем Triton-ядра один раз
        for _ in range(num_warmup):
            self._forward_pass()
        
        # Захват графа
        stream = torch.cuda.Stream()
        stream.begin_capture()
        self._forward_pass()
        self.graph = stream.end_capture()
    
    def _forward_pass(self):
        # Один forward predictor'а — 15 codebook-групп
        hidden = self.predictor.forward(self.input_buf)
        # sampling → self.output_ids
    
    def run(self, input_data):
        # Обновляем входной буфер
        self.input_buf.copy_(input_data)
        
        # Replay графа — без Python!
        torch.cuda.graph_replay(self.graph)  # ← это всё!
        
        return self.output_ids.clone()
```

### TalkerGraph — захват декодера с статическим KV cache

```python
# faster_qwen3_tts/talker_graph.py (упрощённо)
class TalkerGraph:
    def __init__(self, talker_model, config, max_seq_len=2048):
        self.max_seq_len = max_seq_len
        
        # Статический KV cache — один раз на всю жизнь
        num_layers = config.num_hidden_layers   # 28
        num_heads = config.num_attention_heads   # 4
        head_dim = config.hidden_size // num_heads  # 128
        
        self.key_cache = torch.zeros(
            1, num_layers, num_heads, max_seq_len, head_dim,
            device='cuda', dtype=torch.bfloat16
        )
        self.value_cache = torch.zeros_like(self.key_cache)
        
        # Статический буфер вывода
        self.output_buf = torch.zeros(
            1, 1, config.hidden_size,
            device='cuda', dtype=torch.bfloat16
        )
        
        # Position buffer — обновляем каждый шаг
        self.position_buf = torch.tensor([0], device='cuda')
        
        self.graph = None
    
    def prefill_kv(self, dynamic_past_kv):
        """Копируем prefill KV в статический буфер."""
        for layer_idx in range(num_layers):
            k_len = dynamic_past_kv[layer_idx].key.shape[2]
            self.key_cache[:, layer_idx, :, :k_len, :] = dynamic_past_kv[layer_idx].key
            self.value_cache[:, layer_idx, :, :k_len, :] = dynamic_past_kv[layer_idx].value
        return k_len
    
    def capture(self, prefill_len=100, num_warmup=3):
        # Warmup
        for _ in range(num_warmup):
            self._decode_step(prefill_len + 5)
        
        # Захват графа декодирования
        stream = torch.cuda.Stream()
        stream.begin_capture()
        self._decode_step(prefill_len + 10)
        self.graph = stream.end_capture()
    
    def _decode_step(self, position):
        """Один шаг декодирования со статическим KV cache."""
        # Talker forward с фиксированными буферами
        hidden = talker.forward(
            inputs_embeds=self.input_buf,
            static_kv_cache=(self.key_cache, self.value_cache),
            position=position,  # ← где писать новый KV
            output_hidden=True,   # → в self.output_buf
        )
    
    def run(self, input_embed, position):
        """Replay декодирования."""
        self.input_buf.copy_(input_embed)
        self.position_buf[0] = position
        
        torch.cuda.graph_replay(self.graph)  # ← GPU replay!
        
        return self.output_buf.clone()
```

### Основной цикл генерации (быстрый путь)

```python
# faster_qwen3_tts/generate.py — fast_generate()
@torch.inference_mode()
def fast_generate(talker, talker_input_embeds, ..., predictor_graph, talker_graph):
    # === PREFILL (один раз через обычный forward) ===
    out = talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,  # динамический cache только для prefill!
    )
    
    # Копируем prefill KV в статический буфер
    prefill_len = talker_graph.prefill_kv(out.past_key_values)
    
    # Первый токен — обычный forward (для инициализации)
    logits = out.logits[:, -1]
    token = sample(logits)
    
    # === DECODE LOOP (CUDA graphs!) ===
    all_codec_ids = []
    
    for step_idx in range(max_new_tokens):
        if token.item() == eos_id:
            break
        
        # 1. Predictor — replay графа (~2ms!)
        last_hidden = talker.get_input_embeddings()(token)
        pred_input = torch.cat([past_hidden, last_hidden], dim=1)
        codebook_ids = predictor_graph.run(pred_input)  # ← CUDA graph!
        
        # 2. Собираем полный codec token [cb0, cb1, ..., cb15]
        all_cb = torch.cat([token.view(1), codebook_ids])
        all_codec_ids.append(all_cb.detach())
        
        # 3. Строим input embedding для talker
        inputs_embeds = build_codec_embedding(codebook_ids)
        
        # 4. Talker decode — replay графа (~29ms!)
        current_pos = prefill_len + step_idx
        hidden = talker_graph.run(inputs_embeds, position=current_pos)  # ← CUDA graph!
        
        # 5. Логиты и сэмплинг (быстро, маленький тензор)
        logits = talker.codec_head(hidden[:, -1])
        token = sample(logits)
    
    return torch.stack(all_codec_ids), timing
```

---

## 6. Сравнение кода: до и после

### До — стандартный путь (~250ms/step)

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
    
    # ← Один вызов — внутри 130 шагов с динамическим KV cache
    talker_codes_list, _ = self.model.model.generate(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=[language],
        speakers=[speaker],
        non_streaming_mode=True,
        **gen_kwargs,
    )
    
    # ← Chunked decode (медленно — каждый чанк отдельный forward)
    decoder = self.model.model.speech_tokenizer.model.decoder
    for start in range(0, seq_len, 50):
        wav = decoder.chunked_decode(codes[start:end], chunk_size=50)
    
    return final_wav
```

**Что происходит внутри `model.model.generate()`:**
- ~130 итераций decode loop
- Каждая итерация: talker.forward() + 14× code_predictor.forward()
- Динамический KV cache растёт каждый шаг
- Python dispatcher на каждом forward (~150 kernel launches)

### После — CUDA graphs путь (~31ms/step)

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
        yield audio_chunk  # ← звук появляется каждые ~300ms!
```

**Что происходит внутри `generate_custom_voice_streaming()`:**
- Один prefill (динамический forward) → KV копируется в статический буфер
- Decode loop: predictor_graph.run() + talker_graph.run() — CUDA graph replay
- На каждый чанк (8 шагов): ~250ms генерации → декодирование → аудио

---

## 7. Бенчмарк на RTX 5060 Ti

### Конфигурация

| Параметр | Значение |
|----------|---------|
| GPU | NVIDIA GeForce RTX 5060 Ti (16GB) |
| PyTorch | 2.13.0+cu132 |
| CUDA | 13.2 |
| Модель | Qwen3-TTS-12Hz-0.6B-CustomVoice |
| Спикер | Sohee (Russian) |
| Текст | ~4 предложения, ~130 codec токенов |

### Результаты

| Метрика | Стандарт (plain) | CUDA Graphs | Ускорение |
|---------|-----------------|-------------|-----------|
| **ms/step** | 247.7ms | 31.3ms | **7.9x** |
| **RTF** | 0.336 (в 3 раза медленнее реалтайма) | 2.67 (в 2.7 раза быстрее!) | **8.0x** |
| **Время на 11s аудио** | 33.1s | 4.2s | **7.9x** |
| **TTFA (первый звук)** | N/A | 355ms | — |

### Streaming с сегментацией (v10)

| Сегмент | Генерация | TTFA | Аудио | Чанков | RTF |
|---------|-----------|------|-------|--------|-----|
| Меня зовут Александр. | 1376ms | 466ms | 2.48s | 4 | 1.80 |
| Мне двадцать пять лет. | 1170ms | 389ms | 2.48s | 4 | 2.12 |
| Я живу в Санкт-Петербурге. | 1420ms | 372ms | 2.96s | 5 | 2.08 |

**Итого:** 6.4s аудио за 8.0s wall time (RTF ~0.8 с учётом префиллов сегментов).

### Сравнение v9 vs v10

| Метрика | v9 (plain) | v10 (CUDA graphs) | Ускорение |
|---------|-----------|-------------------|-----------|
| Общее время | 29,990ms | 8,042ms | **3.7x** |
| Генерация/сегмент | ~9s | ~1.3s | **~7x** |
| Underruns плеера | 350 | 1 | **350x меньше** |

---

## 8. Почему torch.compile не помог

### Попытка с torch.compile (v8)

```python
# streaming_tts_v8.py — попытка compile
self.model.model = torch.compile(self.model.model, mode='reduce-overhead')
if hasattr(self.model.model, 'talker'):
    self.model.model.talker = torch.compile(self.model.model.talker, mode='reduce-overhead')
```

### Почему не сработало

**Причина 1: Динамические размеры**

torch.compile работает лучше всего с фиксированными размерами тензоров. Qwen3-TTS имеет:
- KV cache растёт каждый шаг (shape меняется)
- Attention mask меняется каждый шаг
- Code predictor получает разные hidden states

```python
# Каждый шаг — разный shape:
step_0: past_key.shape = [1, 4, 1, 128]
step_1: past_key.shape = [1, 4, 2, 128]  ← новый shape → recompilation!
step_2: past_key.shape = [1, 4, 3, 128]  ← новый shape → recompilation!
```

torch.compile пытается **recompile** на каждый новый shape — это медленнее, чем не компилировать вообще.

**Причина 2: Code predictor подцикл**

15 последовательных forward-проходов code_predictor'а с разными input shapes (разные codebook группы) — compile не может захватить этот паттерн эффективно.

**Причина 3: Python overhead в HF generate()**

torch.compile компилирует **только PyTorch ops**, но не оптимизирует:
- Проверки stopping criteria
- Logits processing (top_k, top_p)
- Dynamic cache management
- Attention mask updates

Эти операции занимают ~30% времени и находятся вне compiled graph.

### CUDA Graphs vs torch.compile

| Аспект | torch.compile | CUDA Graphs |
|--------|--------------|-------------|
| Что оптимизирует | PyTorch ops внутри forward() | Весь граф: Python → GPU |
| Динамические размеры | Проблемы (recompilation) | Не проблема (статический буфер) |
| Kernel launch overhead | Уменьшает (fusion) | Полностью устраняет |
| Python dispatcher | Нет | Полностью устраняет |
| Memory allocation | Нет | Статические буферы |
| Ускорение на TTS | ~1.2x (или даже замедление) | **~8x** |

---

## Выводы

### Почему CUDA Graphs — правильный подход для autoregressive generation

1. **Фиксированный размер шага** — каждый decode step имеет одинаковый input/output shape
2. **Предсказуемый граф** — последовательность операций не меняется между шагами
3. **Статический KV cache** — один раз выделили, потом просто пишем в нужную позицию
4. **Устранение Python overhead** — replay графа = 1 CUDA вызов вместо ~150 kernel launches

### Когда НЕ использовать CUDA Graphs

- Переменные input sizes (batch size меняется)
- Control flow зависит от данных (if/else внутри forward)
- Один-два прохода (warmup/capture дороже, чем просто выполнить)
- Очень длинные последовательности (> max_seq_len графа)

### Для Qwen3-TTS это идеальный кейс

- Фиксированный batch size = 1
- Каждый decode step — одинаковый input shape [1, 1, hidden]
- ~130 шагов на сегмент → окупаемость capture мгновенная
- max_seq_len = 2048 → покрывает любые разумные тексты
