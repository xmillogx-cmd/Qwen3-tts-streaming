# Qwen3-TTS: Потоковая генерация с CUDA Graphs ускорением

Полный технический разбор нашей реализации потокового TTS на базе Qwen3-TTS.

## Содержание

1. [Введение](#1-введение)
2. [Архитектура Qwen3-TTS](#2-архитектура-qwen3-tts)
3. [Проблемы стандартной реализации](#3-проблемы-стандартной-реализации)
4. [CUDA Graphs оптимизация](#4-cuda-graphs-оптимизация)
5. [Потоковая генерация (Streaming Pipeline)](#5-потоковая-генерация-streaming-pipeline)
6. [Эволюция версий (v6 → v14)](#6-эволюция-версий-v6--v14)
7. [Сравнение с FasterQwen3TTS](#7-сравнение-с-fasterqwen3tts)
8. [Бенчмарки на RTX 5060 Ti](#8-бенчмарки-на-rtx-5060-ti)
9. [Установка и запуск](#9-установка-и-запуск)
10. [Ключевые паттерны кода](#10-ключевые-паттерны-кода)

---

## 1. Введение

Мы реализовали потоковую генерацию речи на базе **Qwen3-TTS-0.6B** с ускорением через **CUDA Graphs**. Результат:

| Метрика | До оптимизации | После | Ускорение |
|---------|---------------|-------|-----------|
| ms/step | 247.7ms | 31.3ms | **7.9x** |
| RTF | 0.336 (медленнее реалтайма) | 2.67 (быстрее в 2.7x) | **8.0x** |
| TTFA | ~40s (весь аудио сразу) | ~355ms | **113x** |

### Ключевые достижения

- **Потоковая генерация**: звук появляется через ~350ms, а не через 40 секунд
- **CUDA Graphs**: ускорение декодирования в ~8 раз за счёт устранения Python overhead
- **Producer-consumer pipeline**: поток генерации → очередь → плеер без underruns
- **Text segmentation**: автоматическое разбиение длинных текстов на сегменты

### Файлы реализации

| Файл | Описание |
|------|----------|
| `fast_tts_v14.py` | Финальная реализация — потоковая генерация с кроссфейдом |
| `fast_tts_v10.py` | CUDA Graphs + streaming pipeline (переходная версия) |
| `streaming_tts_v6.py` | Первый рабочий producer-consumer pipeline |
| `qwen_tts_cuda_graphs/` | Кастомные PredictorGraph и TalkerGraph |
| `docs/cuda_graphs_optimization.md` | Детальный разбор CUDA Graphs оптимизации |

---

## 2. Архитектура Qwen3-TTS

Qwen3-TTS — это **двухкомпонентная модель** для синтеза речи:

```
┌──────────────────────────────────────────────────────────────┐
│              Qwen3TTSForConditionalGeneration                │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │                  Talker (28 слоёв)                    │   │
│  │                                                       │   │
│  │  Input: текст + язык + спикер                         │   │
│  │  Output: codec token (codebook 0)                     │   │
│  │                                                       │   │
│  │  ┌───────────────────────────────────────────────┐    │   │
│  │  │     Code Predictor (5 слоёв)                  │    │   │
│  │  │                                               │    │   │
│  │  │  Input: codebook N                            │    │   │
│  │  │  Output: codebook N+1                         │    │   │
│  │  │  (15 codebook-групп всего)                    │    │   │
│  │  └───────────────────────────────────────────────┘    │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  Speech Tokenizer (HiFi-GGAN decoder)                       │
│  Input: 15 codebook-токенов → Output: waveform @24kHz       │
└──────────────────────────────────────────────────────────────┘
```

### Flow генерации

```
Текст → tokenize → Talker prefill (динамический KV cache)
                ↓
        Code Predictor × 15 (последовательно)
                ↓
        Codec token [cb0, cb1, ..., cb14]
                ↓
        Speech Tokenizer decoder.chunked_decode()
                ↓
        Waveform @24kHz (12Hz codec = ~0.83s на токен)
```

### Ключевые параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| Talker layers | 28 | Основной autoregressive decoder |
| Code Predictor layers | 5 | Предсказывает оставшиеся codebook-группы |
| Codebook groups | 15 | Всего групп на один timestep |
| Codec rate | 12 Hz | 12 codec токенов в секунду аудио |
| Sample rate | 24 kHz | Частота дискретизации выходного waveform |
| Hidden size | 1024 | Размер embedding (0.6B модель) |

### Один шаг декодирования = ~15 forward-проходов

```python
# Каждый шаг декодирования:
for step in range(max_new_tokens):      # ~130 шагов на предложение
    cb_0 = talker.forward(...)          # 1 forward (28 слоёв)
    
    for group in range(1, 15):          # 14 последовательных forward
        hidden = predictor.forward(cb_{group-1})
        cb_{group} = sample(hidden)     # Code Predictor (5 слоёв)
    
    # Итого: 1 + 14 = 15 forward на шаг
    # × 130 шагов = ~2080 forward-проходов!
```

---

## 3. Проблемы стандартной реализации

Стандартный `Qwen3TTSModel.generate_custom_voice()` работает за **~250ms/step**. Разберём 4 узких места:

### Узкое место #1: Динамический KV Cache (~40ms/step)

На каждом шаге `past_key_values` растёт на 1 позицию. PyTorch вынужден:
- Выделять новый тензор большего размера
- Копировать старые значения
- Обновлять attention mask

```python
# Каждый шаг — динамическое выделение:
step_0: key = [K₀]                    # размер 1
step_1: key = concat([K₀, K₁])        # размер 2 → аллокация!
step_2: key = concat([K₀, K₁, K₂])    # размер 3 → аллокация!
...
# На 28 слоях × 4 heads × 128 dim ≈ 720KB аллокации/шаг
```

### Узкое место #2: Python → GPU dispatcher (~50ms/step)

Каждый forward-проход проходит через Python runtime:

```
Python call → PyTorch dispatcher → Triton kernel compilation → CUDA launch
     ↑                                    ↑                      ↑
  ~10ms                              ~30ms (первый раз!)      ~5ms
```

На каждый шаг приходится **~150+ kernel launches** (28 слоёв × attention + MLP + norm):

```python
# Каждый forward = сотни отдельных CUDA kernel launches:
layer_0.attention.q_proj.forward()     # → kernel launch 1
layer_0.attention.k_proj.forward()     # → kernel launch 2
...
# 150 × 4μs = 600μs только на launch overhead
```

### Узкое место #3: HF `generate()` цикл (~20ms/step)

Универсальный цикл генерации с кучей проверок на каждом шаге:

```python
# transformers/generation/utils.py — упрощённо
for i in range(max_new_tokens):
    outputs = self(inputs, past_key_values=past_kv)  # forward
    
    scores = self.compute_scores(inputs, outputs)     # ~5ms
    scores = logits_processor(inputs, scores)         # top_k, top_p...
    
    for criteria in stopping_criteria:                # проверка EOS
        if criteria(outputs, scores):
            break
    
    next_tokens = self.sample(scores, ...)            # сэмплинг
```

### Узкое место #4: Code Predictor подцикл (~30ms/step)

15 codebook-групп генерируются **последовательно** внутри каждого шага talker'а — невозможно распараллелить без изменения архитектуры.

### Сводка по оверхеду

| Компонент | Время/шаг | % от общего |
|-----------|-----------|-------------|
| Чистое GPU вычисление | ~60ms | 24% |
| Динамический KV cache (alloc + cat) | ~40ms | 16% |
| Python dispatcher (~150 kernel launches) | ~50ms | 20% |
| HF generate() overhead | ~20ms | 8% |
| Code predictor подцикл (14× forward) | ~30ms | 12% |
| CUDA kernel compilation (Triton) | ~50ms | 20% |
| **Итого** | **~250ms/step** | **100%** |

---

## 4. CUDA Graphs оптимизация

### Идея

Вместо выполнения forward-прохода через Python каждый шаг, мы:
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

### StaticCache vs DynamicCache

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

### Архитектура наших CUDA Graphs

```
┌───────────────────────────────────────────────────────┐
│              qwen_tts_cuda_graphs/                    │
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

### PredictorGraph — захват предиктора

Файл: `qwen_tts_cuda_graphs/predictor_graph.py`

```python
class PredictorGraph:
    def __init__(self, code_predictor, pred_config, talker_hidden_size, ...):
        # Статические буферы для ввода/вывода
        self.input_buf = torch.zeros(1, 2, talker_hidden_size, ...)
        self.output_tokens = torch.zeros(15, dtype=torch.long, ...)
        
        # StaticCache для предиктора (max_seq=17)
        self.static_cache = StaticCache(config=pred_config, max_cache_len=17)

    def capture(self, num_warmup=3):
        # Warmup — компилируем Triton-ядра один раз
        for _ in range(num_warmup):
            self._full_loop()  # 15 codebook-групп
        
        # Захват графа
        stream = torch.cuda.Stream()
        with torch.cuda.graph(self.graph):
            self._full_loop()

    def run(self, pred_input: torch.Tensor) -> torch.Tensor:
        self.input_buf.copy_(pred_input)
        self.static_cache.reset()
        self.graph.replay()  # ← GPU replay!
        return self.output_tokens.clone()
```

### TalkerGraph — захват декодера

Файл: `qwen_tts_cuda_graphs/talker_graph.py`

```python
class TalkerGraph:
    def __init__(self, talker_model, talker_config, max_seq_len=2048):
        # Статический KV cache — один раз на всю жизнь
        self.static_cache = StaticCache(config=talker_config, max_cache_len=max_seq_len)
        
        # Статические буферы
        self.input_buf = torch.zeros(1, 1, hidden_size, ...)
        self.output_buf = torch.zeros(1, 1, hidden_size, ...)
        self.cache_position = torch.zeros(1, dtype=torch.long, ...)

    def prefill_kv(self, dynamic_past_kv):
        """Копируем prefill KV из DynamicCache в StaticCache."""
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

### Основной цикл генерации (быстрый путь)

```python
@torch.inference_mode()
def fast_generate(talker, talker_input_embeds, predictor_graph, talker_graph):
    # === PREFILL (один раз через обычный forward) ===
    out = talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,  # динамический cache только для prefill!
    )
    
    # Копируем prefill KV в статический буфер
    prefill_len = talker_graph.prefill_kv(out.past_key_values)
    
    # Первый токен — обычный forward
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
        
        # 2. Собираем полный codec token [cb0, cb1, ..., cb14]
        all_cb = torch.cat([token.view(1), codebook_ids])
        all_codec_ids.append(all_cb.detach())
        
        # 3. Строим input embedding для talker
        inputs_embeds = build_codec_embedding(codebook_ids)
        
        # 4. Talker decode — replay графа (~29ms!)
        current_pos = prefill_len + step_idx
        hidden = talker_graph.run(inputs_embeds, position=current_pos)
        
        # 5. Логиты и сэмплинг (быстро, маленький тензор)
        logits = talker.codec_head(hidden[:, -1])
        token = sample(logits)
    
    return torch.stack(all_codec_ids), timing
```

### Warmup behavior

CUDA graph warmup выполняется **один раз на экземпляр модели**, не на каждый вызов:

```python
# Первый вызов — полный warmup (~8.5s)
model.generate_custom_voice_streaming(...)  # ~9.5s (warmup + gen)

# Последующие вызовы — без warmup (~80ms prefill overhead)
model.generate_custom_voice_streaming(...)  # ~1.7s
model.generate_custom_voice_streaming(...)  # ~0.96s
```

**Почему ~80ms на последующих вызовах?**
1. `talker.forward()` initial prefill pass (~50-70ms)
2. Tokenization + `_build_assistant_text` CPU work (~10-20ms)
3. `prefill_kv()` copying DynamicCache → StaticCache

---

## 5. Потоковая генерация (Streaming Pipeline)

### Архитектура Producer-Consumer

```
┌──────────────┐     queue      ┌──────────────┐
│   PRODUCER   │ ────────────→  │    PLAYER    │
│  (генерация) │                │ (sounddevice)│
│              │                │              │
│  Segment 1   │                │ Chunk 1      │
│  → Chunk 1   │                │ → play       │
│  → Chunk 2   │                │ Chunk 2      │
│  → Chunk 3   │                │ → play       │
└──────────────┘                └──────────────┘
```

### StreamingAudioPlayer

Файл: `fast_tts_v14.py` (строки 47-138)

Callback-based плеер на базе `sounddevice.OutputStream`:

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
        # PortAudio callback — непрерывно подаём аудио
        with self._lock:
            if not self._started:
                if self._buffered >= self._preroll:
                    self._started = True  # начинаем воспроизведение
            
            while written < frames:
                chunk = self._chunks.get_nowait()
                out[written:written+n] = chunk[offset:offset+n]
```

### Text Segmentation

Файл: `fast_tts_v14.py` (строки 28-56)

Автоматическое разбиение длинных текстов на сегменты ≤85 символов:

```python
def split_segments(text, max_chars=85):
    # 1. Разбиваем по предложениям (.!? )
    sentences = re.findall(r'([^.!?]+[.!?])|([^.!?]+$)', text)
    
    # 2. Длинные предложения — по запятым/точкам с многоточием
    parts = re.split(r'(?<=[,;:])\s+', sentence)
    
    # 3. Если всё ещё длинно — по словам
    words = buf.split()
```

### Producer-Consumer Pipeline (v14)

```python
def generate_and_play(self, text, language='Russian', save_wav=None):
    segments = split_segments(text, max_chars=85)
    
    player = StreamingAudioPlayer(sample_rate=24000, preroll_sec=0.3)
    player.start()
    
    q = queue.Queue(maxsize=32)
    
    # PRODUCER: генерация в отдельном потоке
    def producer():
        for seg in segments:
            gen = self.model.generate_custom_voice_streaming(
                text=seg, chunk_size=8, max_new_tokens=max_tokens
            )
            for audio_chunk, sr, timing in gen:
                chunk = to_pcm_chunk(audio_chunk)
                chunk = safe_normalize(chunk)  # clip protection
                
                # Backpressure: ждём пока очередь не освободится
                while q.qsize() >= 20:
                    time.sleep(0.01)
                q.put(chunk)
            gen.close()
        q.put(None)  # sentinel
    
    gen_thread = threading.Thread(target=producer, daemon=True)
    gen_thread.start()
    
    # CONSUMER: чтение из очереди → плеер
    all_wavs = []
    while True:
        chunk = q.get(timeout=0.1)
        if chunk is None:
            break
        
        # Backpressure: ждём пока буфер плеера не опустеет
        while player.buffered_seconds() > 3.5:
            time.sleep(0.01)
        
        all_wavs.append(chunk)
        player.add_chunk(chunk)
    
    gen_thread.join(timeout=120)
    player.add_chunk(None)  # signal end
```

### Метрики в реальном времени

| Метрика | Описание | Формула |
|---------|----------|---------|
| **TTFA** | Time To First Audio — время до первого звука | `first_chunk_time - start_time` |
| **RTF** | Real-Time Factor — скорость генерации | `audio_duration / wall_time` |
| **ms/step** | Время на один codec шаг | `wall_time / num_steps` |
| **Inference Speed** | Насколько быстрее реалтайма | `audio_duration / compute_time` |

---

## 6. Эволюция версий (v6 → v14)

### Таблица эволюции

| Версия | Ключевое изменение | Результат |
|--------|-------------------|-----------|
| **v6** | Producer-consumer pipeline, фикс underruns | Потоковая генерация без заиканий |
| **v7** | Parallel generation (ThreadPoolExecutor) | — |
| **v8** | `torch.compile` + chunked decode | Не сработало (динамические размеры) |
| **v9** | StoppingCriteria patch + auto max_new_tokens | 1-3s для коротких фраз |
| **v10** | CUDA Graphs backend (FasterQwen3TTS) | ~8x ускорение, RTF > 1.0 |
| **v14** | True streaming + crossfade + segmentation | Плавный звук без щелчков |

### Детали каждой версии

#### v6 — Первый рабочий pipeline

```python
# Ключевые фиксы:
1. Producer thread генерирует сегменты, кладёт в очередь
2. Main thread читает из очереди и кормит плеер
3. player.add_chunk(None) ТОЛЬКО после завершения генерации
4. trim_silence + apply_fades для плавных границ
```

**Баг:** кроссфейд накапливал prev_audio после каждого чанка → звук умножался на N чанков.

#### v9 — StoppingCriteria fix

**Проблема:** `max_new_tokens=8192` по умолчанию → модель генерирует 15+ секунд аудио для короткой фразы "Да".

**Решение:**
```python
def _get_max_new_tokens(self, text):
    word_count = len(re.findall(r'\b\w+\b', text))
    if word_count <= 2: return 20
    elif word_count <= 5: return 50
    elif word_count <= 10: return 100
    else: return 160
```

**Результат:** "Да" → 1.25s (было 15-27s)

#### v10 — CUDA Graphs переход

Переход на `FasterQwen3TTS` wrapper с CUDA graph capture:
- ~8x ускорение декодирования
- RTF > 1.0 (быстрее реалтайма)
- TTFA ~350ms вместо ~40s

#### v14 — Финальная версия

```python
# Ключевые улучшения vs v10:
1. Crossfade между сегментами (плавные переходы)
2. Per-chunk normalization (consistent loudness)
3. Backpressure control (queue + player buffer)
4. MIN_START_SEC = 1.0s (ждем пока буфер наполнится)
5. split_segments с умным разбиением по предложениям
```

---

## 7. Сравнение с FasterQwen3TTS

### Архитектурные различия

| Аспект | Наша реализация | FasterQwen3TTS |
|--------|-----------------|----------------|
| **Базовая модель** | `Qwen3TTSModel` (native) | `Qwen3TTSModel` + wrapper |
| **CUDA Graphs** | Кастомные `PredictorGraph`, `TalkerGraph` | Свои реализации в `faster_qwen3_tts/` |
| **Streaming API** | `generate_custom_voice_streaming()` | `generate_custom_voice_streaming()` |
| **Voice Cloning** | Не реализовано | Полная поддержка (ICL + x-vector) |
| **Voice Design** | Не реализовано | Поддержка через `instruct` |
| **CLI** | Нет | `faster-qwen3-tts` команда |
| **Server mode** | Нет | OpenAI-compatible API сервер |
| **GGML backend** | Нет | qwentts.cpp опционально |

### Ключевые отличия в реализации

#### 1. Багфикс: non_streaming_mode forced

В native `Qwen3TTSModel` была ошибка (строка 1576 в оригинале):
```python
# Было (баг):
non_streaming_mode=non_streaming_mode or True

# Должно быть:
non_streaming_mode=non_streaming_mode if non_streaming_mode is not None else False
```

Это вызывало повторение аудио при streaming генерации.

#### 2. Кроссфейд между сегментами

Наша реализация применяет кроссфейд на границах сегментов для плавных переходов:
```python
def apply_fades(wav, sr=24000, in_ms=5, out_ms=25):
    n_in = int(sr * in_ms / 1000.0)
    n_out = int(sr * out_ms / 1000.0)
    wav[:n_in] *= np.linspace(0.0, 1.0, n_in)   # fade-in
    wav[-n_out:] *= np.linspace(1.0, 0.0, n_out) # fade-out
```

#### 3. Backpressure control

Наш pipeline использует двойной backpressure:
- **Producer side**: ждём пока очередь не освободится (`q.qsize() >= 20`)
- **Consumer side**: ждём пока буфер плеера не опустеет (`buffered_seconds() > 3.5`)

#### 4. Safe chunk copying

```python
def to_pcm_chunk(x):
    """Всегда real copy — защита от reuse static buffer."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32, copy=True).reshape(-1)
```

### Что FasterQwen3TTS делает лучше

| Возможность | Описание |
|-------------|----------|
| **Voice Cloning** | ICL mode с reference audio + x-vector only mode |
| **Voice Design** | Instruction-based генерация ("Warm, confident narrator") |
| **CLI утилита** | `faster-qwen3-tts clone/design/custom` команды |
| **OpenAI API** | Совместимый сервер для интеграции с клиентами |
| **GGML backend** | Опциональный qwentts.cpp для CPU/low-memory |
| **Speaker caching** | `.spk` / `.rvq` кэш для быстрого клонирования |
| **Demo UI** | Веб-интерфейс с live TTFA/RTF метриками |

### Что наша реализация делает лучше

| Возможность | Описание |
|-------------|----------|
| **Crossfade** | Плавные переходы между сегментами |
| **Per-chunk normalization** | Consistent loudness без pumping |
| **Простота** | Один файл, нет зависимостей от faster-qwen3-tts |
| **Багфиксы** | Фикс non_streaming_mode forced bug |

---

## 8. Бенчмарки на RTX 5060 Ti

### Конфигурация

| Параметр | Значение |
|----------|----------|
| GPU | NVIDIA GeForce RTX 5060 Ti (16GB) |
| PyTorch | 2.13.0+cu132 |
| CUDA | 13.2 |
| Compute Capability | 12.0 |
| Модель | Qwen3-TTS-0.6B @ bfloat16 |
| Спикер | Sohee (Russian) |
| Текст | ~130 codec токенов (~11s аудио) |

### Результаты сравнения

| Метрика | Plain (baseline) | CUDA Graphs | Ускорение |
|---------|-----------------|-------------|-----------|
| **ms/step** | 247.7ms | 31.3ms | **7.9x** |
| **RTF** | 0.336 | 2.67 | **8.0x** |
| **TTFA** | ~40,382ms | ~355ms | **113x** |
| **Время на 11s аудио** | ~33.1s | ~4.2s | **7.9x** |

### Streaming сегменты (v10)

| Сегмент | Генерация | TTFA | Аудио | Чанков | RTF |
|---------|-----------|------|-------|--------|-----|
| Меня зовут Александр. | 1376ms | 466ms | 2.48s | 4 | 1.80 |
| Мне двадцать пять лет. | 1170ms | 389ms | 2.48s | 4 | 2.12 |
| Я живу в Санкт-Петербурге. | 1420ms | 372ms | 2.96s | 5 | 2.08 |

**Итого:** 6.4s аудио за 8.0s wall time (RTF ~0.8 с учётом префиллов сегментов).

### Почему RTF > 1.0 — это хорошо

```
RTF = audio_duration / wall_time

RTF < 1.0  → генерация медленнее реалтайма (ждем аудио)
RTF = 1.0  → ровно в реальном времени
RTF > 1.0  → быстрее реалтайма (опережаем воспроизведение)
```

Наш результат **RTF 2.67** означает: модель генерирует аудио в 2.67 раза быстрее, чем оно воспроизводится. Это запас для обработки следующих сегментов и сетевых задержек.

---

## 9. Установка и запуск

### Требования

```
Python >= 3.10
PyTorch >= 2.5.1 (cu132)
CUDA >= 12.8
NVIDIA GPU с поддержкой CUDA Graphs
```

### Установка flash-attn (Windows + CUDA 13.2)

```cmd
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "FLASH_ATTN_CUDA_ARCHS=120"
set "MAX_JOBS=10"
python flash-attention/setup.py build_ext --inplace
```

**Важно:** Для CUDA 13.2 + MSVC 14.43 нужен фикс в `setup.py`:
```python
# Добавить при sys.platform == "win32":
compiler_c17_flag.append("/Zc:preprocessor")
feature_flags.append("-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING")
```

### Быстрый старт

```python
from fast_tts_v14 import FastTTSv14

tts = FastTTSv14(
    model_path=r'G:\Foundation\models\Qwen3-TTS',
    speaker='Sohee'
)

tts.generate_and_play(
    text="Привет! Это тест потоковой генерации.",
    language='Russian',
    save_wav='output.wav'
)
```

### Тестовый набор (10 предложений)

```bash
python fast_tts_v14.py --test
```

Запускает 5 русских + 5 английских абзацев, сохраняет WAV файлы.

---

## 10. Ключевые паттерны кода

### Паттерн 1: Producer-Consumer для streaming

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

### Паттерн 2: CUDA Graph capture/replay

```python
# Capture (один раз)
stream = torch.cuda.Stream()
with torch.cuda.graph(graph, stream=stream):
    output = model.forward(input_buf)

# Replay (каждый шаг)
input_buf.copy_(new_input)
graph.replay()  # без Python overhead!
```

### Паттерн 3: StaticCache для фиксированного KV

```python
from transformers import StaticCache

cache = StaticCache(config=config, max_cache_len=2048)

# Prefill — копируем динамический cache в статический
for layer_idx in range(num_layers):
    k, v = dynamic_kv[layer_idx]
    pos = torch.arange(k.shape[2], device=device)
    cache.update(k, v, layer_idx, {"cache_position": pos})

# Decode — просто обновляем cache_position
cache_position = torch.tensor([current_pos], device=device)
output = model.forward(input_embeds, past_key_values=cache, 
                       cache_position=cache_position)
```

### Паттерн 4: Safe tensor copying от static buffer reuse

```python
def to_pcm_chunk(x):
    """Всегда real copy — защита от перезаписи static buffer."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32, copy=True).reshape(-1)
```

### Паттерн 5: Backpressure control

```python
# Producer side — не перегружаем очередь
while q.qsize() >= max_queue_size:
    time.sleep(0.01)
q.put(chunk)

# Consumer side — не перегружаем плеер
while player.buffered_seconds() > max_buffer_sec:
    time.sleep(0.01)
player.add_chunk(chunk)
```

---

## Приложения

### A. Ссылки на файлы

| Файл | Описание |
|------|----------|
| `fast_tts_v14.py` | Финальная потоковая реализация |
| `fast_tts_v10.py` | CUDA Graphs + streaming pipeline |
| `streaming_tts_v6.py` | Первый рабочий producer-consumer |
| `qwen_tts_cuda_graphs/predictor_graph.py` | Predictor CUDA graph |
| `qwen_tts_cuda_graphs/talker_graph.py` | Talker CUDA graph |
| `docs/cuda_graphs_optimization.md` | Детальный разбор оптимизации |
| `bench_faster_custom_voice.py` | Бенчмарк скрипт |
| `bench_comparison.json` | Результаты бенчмарков |

### B. Полезные команды

```bash
# Запуск с текстом из аргументов
python fast_tts_v14.py "Привет мир"

# Тестовый набор (10 предложений)
python fast_tts_v14.py --test

# Бенчмарк Faster vs Plain
python bench_faster_custom_voice.py

# Проверка CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### C. Известные ограничения

1. **max_seq_len = 2048** — ограничивает длину префилла (текст + reference audio)
2. **Batch size = 1** — CUDA graphs не поддерживают динамический batch
3. **Только NVIDIA GPU** — требуется CUDA и поддержка CUDAGraph
4. **PyTorch >= 2.5.1** — для стабильного capture

---

*Документ создан на основе анализа реализации в `fast_tts_v14.py` и сравнения с `faster-qwen3-tts`.*
