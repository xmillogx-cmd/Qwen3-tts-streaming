# Qwen3-TTS Streaming Engine

## English

A streaming speech synthesis engine built on **Qwen3-TTS-0.6B** with **CUDA Graphs** acceleration for real-time playback.

### What's inside

| Component | Description |
|-----------|-------------|
| `fast_tts_v14.py` | Final implementation — true streaming via native `generate_custom_voice_streaming` + crossfade |
| `qwen_tts_cuda_graphs/` | Custom `PredictorGraph` and `TalkerGraph` — ported from [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) with adaptation for native `qwen_tts` |
| `test_native.py`, `test_v14.py` | Tests for native API and v14 |
| `docs/` | Technical documentation on optimizations |

### Results on RTX 5060 Ti (CC 12.0)

| Metric | Before optimization | After | Speedup |
|--------|--------------------|-------|---------|
| ms/step | ~248ms | **31ms** | **7.9x** |
| RTF | 0.336 (slower than realtime) | **2.67** (2.7× faster than realtime) | **8.0x** |
| TTFA (time to first audio) | ~40s | **~355ms** | **113x** |

### Key optimizations

#### CUDA Graphs
- `PredictorGraph` — captures the full 15-step code predictor loop as a single CUDA graph (~26ms vs ~190ms)
- `TalkerGraph` — captures single-token talker decode (~12ms vs ~75ms)
- Uses `transformers.StaticCache` instead of DynamicCache for fixed-size KV buffers

#### Streaming Pipeline (v14)
- Native `generate_custom_voice_streaming()` — no manual token management
- Producer-consumer architecture: generation thread → queue → player
- Automatic long-text segmentation via `split_segments()`
- Crossfade between chunks for smooth transitions
- 0.3s preroll — audio starts in ~350ms

### Installation

#### Clone (with submodules)

```bash
git clone --recursive https://github.com/xmillogx-cmd/Qwen3-tts-streaming.git
cd Qwen3-tts-streaming
# Or if already cloned without --recursive:
git submodule update --init --recursive
```

#### Dependencies

```bash
# Core dependencies
pip install -U qwen-tts transformers accelerate torchaudio soundfile sounddevice

# Flash Attention (optional, reduces memory usage)
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "FLASH_ATTN_CUDA_ARCHS=120"
pip install -U flash-attn --no-build-isolation
```

### Quick start

```python
from fast_tts_v14 import FastTTSv14

tts = FastTTSv14(
    model_path="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    speaker="Sohee",
)

# Streaming generation with live playback
tts.generate_and_play("Hello! This is a streaming TTS test.")
```

### Project structure

```
qwen-tts-streaming/
├── fast_tts_v14.py             # Final version — true streaming + crossfade
├── qwen_tts_cuda_graphs/       # CUDA Graph optimizations
│   ├── __init__.py
│   ├── predictor_graph.py      # 15-step predictor loop capture
│   ├── talker_graph.py         # Single-token talker decode capture
│   └── sampling.py             # Token sampling utilities
├── test_native.py              # Test for native generate_custom_voice_streaming
├── test_v14.py                 # v14 test (10 sentences)
├── docs/
│   ├── cuda_graphs_optimization.md  # Detailed CUDA Graphs breakdown
│   └── qwen3-tts-implementation.md  # Architecture and version history
├── Qwen3-TTS/                  # submodule — original Qwen3-TTS
└── faster-qwen3-tts/           # submodule — source of CUDA Graphs
```

### Generation architecture (v14)

```
Text → split_segments() → generate_custom_voice_streaming() × N segments
                                      ↓
                    Talker prefill + Predictor × 15 (CUDA graphs)
                                      ↓
                    Codec tokens [cb0..cb14] → chunked_decode()
                                      ↓
                    Producer thread → queue → StreamingAudioPlayer
                                      ↓
                    Crossfade chunks → Live playback @24kHz
```

### Dependencies

- Python 3.10+
- PyTorch 2.5.1+ (with CUDA)
- `qwen-tts>=0.1.1`
- `transformers>=4.57,<5`
- `accelerate`, `soundfile`, `sounddevice`

### License

- CUDA Graph optimizations — **MIT** (reference: [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts))
- Base model Qwen3-TTS — **Apache 2.0**, [Alibaba Group / QwenLM](https://github.com/QwenLM/Qwen3-TTS)

---

## Русский

Потоковый движок синтеза речи на базе **Qwen3-TTS-0.6B** с ускорением через **CUDA Graphs**.

### Что внутри

| Компонент | Описание |
|-----------|----------|
| `fast_tts_v14.py` | Финальная реализация — true streaming через нативный `generate_custom_voice_streaming` + кроссфейд |
| `qwen_tts_cuda_graphs/` | Кастомные `PredictorGraph` и `TalkerGraph` — перенесены из [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) с адаптацией под нативный `qwen_tts` |
| `test_native.py`, `test_v14.py` | Тесты нативного API и v14 |
| `docs/` | Техническая документация по оптимизациям |

### Результаты на RTX 5060 Ti (CC 12.0)

| Метрика | До оптимизации | После | Ускорение |
|---------|---------------|-------|-----------|
| ms/step | ~248ms | **31ms** | **7.9x** |
| RTF | 0.336 (медленнее realtime) | **2.67** (быстрее в 2.7×) | **8.0x** |
| TTFA (время до первого аудио) | ~40s | **~355ms** | **113x** |

### Ключевые оптимизации

#### CUDA Graphs
- `PredictorGraph` — захватывает полный 15-шаговый цикл code predictor как один CUDA graph (~26ms вместо ~190ms)
- `TalkerGraph` — захватывает single-token decode talker'а (~12ms вместо ~75ms)
- Используется `transformers.StaticCache` вместо DynamicCache для фиксированных KV-буферов

#### Streaming Pipeline (v14)
- Нативный `generate_custom_voice_streaming()` — без ручного управления токенами
- Producer-consumer архитектура: поток генерации → очередь → плеер
- Автоматическое разбиение длинного текста на сегменты (`split_segments`)
- Crossfade между чанками для плавного перехода
- Преролл 0.3s — звук появляется через ~350ms

### Установка

#### Клонирование (с подмодулями)

```bash
git clone --recursive https://github.com/xmillogx-cmd/Qwen3-tts-streaming.git
cd Qwen3-tts-streaming
# Или если уже склонирован без --recursive:
git submodule update --init --recursive
```

#### Зависимости

```bash
# Базовые зависимости
pip install -U qwen-tts transformers accelerate torchaudio soundfile sounddevice

# Flash Attention (опционально, для снижения памяти)
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "FLASH_ATTN_CUDA_ARCHS=120"
pip install -U flash-attn --no-build-isolation
```

### Быстрый старт

```python
from fast_tts_v14 import FastTTSv14

tts = FastTTSv14(
    model_path="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    speaker="Sohee",
)

# Потоковая генерация с живым воспроизведением
tts.generate_and_play("Привет! Это тест потокового синтеза речи.")
```

### Структура проекта

```
qwen-tts-streaming/
├── fast_tts_v14.py             # Финальная версия — true streaming + crossfade
├── qwen_tts_cuda_graphs/       # CUDA Graph оптимизации
│   ├── __init__.py
│   ├── predictor_graph.py      # 15-step predictor loop capture
│   ├── talker_graph.py         # Single-token talker decode capture
│   └── sampling.py             # Token sampling utilities
├── test_native.py              # Тест нативного generate_custom_voice_streaming
├── test_v14.py                 # Тест v14 (10 предложений)
├── docs/
│   ├── cuda_graphs_optimization.md  # Детальный разбор CUDA Graphs
│   └── qwen3-tts-implementation.md  # Архитектура и эволюция версий
├── Qwen3-TTS/                  # submodule — оригинальный Qwen3-TTS
└── faster-qwen3-tts/           # submodule — источник CUDA Graphs
```

### Архитектура генерации (v14)

```
Текст → split_segments() → generate_custom_voice_streaming() × N сегментов
                                      ↓
                    Talker prefill + Predictor × 15 (CUDA graphs)
                                      ↓
                    Codec tokens [cb0..cb14] → chunked_decode()
                                      ↓
                    Producer thread → queue → StreamingAudioPlayer
                                      ↓
                    Crossfade chunks → Live playback @24kHz
```

### Зависимости

- Python 3.10+
- PyTorch 2.5.1+ (с CUDA)
- `qwen-tts>=0.1.1`
- `transformers>=4.57,<5`
- `accelerate`, `soundfile`, `sounddevice`

### License

- Оптимизации CUDA Graphs — **MIT** (референс: [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts))
- Базовая модель Qwen3-TTS — **Apache 2.0**, [Alibaba Group / QwenLM](https://github.com/QwenLM/Qwen3-TTS)
