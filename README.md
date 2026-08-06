# Qwen3-TTS Streaming Engine

## English

A streaming speech synthesis engine built on **Qwen3-TTS-0.6B** with **CUDA Graphs** acceleration for real-time playback.

### What's inside

| Component | Description |
|-----------|-------------|
| `fast_tts_v14.py` | Final implementation — true streaming via native `generate_custom_voice_streaming`, seamless chunk concatenation |
| `qwen_tts_cuda_graphs/` | Custom `PredictorGraph` and `TalkerGraph` — ported from [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) with adaptation for native `qwen_tts` |
| `run_native.py` | Quick test of native Qwen3TTSModel streaming API |
| `run_faster.py` | Quick test of FasterQwen3TTS (CUDA graphs) streaming API |
| `bench_sdpa.py` | SDPA attention performance benchmark |
| `debug_graphs.py` | CUDA graph timing diagnostics |
| `test_v14.py` | Full test suite — 10 sentences with playback verification |
| `*.bat` | Windows launchers (double-click or from terminal) |
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
- Seamless chunk concatenation (stateful generation preserves phase continuity)
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
# Using the bundled .conda environment (recommended on Windows)
G:\qwen-tts\.conda\python.exe -m pip install -r requirements.txt

# Or with your own venv
pip install -U qwen-tts transformers accelerate torchaudio soundfile sounddevice

# Note: SDPA attention is used by default — it's compatible with CUDA graphs.
# Flash Attention 2 is NOT compatible with CUDA graph capture and will crash.
```

### Quick start

#### From terminal (with --model flag)

```bash
G:\qwen-tts\.conda\python.exe run_native.py --text "Hello world"
G:\qwen-tts\.conda\python.exe run_faster.py --text "Привет мир"
G:\qwen-tts\.conda\python.exe bench_sdpa.py
G:\qwen-tts\.conda\python.exe test_v14.py
```

#### From Windows launcher (.bat files)

```cmd
run_native.bat --text "Hello world"
run_faster.bat
bench_sdpa.bat
test_v14.bat
fast_tts.bat --text "Привет мир"
```

#### In Python

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
├── fast_tts_v14.py             # Final version — true streaming, seamless chunk concatenation
├── run_native.py               # Quick test: native Qwen3TTSModel API
├── run_faster.py               # Quick test: FasterQwen3TTS (CUDA graphs)
├── bench_sdpa.py               # SDPA attention benchmark
├── debug_graphs.py             # CUDA graph timing diagnostics
├── test_v14.py                 # Full test suite — 10 sentences + playback
├── requirements.txt            # Python dependencies
├── *.bat                       # Windows launchers (double-click friendly)
├── qwen_tts_cuda_graphs/       # CUDA Graph optimizations
│   ├── __init__.py
│   ├── predictor_graph.py      # 15-step predictor loop capture
│   ├── talker_graph.py         # Single-token talker decode capture
│   └── sampling.py             # Token sampling utilities
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
                    Concatenate chunks (stateful) → Live playback @24kHz
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
| `fast_tts_v14.py` | Финальная реализация — true streaming через нативный `generate_custom_voice_streaming`, бесшовная склейка чанков |
| `qwen_tts_cuda_graphs/` | Кастомные `PredictorGraph` и `TalkerGraph` — перенесены из [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) с адаптацией под нативный `qwen_tts` |
| `run_native.py` | Быстрый тест нативного API Qwen3TTSModel |
| `run_faster.py` | Быстрый тест FasterQwen3TTS (CUDA graphs) |
| `bench_sdpa.py` | Бенчмарк SDPA attention |
| `debug_graphs.py` | Дебаг таймингов CUDA graph |
| `test_v14.py` | Полный тест — 10 предложений с проигрыванием |
| `*.bat` | Windows-лаунчеры (двойной клик или из терминала) |
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
- Бесшовная склейка чанков (stateful generation сохраняет фазовую непрерывность)
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
# Через встроенный .conda (рекомендуется на Windows)
G:\qwen-tts\.conda\python.exe -m pip install -r requirements.txt

# Или через свой venv
pip install -U qwen-tts transformers accelerate torchaudio soundfile sounddevice

# Примечание: используется SDPA attention по умолчанию — совместима с CUDA graphs.
# Flash Attention 2 НЕ совместима с захватом CUDA graph и вызовет ошибку.
```

### Быстрый старт

#### Из терминала (с флагом --model)

```bash
G:\qwen-tts\.conda\python.exe run_native.py --text "Hello world"
G:\qwen-tts\.conda\python.exe run_faster.py --text "Привет мир"
G:\qwen-tts\.conda\python.exe bench_sdpa.py
G:\qwen-tts\.conda\python.exe test_v14.py
```

#### Из Windows-лаунчеров (.bat файлы)

```cmd
run_native.bat --text "Hello world"
run_faster.bat
bench_sdpa.bat
test_v14.bat
fast_tts.bat --text "Привет мир"
```

#### В Python

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
├── fast_tts_v14.py             # Финальная версия — true streaming, бесшовная склейка чанков
├── run_native.py               # Быстрый тест нативного API Qwen3TTSModel
├── run_faster.py               # Быстрый тест FasterQwen3TTS (CUDA graphs)
├── bench_sdpa.py               # Бенчмарк SDPA attention
├── debug_graphs.py             # Дебаг таймингов CUDA graph
├── test_v14.py                 # Полный тест — 10 предложений + проигрывание
├── requirements.txt            # Python зависимости
├── *.bat                       # Windows-лаунчеры (двойной клик)
├── qwen_tts_cuda_graphs/       # CUDA Graph оптимизации
│   ├── __init__.py
│   ├── predictor_graph.py      # 15-step predictor loop capture
│   ├── talker_graph.py         # Single-token talker decode capture
│   └── sampling.py             # Token sampling utilities
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
                    Concatenate chunks (stateful) → Live playback @24kHz
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
