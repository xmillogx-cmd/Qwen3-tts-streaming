# Qwen3-TTS Streaming Engine

## English

A streaming speech synthesis engine built on **Qwen3-TTS-0.6B** with **CUDA Graphs** acceleration for real-time playback.

### What's inside

| Component | Description |
|-----------|-------------|
| `fast_tts_v14.py` | Final implementation — true streaming via native `generate_custom_voice_streaming`, seamless chunk concatenation |
| `Qwen3-TTS/qwen_tts/inference/` | CUDA Graph code (`PredictorGraph`, `TalkerGraph`) — ported from [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts), patched in-repo (live editable install) |
| `profile_v14.py` | Baseline profiler: load/capture cost, Mimi decode vs context, TTFA/RTF, token-cap usage |
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

#### Clone

```bash
git clone https://github.com/xmillogx-cmd/Qwen3-tts-streaming.git
cd Qwen3-tts-streaming
```

No submodules — the patched `Qwen3-TTS/` source is vendored directly in the repo.
(`faster-qwen3-tts/`, if present, is a gitignored reference-only clone used by `run_faster.py`; not required for anything else.)

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
├── profile_v14.py              # Baseline profiler: load/capture, Mimi vs ctx, TTFA/RTF, token caps
├── requirements.txt            # Python dependencies
├── *.bat                       # Windows launchers (double-click friendly)
├── docs/
│   ├── cuda_graphs_optimization.md  # Detailed CUDA Graphs breakdown
│   └── qwen3-tts-implementation.md  # Architecture and version history
├── Qwen3-TTS/                  # Vendored + patched qwen_tts source (live editable install)
│   └── qwen_tts/inference/     # CUDA Graph code: predictor_graph.py, talker_graph.py, sampling.py
└── faster-qwen3-tts/           # Gitignored reference-only clone (source of the CUDA Graphs port)
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
| `Qwen3-TTS/qwen_tts/inference/` | Код CUDA Graph (`PredictorGraph`, `TalkerGraph`) — перенесён из [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts), патчится прямо в репозитории (live editable install) |
| `profile_v14.py` | Базовый профайлер: стоимость load/capture, Mimi decode vs контекст, TTFA/RTF, расход токенов |
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

#### Клонирование

```bash
git clone https://github.com/xmillogx-cmd/Qwen3-tts-streaming.git
cd Qwen3-tts-streaming
```

Подмодулей нет — патченный исходник `Qwen3-TTS/` закоммичен прямо в репозиторий.
(`faster-qwen3-tts/`, если присутствует, — gitignored-клон только для референса, используется `run_faster.py`; без него всё остальное работает.)

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
├── profile_v14.py              # Базовый профайлер: load/capture, Mimi vs ctx, TTFA/RTF, лимиты токенов
├── requirements.txt            # Python зависимости
├── *.bat                       # Windows-лаунчеры (двойной клик)
├── docs/
│   ├── cuda_graphs_optimization.md  # Детальный разбор CUDA Graphs
│   └── qwen3-tts-implementation.md  # Архитектура и эволюция версий
├── Qwen3-TTS/                  # Вендорный + патченный исходник qwen_tts (live editable install)
│   └── qwen_tts/inference/     # Код CUDA Graph: predictor_graph.py, talker_graph.py, sampling.py
└── faster-qwen3-tts/           # Gitignored-клон только для референса (источник переноса CUDA Graphs)
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
