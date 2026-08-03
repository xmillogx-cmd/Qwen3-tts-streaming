# Qwen3-TTS Streaming Engine

Потоковый движок синтеза речи на базе **Qwen3-TTS-0.6B** с ускорением через **CUDA Graphs**.

## Что внутри

| Компонент | Описание |
|-----------|----------|
| `qwen_tts_cuda_graphs/` | Кастомные `PredictorGraph` и `TalkerGraph` — перенесены из [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) с адаптацией под нативный `qwen_tts` |
| `streaming_tts_v6.py` | Первый рабочий producer-consumer pipeline (фикс underruns) |
| `streaming_tts_v7.py` | Parallel generation через ThreadPoolExecutor |
| `streaming_tts_v8.py` | torch.compile + chunked decode с overlap-add crossfade |
| `docs/` | Техническая документация по оптимизациям |

## Результаты на RTX 5060 Ti (CC 12.0)

| Метрика | До оптимизации | После | Ускорение |
|---------|---------------|-------|-----------|
| ms/step | ~248ms | **31ms** | **7.9x** |
| RTF | 0.336 (медленнее realtime) | **2.67** (быстрее в 2.7×) | **8.0x** |
| TTFA (время до первого аудио) | ~40s | **~355ms** | **113x** |

## Ключевые оптимизации

### CUDA Graphs
- `PredictorGraph` — захватывает полный 15-шаговый цикл code predictor как один CUDA graph (~26ms вместо ~190ms)
- `TalkerGraph` — захватывает single-token decode talker'а (~12ms вместо ~75ms)
- Используется `transformers.StaticCache` вместо DynamicCache для фиксированных KV-буферов

### Streaming Pipeline
- Producer-consumer архитектура: поток генерации → очередь → плеер
- Автоматическое разбиение длинного текста на сегменты (`split_segments`)
- Crossfade между чанками для плавного перехода
- Преролл 0.6s — звук появляется через ~350ms

### torch.compile
- `torch.compile(mode='reduce-overhead')` на модели и talker
- Снижение Python overhead при последовательных forward-проходах

## Установка

```bash
# Базовые зависимости
pip install -U qwen-tts transformers accelerate torchaudio soundfile sounddevice

# Flash Attention (опционально, для снижения памяти)
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "FLASH_ATTN_CUDA_ARCHS=120"
pip install -U flash-attn --no-build-isolation
```

## Быстрый старт

```python
from streaming_tts_v8 import StreamingTTS

tts = StreamingTTS(
    model_path="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    speaker="Vivian",
    language="Russian",
)

# Потоковая генерация с живым воспроизведением
tts.generate("Привет! Это тест потокового синтеза речи.")
```

## Структура проекта

```
qwen-tts-streaming/
├── qwen_tts_cuda_graphs/       # CUDA Graph оптимизации
│   ├── __init__.py
│   ├── predictor_graph.py      # 15-step predictor loop capture
│   ├── talker_graph.py         # Single-token talker decode capture
│   └── sampling.py             # Token sampling utilities
├── streaming_tts_v6.py         # Producer-consumer pipeline (фикс underruns)
├── streaming_tts_v7.py         # Parallel generation
├── streaming_tts_v8.py         # torch.compile + chunked decode
├── docs/
│   ├── cuda_graphs_optimization.md  # Детальный разбор CUDA Graphs
│   └── qwen3-tts-implementation.md  # Архитектура и эволюция версий
├── bench_*.py                  # Бенчмарки
├── profile_*.py                # Профилировщики
├── test_*.py                   # Тесты
└── QWEN.md                     # Заметки по проекту
```

## Архитектура генерации

```
Текст → tokenize → Talker prefill (StaticCache)
                ↓
        Predictor × 15 (CUDA graph, ~26ms/шаг)
                ↓
        Codec tokens [cb0..cb14]
                ↓
        Chunked decode → Waveform @24kHz
                ↓
        Crossfade chunks → Streaming playback
```

## Зависимости

- Python 3.10+
- PyTorch 2.5.1+ (с CUDA)
- `qwen-tts>=0.1.1`
- `transformers>=4.57,<5`
- `accelerate`, `soundfile`, `sounddevice`

## License

Код оптимизаций — MIT (адаптировано из faster-qwen3-tts).
Базовая модель — Qwen3-TTS (Apache 2.0, Alibaba Group).
