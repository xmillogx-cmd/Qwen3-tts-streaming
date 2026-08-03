"""Test v14 with 10 sentences (5 Russian, 5 English)."""
import torch, time, numpy as np, soundfile as sf, sys

model_path = r'G:\Foundation\models\Qwen3-TTS'
speaker = 'Sohee'

sentences = [
    # Russian
    "Привет мир",
    "Как дела? У меня всё хорошо.",
    "Сегодня прекрасная погода для прогулки.",
    "Я люблю слушать музыку по утрам.",
    "Этот тест показывает качество генерации речи.",
    # English
    "Hello world, this is a test.",
    "How are you doing today?",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming every industry.",
    "Natural language processing makes computers understand text.",
]

print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print()

from qwen_tts import Qwen3TTSModel

t0 = time.perf_counter()
model = Qwen3TTSModel.from_pretrained(
    model_path, device_map='cuda:0', dtype=torch.bfloat16,
)
load_ms = (time.perf_counter()-t0)*1000
print(f"Load: {load_ms:.0f}ms", flush=True)

# Warmup
gen = model.generate_custom_voice_streaming(
    text="Привет.", speaker=speaker, language='Russian',
    chunk_size=8, max_new_tokens=10, backend='auto',
)
for _ in gen:
    pass
torch.cuda.synchronize()
print("Warmup done", flush=True)
print()

# Run all sentences
results = []
for i, text in enumerate(sentences):
    lang = 'Russian' if i < 5 else 'English'
    
    t_start = time.perf_counter()
    total_s = 0
    chunks = 0
    
    for audio_c, sr, timing in model.generate_custom_voice_streaming(
        text=text, speaker=speaker, language=lang,
        chunk_size=8, max_new_tokens=300, backend='auto',
    ):
        chunks += 1
        total_s += len(audio_c) / sr
    
    wall_ms = (time.perf_counter() - t_start) * 1000
    
    # Save audio
    all_chunks = []
    for audio_c, sr, timing in model.generate_custom_voice_streaming(
        text=text, speaker=speaker, language=lang,
        chunk_size=8, max_new_tokens=300, backend='auto',
    ):
        all_chunks.append(audio_c.detach().cpu().numpy() if torch.is_tensor(audio_c) else np.array(audio_c))
    
    audio = np.concatenate(all_chunks)
    sf.write(f'test_v14_{i+1:02d}.wav', audio, sr)
    
    results.append({
        'idx': i+1,
        'text': text,
        'lang': lang,
        'chunks': chunks,
        'audio_s': total_s,
        'wall_ms': wall_ms,
        'ms_per_step': (wall_ms / (total_s * 12)) if total_s > 0 else 0,
    })
    
    print(f"[{i+1:02d}] {lang:8s} | chunks={chunks:2d} | audio={total_s:.2f}s | wall={wall_ms:.0f}ms | ms/step={results[-1]['ms_per_step']:.0f}", flush=True)
    print(f"      Text: {text}")

# Summary
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
for r in results:
    print(f"[{r['idx']:02d}] {r['lang']:8s} | {r['chunks']:2d} chunks | {r['audio_s']:.2f}s audio | {r['wall_ms']:.0f}ms wall | {r['ms_per_step']:.0f}ms/step")
    print(f"      \"{r['text']}\"")

# Stats
all_audio = [r['audio_s'] for r in results]
all_wall = [r['wall_ms'] for r in results]
avg_audio = np.mean(all_audio)
avg_wall = np.mean(all_wall)
print()
print(f"Average audio length: {avg_audio:.2f}s")
print(f"Average wall time:    {avg_wall:.0f}ms")
print(f"Total audio:          {sum(all_audio):.2f}s")
