import torch, sys, time
sys.stdout.reconfigure(line_buffering=True)
from qwen_tts import Qwen3TTSModel

model_path = r'G:\Foundation\models\Qwen3-TTS'
text_short = 'Привет мир.'
text_long = 'Меня зовут Александр. Мне двадцать пять лет. Я живу в Санкт-Петербурге. Работаю программистом уже пять лет. Это тест потоковой генерации с длинным текстом для проверки стабильности CUDA графиков при разных длинах входных последовательностей.'

print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)

m = Qwen3TTSModel.from_pretrained(model_path, device_map='cuda:0', dtype=torch.bfloat16)
print('Loaded.', flush=True)

# Call 1: short text
print('\n=== CALL 1 (short) ===', flush=True)
t0 = time.time()
for i, (audio_chunk, sr, timing) in enumerate(m.generate_custom_voice_streaming(
    text=text_short, speaker='Sohee', language='Russian', chunk_size=8, max_new_tokens=20, backend='auto'
)):
    print(f'  chunk {i}: decode={timing["decode_ms"]:.0f}ms prefill={timing.get("prefill_ms",0):.0f}ms', flush=True)
print(f'Total: {(time.time()-t0)*1000:.0f}ms', flush=True)

# Call 2: long text (different prefill length!)
print('\n=== CALL 2 (long) ===', flush=True)
t0 = time.time()
for i, (audio_chunk, sr, timing) in enumerate(m.generate_custom_voice_streaming(
    text=text_long, speaker='Sohee', language='Russian', chunk_size=8, max_new_tokens=40, backend='auto'
)):
    print(f'  chunk {i}: decode={timing["decode_ms"]:.0f}ms prefill={timing.get("prefill_ms",0):.0f}ms', flush=True)
print(f'Total: {(time.time()-t0)*1000:.0f}ms', flush=True)

# Call 3: short text again
print('\n=== CALL 3 (short again) ===', flush=True)
t0 = time.time()
for i, (audio_chunk, sr, timing) in enumerate(m.generate_custom_voice_streaming(
    text=text_short, speaker='Sohee', language='Russian', chunk_size=8, max_new_tokens=20, backend='auto'
)):
    print(f'  chunk {i}: decode={timing["decode_ms"]:.0f}ms prefill={timing.get("prefill_ms",0):.0f}ms', flush=True)
print(f'Total: {(time.time()-t0)*1000:.0f}ms', flush=True)
