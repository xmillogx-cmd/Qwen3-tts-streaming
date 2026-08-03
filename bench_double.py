import torch, sys, time
sys.stdout.reconfigure(line_buffering=True)
from qwen_tts import Qwen3TTSModel

model_path = r'G:\Foundation\models\Qwen3-TTS'
text = 'Привет мир.'

print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)

m = Qwen3TTSModel.from_pretrained(model_path, device_map='cuda:0', dtype=torch.bfloat16)
print('Loaded.', flush=True)

# First call - should warmup + run
print('\n=== CALL 1 ===', flush=True)
t0 = time.time()
for i, (audio_chunk, sr, timing) in enumerate(m.generate_custom_voice_streaming(
    text=text, speaker='Sohee', language='Russian', chunk_size=8, max_new_tokens=20, backend='auto'
)):
    print(f'  chunk {i}: decode={timing["decode_ms"]:.0f}ms prefill={timing.get("prefill_ms",0):.0f}ms', flush=True)
print(f'Call 1 total: {(time.time()-t0)*1000:.0f}ms', flush=True)

# Second call - should reuse warmup
print('\n=== CALL 2 ===', flush=True)
t0 = time.time()
for i, (audio_chunk, sr, timing) in enumerate(m.generate_custom_voice_streaming(
    text=text, speaker='Sohee', language='Russian', chunk_size=8, max_new_tokens=20, backend='auto'
)):
    print(f'  chunk {i}: decode={timing["decode_ms"]:.0f}ms prefill={timing.get("prefill_ms",0):.0f}ms', flush=True)
print(f'Call 2 total: {(time.time()-t0)*1000:.0f}ms', flush=True)
