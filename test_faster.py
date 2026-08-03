import torch, sys
sys.stdout.reconfigure(line_buffering=True)
from faster_qwen3_tts import FasterQwen3TTS

model_path = r'G:\Foundation\models\Qwen3-TTS'
text = 'Меня зовут Александр. Мне двадцать пять лет. Я живу в Санкт-Петербурге. Работаю программистом уже пять лет. Это тест потоковой генерации.'

print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
print(f'Text: {text}', flush=True)

m = FasterQwen3TTS.from_pretrained(model_path, device='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa', max_seq_len=2048)
print('Loaded.', flush=True)

gen = m.generate_custom_voice_streaming(
    text=text, speaker='Sohee', language='Russian',
    chunk_size=8, max_new_tokens=200
)
for audio_chunk, sr, timing in gen:
    print(f'chunk {sr}Hz len={len(audio_chunk)} samples | {timing}', flush=True)
print('Done.', flush=True)
