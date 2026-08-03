import torch, sys, time
sys.stdout.reconfigure(line_buffering=True)
from qwen_tts import Qwen3TTSModel

model_path = r'G:\Foundation\models\Qwen3-TTS'
text = 'Меня зовут Александр. Мне двадцать пять лет. Я живу в Санкт-Петербурге. Работаю программистом уже пять лет. Это тест потоковой генерации.'

print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
print(f'Text: {text}', flush=True)

m = Qwen3TTSModel.from_pretrained(model_path, device_map='cuda:0', dtype=torch.bfloat16)
print('Loaded.', flush=True)

# NO warmup - go straight to streaming
gen = m.generate_custom_voice_streaming(
    text=text, speaker='Sohee', language='Russian',
    chunk_size=8, max_new_tokens=200, backend='auto'
)
for i, (audio_chunk, sr, timing) in enumerate(gen):
    print(f'chunk {i} {sr}Hz len={len(audio_chunk)} samples | {timing}', flush=True)
print('Done.', flush=True)
