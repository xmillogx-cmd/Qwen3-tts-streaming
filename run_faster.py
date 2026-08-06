"""Run FasterQwen3TTS (CUDA graphs) streaming inference."""
import argparse, sys, os
sys.stdout.reconfigure(line_buffering=True)

import torch
from faster_qwen3_tts import FasterQwen3TTS


def main():
    parser = argparse.ArgumentParser(description='FasterQwen3TTS CUDA graph streaming test')
    parser.add_argument('--model', default=os.getenv('MODEL_PATH', r'G:\Foundation\models\Qwen3-TTS'),
                        help='Path to Qwen3-TTS model directory')
    parser.add_argument('--text', default='Меня зовут Александр. Мне двадцать пять лет. Я живу в Санкт-Петербурге. Работаю программистом уже пять лет. Это тест потоковой генерации.',
                        help='Text to synthesize')
    parser.add_argument('--speaker', default='Sohee')
    parser.add_argument('--language', default='Russian')
    args = parser.parse_args()

    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    print(f'Text: {args.text}', flush=True)

    m = FasterQwen3TTS.from_pretrained(args.model, device='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa', max_seq_len=2048)
    print('Loaded.', flush=True)

    gen = m.generate_custom_voice_streaming(
        text=args.text, speaker=args.speaker, language=args.language,
        chunk_size=8, max_new_tokens=200
    )
    for audio_chunk, sr, timing in gen:
        print(f'chunk {sr}Hz len={len(audio_chunk)} samples | {timing}', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
