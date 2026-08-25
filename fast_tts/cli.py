"""Command-line entry point (``fast-tts``) and dev test suite."""
from __future__ import annotations

import argparse
import os
import sys

import sounddevice as sd
import torch

from .engine import FastTTSv14


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Fast TTS v14 — Streaming playback')
    parser.add_argument('--model', default=os.getenv('MODEL_PATH'),
                        help='Path to Qwen3-TTS model directory (or set MODEL_PATH env var)')
    parser.add_argument('--speaker', default='Sohee',
                        help='Speaker name (default: Sohee)')
    parser.add_argument('--text', nargs='*', help='Text to synthesize (or use default)')
    parser.add_argument('--chunk-size', type=int, default=8, choices=[2, 4, 8],
                        help='Audio chunk size in tokens (default: 8)')
    parser.add_argument('--min-start-sec', type=float, default=0.15,
                        help='Minimum buffered seconds before playback starts (default: 0.15)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio output device index (interactive menu if omitted)')
    args = parser.parse_args()

    if not args.model or not os.path.exists(args.model):
        print(f"Error: model path not found: {args.model}", flush=True)
        print("Hint: specify --model <path> or set MODEL_PATH env var", flush=True)
        sys.exit(1)

    # Audio device selection menu
    devices = sd.query_devices()
    print("\nAudio devices:", flush=True)
    for i, d in enumerate(devices):
        name = d['name'][:50]
        inp  = f"in={int(d['max_input_channels'])}" if d['max_input_channels'] else ''
        out  = f"out={int(d['max_output_channels'])}" if d['max_output_channels'] else ''
        flags = ', '.join(filter(None, [inp, out])) or '(no channels)'
        default_mark = ' <-- default' if i == sd.default.device[1] else ''
        print(f"  [{i:2d}] {name}  {flags}{default_mark}", flush=True)

    device_id = args.device
    if device_id is None:
        try:
            device_id = int(input("\nSelect audio device [Enter=default]: "))
        except (ValueError, EOFError):
            device_id = sd.default.device[1] or 0
    print(f"  -> Using device {device_id}: {devices[device_id]['name'][:50]}", flush=True)

    text = ' '.join(args.text) if args.text else (
        "Привет! Это тест потоковой генерации. Звук должен быть плавным без щелчков!"
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print()

    tts = FastTTSv14(args.model, speaker=args.speaker, device_id=device_id)
    try:
        tts.generate_and_play(text, save_wav='tts_output_v14.wav',
                              chunk_size=args.chunk_size, min_start_sec=args.min_start_sec)
    except KeyboardInterrupt:
        print("\n[V14] Interrupted.", flush=True)
        sys.exit(0)
    finally:
        tts.player.stop()


# ============================================================================
# TEST SUITE — 10 sentences (5 Russian, 5 English)
# ============================================================================
def run_test_suite():
    """Run test suite with 10 sentences for audio verification."""
    model_path = os.getenv('MODEL_PATH')

    if not model_path or not os.path.exists(model_path):
        print(f"Error: model path not found: {model_path}", flush=True)
        print("Hint: set MODEL_PATH env var to your Qwen3-TTS model directory", flush=True)
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print()

    tts = FastTTSv14(model_path, speaker='Sohee')

    # 5 Russian + 5 English paragraphs
    test_sentences = [
        ("Russian", "Привет! Меня зовут Александр. Мне двадцать пять лет. Я живу в Санкт-Петербурге и работаю программистом уже пять лет."),
        ("Russian", "Сегодня прекрасная погода для прогулки по городу. Мы с друзьями решили посетить Эрмитаж и затем поужинать в хорошем ресторане."),
        ("Russian", "Технологии меняют наш мир каждый день. Искусственный интеллект помогает врачам ставить диагнозы, а роботы уже работают на производствах."),
        ("Russian", "Книги — это лучшие друзья человека. Чтение развивает воображение и расширяет кругозор. Я читаю минимум одну книгу в неделю."),
        ("Russian", "Это финальный абзац для проверки качества звука. Надеюсь, всё звучит чётко и без заиканий!"),
        ("English", "Hello! My name is John and I have been working as a software engineer for ten years. I live in New York with my wife and two cats."),
        ("English", "The quick brown fox jumps over the lazy dog near the old wooden bridge. It was a beautiful autumn morning with golden leaves falling from the trees."),
        ("English", "Technology is advancing at an incredible pace these days. Machine learning models can now write code, create art, and even compose music."),
        ("English", "Natural language processing makes computers understand us better than ever before. This technology powers virtual assistants like Siri and Alexa."),
        ("English", "This is the final paragraph for audio quality verification. I hope everything sounds clear and smooth without any stuttering!"),
    ]

    print(f"\n{'='*70}", flush=True)
    print("[V14] TEST SUITE — 5 Russian + 5 English Paragraphs", flush=True)
    print(f"{'='*70}", flush=True)

    try:
        for i, (lang, text) in enumerate(test_sentences):
            save_wav = f'tts_test_v14_{i+1:02d}_{lang.lower()}.wav'
            print(f"\n[{i+1}/10] [{lang}] {text}", flush=True)
            tts.generate_and_play(text, language=lang, save_wav=save_wav)

        print(f"\n{'='*70}", flush=True)
        print("[V14] All 10 tests complete!", flush=True)
        print(f"{'='*70}", flush=True)
    finally:
        tts.player.stop()


if __name__ == '__main__':
    if '--test' in sys.argv:
        run_test_suite()
    else:
        main()
