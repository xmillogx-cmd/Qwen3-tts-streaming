"""Debug script to check CUDA graph timing."""
import argparse, os, sys, time
sys.stdout.reconfigure(line_buffering=True)

import torch
from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser(description='CUDA graph timing debug')
    parser.add_argument('--model', default=os.getenv('MODEL_PATH', r'G:\Foundation\models\Qwen3-TTS'),
                        help='Path to Qwen3-TTS model directory')
    parser.add_argument('--speaker', default='Sohee')
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print()

    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map='cuda:0', dtype=torch.bfloat16,
    )
    load_ms = (time.perf_counter()-t0)*1000
    print(f"Load: {load_ms:.0f}ms", flush=True)

    # Check if CUDA graphs are available
    print(f"\nCUDA graphs attr: _graphs_initialized={getattr(model, '_graphs_initialized', False)}", flush=True)

    # Warmup - short text
    print("\n--- WARMUP ---", flush=True)
    gen = model.generate_custom_voice_streaming(
        text="Привет.", speaker=args.speaker, language='Russian',
        chunk_size=8, max_new_tokens=10, backend='auto',
    )
    for audio_c, sr, timing in gen:
        print(f"  Warmup chunk: timing={timing}", flush=True)
    torch.cuda.synchronize()
    print("Warmup done", flush=True)

    # Check after warmup
    print(f"After warmup: _graphs_initialized={getattr(model, '_graphs_initialized', False)}", flush=True)
    print(f"After warmup: _graphs_warmed_up={getattr(model, '_graphs_warmed_up', False)}", flush=True)

    # Test 1: Short Russian
    print("\n--- TEST 1: Привет мир ---", flush=True)
    t_start = time.perf_counter()
    total_s = 0
    chunks = 0
    for audio_c, sr, timing in model.generate_custom_voice_streaming(
        text="Привет мир", speaker=args.speaker, language='Russian',
        chunk_size=8, max_new_tokens=300, backend='auto',
    ):
        chunks += 1
        total_s += len(audio_c) / sr
        print(f"  Chunk {chunks}: timing={timing}", flush=True)
    wall_ms = (time.perf_counter() - t_start) * 1000
    print(f"  TOTAL: chunks={chunks}, audio={total_s:.2f}s, wall={wall_ms:.0f}ms", flush=True)

    # Test 2: Longer Russian
    print("\n--- TEST 2: Как дела? У меня всё хорошо. ---", flush=True)
    t_start = time.perf_counter()
    total_s = 0
    chunks = 0
    for audio_c, sr, timing in model.generate_custom_voice_streaming(
        text="Как дела? У меня всё хорошо.", speaker=args.speaker, language='Russian',
        chunk_size=8, max_new_tokens=300, backend='auto',
    ):
        chunks += 1
        total_s += len(audio_c) / sr
        print(f"  Chunk {chunks}: timing={timing}", flush=True)
    wall_ms = (time.perf_counter() - t_start) * 1000
    print(f"  TOTAL: chunks={chunks}, audio={total_s:.2f}s, wall={wall_ms:.0f}ms", flush=True)

    # Test 3: English
    print("\n--- TEST 3: Hello world ---", flush=True)
    t_start = time.perf_counter()
    total_s = 0
    chunks = 0
    for audio_c, sr, timing in model.generate_custom_voice_streaming(
        text="Hello world, this is a test.", speaker=args.speaker, language='English',
        chunk_size=8, max_new_tokens=300, backend='auto',
    ):
        chunks += 1
        total_s += len(audio_c) / sr
        print(f"  Chunk {chunks}: timing={timing}", flush=True)
    wall_ms = (time.perf_counter() - t_start) * 1000
    print(f"  TOTAL: chunks={chunks}, audio={total_s:.2f}s, wall={wall_ms:.0f}ms", flush=True)

    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
