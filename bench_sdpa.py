"""Benchmark SDPA attention performance."""
import argparse, os, sys, time
sys.stdout.reconfigure(line_buffering=True)

import torch
from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser(description='SDPA attention benchmark')
    parser.add_argument('--model', default=os.getenv('MODEL_PATH', r'G:\Foundation\models\Qwen3-TTS'),
                        help='Path to Qwen3-TTS model directory')
    parser.add_argument('--speaker', default='Sohee')
    args = parser.parse_args()

    attn_impl = 'sdpa'

    print(f"\n{'='*60}", flush=True)
    print(f"Testing attn_implementation={attn_impl}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map='cuda:0', dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    load_ms = (time.perf_counter()-t0)*1000
    print(f"Load: {load_ms:.0f}ms", flush=True)

    # Warmup — captures CUDA graphs
    gen = model.generate_custom_voice_streaming(
        text="Привет.", speaker=args.speaker, language='Russian',
        chunk_size=8, max_new_tokens=10, backend='auto',
    )
    for _ in gen:
        pass
    torch.cuda.synchronize()
    print("Warmup done (CUDA graphs captured)", flush=True)

    # Test
    t_start = time.perf_counter()
    total_s = 0
    chunks = 0
    decode_times = []
    for audio_c, sr, timing in model.generate_custom_voice_streaming(
        text="Привет мир, как дела? Это тест производительности SDPA attention.", speaker=args.speaker, language='Russian',
        chunk_size=8, max_new_tokens=300, backend='auto',
    ):
        chunks += 1
        total_s += len(audio_c) / sr
        decode_ms = timing.get('decode_ms', 0)
        decode_times.append(decode_ms)

    wall_ms = (time.perf_counter() - t_start) * 1000
    avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
    steps = total_s * 12
    ms_per_step = wall_ms / steps if steps > 0 else 0

    print(f"Chunks: {chunks}", flush=True)
    print(f"Audio: {total_s:.2f}s", flush=True)
    print(f"Wall: {wall_ms:.0f}ms", flush=True)
    print(f"Ms/step: {ms_per_step:.0f}", flush=True)
    print(f"Avg decode_ms/chunk: {avg_decode:.0f}ms", flush=True)
    if decode_times:
        print(f"Decode ms per step: {avg_decode/8:.0f}", flush=True)


if __name__ == '__main__':
    main()
