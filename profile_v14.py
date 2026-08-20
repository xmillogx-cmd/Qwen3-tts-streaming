"""Baseline profiler for Qwen3-TTS streaming pipeline (v14).

Measures:
  A. Model load time
  B. CUDA graph capture cost (first streaming call)
  C. Per-chunk generation timing + Mimi decoder cost at different context sizes
  D. TTFA / RTF on representative texts
  E. Actual token usage vs v14 max_new_tokens caps (truncation risk)

Run: G:\qwen-tts\.conda\python.exe profile_v14.py [--model PATH]
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

# ------------------------------------------------------------------ run log
# Mirror stdout/stderr to .local/logs/profile_v14_<ts>.log (gitignored),
# so every run leaves a readable history even if the console window closes.
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".local", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_PATH = os.path.join(_LOG_DIR, f"profile_v14_{time.strftime('%Y%m%d_%H%M%S')}.log")
_log_file = open(_LOG_PATH, "w", encoding="utf-8")

class _Tee:
    def __init__(self, stream):
        self.stream = stream
    def write(self, s):
        self.stream.write(s)
        _log_file.write(s)
        if "\n" in s:
            self.stream.flush()
    def flush(self):
        self.stream.flush()
        _log_file.flush()

sys.stdout = _Tee(sys.stdout)
sys.stderr = _Tee(sys.stderr)
print(f"[LOG] saving run log to {_LOG_PATH}")

import numpy as np
import torch


def cuda_time(fn, iters=5):
    """Time a callable on GPU using events (median of `iters`)."""
    for _ in range(2):  # warmup
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.getenv('MODEL_PATH', r'G:\Foundation\models\Qwen3-TTS'))
    parser.add_argument('--speaker', default='Sohee')
    parser.add_argument('--skip-ab', action='store_true', help='Skip load+capture timing (still loads model)')
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA mem reserved at start: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print()

    # ------------------------------------------------------------------ A. Load
    from qwen_tts import Qwen3TTSModel
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(args.model, device_map='cuda:0', dtype=torch.bfloat16)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"[A] Model load: {load_ms:.0f} ms")

    st = model.model.speech_tokenizer
    mimi = st.model.decoder  # Mimi decoder module

    # ------------------------------------------------------------------ B. Graph capture
    warm_text = "Привет! Как дела? Это тест."
    t0 = time.perf_counter()
    gen = model.generate_custom_voice_streaming(
        text=warm_text, speaker=args.speaker, language='Russian',
        chunk_size=8, max_new_tokens=40, backend='auto')
    n_warm_chunks = 0
    for _ in gen:
        n_warm_chunks += 1
    capture_ms = (time.perf_counter() - t0) * 1000
    print(f"[B] First call incl. graph capture + {n_warm_chunks} chunks: {capture_ms:.0f} ms")

    # Second call — no capture, steady state
    t0 = time.perf_counter()
    gen = model.generate_custom_voice_streaming(
        text=warm_text, speaker=args.speaker, language='Russian',
        chunk_size=8, max_new_tokens=40, backend='auto')
    for _ in gen:
        pass
    second_ms = (time.perf_counter() - t0) * 1000
    print(f"[B] Second call (steady state, same text): {second_ms:.0f} ms")

    # ------------------------------------------------------------------ C. Mimi decode cost by context size
    print("\n[C] Mimi decoder cost vs window length (chunk_size=8, random codes):")
    Q = 16  # talker first token + 15 predictor codebooks
    for n_ctx in [0, 8, 16, 25]:
        T = n_ctx + 8
        codes = torch.randint(0, 128, (T, Q), dtype=torch.long)

        def run(codes=codes):
            st.decode({"audio_codes": codes.unsqueeze(0)})

        ms = cuda_time(run, iters=5)
        print(f"    ctx={n_ctx:2d} + new=8  (total {T:2d} frames): {ms:7.1f} ms")

    # Also: raw Mimi forward without the wrapper overhead (clamp/trim/chunked_decode)
    for T in [8, 33]:
        codes = torch.randint(0, 128, (T, Q), dtype=torch.long).unsqueeze(0).cuda()  # [1,T,Q]

        def run_raw(codes=codes):
            mimi(codes.transpose(1, 2))

        ms = cuda_time(run_raw, iters=5)
        print(f"    raw Mimi forward T={T:2d}: {ms:7.1f} ms")

    # ------------------------------------------------------------------ D. TTFA / RTF
    texts = [
        ('Russian', "Привет! Меня зовут Александр. Мне двадцать пять лет. Я живу в Санкт-Петербурге и работаю программистом уже пять лет."),
        ('English', "Hello! My name is John and I have been working as a software engineer for ten years. I live in New York with my wife and two cats."),
    ]
    print("\n[D] TTFA / RTF (chunk_size=8, max_new_tokens=400):")
    for lang, text in texts:
        t_start = time.perf_counter()
        first_chunk_ms = None
        total_audio_s = 0.0
        n_chunks = 0
        steps_total = 0
        gen = model.generate_custom_voice_streaming(
            text=text, speaker=args.speaker, language=lang,
            chunk_size=8, max_new_tokens=400, backend='auto')
        for audio_c, sr, timing in gen:
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter() - t_start) * 1000
            total_audio_s += len(audio_c) / sr
            n_chunks += 1
            steps_total = timing.get('total_steps_so_far', 0)
        wall_ms = (time.perf_counter() - t_start) * 1000
        rtf = total_audio_s / (wall_ms / 1000.0) if wall_ms > 0 else 0
        print(f"    [{lang:7s}] chunks={n_chunks:2d} steps={steps_total:3d} audio={total_audio_s:5.2f}s "
              f"TTFA={first_chunk_ms:6.0f}ms wall={wall_ms:6.0f}ms RTF={rtf:.2f}")

    # ------------------------------------------------------------------ E. Token usage vs caps
    print("\n[E] Actual tokens used per segment (cap=512) — check v14 cap tightness:")
    from fast_tts_v14 import split_segments, FastTTSv14
    long_text = ("Привет! Меня зовут Александр. Мне двадцать пять лет. Я живу в Санкт-Петербурге и работаю программистом уже пять лет. "
                 "Сегодня прекрасная погода для прогулки по городу. Мы с друзьями решили посетить Эрмитаж и затем поужинать в хорошем ресторане. "
                 "Технологии меняют наш мир каждый день. Искусственный интеллект помогает врачам ставить диагнозы, а роботы уже работают на производствах.")
    segments = split_segments(long_text, max_chars=85)
    for i, seg in enumerate(segments):
        words = len(seg.split())
        # v14 caps: <=2w->20, <=5w->50, <=10w->100, else 160
        cap = 20 if words <= 2 else 50 if words <= 5 else 100 if words <= 10 else 160
        gen = model.generate_custom_voice_streaming(
            text=seg, speaker=args.speaker, language='Russian',
            chunk_size=8, max_new_tokens=512, backend='auto')
        steps_total = 0
        audio_s = 0.0
        for _a, sr, timing in gen:
            steps_total = timing.get('total_steps_so_far', 0)
            audio_s += len(_a) / sr
        hit_cap = "HIT-CAP!" if steps_total >= 512 else ""
        print(f"    seg{i+1} words={words:2d} v14cap={cap:3d} actual_steps={steps_total:3d} audio={audio_s:4.2f}s {hit_cap}")

    print("\nDone.")


if __name__ == '__main__':
    main()
