"""
Benchmark: FasterQwen3TTS (CUDA graphs) vs plain Qwen3TTSModel on our local model.

Measures:
  - TTFA (Time To First Audio) via streaming, chunk_size=8
  - RTF (Real-Time Factor) = audio_duration / wall_time
  - ms/step for non-streaming generation
"""
import os, sys, time, json
import numpy as np
import soundfile as sf

MODEL_PATH = r'G:\Foundation\models\Qwen3-TTS'
SPEAKER   = 'Sohee'
LANGUAGE  = 'Russian'
TEXT      = "Меня зовут Александр. Мне двадцать пять лет. Я живу в Санкт-Петербурге. Работаю программистом уже пять лет. Это тест потоковой генерации."

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")


# ============================================================
# 1) FasterQwen3TTS (CUDA graphs path)
# ============================================================
def bench_faster():
    from faster_qwen3_tts import FasterQwen3TTS

    print("\n=== FasterQwen3TTS (CUDA graphs) ===")
    print("Loading model...")
    t_load = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(
        MODEL_PATH,
        device='cuda',
        dtype=torch.bfloat16,
        attn_implementation='sdpa',
        max_seq_len=2048,
    )
    print(f"Load time: {time.perf_counter() - t_load:.1f}s")

    # Check model type
    tts_type = model.model.model.tts_model_type
    print(f"Model type: {tts_type}")

    spk_dict = model.model.model.config.talker_config.spk_id
    speakers = list(spk_dict.keys()) if spk_dict else []
    print(f"Available speakers (first 5): {speakers[:5]}")

    # Warmup (captures CUDA graphs)
    print("\nWarmup (CUDA graph capture)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate_custom_voice(
        text="Привет", speaker=SPEAKER, language=LANGUAGE, max_new_tokens=20,
    )
    torch.cuda.synchronize()
    print(f"Warmup: {(time.perf_counter() - t0)*1000:.0f}ms")

    results = {}

    # --- TTFA via streaming (5 runs) ---
    chunk_size = 8
    print(f"\nTTFA streaming ({chunk_size} chunk, 5 runs)...")
    ttfas = []
    for run in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = model.generate_custom_voice_streaming(
            text=TEXT, speaker=SPEAKER, language=LANGUAGE,
            chunk_size=chunk_size, max_new_tokens=200,
        )
        first_chunk, sr, timing = next(gen)
        torch.cuda.synchronize()
        ttfa_ms = (time.perf_counter() - t0) * 1000
        ttfas.append(ttfa_ms)
        gen.close()
        print(f"  Run {run+1}: {ttfa_ms:.0f}ms")

    results['ttfa_mean'] = float(np.mean(ttfas))
    results['ttfa_std']  = float(np.std(ttfas))
    print(f"  => TTFA: {results['ttfa_mean']:.0f}ms ± {results['ttfa_std']:.0f}ms")

    # --- RTF non-streaming (3 runs) ---
    print(f"\nRTF non-streaming (3 runs)...")
    rtfs = []
    ms_per_steps = []
    for run in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        audio_list, sr = model.generate_custom_voice(
            text=TEXT, speaker=SPEAKER, language=LANGUAGE, max_new_tokens=200,
        )
        torch.cuda.synchronize()
        total_s = time.perf_counter() - t0
        audio = audio_list[0]
        audio_dur = len(audio) / sr
        rtf = audio_dur / total_s if total_s > 0 else 0
        n_steps = int(round(audio_dur * 12))
        ms_step = (total_s / max(n_steps, 1)) * 1000
        rtfs.append(rtf)
        ms_per_steps.append(ms_step)
        print(f"  Run {run+1}: {n_steps} steps, {ms_step:.1f}ms/step, audio={audio_dur:.2f}s, wall={total_s:.2f}s, RTF={rtf:.3f}")

    results['rtf_mean']     = float(np.mean(rtfs))
    results['rtf_std']      = float(np.std(rtfs))
    results['ms_per_step']  = float(np.mean(ms_per_steps))

    # Save sample audio
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_faster_output.wav')
    sf.write(out_path, audio_list[0], sr)
    print(f"\nSaved: {out_path}")

    # Explicitly delete model to free GPU before baseline run
    del model
    torch.cuda.empty_cache()

    return results


# ============================================================
# 2) Plain Qwen3TTSModel (our v9 baseline — no CUDA graphs)
# ============================================================
def bench_baseline():
    from qwen_tts import Qwen3TTSModel

    print("\n=== Baseline: plain Qwen3TTSModel (no CUDA graphs) ===")
    print("Loading model...")
    t_load = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH, device_map='cuda', dtype=torch.bfloat16, attn_implementation='sdpa',
    )
    print(f"Load time: {time.perf_counter() - t_load:.1f}s")

    # Warmup
    print("Warmup...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate_custom_voice(text="Привет", language=LANGUAGE, speaker=SPEAKER, max_new_tokens=20)
    torch.cuda.synchronize()
    print(f"Warmup: {(time.perf_counter() - t0)*1000:.0f}ms")

    results = {}

    # --- TTFA (first token + decode time) ---
    print(f"\nTTFA non-streaming (5 runs)...")
    ttfas = []
    for run in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        audio_list, sr = model.generate_custom_voice(
            text=TEXT, language=LANGUAGE, speaker=SPEAKER, max_new_tokens=200,
        )
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000
        ttfas.append(total_ms)
        print(f"  Run {run+1}: {total_ms:.0f}ms")

    results['ttfa_mean'] = float(np.mean(ttfas))
    results['ttfa_std']  = float(np.std(ttfas))

    # --- RTF (3 runs) ---
    print(f"\nRTF non-streaming (3 runs)...")
    rtfs = []
    ms_per_steps_list = []
    for run in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        audio_list, sr = model.generate_custom_voice(
            text=TEXT, language=LANGUAGE, speaker=SPEAKER, max_new_tokens=200,
        )
        torch.cuda.synchronize()
        total_s = time.perf_counter() - t0
        audio_dur = len(audio_list[0]) / sr
        rtf = audio_dur / total_s if total_s > 0 else 0
        n_steps = int(round(audio_dur * 12))
        ms_step = (total_s / max(n_steps, 1)) * 1000
        rtfs.append(rtf)
        ms_per_steps_list.append(ms_step)
        print(f"  Run {run+1}: {n_steps} steps, {ms_step:.1f}ms/step, audio={audio_dur:.2f}s, wall={total_s:.2f}s, RTF={rtf:.3f}")

    results['rtf_mean']     = float(np.mean(rtfs))
    results['rtf_std']      = float(np.std(rtfs))
    results['ms_per_step']  = float(np.mean(ms_per_steps_list))

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    faster_results = bench_faster()  # model already deleted inside

    baseline_results = bench_baseline()

    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} {'Faster (CUDA graphs)':>20} {'Baseline (plain)':>20}")
    print("-"*65)

    # TTFA
    f_ttfa = faster_results['ttfa_mean']
    b_ttfa = baseline_results['ttfa_mean']
    speedup_ttfa = b_ttfa / f_ttfa if f_ttfa > 0 else 0
    print(f"{'TTFA (ms)':<25} {f_ttfa:>20.0f} {b_ttfa:>20.0f}")
    print(f"{'  -> speedup':<25} {'':>19}{speedup_ttfa:.2f}x")

    # RTF
    f_rtf = faster_results['rtf_mean']
    b_rtf = baseline_results['rtf_mean']
    rtf_ratio = f_rtf / b_rtf if b_rtf > 0 else 0
    print(f"{'RTF':<25} {f_rtf:>20.3f} {b_rtf:>20.3f}")
    print(f"{'  -> ratio (higher=better)':<25} {'':>19}{rtf_ratio:.2f}x")

    # ms/step
    f_ms = faster_results['ms_per_step']
    b_ms = baseline_results['ms_per_step']
    speedup_step = b_ms / f_ms if f_ms > 0 else 0
    print(f"{'ms/step':<25} {f_ms:>20.1f} {b_ms:>20.1f}")
    print(f"{'  -> speedup':<25} {'':>19}{speedup_step:.2f}x")

    # Save JSON
    out = {
        'model': MODEL_PATH,
        'speaker': SPEAKER,
        'text': TEXT,
        'gpu': torch.cuda.get_device_name(0),
        'pytorch': torch.__version__,
        'faster_cuda_graphs': faster_results,
        'baseline_plain': baseline_results,
    }
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {json_path}")
