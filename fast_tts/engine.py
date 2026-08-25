"""FastTTSv14 — true streaming Qwen3-TTS with CUDA-graph acceleration.

- CUDA graph warmup done ONCE during model loading
- TRUE streaming: producer thread -> queue -> player (no two-phase collection)
- Text segmentation for long inputs (split_segments)
- Peak-normalized WAV output (live playback uses raw model output)
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
import queue
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch

from .player import StreamingAudioPlayer


# ============================================================================
# HELPERS
# ============================================================================
def to_pcm_chunk(x) -> np.ndarray:
    """Safely convert audio chunk to float32 PCM."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def global_peak_normalize(audio_chunks: List[np.ndarray], limit: float = 0.95) -> List[np.ndarray]:
    """Peak-normalize the concatenated audio (applied to saved WAV only)."""
    if not audio_chunks:
        return audio_chunks
    full = np.concatenate(audio_chunks)
    peak = float(np.max(np.abs(full))) if full.size else 0.0
    if peak > limit and peak > 0:
        scale = limit / peak
        return [c * scale for c in audio_chunks]
    return audio_chunks


def split_segments(text: str, max_chars: int = 85) -> List[str]:
    """Split text into segments of at most max_chars characters.

    Splits on sentence boundaries first (.!?), then on commas/semicolons/colons,
    then on word boundaries if still too long.
    """
    pattern = r'([^.!?]+[.!?])|([^.!?]+$)'
    sentences = [m[0] if m[0] else m[1] for m in re.findall(pattern, text)]
    sentences = [s.strip() for s in sentences if s.strip()]
    segments = []
    for s in sentences:
        if len(s) <= max_chars:
            segments.append(s)
            continue
        parts = re.split(r'(?<=[,;:])\s+', s)
        buf = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if not buf:
                buf = p
            elif len(buf) + 1 + len(p) <= max_chars:
                buf = f"{buf} {p}"   # fixed: was missing space
            else:
                segments.append(buf)
                buf = p
        if buf:
            if len(buf) <= max_chars:
                segments.append(buf)
            else:
                words = buf.split()
                cur = ""
                for w in words:
                    if not cur:
                        cur = w
                    elif len(cur) + 1 + len(w) <= max_chars:
                        cur = f"{cur} {w}"  # one space between words
                    else:
                        segments.append(cur)
                        cur = w
                if cur:
                    segments.append(cur)
    return [s.strip() for s in segments if s.strip()]


# ============================================================================
# FAST TTS V14 — True Streaming
# ============================================================================
class FastTTSv14:
    """Fast streaming TTS with true producer-consumer pipeline."""

    def __init__(self, model_path: str, device: str = 'cuda:0', speaker: str = 'Sohee', device_id: Optional[int] = None) -> None:
        self.device = device
        self.speaker = speaker
        print(f"[V14] Loading Qwen3TTSModel from {model_path}...", flush=True)
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            print("Error: 'qwen_tts' package not found.", flush=True)
            print("Install it: pip install -U qwen-tts", flush=True)
            sys.exit(1)

        t0 = time.perf_counter()
        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
        )
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"[V14] Load: {load_ms:.0f}ms", flush=True)

        # Fail fast if this qwen-tts build lacks any API the streaming overlay needs.
        from ._compat import probe_model_api
        probe_model_api(self.model)

        # Preflight warmup — capture CUDA graphs and reach steady state.
        # Graphs are shape-based, not content-based: the talker graph has a position/
        # attention-mask table for every position up to 2048 (any prefill length works),
        # and the predictor graph is fixed-size (batch=1, seq=1). Neither language nor
        # chunk_size changes what gets captured — so a few short calls suffice.
        # (Was: 6 languages x 3 chunk sizes = 18 calls, ~30s of pure startup waste.)
        print("[V14] Preflight warmup...", flush=True)
        t0 = time.perf_counter()

        warmup_texts = {
            'Russian':   "Привет! Как дела? Это тест потоковой генерации речи.",
            'English':   "Hello! How are you doing today?",
            'Chinese':   "你好！我叫李明。",
        }

        for lang, text in warmup_texts.items():
            gen = self.model.generate_custom_voice_streaming(
                text=text,
                speaker=self.speaker,
                language=lang,
                chunk_size=8,
                max_new_tokens=50,
                backend='auto',
            )
            for _ in gen:
                pass
            gen.close()

        torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - t0) * 1000
        langs_list = ', '.join(warmup_texts.keys())
        print(f"[V14] Warmup: {warmup_ms:.0f}ms | Languages warmed up: {langs_list}", flush=True)

        # fixed: create player once and reuse across generate_and_play calls
        self.player = StreamingAudioPlayer(sample_rate=24000, device_id=device_id, preroll_sec=0.3)
        self.player.start()

        import atexit
        atexit.register(self.player.stop)

    def _get_max_new_tokens(self, text: str) -> int:
        # Caps are a safety net against runaway generation — normal speech ends at EOS.
        # profile_v14 section E measured a 1-word segment needing up to 39 codec steps,
        # so the old cap of 20 truncated short segments mid-speech.
        word_count = len(re.findall(r'\b\w+\b', text))
        if word_count <= 2: return 64
        elif word_count <= 5: return 50
        elif word_count <= 10: return 100
        else: return 160

    @torch.inference_mode()
    def generate_and_play(self, text: str, language: str = 'Russian', save_wav: Optional[str] = None,
                          chunk_size: int = 8, min_start_sec: float = 0.15) -> None:
        """True streaming: producer thread -> queue -> player."""
        segments = split_segments(text, max_chars=85)
        print(f"\n[V14] Generating {len(segments)} segments:", flush=True)
        for i, s in enumerate(segments):
            print(f"  [{i+1}] {s}", flush=True)

        if not segments:
            print("[V14] No text to generate.", flush=True)
            return

        # fixed: reuse self.player instead of creating new one each call
        self.player.reset()
        q = queue.Queue(maxsize=32)
        errors = []
        t_start = time.perf_counter()

        # TTFA tracking
        first_chunk_time = [None]
        started = [False]
        chunk_count_ref = [0]
        all_wavs = []

        # ------------------------------------------------------------------
        # PRODUCER: generate chunks, send to queue
        # fixed: q.put(None) moved OUTSIDE the for seg loop (was causing deadlock)
        # ------------------------------------------------------------------
        def producer():
            try:
                for seg in segments:
                    max_tokens = self._get_max_new_tokens(seg)

                    gen = self.model.generate_custom_voice_streaming(
                        text=seg,
                        speaker=self.speaker,
                        language=language,
                        chunk_size=chunk_size,
                        max_new_tokens=max_tokens,
                        backend='auto',
                    )

                    try:
                        for audio_chunk, sr, timing in gen:
                            if first_chunk_time[0] is None:
                                first_chunk_time[0] = time.perf_counter() - t_start

                            chunk = to_pcm_chunk(audio_chunk)
                            if chunk.size == 0:
                                continue

                            chunk_count_ref[0] += 1
                            decode_ms = timing.get('decode_ms', 0)
                            print(f"  Chunk {chunk_count_ref[0]}: len={len(chunk)} samples, decode={decode_ms:.0f}ms", flush=True)
                            all_wavs.append(chunk)

                            # fixed: queue.put blocks automatically when full (maxsize=32)
                            q.put(chunk)

                    finally:
                        gen.close()

                # fixed: sentinel AFTER all segments are processed
                q.put(None)

            except Exception as e:
                errors.append(e)
                print(f"  Producer error: {e}", flush=True)
                import traceback; traceback.print_exc()
                try:
                    q.put(None)
                except Exception:
                    pass

        gen_thread = threading.Thread(target=producer, daemon=True)
        gen_thread.start()

        # ------------------------------------------------------------------
        # CONSUMER: read from queue -> player (immediate playback)
        # ------------------------------------------------------------------
        max_buffer_sec = 3.5

        while True:
            try:
                chunk = q.get(timeout=0.1)
            except queue.Empty:
                if not gen_thread.is_alive():
                    break
                continue

            if chunk is None:
                break

            if chunk.size == 0:
                continue

            # Backpressure: wait BEFORE adding to player
            while self.player.buffered_seconds() > max_buffer_sec and not self.player.is_finished():
                time.sleep(0.01)

            self.player.add_chunk(chunk)

            # Start playback when buffer is warm
            if not started[0]:
                if self.player.buffered_seconds() >= min_start_sec:
                    self.player.request_start()
                    started[0] = True

        gen_thread.join(timeout=120)
        if gen_thread.is_alive():
            print("WARNING: generator thread did not finish in 120s", flush=True)

        self.player.add_chunk(None)

        if errors:
            raise errors[0]

        # Peak-normalize for the saved WAV only. Live playback already used raw chunks
        # (streamed audio cannot be rescaled retroactively), so the file may differ from
        # what was heard by a constant gain — expected, not a bug.
        all_wavs = global_peak_normalize(all_wavs)

        total_ms = (time.perf_counter() - t_start) * 1000
        ttfa_ms = first_chunk_time[0] * 1000 if first_chunk_time[0] else 0
        audio_total_s = sum(len(c) for c in all_wavs) / 24000.0

        print(f"\n[V14] Done!", flush=True)
        print(f"  Audio chunks: {chunk_count_ref[0]}", flush=True)
        print(f"  TTFA (time to first audio): {ttfa_ms:.0f}ms", flush=True)
        print(f"  Total wall time: {total_ms:.0f}ms", flush=True)
        print(f"  Total audio: {audio_total_s:.2f}s", flush=True)

        print("  Waiting for playback...", flush=True)
        self.player.wait(timeout=60.0)

        if save_wav and all_wavs:
            full_audio = np.concatenate(all_wavs)
            sf.write(save_wav, full_audio, 24000)
            print(f"  Saved: {save_wav}", flush=True)
