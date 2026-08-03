"""
True Streaming TTS v7 - PARALLEL LLM generation!
=================================================
Key insight: LLM generation is 91% of time, decoder is fast (8%).
Solution: Generate multiple segments in parallel using thread pool.
While segment 1 plays, segments 2-4 generate simultaneously on GPU.
"""
import torch, time, numpy as np, threading, queue, soundfile as sf, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from qwen_tts import Qwen3TTSModel


# ============================================================================
# STREAMING AUDIO PLAYER (callback-based)
# ============================================================================
class StreamingAudioPlayer:
    """Continuous callback-based audio player using OutputStream."""

    def __init__(self, sample_rate=24000, device_id=3, blocksize=1024, preroll_sec=0.6):
        self.sample_rate = sample_rate
        self.device_id = device_id
        self.blocksize = blocksize

        import sounddevice as sd
        self.sd = sd

        self._lock = threading.Lock()
        self._chunks = deque()
        self._current = None
        self._offset = 0
        self._buffered = 0

        self._done = False
        self._started = False
        self._finished = threading.Event()
        self._underruns = 0
        self._preroll = int(sample_rate * preroll_sec)
        self._stream = None

        name = sd.query_devices()[device_id]["name"]
        print(f"    Player: dev={device_id}, sr={sample_rate}, preroll={preroll_sec:.2f}s", flush=True)

    def start(self):
        self._stream = self.sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            device=self.device_id, blocksize=self.blocksize, latency="high",
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status):
        out = outdata[:, 0]
        out.fill(0.0)

        with self._lock:
            if self._finished.is_set():
                return

            if not self._started:
                if self._buffered >= self._preroll or self._done:
                    self._started = True
                else:
                    return

            written = 0
            while written < frames:
                if self._current is None:
                    if not self._chunks:
                        self._underruns += 1
                        break
                    self._current = self._chunks.popleft()
                    self._offset = 0

                n = min(frames - written, len(self._current) - self._offset)
                out[written:written + n] = self._current[self._offset:self._offset + n]
                self._offset += n
                written += n
                self._buffered -= n

                if self._offset >= len(self._current):
                    self._current = None

            if self._done and self._buffered == 0 and self._current is None and not self._chunks:
                self._finished.set()

    def add_chunk(self, chunk):
        if chunk is None:
            with self._lock:
                self._done = True
            return
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        peak = float(np.max(np.abs(chunk)))
        if peak > 1.0:
            chunk = chunk * (0.99 / peak)
        with self._lock:
            self._chunks.append(np.ascontiguousarray(chunk))
            self._buffered += chunk.size

    def buffered_seconds(self):
        with self._lock:
            return self._buffered / float(self.sample_rate)

    def request_start(self):
        with self._lock:
            self._started = True

    def is_finished(self):
        return self._finished.is_set()

    def wait(self, timeout=None):
        return self._finished.wait(timeout)

    def stop(self, timeout=5.0):
        with self._lock:
            self._done = True
        self.wait(timeout)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        print(f"    Underruns: {self._underruns}", flush=True)


# ============================================================================
# AUDIO PROCESSING
# ============================================================================
def trim_silence(wav, sr=24000, threshold=0.0006, keep_sec=0.012):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return wav
    idx = np.where(np.abs(wav) > threshold)[0]
    if idx.size == 0:
        return np.zeros(0, dtype=np.float32)
    keep = int(keep_sec * sr)
    start = max(0, int(idx[0]) - keep)
    end = min(wav.size, int(idx[-1]) + keep + 1)
    return wav[start:end]


def apply_fades(wav, sr=24000, in_ms=5, out_ms=25):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return wav
    n_in = int(sr * in_ms / 1000.0)
    n_out = int(sr * out_ms / 1000.0)
    if n_in > 1 and wav.size >= n_in:
        t = np.linspace(0.0, 1.0, n_in, dtype=np.float32)
        wav[:n_in] *= t
    if n_out > 1 and wav.size >= n_out:
        t = np.linspace(1.0, 0.0, n_out, dtype=np.float32)
        wav[-n_out:] *= t
    return wav


# ============================================================================
# TEXT SEGMENTATION
# ============================================================================
def split_segments(text, max_chars=85):
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
                buf = f"{buf} {p}"
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
                        cur = f"{cur} {w}"
                    else:
                        segments.append(cur)
                        cur = w
                if cur:
                    segments.append(cur)

    return [s.strip() for s in segments if s.strip()]


# ============================================================================
# PARALLEL STREAMING TTS v7
# ============================================================================
class StreamingTTSv7:
    """Parallel generation: multiple segments at once on GPU."""

    def __init__(self, model_path, device='cuda:0'):
        self.device = device
        print(f"[V7] Loading...", flush=True)
        self.model = Qwen3TTSModel.from_pretrained(
            model_path, device_map=device, dtype=torch.bfloat16,
            attn_implementation='sdpa',
        )
        torch.cuda.synchronize()

        # Warmup
        print("[V7] Warming up...", flush=True)
        for _ in range(3):
            _ = self.model.generate_custom_voice(
                text="Привет", language='Russian', speaker='Sohee', max_new_tokens=15
            )
        torch.cuda.synchronize()
        print("[V7] Ready!", flush=True)

    def _get_max_new_tokens(self, text, language):
        word_count = len(re.findall(r'\b\w+\b', text)) if language != 'Chinese' else len(text)
        if word_count <= 2: return 15
        elif word_count <= 5: return 45
        elif word_count <= 10: return 90
        else: return 150

    def _decode_chunked(self, codes):
        """Fast chunked decode to numpy."""
        tokenizer_wrapper = self.model.model.speech_tokenizer
        decoder = tokenizer_wrapper.model.decoder

        if codes.dim() == 2:
            codes_t = codes.unsqueeze(0).transpose(1, 2)
        else:
            codes_t = codes.unsqueeze(0).unsqueeze(1)

        chunk_size = 50
        all_wavs = []
        seq_len = codes.shape[0]

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            x = codes_t[..., start:end]
            wav = decoder.chunked_decode(x, chunk_size=chunk_size, left_context_size=25)
            wav = np.asarray(wav.float().detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            if wav.size > 0 and np.abs(wav).max() > 0.001:
                all_wavs.append(wav)

        if all_wavs:
            return np.concatenate(all_wavs)
        return np.zeros(0, dtype=np.float32)

    def _generate_segment(self, text, language, speaker):
        """Generate and decode a single segment. Returns (wav, gen_time_ms)."""
        if not text.strip():
            return np.zeros(0, dtype=np.float32), 0.0

        max_new_tokens = self._get_max_new_tokens(text, language)
        gen_kwargs = self.model._merge_generate_kwargs(
            max_new_tokens=max_new_tokens,
            do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
        )

        input_ids = self.model._tokenize_texts([self.model._build_assistant_text(text)])
        instruct_ids = [None]

        t0 = time.perf_counter()
        talker_codes_list, _ = self.model.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=[language],
            speakers=[speaker],
            non_streaming_mode=True,
            **gen_kwargs,
        )
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - t0) * 1000

        codes = talker_codes_list[0]
        wav = self._decode_chunked(codes)
        wav = trim_silence(wav, sr=24000, threshold=0.0006, keep_sec=0.012)
        wav = apply_fades(wav, sr=24000, in_ms=5, out_ms=25)

        return wav, gen_ms

    @torch.no_grad()
    def generate_streaming(self, text, language='Russian', speaker='Sohee'):
        """Parallel generation: all segments start at once!"""

        segments = split_segments(text, max_chars=85)
        print(f"\n▶ Streaming {len(segments)} segments (PARALLEL):", flush=True)
        for i, s in enumerate(segments):
            print(f"  [{i+1}] {s}", flush=True)

        if not segments:
            return [np.zeros(0, dtype=np.float32)], 24000, 0.0

        # Start player
        player = StreamingAudioPlayer(sample_rate=24000, device_id=3, preroll_sec=0.6)
        player.start()

        t_start = time.perf_counter()
        all_wavs = []
        completed = {}  # index -> wav

        # ===================================================================
        # PARALLEL GENERATION: start ALL segments at once!
        # GPU can process multiple forward passes in parallel.
        # Python GIL is released during CUDA kernel execution.
        # ===================================================================
        print(f"\n[PARALLEL] Starting {len(segments)} generators...", flush=True)

        def generate_and_queue(seg_idx, text):
            wav, gen_ms = self._generate_segment(text, language, speaker)
            completed[seg_idx] = wav
            print(f"  [Done {seg_idx+1}] {gen_ms:.0f}ms, audio={len(wav)/24000:.2f}s", flush=True)

        # Use thread pool - GIL released during GPU ops!
        with ThreadPoolExecutor(max_workers=min(4, len(segments))) as executor:
            futures = {
                executor.submit(generate_and_queue, i, s): i
                for i, s in enumerate(segments)
            }

            # Main loop: feed player as segments complete
            first = True
            max_buffer_sec = 4.0

            while len(completed) < len(segments):
                # Check for completed futures
                for future in list(futures.keys()):
                    if future.done() and not future.exception():
                        seg_idx = futures[future]
                        wav = completed.get(seg_idx)
                        if wav is not None and wav.size > 0:
                            all_wavs.append(wav)
                            player.add_chunk(wav)

                            if first:
                                player.request_start()
                                first = False

                            # Throttle buffer
                            while player.buffered_seconds() > max_buffer_sec and not player.is_finished():
                                time.sleep(0.01)

                        del futures[future]

                if not futures:
                    break
                time.sleep(0.05)

        # Signal end
        player.add_chunk(None)

        # Wait for playback
        if all_wavs:
            final_wav = np.concatenate(all_wavs)
        else:
            final_wav = np.zeros(0, dtype=np.float32)

        expected_duration = len(final_wav) / 24000 + 5.0
        print(f"\n    Waiting {expected_duration:.1f}s for playback...", flush=True)
        player.wait(timeout=expected_duration)
        player.stop()

        total_ms = (time.perf_counter() - t_start) * 1000
        print(f"\nTotal: {total_ms:.0f}ms | Audio: {len(final_wav)/24000:.2f}s", flush=True)

        return [final_wav], 24000, total_ms


# ============================================================================
# MAIN
# ============================================================================
def main():
    import sys
    model_path = r'G:\Foundation\models\Qwen3-TTS'

    text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Меня зовут Александр. Мне двадцать пять лет. Я живу в Санкт-Петербурге. "
        "Работаю программистом уже пять лет. Это тест потоковой генерации."
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Text: {text}", flush=True)

    tts = StreamingTTSv7(model_path)
    wavs, sr, dt_ms = tts.generate_streaming(text)

    if len(wavs[0]) > 0:
        sf.write('tts_output_streaming_v7.wav', wavs[0], sr)
        print(f"\nSaved: tts_output_streaming_v7.wav", flush=True)
    else:
        print("\nWarning: Empty audio!", flush=True)


if __name__ == '__main__':
    main()
