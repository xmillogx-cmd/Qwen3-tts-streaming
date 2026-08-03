"""
True Streaming TTS v6 - Correct producer-consumer pipeline
===========================================================
Key fixes:
1. Producer thread generates segments, puts into queue
2. Main thread consumes from queue and feeds player
3. player.add_chunk(None) ONLY after generator finishes
4. trim_silence + apply_fades for smooth boundaries
5. Use wrapper.decode() for safety first
"""
import torch, time, numpy as np, threading, queue, soundfile as sf, re
from collections import deque
from qwen_tts import Qwen3TTSModel


# ============================================================================
# STREAMING AUDIO PLAYER (callback-based)
# ============================================================================
class StreamingAudioPlayer:
    """Continuous callback-based audio player using OutputStream."""

    def __init__(
        self,
        sample_rate=24000,
        device_id=3,
        blocksize=1024,
        preroll_sec=0.6,
        latency="high",
    ):
        self.sample_rate = sample_rate
        self.device_id = device_id
        self.blocksize = blocksize
        self.latency = latency

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
        print(
            f"    Audio player: device={device_id}, name={name}, "
            f"sr={sample_rate}, block={blocksize}, preroll={preroll_sec:.2f}s",
            flush=True,
        )

    def start(self):
        self._stream = self.sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device_id,
            blocksize=self.blocksize,
            latency=self.latency,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status):
        """PortAudio callback - feed audio data continuously."""
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

            if (
                self._done
                and self._buffered == 0
                and self._current is None
                and not self._chunks
            ):
                self._finished.set()

    def add_chunk(self, chunk):
        """Add audio chunk to playback queue."""
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

        chunk = np.ascontiguousarray(chunk)

        with self._lock:
            self._chunks.append(chunk)
            self._buffered += chunk.size

    def buffered_seconds(self):
        """Get current buffer duration in seconds."""
        with self._lock:
            return self._buffered / float(self.sample_rate)

    def request_start(self):
        """Force start playback immediately (bypass preroll)."""
        with self._lock:
            self._started = True

    def is_finished(self):
        """Check if playback has finished."""
        return self._finished.is_set()

    def wait(self, timeout=None):
        """Wait for playback to finish."""
        return self._finished.wait(timeout)

    def stop(self, timeout=5.0):
        """Stop playback with graceful drain."""
        with self._lock:
            self._done = True

        self.wait(timeout)

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

        print(f"    Player underruns: {self._underruns}", flush=True)


# ============================================================================
# AUDIO PROCESSING HELPERS
# ============================================================================
def trim_silence(wav, sr=24000, threshold=0.0006, keep_sec=0.012):
    """Trim leading/trailing silence from audio."""
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
    """Apply short fades to avoid clicks at segment boundaries."""
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
def split_sentences(text):
    """Split text into sentences by punctuation."""
    pattern = r'([^.!?]+[.!?])|([^.!?]+$)'
    matches = re.findall(pattern, text)
    
    sentences = []
    for match in matches:
        sentence = match[0] if match[0] else match[1]
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence)
    
    return sentences


def split_segments(text, max_chars=85):
    """
    Split text into shorter TTS segments.
    
    Short segments => faster first audio and easier parallel pipeline.
    Too short segments => worse prosody.
    """
    sentences = split_sentences(text)
    segments = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if len(s) <= max_chars:
            segments.append(s)
            continue

        # Split long sentence by commas / semicolons / colons.
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
            # If still too long, split by words.
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
# STREAMING TTS v6 (correct producer-consumer pipeline)
# ============================================================================
class StreamingTTSv6:
    """True streaming TTS with correct producer-consumer pipeline."""

    def __init__(self, model_path, device='cuda:0'):
        self.device = device
        print(f"[V6] Loading...", flush=True)
        self.model = Qwen3TTSModel.from_pretrained(
            model_path, device_map=device, dtype=torch.bfloat16,
            attn_implementation='sdpa',
        )
        torch.cuda.synchronize()

        # Warmup
        print("[V6] Warming up...", flush=True)
        for _ in range(3):
            _ = self.model.generate_custom_voice(
                text="Привет", language='Russian', speaker='Sohee', max_new_tokens=15
            )
        torch.cuda.synchronize()
        print("[V6] Ready!", flush=True)

    def _get_max_new_tokens(self, text, language):
        if language == 'Chinese':
            word_count = len(text)
        else:
            word_count = len(re.findall(r'\b\w+\b', text))

        if word_count <= 2: return 15
        elif word_count <= 5: return 45
        elif word_count <= 10: return 90
        else: return 150

    @torch.no_grad()
    def _generate_and_decode_chunked(self, text, language, speaker, q):
        """Generate tokens and decode in chunks, feeding player immediately."""
        if not text.strip():
            return

        max_new_tokens = self._get_max_new_tokens(text, language)

        gen_kwargs = self.model._merge_generate_kwargs(
            max_new_tokens=max_new_tokens,
            do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
        )

        input_ids = self.model._tokenize_texts([self.model._build_assistant_text(text)])
        instruct_ids = [None]

        # Generate codec tokens (FAST - milliseconds)
        talker_codes_list, _ = self.model.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=[language],
            speakers=[speaker],
            non_streaming_mode=True,
            **gen_kwargs,
        )

        torch.cuda.synchronize()
        codes = talker_codes_list[0]
        seq_len = codes.shape[0]

        print(f"    Tokens: {seq_len}, decoding in chunks...", flush=True)

        # Get decoder for chunked decoding
        tokenizer_wrapper = self.model.model.speech_tokenizer
        decoder = tokenizer_wrapper.model.decoder

        # Prepare input shape (1, Q, T)
        if codes.dim() == 2:
            codes_t = codes.unsqueeze(0).transpose(1, 2)
        else:
            codes_t = codes.unsqueeze(0).unsqueeze(1)

        # Decode in chunks and feed player immediately
        chunk_size = 50
        all_wavs = []

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            x = codes_t[..., start:end]

            wav = decoder.chunked_decode(
                x,
                chunk_size=chunk_size,
                left_context_size=25
            )

            wav = np.asarray(wav.float().detach().cpu().numpy(), dtype=np.float32).reshape(-1)

            if wav.size > 0 and np.abs(wav).max() > 0.001:
                # Trim silence and apply fades
                wav = trim_silence(wav, sr=24000, threshold=0.0006, keep_sec=0.012)
                wav = apply_fades(wav, sr=24000, in_ms=5, out_ms=25)

                if wav.size > 0:
                    all_wavs.append(wav)
                    q.put(wav)  # Feed to player immediately!

        print(f"    Decoded {len(all_wavs)} chunks", flush=True)

    @torch.no_grad()
    def generate_streaming(self, text, language='Russian', speaker='Sohee'):
        """True parallel pipeline: producer -> queue -> player."""
        
        # Split into segments
        segments = split_segments(text, max_chars=85)
        print(f"\n▶ Streaming {len(segments)} segments:", flush=True)
        for i, s in enumerate(segments):
            print(f"  [{i+1}] {s}", flush=True)

        if not segments:
            return [np.zeros(0, dtype=np.float32)], 24000, 0.0

        # Start playback
        player = StreamingAudioPlayer(
            sample_rate=24000,
            device_id=3,
            blocksize=1024,
            preroll_sec=0.6,
            latency="high",
        )
        player.start()

        # Queue for producer-consumer
        q = queue.Queue(maxsize=2)
        errors = []

        t_start = time.perf_counter()

        # ------------------------------------------------------------------
        # PRODUCER: generates and decodes chunks in background
        # ------------------------------------------------------------------
        def producer():
            try:
                for i, s in enumerate(segments):
                    sent_start = time.perf_counter()

                    self._generate_and_decode_chunked(
                        s, language, speaker, q
                    )

                    gen_ms = (time.perf_counter() - sent_start) * 1000

                    print(
                        f"  [Seg {i+1}/{len(segments)}] "
                        f"{gen_ms:.0f}ms total",
                        flush=True,
                    )

                # Signal end of generation
                q.put(None)

            except Exception as e:
                errors.append(e)
                print(f"    Producer error: {e}", flush=True)
                q.put(None)

        gen_thread = threading.Thread(target=producer, daemon=True)
        gen_thread.start()

        # ------------------------------------------------------------------
        # MAIN FEEDER: consumes from queue and feeds player
        # ------------------------------------------------------------------
        all_wavs = []
        first = True
        max_buffer_sec = 3.5

        while True:
            try:
                item = q.get(timeout=0.2)
            except queue.Empty:
                if not gen_thread.is_alive():
                    break
                continue

            if item is None:
                break

            wav = item
            if wav.size == 0:
                continue

            all_wavs.append(wav)
            player.add_chunk(wav)

            # Low-latency start: start playback as soon as first audio arrives
            if first:
                player.request_start()
                first = False

            # Throttle if buffer gets too large
            while player.buffered_seconds() > max_buffer_sec and not player.is_finished():
                time.sleep(0.01)

        # CRITICAL FIX: wait for ALL segments to be generated FIRST
        gen_thread.join(timeout=120)

        # NOW signal end of stream (all audio has been generated and queued)
        player.add_chunk(None)

        if errors:
            player.stop()
            raise errors[0]

        # Wait for playback to finish
        if all_wavs:
            final_wav = np.concatenate(all_wavs)
        else:
            final_wav = np.zeros(0, dtype=np.float32)

        expected_duration = len(final_wav) / 24000 + 6.0
        print(f"\n    Waiting {expected_duration:.1f}s for playback...", flush=True)
        player.wait(timeout=expected_duration)
        player.stop()

        total_ms = (time.perf_counter() - t_start) * 1000
        
        print(f"\nTotal time: {total_ms:.0f}ms", flush=True)
        print(f"Audio duration: {len(final_wav)/24000:.2f}s", flush=True)

        return [final_wav], 24000, total_ms


# ============================================================================
# MAIN
# ============================================================================
def main():
    import sys
    model_path = r'G:\Foundation\models\Qwen3-TTS'
    
    # Multi-sentence text to test pipeline
    text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Меня зовут Александр. Мне двадцать пять лет. Я живу в Санкт-Петербурге. "
        "Работаю программистом уже пять лет. Это тест потоковой генерации."
    )
    
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Text: {text}", flush=True)

    # Load streaming TTS
    tts = StreamingTTSv6(model_path)

    # Generate with pipeline streaming
    wavs, sr, dt_ms = tts.generate_streaming(text)

    # Save output
    if len(wavs[0]) > 0:
        sf.write('tts_output_streaming_v6.wav', wavs[0], sr)
        print(f"\nSaved: tts_output_streaming_v6.wav", flush=True)
    else:
        print("\nWarning: Empty audio output!", flush=True)


if __name__ == '__main__':
    main()
