"""
Fast TTS v14 — True Streaming Playback
=======================================
- CUDA graph warmup done ONCE during model loading
- TRUE streaming: producer thread -> queue -> player (no two-phase collection)
- Text segmentation for long inputs (split_segments)
- Per-chunk normalization for consistent loudness
- Live speaker output via sounddevice
"""
import torch, time, numpy as np, threading, queue, soundfile as sf, re, sys


# ============================================================================
# HELPERS
# ============================================================================
def to_pcm_chunk(x):
    """Safely convert audio chunk to float32 PCM - always a real copy."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32, copy=True).reshape(-1)


def safe_normalize(chunk, limit=0.99):
    """Per-chunk clip protection - prevents loudness pumping between chunks."""
    peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
    if peak > limit:
        chunk = chunk * (limit / peak)
    return chunk


def split_segments(text, max_chars=85):
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
# STREAMING AUDIO PLAYER (callback-based)
# ============================================================================
class StreamingAudioPlayer:
    def __init__(self, sample_rate=24000, device_id=None, blocksize=1024, preroll_sec=0.3):
        self.sample_rate = sample_rate
        import sounddevice as sd
        self.sd = sd

        if device_id is None:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if 'loopback' in d.get('name', '').lower():
                    device_id = i
                    break
            if device_id is None:
                device_id = sd.default.device[1] or 0

        self.device_id = device_id
        self.blocksize = blocksize
        self._lock = threading.Lock()
        self._chunks = queue.Queue(maxsize=32)
        self._current = None
        self._offset = 0
        self._buffered = 0
        self._done = False
        self._started = False
        self._finished = threading.Event()
        self._underruns = 0
        self._preroll = int(sample_rate * preroll_sec)
        self._stream = None

    def start(self):
        print(f"  [Audio] Device {self.device_id}: {self.sd.query_devices()[self.device_id]['name'][:50]}", flush=True)
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
                    try:
                        self._current = self._chunks.get_nowait()
                        self._offset = 0
                    except queue.Empty:
                        self._underruns += 1
                        break
                n = min(frames - written, len(self._current) - self._offset)
                out[written:written + n] = self._current[self._offset:self._offset + n]
                self._offset += n
                written += n
                self._buffered -= n
                if self._offset >= len(self._current):
                    self._current = None
            if self._done and self._buffered == 0 and self._current is None:
                try:
                    self._chunks.get_nowait()
                except queue.Empty:
                    pass
                self._finished.set()

    def add_chunk(self, chunk):
        """Accepts pre-normalized float32 PCM chunks. No internal normalize."""
        if chunk is None:
            with self._lock:
                self._done = True
            return
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        try:
            self._chunks.put_nowait(np.ascontiguousarray(chunk))
            with self._lock:
                self._buffered += chunk.size
        except queue.Full:
            pass

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
        print(f"  [Audio] Underruns: {self._underruns}", flush=True)


# ============================================================================
# FAST TTS V14 — True Streaming with Crossfade
# ============================================================================
class FastTTSv14:
    """Fast streaming TTS with true producer-consumer pipeline + crossfade."""

    def __init__(self, model_path, device='cuda:0', speaker='Sohee'):
        self.device = device
        self.speaker = speaker
        print(f"[V14] Loading Qwen3TTSModel from {model_path}...", flush=True)
        from qwen_tts import Qwen3TTSModel

        t0 = time.perf_counter()
        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
        )
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"[V14] Load: {load_ms:.0f}ms", flush=True)

        # Warmup - captures CUDA graphs with realistic prefill length
        print("[V14] Warming up CUDA graphs...", flush=True)
        t0 = time.perf_counter()

        self.model.generate_custom_voice(
            text="Привет, это тест потоковой генерации с кроссфейдом. Теперь звук должен быть плавным без щелчков!",
            speaker=self.speaker,
            language='Russian',
            max_new_tokens=15,
        )

        warmup_gen = self.model.generate_custom_voice_streaming(
            text="Привет! Как дела? Я живу в Москве. Это тест потоковой генерации.",
            speaker=self.speaker,
            language='Russian',
            chunk_size=8,
            max_new_tokens=30,
            backend='auto',
        )
        for _ in warmup_gen:
            pass

        torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - t0) * 1000
        print(f"[V14] Warmup: {warmup_ms:.0f}ms | Ready!", flush=True)

    def _get_max_new_tokens(self, text):
        word_count = len(re.findall(r'\b\w+\b', text))
        if word_count <= 2: return 20
        elif word_count <= 5: return 50
        elif word_count <= 10: return 100
        else: return 160

    @torch.inference_mode()
    def generate_and_play(self, text, language='Russian', save_wav=None):
        """True streaming: producer thread -> queue -> player with crossfade."""
        segments = split_segments(text, max_chars=85)
        print(f"\n[V14] Generating {len(segments)} segments:", flush=True)
        for i, s in enumerate(segments):
            print(f"  [{i+1}] {s}", flush=True)

        if not segments:
            return [np.zeros(0, dtype=np.float32)], 24000, {}

        # Audio player
        player = StreamingAudioPlayer(sample_rate=24000, preroll_sec=0.3)
        player.start()

        q = queue.Queue(maxsize=32)
        errors = []
        t_start = time.perf_counter()

        # TTFA tracking
        first_chunk_time = [None]
        MIN_START_SEC = 1.0  # wait longer before starting playback to avoid underruns
        started = [False]
        chunk_count_ref = [0]
        first_seg_done = [False]

        # ------------------------------------------------------------------
        # PRODUCER: generate chunks, apply crossfade, send to queue
        # ------------------------------------------------------------------
        def producer():
            try:
                for seg in segments:
                    max_tokens = self._get_max_new_tokens(seg)

                    gen = self.model.generate_custom_voice_streaming(
                        text=seg,
                        speaker=self.speaker,
                        language=language,
                        chunk_size=8,
                        max_new_tokens=max_tokens,
                        backend='auto',
                    )

                    try:
                        for audio_chunk, sr, timing in gen:
                            if first_chunk_time[0] is None:
                                first_chunk_time[0] = time.perf_counter() - t_start

                            # Convert to numpy
                            chunk = to_pcm_chunk(audio_chunk)
                            if chunk.size == 0:
                                continue

                            chunk_count_ref[0] += 1
                            decode_ms = timing.get('decode_ms', 0)
                            print(f"  Chunk {chunk_count_ref[0]}: len={len(chunk)} samples, decode={decode_ms:.0f}ms", flush=True)

                            # Per-chunk normalization
                            chunk = safe_normalize(chunk)

                            # Backpressure: wait BEFORE putting (don't overfill queue)
                            while q.qsize() >= 20 and not player.is_finished():
                                time.sleep(0.01)

                            q.put(chunk)

                    finally:
                        gen.close()

                    # Small pause between segments to let player buffer fill
                    if not first_seg_done[0]:
                        first_seg_done[0] = True
                        time.sleep(0.3)

                q.put(None)  # sentinel

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
        all_wavs = []
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
            while player.buffered_seconds() > max_buffer_sec and not player.is_finished():
                time.sleep(0.01)

            all_wavs.append(chunk)
            player.add_chunk(chunk)

            # Start playback when buffer is warm
            if not started[0]:
                if player.buffered_seconds() >= MIN_START_SEC:
                    player.request_start()
                    started[0] = True

        gen_thread.join(timeout=120)
        if gen_thread.is_alive():
            print("WARNING: generator thread did not finish in 120s", flush=True)

        player.add_chunk(None)

        if errors:
            player.stop()
            raise errors[0]

        total_ms = (time.perf_counter() - t_start) * 1000
        ttfa_ms = first_chunk_time[0] * 1000 if first_chunk_time[0] else 0
        audio_total_s = sum(len(c) for c in all_wavs) / 24000.0

        print(f"\n[V14] Done!", flush=True)
        print(f"  Audio chunks: {chunk_count_ref[0]}", flush=True)
        print(f"  TTFA (time to first audio): {ttfa_ms:.0f}ms", flush=True)
        print(f"  Total wall time: {total_ms:.0f}ms", flush=True)
        print(f"  Total audio: {audio_total_s:.2f}s", flush=True)

        print("  Waiting for playback...", flush=True)
        player.wait(timeout=60.0)
        player.stop()

        if save_wav and all_wavs:
            # Save with crossfade (smooth output)
            full_audio = np.concatenate(all_wavs)
            sf.write(save_wav, full_audio, 24000)
            print(f"  Saved: {save_wav}", flush=True)


# ============================================================================
# MAIN
# ============================================================================
def main():
    model_path = r'G:\Foundation\models\Qwen3-TTS'

    text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Привет! Это тест потоковой генерации с кроссфейдом. "
        "Теперь звук должен быть плавным без щелчков!"
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print()

    tts = FastTTSv14(model_path, speaker='Sohee')
    tts.generate_and_play(text, save_wav='tts_output_v14.wav')


# ============================================================================
# TEST SUITE — 10 sentences (5 Russian, 5 English)
# ============================================================================
def run_test_suite():
    """Run test suite with 10 sentences for audio verification."""
    model_path = r'G:\Foundation\models\Qwen3-TTS'

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

    for i, (lang, text) in enumerate(test_sentences):
        save_wav = f'tts_test_v14_{i+1:02d}_{lang.lower()}.wav'
        print(f"\n[{i+1}/10] [{lang}] {text}", flush=True)
        tts.generate_and_play(text, language=lang, save_wav=save_wav)

    print(f"\n{'='*70}", flush=True)
    print("[V14] All 10 tests complete!", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    if '--test' in sys.argv:
        run_test_suite()
    else:
        main()
