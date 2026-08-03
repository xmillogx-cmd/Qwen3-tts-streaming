"""Test v14 — Streaming playback with proper warmup."""
import torch, time, numpy as np, soundfile as sf, sounddevice as sd, sys, threading, queue

model_path = r'G:\Foundation\models\Qwen3-TTS'
speaker = 'Sohee'

sentences = [
    # Russian
    "Привет мир",
    "Как дела? У меня всё хорошо.",
    "Сегодня прекрасная погода для прогулки.",
    "Я люблю слушать музыку по утрам.",
    "Этот тест показывает качество генерации речи.",
    # English
    "Hello world, this is a test.",
    "How are you doing today?",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming every industry.",
    "Natural language processing makes computers understand text.",
]

print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print()

from qwen_tts import Qwen3TTSModel

t0 = time.perf_counter()
model = Qwen3TTSModel.from_pretrained(
    model_path, device_map='cuda:0', dtype=torch.bfloat16,
    attn_implementation='sdpa',  # NVIDIA optimized attention for CUDA graphs
)
load_ms = (time.perf_counter()-t0)*1000
print(f"Load: {load_ms:.0f}ms", flush=True)

# ============================================================================
# PREFLIGHT WARMUP — single pass with chunk_size=8 (what we use in tests)
# ============================================================================
print("\n[Preflight warmup]...", flush=True)
t0 = time.perf_counter()

warmup_texts = {
    'Russian': "Привет! Как дела? Я живу в Москве. Это тест потоковой генерации речи.",
    'English': "Hello! How are you doing today? This is a test of streaming speech generation.",
}

for lang in ['Russian', 'English']:
    gen = model.generate_custom_voice_streaming(
        text=warmup_texts[lang], speaker=speaker, language=lang,
        chunk_size=8, max_new_tokens=50, backend='auto',
    )
    for _ in gen:
        pass
    gen.close()

torch.cuda.synchronize()
warmup_ms = (time.perf_counter()-t0)*1000
print(f"[Preflight warmup] {warmup_ms:.0f}ms | Ready!", flush=True)
print()

# ============================================================================
# STREAMING AUDIO PLAYER
# ============================================================================
class StreamingAudioPlayer:
    def __init__(self, sample_rate=24000, device_id=None, blocksize=1024, preroll_sec=0.3):
        self.sample_rate = sample_rate
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
        print(f"  [Audio] Device {self.device_id}: {sd.query_devices()[self.device_id]['name'][:50]}", flush=True)
        self._stream = sd.OutputStream(
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
        if chunk is None:
            with self._lock:
                self._done = True
            return
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        with self._lock:
            self._buffered += chunk.size
        while True:
            try:
                self._chunks.put_nowait(np.ascontiguousarray(chunk))
                break
            except queue.Full:
                time.sleep(0.005)

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

    def reset(self):
        with self._lock:
            while not self._chunks.empty():
                try:
                    self._chunks.get_nowait()
                except queue.Empty:
                    break
            self._current = None
            self._offset = 0
            self._buffered = 0
            self._done = False
            self._started = False
        self._finished.clear()


# ============================================================================
# RUN TESTS WITH STREAMING PLAYBACK
# ============================================================================
player = StreamingAudioPlayer(sample_rate=24000, preroll_sec=0.3)
player.start()

results = []
for i, text in enumerate(sentences):
    lang = 'Russian' if i < 5 else 'English'
    
    t_start = time.perf_counter()
    first_chunk_time = [None]
    chunk_count_ref = [0]
    all_wavs = []
    started = [False]
    MIN_START_SEC = 1.0
    max_buffer_sec = 3.5

    # Producer-consumer pipeline
    q = queue.Queue(maxsize=32)
    
    def producer():
        gen = model.generate_custom_voice_streaming(
            text=text, speaker=speaker, language=lang,
            chunk_size=8, max_new_tokens=300, backend='auto',
        )
        try:
            for audio_c, sr, timing in gen:
                if first_chunk_time[0] is None:
                    first_chunk_time[0] = time.perf_counter() - t_start
                
                chunk = np.asarray(audio_c.detach().cpu().numpy() if torch.is_tensor(audio_c) else audio_c, dtype=np.float32).reshape(-1)
                if chunk.size == 0:
                    continue
                
                chunk_count_ref[0] += 1
                decode_ms = timing.get('decode_ms', 0)
                print(f"  Chunk {chunk_count_ref[0]}: len={len(chunk)} samples, decode={decode_ms:.0f}ms", flush=True)
                all_wavs.append(chunk)
                q.put(chunk)
        finally:
            gen.close()
        q.put(None)

    gen_thread = threading.Thread(target=producer, daemon=True)
    gen_thread.start()

    # Consumer
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
        
        # Backpressure
        while player.buffered_seconds() > max_buffer_sec and not player.is_finished():
            time.sleep(0.01)
        
        player.add_chunk(chunk)
        
        # Start playback when buffer is warm
        if not started[0]:
            if player.buffered_seconds() >= MIN_START_SEC:
                player.request_start()
                started[0] = True

    gen_thread.join(timeout=120)
    # Don't signal end between sentences — keep playback flowing

    wall_ms = (time.perf_counter() - t_start) * 1000
    audio_total_s = sum(len(c) for c in all_wavs) / 24000.0
    ttfa_ms = first_chunk_time[0] * 1000 if first_chunk_time[0] else 0

    # Save WAV
    full_audio = np.concatenate(all_wavs)
    sf.write(f'test_v14_{i+1:02d}.wav', full_audio, 24000)

    results.append({
        'idx': i+1,
        'text': text,
        'lang': lang,
        'chunks': chunk_count_ref[0],
        'audio_s': audio_total_s,
        'wall_ms': wall_ms,
        'ttfa_ms': ttfa_ms,
    })

    print(f"[{i+1:02d}] {lang:8s} | chunks={chunk_count_ref[0]:2d} | audio={audio_total_s:.2f}s | wall={wall_ms:.0f}ms | TTFA={ttfa_ms:.0f}ms", flush=True)
    print(f"      Text: {text}")

# Signal end only after all sentences
player.add_chunk(None)
player.stop()

# Summary
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
for r in results:
    print(f"[{r['idx']:02d}] {r['lang']:8s} | {r['chunks']:2d} chunks | {r['audio_s']:.2f}s audio | {r['wall_ms']:.0f}ms wall | TTFA={r['ttfa_ms']:.0f}ms")
    print(f"      \"{r['text']}\"")

all_audio = [r['audio_s'] for r in results]
all_wall = [r['wall_ms'] for r in results]
print()
print(f"Average audio length: {np.mean(all_audio):.2f}s")
print(f"Average wall time:    {np.mean(all_wall):.0f}ms")
print(f"Total audio:          {sum(all_audio):.2f}s")
