"""
Fast TTS v14 — True Streaming Playback
=======================================
- CUDA graph warmup done ONCE during model loading
- TRUE streaming: producer thread -> queue -> player (no two-phase collection)
- Text segmentation for long inputs (split_segments)
- Global RMS normalization for consistent loudness across chunks
- Live speaker output via sounddevice
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
import queue
from typing import Optional, List

import numpy as np
import torch
import soundfile as sf
import sounddevice as sd


# ============================================================================
# HELPERS
# ============================================================================
def to_pcm_chunk(x) -> np.ndarray:
    """Safely convert audio chunk to float32 PCM."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def global_normalize(audio_chunks: List[np.ndarray], limit: float = 0.95) -> List[np.ndarray]:
    """Normalize all chunks together by global peak to avoid loudness pumping."""
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
                    elif len(cur) + 1 + 1 + len(w) <= max_chars:
                        cur = f"{cur} {w}"  # fixed: was missing space
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
    def __init__(self, sample_rate: int = 24000, device_id: Optional[int] = None, blocksize: int = 1024, preroll_sec: float = 0.3) -> None:
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

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
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

    def add_chunk(self, chunk: Optional[np.ndarray]) -> None:
        """Accepts pre-normalized float32 PCM chunks. Blocks if queue is full."""
        if chunk is None:
            with self._lock:
                self._done = True
            return
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        # fixed: update _buffered under lock BEFORE put to avoid race condition
        with self._lock:
            self._buffered += chunk.size
        # fixed: blocking put with retry instead of dropping chunks on Full
        while True:
            try:
                self._chunks.put_nowait(np.ascontiguousarray(chunk))
                break
            except queue.Full:
                time.sleep(0.005)

    def buffered_seconds(self) -> float:
        with self._lock:
            return self._buffered / float(self.sample_rate)

    def request_start(self) -> None:
        with self._lock:
            self._started = True

    def is_finished(self) -> bool:
        return self._finished.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._finished.wait(timeout)

    def stop(self, timeout: float = 5.0) -> None:
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

    def reset(self) -> None:
        """Reset player state for reuse without reopening the stream."""
        with self._lock:
            # fixed: drain queue properly instead of touching internal .queue
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

        # Preflight warmup - capture CUDA graphs for ALL chunk_size variants
        # This prevents TTFA spikes on first real request
        print("[V14] Preflight warmup (all chunk sizes)...", flush=True)
        t0 = time.perf_counter()

        # Warmup texts covering all supported languages — ensures CUDA graphs
        # and language-specific embedding paths are captured before first real call.
        warmup_texts = {
            'Russian':   "Привет! Как дела? Я живу в Москве. Это тест потоковой генерации речи.",
            'English':   "Hello! How are you doing today? This is a test of streaming speech generation.",
            'German':    "Hallo! Wie geht es dir? Ich wohne in Berlin und arbeite als Softwareentwickler.",
            'Spanish':   "Hola! ¿Cómo estás? Vivo en Madrid y trabajo como ingeniero de software.",
            'French':    "Bonjour! Comment allez-vous? Je vis à Paris et je travaille comme développeur.",
            'Chinese':   "你好！我叫李明。我住在北京，是一名软件工程师。",
        }

        for lang, text in warmup_texts.items():
            for chunk_size in [2, 4, 8]:
                gen = self.model.generate_custom_voice_streaming(
                    text=text,
                    speaker=self.speaker,
                    language=lang,
                    chunk_size=chunk_size,
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
        word_count = len(re.findall(r'\b\w+\b', text))
        if word_count <= 2: return 20
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
            return [np.zeros(0, dtype=np.float32)], 24000, {}

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

        # Global RMS normalization (avoids loudness pumping between chunks)
        all_wavs = global_normalize(all_wavs)

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


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Fast TTS v14 — Streaming playback')
    parser.add_argument('--model', default=os.getenv('MODEL_PATH', r'G:\Foundation\models\Qwen3-TTS'),
                        help='Path to Qwen3-TTS model directory')
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

    if not os.path.exists(args.model):
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
    model_path = os.getenv('MODEL_PATH', r'G:\Foundation\models\Qwen3-TTS')

    if not os.path.exists(model_path):
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
