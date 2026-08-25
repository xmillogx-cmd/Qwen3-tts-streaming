"""Callback-based streaming audio player for live TTS playback."""
from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd


class StreamingAudioPlayer:
    # blocksize 4800 (~200ms @24kHz) instead of 1024: fewer PortAudio callbacks
    # means less GIL/DPC contention with the generation thread on USB audio devices.
    def __init__(self, sample_rate: int = 24000, device_id: Optional[int] = None, blocksize: int = 4800, preroll_sec: float = 0.3) -> None:
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
        # fixed: blocking put — waits for the consumer instead of polling; never drops chunks
        self._chunks.put(np.ascontiguousarray(chunk))

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
