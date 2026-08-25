"""Backward-compat shim — the engine now lives in the ``fast_tts`` package.

Kept so existing launchers (``*.bat``) and dev scripts keep working unchanged:
    python fast_tts_v14.py --text "..."     # CLI
    python fast_tts_v14.py --test           # 10-sentence test suite
"""
import os
import sys

from fast_tts import FastTTSv14, StreamingAudioPlayer, global_peak_normalize, split_segments, to_pcm_chunk
from fast_tts.cli import main, run_test_suite

__all__ = ["FastTTSv14", "StreamingAudioPlayer", "split_segments", "to_pcm_chunk", "global_peak_normalize"]


if __name__ == '__main__':
    # Local dev convenience: keep the historical default model path when neither
    # --model nor MODEL_PATH is provided (the released CLI has no hardcoded paths).
    os.environ.setdefault('MODEL_PATH', r'G:\Foundation\models\Qwen3-TTS')

    if '--test' in sys.argv:
        run_test_suite()
    else:
        main()
