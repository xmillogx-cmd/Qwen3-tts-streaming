"""fast_tts — true streaming Qwen3-TTS with CUDA Graphs acceleration.

Importing this package attaches the streaming + CUDA-graph methods onto the
stock PyPI ``qwen_tts.Qwen3TTSModel`` (see :mod:`fast_tts._patch`), so the
engine works against a plain ``pip install qwen-tts`` — no patched or
editable qwen_tts copy required.

Quick start::

    from fast_tts import FastTTSv14

    tts = FastTTSv14(model_path="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", speaker="Sohee")
    tts.generate_and_play("Hello! This is a streaming TTS test.")
"""
from ._patch import apply_patch

apply_patch()  # idempotent; must run before any FastTTSv14 instantiation

from .engine import FastTTSv14, global_peak_normalize, split_segments, to_pcm_chunk
from .player import StreamingAudioPlayer
from ._patch.predictor_graph import PredictorGraph
from ._patch.talker_graph import TalkerGraph

__version__ = "0.1.0"

__all__ = [
    "FastTTSv14",
    "StreamingAudioPlayer",
    "PredictorGraph",
    "TalkerGraph",
    "split_segments",
    "to_pcm_chunk",
    "global_peak_normalize",
    "apply_patch",
]
