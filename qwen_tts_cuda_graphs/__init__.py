"""CUDA graph capture for Qwen3-TTS talker and predictor.

Перенесено из faster-qwen3-tts с адаптацией для нативного qwen_tts.
"""

from .predictor_graph import PredictorGraph
from .talker_graph import TalkerGraph

__all__ = ["PredictorGraph", "TalkerGraph"]
