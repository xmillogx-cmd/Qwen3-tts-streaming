"""PyPI verification harness: run test_v14.py against plain PyPI qwen-tts.

Imports fast_tts first so its apply_patch() attaches the streaming +
CUDA-graph methods to the stock Qwen3TTSModel — exactly what a real user of
the pip package gets. test_v14.py itself stays untouched (no fast_tts import).

Run via test_v14_pypi.bat (sets PYTHONPATH to the extracted PyPI wheel and
pins CUDA_VISIBLE_DEVICES=1; GPU 0 is reserved for manual work).
"""
import os
import runpy

import fast_tts  # noqa: F401 — patches qwen_tts.Qwen3TTSModel at import time

runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_v14.py"),
    run_name="__main__",
)
