"""Runtime patch: attach streaming + CUDA-graph methods to stock Qwen3TTSModel.

The PyPI ``qwen-tts`` wheel (>=0.1.1) ships without the streaming API and the
CUDA-graph modules used by this package. All of those additions are purely
additive (new methods only, no modifications to existing code), so we attach
them at import time instead of shadowing the whole ``qwen_tts`` module — which
keeps plain ``qwen-tts`` installable alongside this one and stays compatible
with future upstream releases.
"""


def apply_patch() -> None:
    """Idempotently attach the streaming methods onto Qwen3TTSModel."""
    import qwen_tts.inference.qwen3_tts_model as _stock_mod
    from .streaming import Qwen3TTSStreamingMixin

    stock_cls = _stock_mod.Qwen3TTSModel
    if getattr(stock_cls, "_fast_tts_patched", False):
        return

    for name in dir(Qwen3TTSStreamingMixin):
        if name.startswith("__"):
            continue
        if not hasattr(stock_cls, name):
            setattr(stock_cls, name, getattr(Qwen3TTSStreamingMixin, name))
    stock_cls._fast_tts_patched = True


__all__ = ["apply_patch"]
