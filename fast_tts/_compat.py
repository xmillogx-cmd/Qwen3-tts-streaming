"""Compatibility bridge: fail fast when the qwen-tts API drifts.

The streaming methods in :mod:`fast_tts._patch.streaming` are an additive
overlay onto the stock ``qwen_tts.Qwen3TTSModel`` API (methods, config fields
and module attributes). If a future ``qwen-tts`` release renames or removes any
of those touchpoints, attaching the mixin would still "succeed" but every
streaming call would crash deep inside with an obscure AttributeError.

This module probes the installed package at two points:

- :func:`check_qwen_tts_version` — version pin (>=0.1.1,<0.2), run in
  ``apply_patch()`` before anything is attached;
- :func:`probe_model_class` — class-level method probe, also run in
  ``apply_patch()``;
- :func:`probe_model_api` — instance-level attribute/config-field probe, run in
  ``FastTTSv14.__init__`` right after ``from_pretrained``.

All probes raise :class:`RuntimeError` listing every missing touchpoint at once.
"""
from __future__ import annotations

SUPPORTED_QWEN_TTS = ">=0.1.1,<0.2"

_MIN_VERSION = (0, 1, 1)
_MAX_VERSION_EXCL = (0, 2, 0)


def _parse_version(version: str):
    """Parse 'X.Y.Z[...]' into a numeric tuple; trailing tags are ignored."""
    parts = []
    for comp in version.split(".")[:3]:
        digits = ""
        for ch in comp:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def check_qwen_tts_version() -> None:
    """Raise RuntimeError unless the installed qwen-tts is within SUPPORTED_QWEN_TTS."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("qwen-tts")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "The 'qwen-tts' package is not installed. Install it with: pip install qwen-tts"
        ) from exc

    v = _parse_version(installed)
    if not (_MIN_VERSION <= v < _MAX_VERSION_EXCL):
        raise RuntimeError(
            f"fast_tts requires qwen-tts {SUPPORTED_QWEN_TTS}, but {installed} is installed. "
            "The streaming overlay depends on the exact Qwen3TTSModel API of that range; "
            "upgrade fast_tts before upgrading qwen-tts."
        )


# Methods the overlay calls on Qwen3TTSModel (class-level probe).
REQUIRED_MODEL_METHODS = (
    "_build_assistant_text",
    "_tokenize_texts",
    "_build_instruct_text",
    "_validate_languages",
    "_validate_speakers",
    "create_voice_clone_prompt",
    "_prompt_items_to_voice_clone_prompt",
    "_build_ref_text",
)

# Dotted attribute paths on a loaded Qwen3TTSModel instance (instance probe).
REQUIRED_INSTANCE_ATTRS = (
    "model",
    "device",
    "model.tts_model_type",
    "model.tts_model_size",
    "model.config",
    "model.speech_tokenizer",
    # Voice-clone/ICL helpers live on the inner model, not the wrapper.
    "model.generate_speaker_prompt",
    "model.generate_icl_prompt",
    "model.talker",
    "model.talker.device",
    "model.talker.model",
    "model.talker.codec_head",
    "model.talker.text_projection",
    "model.talker.get_input_embeddings",
    "model.talker.get_text_embeddings",
    "model.talker.code_predictor",
    "model.talker.code_predictor.small_to_mtp_projection",
    "model.talker.code_predictor.lm_head",
    "model.talker.code_predictor.model",
    "model.talker.code_predictor.model.config",
    "model.talker.code_predictor.model.codec_embedding",
    "model.talker.code_predictor.get_input_embeddings",
    # Backbone (talker.model) config field used by TalkerGraph mask selection.
    "model.talker.model.config.sliding_window",
    "model.config.tts_bos_token_id",
    "model.config.tts_eos_token_id",
    "model.config.tts_pad_token_id",
)

# Fields of model.config.talker_config used by the overlay and both graphs.
REQUIRED_TALKER_CONFIG_FIELDS = (
    "spk_id",
    "hidden_size",
    "num_hidden_layers",
    "codec_pad_id",
    "codec_bos_id",
    "codec_eos_token_id",
    "vocab_size",
    "num_code_groups",
    "codec_language_id",
    "spk_is_dialect",
    "codec_nothink_id",
    "codec_think_bos_id",
    "codec_think_eos_id",
    "codec_think_id",
)

# Fields of model.talker.code_predictor.model.config used by PredictorGraph.
REQUIRED_PREDICTOR_CONFIG_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "num_code_groups",
)


def _resolve(obj, path: str):
    """Walk a dotted attribute path; raise AttributeError naming the full path."""
    for part in path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise AttributeError(f"missing '{path}': no such attribute '{part}'") from exc
    return obj


def _probe_attr_paths(model, paths):
    missing = []
    for path in paths:
        try:
            _resolve(model, path)
        except AttributeError:
            missing.append(path)
    return missing


def probe_model_class(cls) -> None:
    """Check that the Qwen3TTSModel class exposes every method the overlay calls."""
    missing = [name for name in REQUIRED_MODEL_METHODS if not hasattr(cls, name)]
    if missing:
        raise RuntimeError(
            "fast_tts is incompatible with this qwen-tts version: Qwen3TTSModel is missing "
            f"required method(s): {', '.join(missing)}. Supported range: {SUPPORTED_QWEN_TTS}."
        )


def probe_model_api(model) -> None:
    """Check that a loaded model instance exposes every attribute/config field the overlay uses."""
    missing = _probe_attr_paths(model, REQUIRED_INSTANCE_ATTRS)

    for path, fields in (
        ("model.config.talker_config", REQUIRED_TALKER_CONFIG_FIELDS),
        ("model.talker.code_predictor.model.config", REQUIRED_PREDICTOR_CONFIG_FIELDS),
    ):
        try:
            cfg = _resolve(model, path)
        except AttributeError:
            if path not in missing:
                missing.append(path)
            continue
        for field in fields:
            if not hasattr(cfg, field):
                missing.append(f"{path}.{field}")

    if missing:
        raise RuntimeError(
            "fast_tts is incompatible with this qwen-tts model API: the loaded Qwen3TTSModel "
            f"is missing required attribute(s)/config field(s): {', '.join(missing)}. "
            f"Supported range: {SUPPORTED_QWEN_TTS}."
        )


__all__ = [
    "SUPPORTED_QWEN_TTS",
    "check_qwen_tts_version",
    "probe_model_class",
    "probe_model_api",
]
