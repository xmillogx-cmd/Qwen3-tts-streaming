"""Token sampling primitives for talker and predictor generation.

Our implementation of the standard HF-style sampler chain: suppression ->
temperature scaling -> top-k thresholding -> nucleus filtering -> multinomial
sampling (or argmax when ``do_sample`` is off). Nucleus sampling runs over
the surviving top-k support instead of sorting the full vocabulary on every
step, which keeps the captured-graph path allocation-light while producing
the same distribution as a sequential TopK + TopP warper pair.

Approach cross-referenced with faster-qwen3-tts (MIT).
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F

_NEG_INF = float("-inf")


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply an HF-style repetition penalty to ``logits`` in place.

    Args:
        logits: Tensor shaped [1, 1, vocab] or [1, vocab]; modified and returned.
        token_history: 1-D tensor of previously generated token ids.
        repetition_penalty: Penalty factor (>1.0 penalizes repeats).
    """
    if repetition_penalty == 1.0 or not token_history.numel():
        return logits

    picked = token_history.unique()
    tok_logits = logits[..., picked]
    adjusted = torch.where(
        tok_logits > 0, tok_logits / repetition_penalty, tok_logits * repetition_penalty
    )
    logits[..., picked].copy_(adjusted)
    return logits


def _suppress_tokens(
    logits: torch.Tensor,
    suppress_mask: Optional[torch.Tensor] = None,
    suppress_tokens: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Mask forbidden ids to -inf; clones only when a write-back is needed."""
    if suppress_mask is None and not suppress_tokens:
        return logits
    out = logits.clone()
    if suppress_mask is not None:
        out[..., suppress_mask] = _NEG_INF
    if suppress_tokens:
        out[..., list(suppress_tokens)] = _NEG_INF
    return out


def _topk_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only the ``top_k`` largest logits (threshold style)."""
    k = min(top_k, logits.size(-1))
    threshold, _ = torch.topk(logits, k)
    cutoff = threshold[..., -1:]
    return torch.where(logits < cutoff, torch.full_like(logits, _NEG_INF), logits)


def _nucleus_filter(logits: torch.Tensor, top_p: float, support_size: int) -> torch.Tensor:
    """Restrict the sampling support to the nucleus of ``top_p`` mass.

    Operates on the largest ``support_size`` logits (the surviving set after
    a preceding top-k pass) instead of sorting the full vocabulary — the same
    distribution as HF's sequential TopK + TopP warper pair, but without a
    full-vocabulary sort per step.
    """
    k = min(support_size, logits.size(-1))
    vals, idx = torch.topk(logits, k)
    probs = F.softmax(vals, dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)
    drop = cumsum > top_p
    drop[..., 0] = False
    kept = torch.where(drop, torch.full_like(vals, _NEG_INF), vals)
    logits.scatter_(-1, idx, kept)
    return logits


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    suppress_mask: Optional[torch.Tensor] = None,
    suppress_tokens: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Sample one token id from ``logits`` (last dim = vocabulary).

    Warper order matches HF: suppression -> temperature scaling -> top-k
    thresholding -> nucleus filtering -> sampling. With ``do_sample`` off the
    argmax is returned directly after suppression.
    """
    logits = _suppress_tokens(logits, suppress_mask, suppress_tokens)
    if not do_sample:
        return torch.argmax(logits, dim=-1)

    vocab_size = logits.size(-1)
    support_size = min(top_k, vocab_size) if top_k > 0 else vocab_size
    logits = logits / temperature
    if top_k > 0:
        logits = _topk_filter(logits, top_k)
    if top_p < 1.0:
        logits = _nucleus_filter(logits, top_p, support_size)

    return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
