"""CUDA-graph capture for the Qwen3-TTS code predictor.

The code predictor emits one token per codebook, autoregressively over all 15
codebooks: a 2-token prefill (talker hidden state + first-codebook embedding)
followed by 14 single-token decode steps, where each step embeds the token
sampled from the previous codebook through that codebook's own embedding
table.

This module captures the entire 15-step pipeline as a single CUDA graph so a
full codebook frame costs one ``graph.replay()`` instead of ~30 kernel
launches. The capture strategy (fixed-shape StaticCache, unrolled loop)
follows the approach published in faster-qwen3-tts (MIT); this is our own
implementation with a precomputed per-step plan table and an optional shared
graph memory pool.

Usage::

    pg = PredictorGraph(code_predictor, pred_config, talker_hidden_size)
    pg.capture()
    codes = pg.run(pred_input)   # pred_input: [1, 2, H_talker] -> [15] long
"""
import torch
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from .sampling import sample_logits


def _device_index(device) -> int:
    """Resolve a device string/tensor to a concrete CUDA index."""
    idx = torch.device(device).index
    return idx if idx is not None else torch.cuda.current_device()


class PredictorGraph:
    """Captures the full 15-step predictor loop as one CUDA graph.

    Every tensor touched by the captured region is a static buffer allocated
    in ``__init__``; replays only copy new inputs into them, so no
    allocations or CPU->GPU transfers happen inside the graph.
    """

    def __init__(self, code_predictor, pred_config, talker_hidden_size, device='cuda',
                 dtype=torch.bfloat16, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
                 pool=None):
        self.device = device
        self._dev_idx = _device_index(device)
        self.dtype = dtype

        cfg = pred_config
        self.num_layers = cfg.num_hidden_layers
        self.hidden_size = cfg.hidden_size
        self.num_code_groups = cfg.num_code_groups
        self.num_codebooks = self.num_code_groups - 1   # 15
        self.max_seq = 2 + self.num_codebooks          # 17
        self.do_sample, self.top_k, self.top_p, self.temperature = (
            do_sample, top_k, top_p, temperature)

        # References into the code predictor (not copies).
        cp = code_predictor
        self._proj = cp.small_to_mtp_projection       # talker H -> predictor H
        self._backbone = cp.model                     # 5-layer transformer
        self._heads = cp.lm_head                      # ModuleList[15]
        self._cb_embeds = cp.model.codec_embedding    # ModuleList[15]
        self._has_sliding = "sliding_attention" in getattr(self._backbone.config, "layer_types", [])

        self._cache = StaticCache(config=cfg, max_cache_len=self.max_seq)

        # Per-step cache positions (pre-allocated: no CPU->GPU inside the graph).
        self._prefill_pos = torch.arange(2, device=device)
        self._decode_positions = [torch.tensor([2 + i], device=device)
                                  for i in range(self.num_codebooks - 1)]

        # I/O buffers.
        self._in_buf = torch.zeros(1, 2, talker_hidden_size, dtype=dtype, device=device)
        self._out_tokens = torch.zeros(self.num_codebooks, dtype=torch.long, device=device)

        self.graph = None
        self.captured = False
        self._prefill_mask = None
        self._decode_masks = []

    def _prime_kv_buffers(self):
        """Materialize StaticCache layer buffers before graph capture."""
        cfg = self._backbone.config
        num_kv_heads = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
        head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
        dummy_k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=self.dtype, device=self.device)
        for layer in self._cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy_k)

    def _layer_masks(self, input_embeds: torch.Tensor, cache_position: torch.Tensor):
        """Causal mask(s) for one step; always a dict keyed by attention type."""
        full = create_causal_mask(
            config=self._backbone.config,
            input_embeds=input_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=self._cache,
        )
        if not self._has_sliding:
            return {"full_attention": full}
        sliding = create_sliding_window_causal_mask(
            config=self._backbone.config,
            input_embeds=input_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=self._cache,
        )
        return {"full_attention": full, "sliding_attention": sliding}

    def _build_step_plan(self):
        """Precompute masks for the prefill and all 14 decode steps once."""
        dummy_prefill = torch.zeros(1, 2, self.hidden_size, dtype=self.dtype, device=self.device)
        dummy_decode = torch.zeros(1, 1, self.hidden_size, dtype=self.dtype, device=self.device)
        self._prefill_mask = self._layer_masks(dummy_prefill, self._prefill_pos)
        self._decode_masks = [self._layer_masks(dummy_decode, pos) for pos in self._decode_positions]

    def _project(self, embeds: torch.Tensor) -> torch.Tensor:
        return self._proj(embeds)

    def _forward_prefill(self, h: torch.Tensor) -> torch.Tensor:
        out = self._backbone(
            inputs_embeds=h,
            attention_mask=self._prefill_mask,
            past_key_values=self._cache,
            cache_position=self._prefill_pos,
            use_cache=True,
        )
        return out.last_hidden_state

    def _forward_decode(self, emb: torch.Tensor, step_idx: int) -> torch.Tensor:
        out = self._backbone(
            inputs_embeds=emb,
            attention_mask=self._decode_masks[step_idx],
            past_key_values=self._cache,
            cache_position=self._decode_positions[step_idx],
            use_cache=True,
        )
        return out.last_hidden_state

    def _sample(self, head_module: torch.nn.Module, hidden_last: torch.Tensor) -> torch.Tensor:
        logits = head_module(hidden_last)[:, 0]
        return sample_logits(
            logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )

    def _run_pipeline(self):
        """The full 15-step loop on static buffers (the captured region)."""
        h = self._project(self._in_buf)                    # [1, 2, H_pred]
        h = self._forward_prefill(h)                       # [1, 2, H_pred]

        tok = self._sample(self._heads[0], h[:, -1:, :])  # [1] long
        self._out_tokens[0] = tok[0]

        for step_idx in range(1, self.num_codebooks):
            emb = self._project(self._cb_embeds[step_idx - 1](tok.unsqueeze(0)))
            h = self._forward_decode(emb, step_idx - 1)
            tok = self._sample(self._heads[step_idx], h[:, -1:, :])
            self._out_tokens[step_idx] = tok[0]

    @torch.inference_mode()
    def capture(self, num_warmup=3, pool=None):
        """Warm up and capture the CUDA graph.

        ``pool`` may be a shared ``torch.cuda.graphs.graph_pool_handle()`` so
        several graphs (predictor + talker) draw from one memory pool.
        """
        print(f"Warming up predictor ({num_warmup} runs)...")
        self._prime_kv_buffers()
        self._build_step_plan()

        for _ in range(num_warmup):
            self._cache.reset()
            self._run_pipeline()
        torch.cuda.synchronize()

        print("Capturing CUDA graph for predictor...")
        with torch.cuda.device(self._dev_idx):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self._cache.reset()
                self._run_pipeline()          # warmup on the capture stream
                torch.cuda.synchronize()

                self._cache.reset()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, pool=pool):
                    self._run_pipeline()

        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.captured = True
        print("CUDA graph captured!")

    @torch.inference_mode()
    def run(self, pred_input: torch.Tensor) -> torch.Tensor:
        """Replay the captured loop.

        Args:
            pred_input: [1, 2, talker_hidden_size] — past_hidden cat first_codebook_embed.
        Returns:
            [15] long tensor of codebook tokens (a fresh copy).
        """
        if not self.captured:
            raise RuntimeError("PredictorGraph.capture() must be called before run()")
        self._in_buf.copy_(pred_input)
        self._cache.reset()
        self.graph.replay()
        return self._out_tokens.clone()
