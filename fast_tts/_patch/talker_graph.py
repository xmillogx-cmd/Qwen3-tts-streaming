"""CUDA-graph capture for the Qwen3-TTS talker's single-token decode step.

The talker backbone is a 28-layer transformer; instead of reimplementing its
forward pass we capture the model's own single-token forward with a fixed-
shape StaticCache, so each decode step replays as one graph (the cache
position and attention mask are updated via static buffers between replays).

Attention masks are built lazily per sequence position: the StaticCache
reports a constant kv length regardless of cache state, so an entry built on
demand is bit-identical to one pre-built at capture time — this avoids
materialising ``max_seq_len`` mask tensors up front. The capture strategy
follows the approach published in faster-qwen3-tts (MIT); this is our own
implementation with lazy masks and an optional shared graph memory pool.

Usage::

    tg = TalkerGraph(talker.model, talker_config, max_seq_len=2048)
    tg.capture(prefill_len=100)
    seq_len = tg.prefill_kv(past_key_values_from_prefill)
    hidden = tg.run(embeds, position)   # consume before the next call
"""
import torch
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


def _device_index(device) -> int:
    """Resolve a device string/tensor to a concrete CUDA index."""
    idx = torch.device(device).index
    return idx if idx is not None else torch.cuda.current_device()


class TalkerGraph:
    """Captures one talker decode step as a CUDA graph.

    All tensors touched by the captured region are static buffers allocated
    in ``__init__``; replays only copy new inputs into them, so no
    allocations or CPU->GPU transfers happen inside the graph.
    """

    def __init__(self, talker_model, talker_config, device='cuda', dtype=torch.bfloat16,
                 max_seq_len=512, pool=None):
        self.device = device
        self._dev_idx = _device_index(device)
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.hidden_size = talker_config.hidden_size
        self.num_layers = talker_config.num_hidden_layers

        # Reference to the inner model (transformer backbone).
        self._model = talker_model

        # Fixed-size KV cache compatible with CUDA graphs.
        self._cache = StaticCache(config=talker_config, max_cache_len=max_seq_len)

        # Static I/O buffers for the captured step.
        self._in_buf = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)
        self._out_buf = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)
        self._pos_buf = torch.zeros(1, dtype=torch.long, device=device)
        # Rope deltas from prefill ([batch, 1]) and the M-RoPE position buffer.
        self._rope_deltas = torch.zeros(1, 1, dtype=torch.float32, device=device)
        self._position_ids = torch.zeros(3, 1, 1, dtype=torch.float32, device=device)

        # All lazily built masks share one shape ([1, 1, 1, max_seq_len]), so a
        # single static buffer receives whichever entry is active for replay.
        self._attn_buf = torch.zeros(1, 1, 1, max_seq_len, dtype=dtype, device=device)

        self.graph = None
        self.captured = False
        self._mask_entries = [None] * max_seq_len
        self._mask_key = None
        self._full_attn_mask = None

    def _prime_kv_buffers(self):
        """Materialize StaticCache layer buffers before graph capture."""
        cfg = self._model.config
        num_kv_heads = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
        head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
        dummy_k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=self.dtype, device=self.device)
        for layer in self._cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy_k)

    def _ensure_mask(self, position: int) -> torch.Tensor:
        """Build (once per position / padding signature) the causal mask."""
        entry = self._mask_entries[position]
        if entry is not None:
            return entry
        dummy = torch.zeros(1, 1, self.hidden_size, dtype=self.dtype, device=self.device)
        pos_tensor = torch.tensor([position], device=self.device)
        mask_fn = (create_causal_mask if self._model.config.sliding_window is None
                   else create_sliding_window_causal_mask)
        entry = mask_fn(
            config=self._model.config,
            input_embeds=dummy,
            attention_mask=self._full_attn_mask,
            cache_position=pos_tensor,
            past_key_values=self._cache,
        )
        self._mask_entries[position] = entry
        return entry

    def _decode_step(self):
        """Single-token decode through the model's own forward."""
        out = self._model(
            inputs_embeds=self._in_buf,
            attention_mask=self._attn_buf,
            past_key_values=self._cache,
            cache_position=self._pos_buf,
            position_ids=self._position_ids,
            use_cache=True,
        )
        self._out_buf.copy_(out.last_hidden_state)

    @torch.inference_mode()
    def capture(self, prefill_len=100, num_warmup=3, pool=None):
        """Warm up and capture the decode-step graph.

        ``prefill_len`` is only a simulated position for warmup (the captured
        step is position-independent). ``pool`` may be a shared
        ``torch.cuda.graphs.graph_pool_handle()`` so several graphs draw from
        one memory pool.
        """
        print(f"Warming up talker graph ({num_warmup} runs)...")
        self._prime_kv_buffers()

        # Set the static buffers for warmup at a representative position.
        self._pos_buf[0] = prefill_len
        self._attn_buf.copy_(self._ensure_mask(prefill_len))

        for _ in range(num_warmup):
            self._decode_step()
        torch.cuda.synchronize()

        print("Capturing CUDA graph for talker decode...")
        with torch.cuda.device(self._dev_idx):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self._decode_step()          # warmup on the capture stream
                torch.cuda.synchronize()

                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, pool=pool):
                    self._decode_step()

        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.captured = True
        print("Talker CUDA graph captured!")

    def reset(self, prefill_len: int):
        """Reset the KV cache for a new sequence."""
        self._cache.reset()

    def prefill_kv(self, past_key_values) -> int:
        """Copy HF DynamicCache from prefill into our StaticCache.

        Args:
            past_key_values: per-layer (k, v), each [1, kv_heads, seq_len, head_dim].
        Returns:
            The prefill sequence length.
        """
        self._cache.reset()
        seq_len = 0
        for layer_idx in range(self.num_layers):
            k, v = past_key_values[layer_idx]
            seq_len = k.shape[2]
            if seq_len > self.max_seq_len:
                raise RuntimeError(
                    f"Input is too long: prefill has {seq_len} tokens but max_seq_len={self.max_seq_len}. "
                    "Use shorter text or shorter reference audio."
                )
            cache_pos = torch.arange(seq_len, device=self.device)
            self._cache.update(k, v, layer_idx, {"cache_position": cache_pos})
        return seq_len

    def set_generation_state(self, attention_mask: torch.Tensor | None, rope_deltas: torch.Tensor | None):
        """Set padding-aware masking and rope deltas for decode parity.

        When the padding signature changes, previously built masks are
        invalidated and rebuilt lazily on demand (only positions actually
        visited during generation).
        """
        mask_key = None
        full_attention_mask = None
        if attention_mask is not None:
            pad_counts = (attention_mask == 0).sum(dim=-1)
            mask_key = tuple(pad_counts.tolist())
            full_attention_mask = torch.ones(
                attention_mask.shape[0], self.max_seq_len,
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            for b, pads in enumerate(pad_counts.tolist()):
                if pads > 0:
                    full_attention_mask[b, :pads] = 0
        if mask_key != self._mask_key:
            self._full_attn_mask = full_attention_mask
            self._mask_entries = [None] * self.max_seq_len
            self._mask_key = mask_key

        if rope_deltas is None:
            self._rope_deltas.zero_()
        else:
            if rope_deltas.dim() == 1:
                rope_deltas = rope_deltas.unsqueeze(1)
            self._rope_deltas.copy_(rope_deltas.to(self._rope_deltas.device, dtype=self._rope_deltas.dtype))

    @torch.inference_mode()
    def run(self, input_embeds: torch.Tensor, position: int) -> torch.Tensor:
        """Run one decode step.

        Args:
            input_embeds: [1, 1, hidden_size]
            position: current sequence position
        Returns:
            [1, 1, hidden_size] hidden states — a static buffer; consume or
            clone before the next call.
        """
        if not self.captured:
            raise RuntimeError("TalkerGraph.capture() must be called before run()")
        self._in_buf.copy_(input_embeds)
        self._pos_buf[0] = position
        self._attn_buf.copy_(self._ensure_mask(position))

        # M-RoPE positions for the single new token: cache_position + rope delta.
        delta = self._rope_deltas + self._pos_buf[0].to(self._rope_deltas.dtype)
        self._position_ids.copy_(delta.unsqueeze(0).expand(3, -1, -1))

        self.graph.replay()
        return self._out_buf  # static buffer — caller should use immediately or clone
