"""CUDA graph capture for the code predictor's 15-step decode loop.

Перенесено из faster-qwen3-tts с адаптацией импортов.
Использует transformers StaticCache для фиксированных KV буферов.
"""
import torch
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from .sampling import sample_logits


class PredictorGraph:
    """Captures the full predictor 15-step loop as a CUDA graph."""

    def __init__(self, code_predictor, pred_config, talker_hidden_size, device='cuda', dtype=torch.bfloat16,
                 do_sample=True, top_k=50, top_p=1.0, temperature=0.9):
        self.device = device
        device_index = torch.device(device).index
        device_index = device_index if device_index is not None else torch.cuda.current_device()
        self.device_index = device_index

        self.dtype = dtype
        self.num_layers = pred_config.num_hidden_layers
        self.hidden_size = pred_config.hidden_size
        self.num_code_groups = pred_config.num_code_groups
        self.num_codebooks = self.num_code_groups - 1  # 15
        self.max_seq = 2 + self.num_codebooks  # 17
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        cp = code_predictor
        self.small_to_mtp = cp.small_to_mtp_projection
        self.pred_model = cp.model
        self.lm_heads = cp.lm_head
        self.codec_embeds = cp.model.codec_embedding
        self.has_sliding_layers = "sliding_attention" in getattr(self.pred_model.config, "layer_types", [])

        self.static_cache = StaticCache(config=pred_config, max_cache_len=self.max_seq)

        self.prefill_cache_pos = torch.arange(2, device=device)
        self.decode_cache_positions = [
            torch.tensor([2 + i], device=device) for i in range(self.num_codebooks - 1)
        ]

        self.input_buf = torch.zeros(1, 2, talker_hidden_size, dtype=dtype, device=device)
        self.output_tokens = torch.zeros(self.num_codebooks, dtype=torch.long, device=device)

        self.graph = None
        self.captured = False
        self.prefill_attn = None
        self.decode_attn = None

    def _init_cache_layers(self):
        config = self.pred_model.config
        num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        dummy_k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=self.dtype, device=self.device)
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy_k)

    def _make_attn_mask(self, input_embeds: torch.Tensor, cache_position: torch.Tensor):
        mask = create_causal_mask(
            config=self.pred_model.config,
            input_embeds=input_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=self.static_cache,
        )
        if self.has_sliding_layers:
            sliding = create_sliding_window_causal_mask(
                config=self.pred_model.config,
                input_embeds=input_embeds,
                attention_mask=None,
                cache_position=cache_position,
                past_key_values=self.static_cache,
            )
            return {"full_attention": mask, "sliding_attention": sliding}
        return {"full_attention": mask}

    def _build_attention_masks(self):
        dummy_prefill = torch.zeros(1, 2, self.hidden_size, dtype=self.dtype, device=self.device)
        dummy_decode = torch.zeros(1, 1, self.hidden_size, dtype=self.dtype, device=self.device)
        self.prefill_attn = self._make_attn_mask(dummy_prefill, self.prefill_cache_pos)
        self.decode_attn = []
        for pos in self.decode_cache_positions:
            self.decode_attn.append(self._make_attn_mask(dummy_decode, pos))

    def _full_loop(self):
        h = self.small_to_mtp(self.input_buf)  # [1, 2, hidden]

        out = self.pred_model(
            inputs_embeds=h,
            attention_mask=self.prefill_attn,
            past_key_values=self.static_cache,
            cache_position=self.prefill_cache_pos,
            use_cache=True,
        )
        h = out.last_hidden_state

        logits = self.lm_heads[0](h[:, -1:, :])
        tok = sample_logits(
            logits[:, 0, :],
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, do_sample=self.do_sample,
        )
        self.output_tokens[0] = tok[0]

        for cb_idx in range(1, self.num_codebooks):
            emb = self.codec_embeds[cb_idx - 1](tok.unsqueeze(0))
            emb = self.small_to_mtp(emb)

            out = self.pred_model(
                inputs_embeds=emb,
                attention_mask=self.decode_attn[cb_idx - 1],
                past_key_values=self.static_cache,
                cache_position=self.decode_cache_positions[cb_idx - 1],
                use_cache=True,
            )
            h = out.last_hidden_state

            logits = self.lm_heads[cb_idx](h[:, -1:, :])
            tok = sample_logits(
                logits[:, 0, :],
                temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, do_sample=self.do_sample,
            )
            self.output_tokens[cb_idx] = tok[0]

        return self.output_tokens

    @torch.inference_mode()
    def capture(self, num_warmup=3):
        print(f"[PredictorGraph] Warming up ({num_warmup} runs)...", flush=True)
        self._init_cache_layers()
        self._build_attention_masks()

        for _ in range(num_warmup):
            self.static_cache.reset()
            self._full_loop()
        torch.cuda.synchronize()

        print("[PredictorGraph] Capturing CUDA graph...", flush=True)
        with torch.cuda.device(self.device_index):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.graph = torch.cuda.CUDAGraph()
                self.static_cache.reset()
                self._full_loop()
                torch.cuda.synchronize()

                self.static_cache.reset()
                with torch.cuda.graph(self.graph):
                    self._full_loop()

        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.captured = True
        print("[PredictorGraph] Captured!", flush=True)

    @torch.inference_mode()
    def run(self, pred_input: torch.Tensor) -> torch.Tensor:
        self.input_buf.copy_(pred_input)
        self.static_cache.reset()
        self.graph.replay()
        return self.output_tokens.clone()
