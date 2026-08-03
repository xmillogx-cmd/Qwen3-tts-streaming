"""Test torch.compile on full predictor loop (no CUDA graph)."""
import torch, time, sys
sys.path.insert(0, r'G:\qwen-tts')

from qwen_tts import Qwen3TTSModel
from fast_tts_v12 import PredictorGraph, _build_talker_inputs, sample_logits
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask

model_path = r'G:\Foundation\models\Qwen3-TTS'
device = 'cuda:0'
text = "Привет мир."
language = 'Russian'
speaker = 'Sohee'

print(f"Loading model...", flush=True)
tts_model = Qwen3TTSModel.from_pretrained(model_path, device_map=device, dtype=torch.bfloat16)
inner = tts_model.model
talker = inner.talker
talker_config = inner.config.talker_config
predictor = talker.code_predictor
pred_config = predictor.config if hasattr(predictor, 'config') else predictor.model.config

# CUDA graph baseline
pg = PredictorGraph(predictor, pred_config, talker_config.hidden_size, device=device, dtype=torch.bfloat16)
pg.capture(num_warmup=3)

tie, tam, tth, tts_pad = _build_talker_inputs(tts_model, text, language, speaker)
out = talker.forward(
    inputs_embeds=tie, attention_mask=tam, use_cache=True, output_hidden_states=True, return_dict=True,
    trailing_text_hidden=tth, tts_pad_embed=tts_pad,
    generation_step=None, past_hidden=None, past_key_values=None,
)
past_hidden = out.past_hidden
logits = out.logits[:, -1, :]
token = sample_logits(logits, temperature=0.9, top_k=50, top_p=1.0, do_sample=True)

talker_codec_embed = talker.get_input_embeddings()
last_id_hidden = talker_codec_embed(token.unsqueeze(1))
pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)

sync = torch.cuda.synchronize

# CUDA graph baseline
with torch.inference_mode():
    for _ in range(5):
        pg.run(pred_input)
sync()

t0 = time.perf_counter()
for _ in range(20):
    pg.run(pred_input)
sync()
cuda_graph_ms = (time.perf_counter() - t0) / 20 * 1000
print(f"CUDA graph: {cuda_graph_ms:.1f}ms/step")

# --- Eager mode with torch.compile ---
print("\nBuilding compiled predictor...", flush=True)

small_to_mtp = predictor.small_to_mtp_projection
pred_model = predictor.model
lm_heads = predictor.lm_head
codec_embeds = predictor.model.codec_embedding

# Build attention masks (same as PredictorGraph)
mask_fn = create_causal_mask if pred_model.config.sliding_window is None else create_sliding_window_causal_mask

def build_masks():
    cache = StaticCache(config=pred_config, max_cache_len=17)
    # Init layers
    num_kv_heads = getattr(pred_config, 'num_key_value_heads', pred_config.num_attention_heads)
    head_dim = getattr(pred_config, 'head_dim', pred_config.hidden_size // pred_config.num_attention_heads)
    dummy_k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=torch.bfloat16, device=device)
    for layer in cache.layers:
        if not layer.is_initialized:
            layer.lazy_initialization(dummy_k)

    prefill_pos = torch.arange(2, device=device)
    dummy_prefill = torch.zeros(1, 2, pred_config.hidden_size, dtype=torch.bfloat16, device=device)
    prefill_attn = mask_fn(config=pred_config, input_embeds=dummy_prefill, attention_mask=None,
                           cache_position=prefill_pos, past_key_values=cache)

    decode_positions = [torch.tensor([2 + i], device=device) for i in range(14)]
    decode_attn = []
    dummy_decode = torch.zeros(1, 1, pred_config.hidden_size, dtype=torch.bfloat16, device=device)
    for pos in decode_positions:
        decode_attn.append(mask_fn(config=pred_config, input_embeds=dummy_decode, attention_mask=None,
                                   cache_position=pos, past_key_values=cache))

    return cache, prefill_pos, prefill_attn, decode_positions, decode_attn

# Compile the inner model forward
pred_model_compiled = torch.compile(pred_model, dynamic=False)

def eager_predictor_loop(pred_input):
    """Eager predictor with compiled inner model."""
    cache, prefill_pos, prefill_attn, decode_positions, decode_attn = build_masks()

    h = small_to_mtp(pred_input)  # [1, 2, H]

    # Prefill
    out = pred_model_compiled(
        inputs_embeds=h, attention_mask=prefill_attn,
        past_key_values=cache, cache_position=prefill_pos,
        use_cache=True,
    )
    h = out.last_hidden_state
    logits = lm_heads[0](h[:, -1:, :])
    tok = sample_logits(logits[:, 0, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
    tokens = [tok]

    for cb_idx in range(1, 15):
        emb = codec_embeds[cb_idx - 1](tok.unsqueeze(0))
        emb = small_to_mtp(emb)
        out = pred_model_compiled(
            inputs_embeds=emb, attention_mask=decode_attn[cb_idx-1],
            past_key_values=cache, cache_position=decode_positions[cb_idx-1],
            use_cache=True,
        )
        h = out.last_hidden_state
        logits = lm_heads[cb_idx](h[:, -1:, :])
        tok = sample_logits(logits[:, 0, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
        tokens.append(tok)

    return torch.stack(tokens)

# Warmup + compile
print("First run (compiling...)...", flush=True)
with torch.inference_mode():
    eager_predictor_loop(pred_input.clone())
sync()
print("Compiled!", flush=True)

# Benchmark
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(20):
        eager_predictor_loop(pred_input.clone())
sync()
compiled_ms = (time.perf_counter() - t0) / 20 * 1000
print(f"Compiled (build_masks each step): {compiled_ms:.1f}ms/step")

# Try with pre-built masks (reuse across steps — but StaticCache resets break this)
# So let's try: compile the whole function including mask building
print("\nTrying full compile with dynamic shapes...", flush=True)

@torch.compile(dynamic=False)
def compiled_full_loop(pred_input, small_to_mtp, pred_model, lm_heads, codec_embeds,
                       cache, prefill_pos, prefill_attn, decode_positions, decode_attn):
    h = small_to_mtp(pred_input)
    out = pred_model(
        inputs_embeds=h, attention_mask=prefill_attn,
        past_key_values=cache, cache_position=prefill_pos, use_cache=True,
    )
    h = out.last_hidden_state
    logits = lm_heads[0](h[:, -1:, :])
    tok = sample_logits(logits[:, 0, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
    tokens = [tok]

    for cb_idx in range(1, 15):
        emb = codec_embeds[cb_idx - 1](tok.unsqueeze(0))
        emb = small_to_mtp(emb)
        out = pred_model(
            inputs_embeds=emb, attention_mask=decode_attn[cb_idx-1],
            past_key_values=cache, cache_position=decode_positions[cb_idx-1], use_cache=True,
        )
        h = out.last_hidden_state
        logits = lm_heads[cb_idx](h[:, -1:, :])
        tok = sample_logits(logits[:, 0, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
        tokens.append(tok)

    return torch.stack(tokens)

# This won't work well because StaticCache state changes between calls
# Let's try a simpler approach: compile just the transformer forward passes

print("\n=== Summary ===")
print(f"CUDA graph (baseline): {cuda_graph_ms:.1f}ms/step")
print(f"Compiled eager:        {compiled_ms:.1f}ms/step ({cuda_graph_ms/compiled_ms:.2f}x)")
