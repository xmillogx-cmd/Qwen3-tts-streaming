"""Test torch.compile on predictor loop vs CUDA graph."""
import torch, time, sys
sys.path.insert(0, r'G:\qwen-tts')

from qwen_tts import Qwen3TTSModel
from fast_tts_v12 import PredictorGraph, _build_talker_inputs, sample_logits

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

# Build CUDA graph (baseline)
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
print(f"CUDA graph (baseline): {cuda_graph_ms:.1f}ms/step")

# Now test compiled eager predictor
print("\nBuilding compiled eager predictor...", flush=True)

# Extract components
cp = predictor
small_to_mtp_compiled = torch.compile(cp.small_to_mtp, mode='max-autotune')
pred_model_compiled = torch.compile(cp.model, mode='max-autotune')

# Warmup compiled
codec_embeds = cp.model.codec_embedding
lm_heads = cp.lm_head
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask

pred_cache = StaticCache(config=pred_config, max_cache_len=17)
prefill_pos = torch.arange(2, device=device)
decode_positions = [torch.tensor([2 + i], device=device) for i in range(14)]

def eager_pred_step(pred_input):
    """Eager predictor loop with compiled components."""
    h = small_to_mtp_compiled(pred_input)  # [1, 2, H]

    # Prefill
    out = pred_model_compiled(
        inputs_embeds=h, cache_position=prefill_pos,
        past_key_values=pred_cache, use_cache=True,
    )
    h = out.last_hidden_state
    logits = lm_heads[0](h[:, -1:, :])
    tok = sample_logits(logits[:, 0, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
    tokens = [tok]

    for cb_idx in range(1, 15):
        emb = codec_embeds[cb_idx - 1](tok.unsqueeze(0))
        emb = small_to_mtp_compiled(emb)
        out = pred_model_compiled(
            inputs_embeds=emb, cache_position=decode_positions[cb_idx-1],
            past_key_values=pred_cache, use_cache=True,
        )
        h = out.last_hidden_state
        logits = lm_heads[cb_idx](h[:, -1:, :])
        tok = sample_logits(logits[:, 0, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
        tokens.append(tok)

    return torch.stack(tokens)

# Warmup (first run compiles!)
print("First compiled run (compiling...)...", flush=True)
with torch.inference_mode():
    pred_cache.reset()
    eager_pred_step(pred_input.clone())
sync()
print("Compilation done!", flush=True)

# Benchmark compiled
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(20):
        pred_cache.reset()
        eager_pred_step(pred_input.clone())
sync()
compiled_ms = (time.perf_counter() - t0) / 20 * 1000
print(f"Compiled eager: {compiled_ms:.1f}ms/step")

# Test full compile on the entire loop
print("\nBuilding fully compiled predictor...", flush=True)

@torch.compile(mode='max-autotune')
def compiled_pred_loop_full(pred_input, cache):
    """Fully compiled predictor."""
    h = small_to_mtp_compiled(pred_input)
    out = pred_model_compiled(
        inputs_embeds=h, cache_position=prefill_pos,
        past_key_values=cache, use_cache=True,
    )
    h = out.last_hidden_state
    logits = lm_heads[0](h[:, -1:, :])
    tok = sample_logits(logits[:, 0, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
    tokens = [tok]

    for cb_idx in range(1, 15):
        emb = codec_embeds[cb_idx - 1](tok.unsqueeze(0))
        emb = small_to_mtp_compiled(emb)
        out = pred_model_compiled(
            inputs_embeds=emb, cache_position=decode_positions[cb_idx-1],
            past_key_values=cache, use_cache=True,
        )
        h = out.last_hidden_state
        logits = lm_heads[cb_idx](h[:, -1:, :])
        tok = sample_logits(logits[:, 0, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
        tokens.append(tok)

    return torch.stack(tokens)

pred_cache2 = StaticCache(config=pred_config, max_cache_len=17)
print("First fully-compiled run (compiling...)...", flush=True)
with torch.inference_mode():
    pred_cache2.reset()
    compiled_pred_loop_full(pred_input.clone(), pred_cache2)
sync()
print("Full compilation done!", flush=True)

t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(20):
        pred_cache2.reset()
        compiled_pred_loop_full(pred_input.clone(), pred_cache2)
sync()
full_compiled_ms = (time.perf_counter() - t0) / 20 * 1000
print(f"Fully compiled: {full_compiled_ms:.1f}ms/step")

# Summary
print(f"\n{'='*50}")
print(f"CUDA graph (baseline):   {cuda_graph_ms:.1f}ms/step")
print(f"Compiled eager:          {compiled_ms:.1f}ms/step ({cuda_graph_ms/compiled_ms:.2f}x)")
print(f"Fully compiled:          {full_compiled_ms:.1f}ms/step ({cuda_graph_ms/full_compiled_ms:.2f}x)")
