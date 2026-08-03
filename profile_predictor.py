"""Profile predictor: CUDA graph vs eager, with/without clone."""
import torch, time, sys
sys.path.insert(0, r'G:\qwen-tts')

from qwen_tts import Qwen3TTSModel
from fast_tts_v11 import PredictorGraph, _build_talker_inputs, sample_logits, apply_repetition_penalty

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

# Build CUDA graph
pg = PredictorGraph(predictor, pred_config, talker_config.hidden_size, device=device, dtype=torch.bfloat16)
pg.capture(num_warmup=3)

# Build inputs for prefill to get past_hidden
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

# Warmup
for _ in range(5):
    pg.run(pred_input)
sync()

print("\n=== Predictor: clone vs copy_ ===")

# 1. clone() (current)
t0 = time.perf_counter()
for _ in range(20):
    out = pg.run(pred_input)
sync()
t_clone = (time.perf_counter() - t0) / 20 * 1000

# 2. Direct buffer access with copy_ instead of clone
buf = pg.output_tokens
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(20):
        pg.input_buf.copy_(pred_input)
        pg.static_cache.reset()
        pg.graph.replay()
        out = torch.empty_like(buf)
        out.copy_(buf)
sync()
t_copy = (time.perf_counter() - t0) / 20 * 1000

# 3. No output copy at all — just replay
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(20):
        pg.input_buf.copy_(pred_input)
        pg.static_cache.reset()
        pg.graph.replay()
sync()
t_nocopy = (time.perf_counter() - t0) / 20 * 1000

print(f"clone():     {t_clone:.1f}ms")
print(f"copy_():     {t_copy:.1f}ms")
print(f"no copy:     {t_nocopy:.1f}ms (graph replay only)")
print(f"Savings from removing clone: {(t_clone - t_copy)/t_clone*100:.0f}%")

# 4. Eager mode baseline
print("\n=== Eager mode baseline ===")

def eager_predictor(pred_input, predictor, talker_codec_embed):
    """Run predictor in eager mode (no CUDA graph)."""
    h = predictor.small_to_mtp(pred_input)
    out = predictor.model(
        inputs_embeds=h, cache_position=torch.arange(2, device=device),
        use_cache=True, past_key_values=None,
    )
    h = out.last_hidden_state
    logits = predictor.lm_head[0](h[:, -1:, :])
    tok = sample_logits(logits[:, 0, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
    tokens = [tok]

    for cb_idx in range(1, 15):
        emb = predictor.model.codec_embedding[cb_idx - 1](tok.unsqueeze(0))
        emb = predictor.small_to_mtp(emb)
        pos = torch.tensor([2 + cb_idx], device=device)
        out = predictor.model(
            inputs_embeds=emb, cache_position=pos, use_cache=True, past_key_values=None,
        )
        h = out.last_hidden_state
        logits = predictor.lm_head[cb_idx](h[:, -1:, :])
        tok = sample_logits(logits[:, 0, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
        tokens.append(tok)

    return torch.stack(tokens)

# Warmup eager
for _ in range(5):
    eager_predictor(pred_input.clone(), predictor, talker_codec_embed)
sync()

t0 = time.perf_counter()
for _ in range(20):
    eager_predictor(pred_input.clone(), predictor, talker_codec_embed)
sync()
t_eager = (time.perf_counter() - t0) / 20 * 1000
print(f"Eager (no cache): {t_eager:.1f}ms")

# 5. Check static_cache.reset() cost
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(20):
        pg.static_cache.reset()
sync()
t_reset = (time.perf_counter() - t0) / 20 * 1000
print(f"static_cache.reset(): {t_reset:.3f}ms")

# 6. Graph replay only (no reset, no copy)
t0 = time.perf_counter()
for _ in range(20):
    pg.graph.replay()
sync()
t_replay = (time.perf_counter() - t0) / 20 * 1000
print(f"graph.replay() only: {t_replay:.1f}ms")
