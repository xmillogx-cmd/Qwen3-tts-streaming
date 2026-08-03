"""Profile pipeline overlap: predictor(N+1) on stream 1 overlaps with codec+talker(N) on default."""
import torch, time, sys
sys.path.insert(0, r'G:\qwen-tts')

from qwen_tts import Qwen3TTSModel
from fast_tts_v12 import PredictorGraph, TalkerGraph, _build_talker_inputs, sample_logits

model_path = r'G:\Foundation\models\Qwen3-TTS'
device = 'cuda:0'
text = "Привет мир. Это тест для проверки скорости синтеза речи с использованием оптимизированного конвейера."
language = 'Russian'
speaker = 'Sohee'

print("Loading model...", flush=True)
tts_model = Qwen3TTSModel.from_pretrained(model_path, device_map=device, dtype=torch.bfloat16)
inner = tts_model.model
talker = inner.talker
tc = inner.config.talker_config
predictor = talker.code_predictor
pred_config = predictor.config if hasattr(predictor, 'config') else predictor.model.config

# Build CUDA graphs
pg = PredictorGraph(predictor, pred_config, tc.hidden_size, device=device, dtype=torch.bfloat16)
pg.capture(num_warmup=3)

tg = TalkerGraph(talker.model, tc, device=device, max_seq_len=2048)
tg.capture(prefill_len=40, num_warmup=3)

tie, tam, tth, tts_pad = _build_talker_inputs(tts_model, text, language, speaker)

# Prefill
out = talker.forward(
    inputs_embeds=tie, attention_mask=tam, use_cache=True, output_hidden_states=True, return_dict=True,
    trailing_text_hidden=tth, tts_pad_embed=tts_pad,
    generation_step=None, past_hidden=None, past_key_values=None,
)
past_hidden = out.past_hidden
logits = out.logits[:, -1, :]
token = sample_logits(logits, temperature=0.9, top_k=50, top_p=1.0, do_sample=True)

prefill_len = tg.prefill_kv(out.past_key_values)
tg.set_generation_state(tam, getattr(talker, "rope_deltas", None))

talker_codec_embed = talker.get_input_embeddings()
talker_codec_head = talker.codec_head
predictor_codec_embeds = predictor.get_input_embeddings()
num_code_groups = tc.num_code_groups
eos_id = tc.codec_eos_token_id

sync = torch.cuda.synchronize
sync()

# === Sequential baseline (v11) ===
print("\n=== SEQUENTIAL BASELINE ===", flush=True)

# Reset graphs for fair comparison
pg2 = PredictorGraph(predictor, pred_config, tc.hidden_size, device=device, dtype=torch.bfloat16)
pg2.capture(num_warmup=3)
tg2 = TalkerGraph(talker.model, tc, device=device, max_seq_len=2048)
tg2.capture(prefill_len=prefill_len, num_warmup=3)

# Re-prefill for sequential run
out2 = talker.forward(
    inputs_embeds=tie, attention_mask=tam, use_cache=True, output_hidden_states=True, return_dict=True,
    trailing_text_hidden=tth, tts_pad_embed=tts_pad,
    generation_step=None, past_hidden=None, past_key_values=None,
)
past_hidden2 = out2.past_hidden
token2 = sample_logits(out2.logits[:, -1, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
prefill_len2 = tg2.prefill_kv(out2.past_key_values)
tg2.set_generation_state(tam, getattr(talker, "rope_deltas", None))

sync()
t0 = time.perf_counter()
steps_seq = 0
for step_idx in range(100):
    if token2.item() == eos_id:
        break
    last_id_hidden = talker_codec_embed(token2.unsqueeze(1))
    pred_input = torch.cat((past_hidden2, last_id_hidden), dim=1)
    codebook_token_ids = pg2.run(pred_input)

    codec_hiddens = [last_id_hidden]
    for i in range(num_code_groups - 1):
        codec_hiddens.append(predictor_codec_embeds[i](codebook_token_ids[i].unsqueeze(0).unsqueeze(0)))
    inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
    if step_idx < tth.shape[1]:
        inputs_embeds = inputs_embeds + tth[:, step_idx].unsqueeze(1)
    else:
        inputs_embeds = inputs_embeds + tts_pad

    hidden_states = tg2.run(inputs_embeds, position=prefill_len2 + step_idx)
    logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)
    token2 = sample_logits(logits.squeeze(0), temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
    past_hidden2 = hidden_states[:, -1:, :].clone()
    steps_seq += 1

sync()
seq_ms = (time.perf_counter()-t0)/steps_seq*1000 if steps_seq else 0
print(f"Sequential: {steps_seq} steps, {seq_ms:.1f}ms/step ({steps_seq/(seq_ms/1000):.0f} step/s)", flush=True)

# === Pipelined (predictor on separate stream) ===
print("\n=== PIPELINED (2 streams) ===", flush=True)

# Re-prefill for pipelined run
out3 = talker.forward(
    inputs_embeds=tie, attention_mask=tam, use_cache=True, output_hidden_states=True, return_dict=True,
    trailing_text_hidden=tth, tts_pad_embed=tts_pad,
    generation_step=None, past_hidden=None, past_key_values=None,
)
past_hidden3 = out3.past_hidden
token3 = sample_logits(out3.logits[:, -1, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
prefill_len3 = tg.prefill_kv(out3.past_key_values)
tg.set_generation_state(tam, getattr(talker, "rope_deltas", None))

pred_stream = torch.cuda.Stream(device=device)
sync()

t0 = time.perf_counter()
steps_pipe = 0
pred_launched = False

for step_idx in range(100):
    if token3.item() == eos_id:
        break

    # --- Launch predictor for THIS step on pred_stream (async) ---
    with torch.cuda.stream(pred_stream):
        last_id_hidden_pipe = talker_codec_embed(token3.unsqueeze(1))
        pred_input_pipe = torch.cat((past_hidden3, last_id_hidden_pipe), dim=1)
        pg.input_buf.copy_(pred_input_pipe)
        pg.static_cache.reset()
        pg.graph.replay()

    # --- Codec + talker on default stream (overlaps with predictor finishing) ---
    # But we need to wait for predictor to finish first...
    # The overlap is: while codec+talker(N-1) runs, predictor(N) starts
    # So the pattern should be:
    #   1. Wait for predictor(N) to finish (launched at end of step N-1)
    #   2. Codec assemble + talker(N) on default stream
    #   3. Launch predictor(N+1) on pred_stream (overlaps with next codec+talker)

    if pred_launched:
        # Wait for predictor from previous iteration
        pred_stream.synchronize()
        codebook_token_ids = pg.output_tokens.clone()
    else:
        # First step: just use the result we already computed above
        codebook_token_ids = pg.output_tokens.clone()

    # Codec assembly (default stream)
    last_id_hidden = talker_codec_embed(token3.unsqueeze(1))
    codec_hiddens = [last_id_hidden]
    for i in range(num_code_groups - 1):
        codec_hiddens.append(predictor_codec_embeds[i](codebook_token_ids[i].unsqueeze(0).unsqueeze(0)))
    inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
    if step_idx < tth.shape[1]:
        inputs_embeds = inputs_embeds + tth[:, step_idx].unsqueeze(1)
    else:
        inputs_embeds = inputs_embeds + tts_pad

    # Talker (default stream)
    hidden_states = tg.run(inputs_embeds, position=prefill_len3 + step_idx)
    logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)
    token3 = sample_logits(logits.squeeze(0), temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
    past_hidden3 = hidden_states[:, -1:, :].clone()
    pred_launched = True
    steps_pipe += 1

sync()
pipe_ms = (time.perf_counter()-t0)/steps_pipe*1000 if steps_pipe else 0
print(f"Pipelined: {steps_pipe} steps, {pipe_ms:.1f}ms/step ({steps_pipe/(pipe_ms/1000):.0f} step/s)", flush=True)

# === Summary ===
print(f"\n{'='*50}")
print(f"Sequential:  {seq_ms:.1f}ms/step ({steps_seq/(seq_ms/1000):.0f} step/s)")
print(f"Pipelined:   {pipe_ms:.1f}ms/step ({steps_pipe/(pipe_ms/1000):.0f} step/s)")
if pipe_ms < seq_ms:
    print(f"Speedup:   {seq_ms/pipe_ms:.2f}x")
else:
    print(f"No improvement (pipeline overhead or no overlap)")
