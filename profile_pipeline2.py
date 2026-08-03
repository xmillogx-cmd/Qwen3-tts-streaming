"""Correct pipeline: predictor(N+1) launched at end of step N, overlaps with codec+talker(N+1).

Timeline per step (steady state):
  ────────────────────── step N+1 ──────────────────────
  pred_stream:   [predictor(N+1) ────16ms────→]
  default:                  [codec(1ms)][talker(12ms)]
  
  predictor starts at end of step N, finishes during codec+talker of step N+1.
  Effective time ≈ max(predictor, codec+talker) ≈ 16ms/step (vs ~30ms sequential)
"""
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

eos_id = tc.codec_eos_token_id
num_code_groups = tc.num_code_groups
talker_codec_embed = talker.get_input_embeddings()
talker_codec_head = talker.codec_head
predictor_codec_embeds = predictor.get_input_embeddings()
sync = torch.cuda.synchronize

def do_prefill():
    """Fresh prefill, returns all state needed for decode."""
    pg = PredictorGraph(predictor, pred_config, tc.hidden_size, device=device, dtype=torch.bfloat16)
    pg.capture(num_warmup=3)
    tg = TalkerGraph(talker.model, tc, device=device, max_seq_len=2048)
    tg.capture(prefill_len=40, num_warmup=3)

    tie, tam, tth, tts_pad = _build_talker_inputs(tts_model, text, language, speaker)
    
    out = talker.forward(
        inputs_embeds=tie, attention_mask=tam, use_cache=True, output_hidden_states=True, return_dict=True,
        trailing_text_hidden=tth, tts_pad_embed=tts_pad,
        generation_step=None, past_hidden=None, past_key_values=None,
    )
    past_hidden = out.past_hidden
    token = sample_logits(out.logits[:, -1, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
    
    prefill_len = tg.prefill_kv(out.past_key_values)
    tg.set_generation_state(tam, getattr(talker, "rope_deltas", None))
    sync()
    return pg, tg, past_hidden, token, prefill_len, tth, tts_pad

# === Sequential baseline ===
print("\n=== SEQUENTIAL BASELINE ===", flush=True)
pg_seq, tg_seq, ph_seq, tok_seq, pl_seq, tth_seq, tpad_seq = do_prefill()

t0 = time.perf_counter()
steps_seq = 0
with torch.no_grad():
    for step_idx in range(100):
        if tok_seq.item() == eos_id:
            break
        
        last_id_hidden = talker_codec_embed(tok_seq.unsqueeze(1))
        pred_input = torch.cat((ph_seq, last_id_hidden), dim=1)
        codebook_token_ids = pg_seq.run(pred_input)

        codec_hiddens = [last_id_hidden]
        for i in range(num_code_groups - 1):
            codec_hiddens.append(predictor_codec_embeds[i](codebook_token_ids[i].unsqueeze(0).unsqueeze(0)))
    inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
    if step_idx < tth_seq.shape[1]:
        inputs_embeds = inputs_embeds + tth_seq[:, step_idx].unsqueeze(1)
    else:
        inputs_embeds = inputs_embeds + tpad_seq

    hidden_states = tg_seq.run(inputs_embeds, position=pl_seq + step_idx)
    logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)
    tok_seq = sample_logits(logits.squeeze(0), temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
    ph_seq = hidden_states[:, -1:, :].clone()
    steps_seq += 1

sync()
seq_elapsed = time.perf_counter() - t0
seq_ms = seq_elapsed / steps_seq * 1000 if steps_seq else 0
print(f"Sequential: {steps_seq} steps, {seq_ms:.1f}ms/step ({steps_seq/(seq_elapsed):.0f} step/s, total={seq_elapsed*1000:.0f}ms)", flush=True)

# === PIPELINED (predictor overlaps with codec+talker) ===
print("\n=== PIPELINED (2 CUDA streams) ===", flush=True)
pg_pipe, tg_pipe, ph_pipe, tok_pipe, pl_pipe, tth_pipe, tpad_pipe = do_prefill()

pred_stream = torch.cuda.Stream(device=device)
sync()

t0 = time.perf_counter()
steps_pipe = 0

# Launch predictor for step 0 on pred_stream (first step, no overlap yet)
with torch.cuda.stream(pred_stream):
    last_id_h_0 = talker_codec_embed(tok_pipe.unsqueeze(1))
    pred_inp_0 = torch.cat((ph_pipe, last_id_h_0), dim=1)
    pg_pipe.input_buf.copy_(pred_inp_0)
    pg_pipe.static_cache.reset()
    pg_pipe.graph.replay()

for step_idx in range(100):
    if tok_pipe.item() == eos_id:
        break
    
    # Wait for predictor to finish (launched at end of previous step or above)
    pred_stream.synchronize()
    
    # Collect codebook tokens from predictor output buffer
    codebook_token_ids = pg_pipe.output_tokens.clone()
    
    # Compute last_id_hidden on default stream (needed for codec assembly)
    last_id_hidden = talker_codec_embed(tok_pipe.unsqueeze(1))

    # Codec assembly (default stream) — this overlaps with NEXT predictor
    codec_hiddens = [last_id_hidden]
    for i in range(num_code_groups - 1):
        codec_hiddens.append(predictor_codec_embeds[i](codebook_token_ids[i].unsqueeze(0).unsqueeze(0)))
    inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
    if step_idx < tth_pipe.shape[1]:
        inputs_embeds = inputs_embeds + tth_pipe[:, step_idx].unsqueeze(1)
    else:
        inputs_embeds = inputs_embeds + tpad_pipe

    # Talker (default stream) — this overlaps with NEXT predictor
    hidden_states = tg_pipe.run(inputs_embeds, position=pl_pipe + step_idx)
    logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)
    tok_next = sample_logits(logits.squeeze(0), temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
    ph_next = hidden_states[:, -1:, :].clone()

    # Launch predictor for NEXT step on pred_stream (async — overlaps with next iteration's codec+talker)
    with torch.cuda.stream(pred_stream):
        last_id_h_next = talker_codec_embed(tok_next.unsqueeze(1))
        pred_inp_next = torch.cat((ph_next, last_id_h_next), dim=1)
        pg_pipe.input_buf.copy_(pred_inp_next)
        pg_pipe.static_cache.reset()
        pg_pipe.graph.replay()

    tok_pipe = tok_next
    ph_pipe = ph_next
    steps_pipe += 1

sync()
pipe_elapsed = time.perf_counter() - t0
pipe_ms = pipe_elapsed / steps_pipe * 1000 if steps_pipe else 0
print(f"Pipelined: {steps_pipe} steps, {pipe_ms:.1f}ms/step ({steps_pipe/(pipe_elapsed):.0f} step/s, total={pipe_elapsed*1000:.0f}ms)", flush=True)

# Summary
print(f"\n{'='*50}")
print(f"Sequential:  {seq_ms:.1f}ms/step ({steps_seq/(seq_elapsed):.0f} step/s)")
print(f"Pipelined:   {pipe_ms:.1f}ms/step ({steps_pipe/(pipe_elapsed):.0f} step/s)")
if pipe_ms < seq_ms:
    print(f"Speedup:   {seq_ms/pipe_ms:.2f}x ({(seq_ms-pipe_ms):.1f}ms saved per step)")
else:
    print(f"No speedup — pipeline overhead or no overlap achieved")
