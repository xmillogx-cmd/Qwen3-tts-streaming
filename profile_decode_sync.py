"""Profile decode loop with proper CUDA sync to get accurate timings."""
import torch, time, sys
sys.path.insert(0, r'G:\qwen-tts')

from qwen_tts import Qwen3TTSModel
from fast_tts_v11 import (
    PredictorGraph, TalkerGraph, _build_talker_inputs, sample_logits, apply_repetition_penalty,
)

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

# Build graphs
pg = PredictorGraph(predictor, pred_config, talker_config.hidden_size, device=device, dtype=torch.bfloat16)
pg.capture(num_warmup=3)
tg = TalkerGraph(talker.model, talker_config, device=device, max_seq_len=2048)
tg.capture(prefill_len=40, num_warmup=3)

# Build inputs
tie, tam, tth, tts_pad = _build_talker_inputs(tts_model, text, language, speaker)

# Prefill
out = talker.forward(
    inputs_embeds=tie, attention_mask=tam, use_cache=True, output_hidden_states=True, return_dict=True,
    trailing_text_hidden=tth, tts_pad_embed=tts_pad,
    generation_step=None, past_hidden=None, past_key_values=None,
)
past_kv = out.past_key_values
past_hidden = out.past_hidden
gen_step = out.generation_step

logits = out.logits[:, -1, :]
token = sample_logits(logits, temperature=0.9, top_k=50, top_p=1.0, do_sample=True)

prefill_len = tg.prefill_kv(past_kv)
tg.set_generation_state(tam, getattr(talker, "rope_deltas", None))

# Setup for decode loop
eos_id = talker_config.codec_eos_token_id
vocab_size = talker_config.vocab_size
num_code_groups = talker_config.num_code_groups

suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
suppress_start = max(0, vocab_size - 1024)
for i in range(suppress_start, vocab_size):
    if i != eos_id:
        suppress_mask[i] = True

predictor_codec_embeds = predictor.get_input_embeddings()
talker_codec_embed = talker.get_input_embeddings()
talker_codec_head = talker.codec_head

# Pre-allocate history buffer (avoid torch.tensor creation each step)
history_buf = torch.zeros(200, dtype=torch.long, device=device)

print(f"\nProfiling decode loop with CUDA sync (prefill_len={prefill_len}, first_token={token.item()}, eos={eos_id})", flush=True)
print("=" * 60)

sync = torch.cuda.synchronize

with torch.inference_mode():
    t_total = 0
    steps_done = 0

    for step_idx in range(100):
        if token.item() == eos_id:
            print(f"EOS at step {step_idx}", flush=True)
            break

        sync()
        t_step_start = time.perf_counter()

        # --- Predictor ---
        last_id_hidden = talker_codec_embed(token.unsqueeze(1))
        pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
        codebook_token_ids = pg.run(pred_input)
        all_cb = torch.cat([token.view(1), codebook_token_ids])

        sync()
        t_pred_end = time.perf_counter()

        # --- Codec embedding assembly ---
        codec_hiddens = [last_id_hidden]
        for i in range(num_code_groups - 1):
            codec_hiddens.append(predictor_codec_embeds[i](codebook_token_ids[i].unsqueeze(0).unsqueeze(0)))
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)

        if gen_step < tth.shape[1]:
            inputs_embeds = inputs_embeds + tth[:, gen_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad

        sync()
        t_codec_end = time.perf_counter()

        # --- Talker decode ---
        current_pos = prefill_len + step_idx
        if current_pos >= tg.max_seq_len - 1:
            break

        hidden_states = tg.run(inputs_embeds, position=current_pos)
        logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)

        sync()
        t_talker_end = time.perf_counter()

        # --- Sampling ---
        if step_idx > 0:
            history_buf[step_idx] = all_cb[0]
            history = history_buf[:step_idx + 1]
            logits = apply_repetition_penalty(logits, history, 1.05)
        token = sample_logits(logits.squeeze(0), temperature=0.9, top_k=50, top_p=1.0,
                              do_sample=True, suppress_mask=suppress_mask)
        past_hidden = hidden_states[:, -1:, :].clone()
        gen_step += 1

        sync()
        t_sample_end = time.perf_counter()

        step_ms = (t_sample_end - t_step_start) * 1000
        pred_ms = (t_pred_end - t_step_start) * 1000
        codec_ms = (t_codec_end - t_pred_end) * 1000
        talker_ms = (t_talker_end - t_codec_end) * 1000
        sample_ms = (t_sample_end - t_talker_end) * 1000

        if step_idx < 5 or step_idx % 5 == 0:
            print(f"Step {step_idx}: total={step_ms:.1f}ms | pred={pred_ms:.1f}ms | codec={codec_ms:.1f}ms | talker={talker_ms:.1f}ms | sample={sample_ms:.1f}ms | token={token.item()}", flush=True)

        t_total += step_ms
        steps_done += 1

avg_ms = t_total / steps_done if steps_done else 0
print(f"\nAverage over {steps_done} steps: {avg_ms:.1f}ms/step ({steps_done/t_total*1000:.1f} step/s)", flush=True)
