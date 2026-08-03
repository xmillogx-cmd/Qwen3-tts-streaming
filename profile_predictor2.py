"""Deep profile of predictor graph components."""
import torch, time, sys
sys.path.insert(0, r'G:\qwen-tts')

from qwen_tts import Qwen3TTSModel
from fast_tts_v11 import PredictorGraph, _build_talker_inputs, sample_logits

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

# Warmup
with torch.inference_mode():
    for _ in range(5):
        pg.input_buf.copy_(pred_input)
        pg.static_cache.reset()
        pg.graph.replay()
sync()

print("\n=== Component breakdown ===")

# 1. Just graph.replay (no reset, no input copy)
t0 = time.perf_counter()
for _ in range(30):
    pg.graph.replay()
sync()
print(f"graph.replay only:     {(time.perf_counter()-t0)/30*1000:.2f}ms")

# 2. reset + replay (no input copy)
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(30):
        pg.static_cache.reset()
        pg.graph.replay()
sync()
print(f"reset + replay:        {(time.perf_counter()-t0)/30*1000:.2f}ms")

# 3. input copy + reset + replay
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(30):
        pg.input_buf.copy_(pred_input)
        pg.static_cache.reset()
        pg.graph.replay()
sync()
print(f"copy + reset + replay: {(time.perf_counter()-t0)/30*1000:.2f}ms")

# 4. Just input copy
t0 = time.perf_counter()
for _ in range(30):
    pg.input_buf.copy_(pred_input)
sync()
print(f"input_buf.copy only:   {(time.perf_counter()-t0)/30*1000:.4f}ms")

# 5. Just reset
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(30):
        pg.static_cache.reset()
sync()
print(f"static_cache.reset:    {(time.perf_counter()-t0)/30*1000:.4f}ms")

# 6. Check cache size
total_params = 0
for layer in pg.static_cache.layers:
    total_params += layer.keys.numel() + layer.values.numel()
print(f"\nStaticCache size: {total_params*2/1e6:.1f}MB (bfloat16)")
print(f"Reset = zeroing {total_params*2/1e6:.1f}MB each step")

# 7. Profile talker graph for comparison
from fast_tts_v11 import TalkerGraph
tg = TalkerGraph(talker.model, talker_config, device=device, max_seq_len=2048)
tg.capture(prefill_len=40, num_warmup=3)

dummy_input = torch.zeros(1, 1, talker_config.hidden_size, dtype=torch.bfloat16, device=device)

# Talker: replay only
t0 = time.perf_counter()
for _ in range(30):
    tg.graph.replay()
sync()
print(f"\ntalker graph.replay only: {(time.perf_counter()-t0)/30*1000:.2f}ms")

# Talker: full run (input copy + attn_mask + replay)
t0 = time.perf_counter()
for _ in range(30):
    tg.run(dummy_input, position=50)
sync()
print(f"talker full run:        {(time.perf_counter()-t0)/30*1000:.2f}ms")
