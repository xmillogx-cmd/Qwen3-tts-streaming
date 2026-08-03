"""Test torch.compile with reduce-overhead mode (avoids Inductor/Triton)."""
import torch, time, sys, traceback
sys.path.insert(0, r'G:\qwen-tts')

from qwen_tts import Qwen3TTSModel
from fast_tts_v12 import TalkerGraph, _build_talker_inputs
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask

model_path = r'G:\Foundation\models\Qwen3-TTS'
device = 'cuda:0'

print("Loading model...", flush=True)
m = Qwen3TTSModel.from_pretrained(model_path, device_map=device, dtype=torch.bfloat16)
tc = m.model.config.talker_config
talker_model = m.model.talker.model

# CUDA graph baseline
tg = TalkerGraph(talker_model, tc, device=device, max_seq_len=2048)
tg.capture(prefill_len=40, num_warmup=3)

tie, tam, tth, tts_pad = _build_talker_inputs(m, 'Hi', 'Russian', 'Sohee')
out = m.model.talker.forward(
    inputs_embeds=tie, attention_mask=tam, use_cache=True, output_hidden_states=True, return_dict=True,
    trailing_text_hidden=tth, tts_pad_embed=tts_pad,
    generation_step=None, past_hidden=None, past_key_values=None,
)
tg.prefill_kv(out.past_key_values)
tg.set_generation_state(tam, None)

di = torch.zeros(1, 1, tc.hidden_size, dtype=torch.bfloat16, device=device)
sync = torch.cuda.synchronize

# CUDA graph benchmark
for _ in range(5): tg.run(di, 50)
sync()
t0 = time.perf_counter()
for _ in range(30): tg.run(di, 50)
sync()
cg_ms = (time.perf_counter()-t0)/30*1000
print(f"CUDA graph: {cg_ms:.1f}ms/step", flush=True)

# --- torch.compile with reduce-overhead ---
print("\nTrying mode='reduce-overhead'...", flush=True)
try:
    tm_compiled = torch.compile(talker_model, dynamic=False, mode='reduce-overhead')

    cache = StaticCache(config=tc, max_cache_len=2048)
    nh = getattr(tc, 'num_key_value_heads', tc.num_attention_heads)
    hd = getattr(tc, 'head_dim', tc.hidden_size // tc.num_attention_heads)
    dk = torch.zeros(1, nh, 1, hd, dtype=torch.bfloat16, device=device)
    for l in cache.layers:
        if not l.is_initialized: l.lazy_initialization(dk)
    for li in range(tc.num_hidden_layers):
        k,v = out.past_key_values[li]
        cache.update(k,v,li,{"cache_position": torch.arange(k.shape[2], device=device)})

    mask_fn = create_causal_mask if talker_model.config.sliding_window is None else create_sliding_window_causal_mask
    attn = mask_fn(config=tc, input_embeds=di, attention_mask=tam, cache_position=torch.tensor([52], device=device), past_key_values=cache)

    def step(inp):
        cp = torch.tensor([52], device=device)
        pd = torch.full((3,1,1), 52.0, device=device)
        o = tm_compiled(inputs_embeds=inp, attention_mask=attn, past_key_values=cache, cache_position=cp, position_ids=pd, use_cache=True)
        return o.last_hidden_state

    print("First run (compiling)...", flush=True)
    with torch.inference_mode(): step(di.clone())
    sync()
    print("Compiled!", flush=True)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(30): step(di.clone())
    sync()
    comp_ms = (time.perf_counter()-t0)/30*1000
    print(f"reduce-overhead: {comp_ms:.1f}ms/step ({cg_ms/comp_ms:.2f}x)", flush=True)

except Exception as e:
    print(f"ERROR reduce-overhead: {e}", flush=True)
    traceback.print_exc()

# --- Try eager baseline for comparison ---
print("\nEager baseline...", flush=True)
cache2 = StaticCache(config=tc, max_cache_len=2048)
dk2 = torch.zeros(1, nh, 1, hd, dtype=torch.bfloat16, device=device)
for l in cache2.layers:
    if not l.is_initialized: l.lazy_initialization(dk2)
for li in range(tc.num_hidden_layers):
    k,v = out.past_key_values[li]
    cache2.update(k,v,li,{"cache_position": torch.arange(k.shape[2], device=device)})

attn2 = mask_fn(config=tc, input_embeds=di, attention_mask=tam, cache_position=torch.tensor([53], device=device), past_key_values=cache2)

def eager_step(inp):
    cp = torch.tensor([53], device=device)
    pd = torch.full((3,1,1), 53.0, device=device)
    o = talker_model(inputs_embeds=inp, attention_mask=attn2, past_key_values=cache2, cache_position=cp, position_ids=pd, use_cache=True)
    return o.last_hidden_state

sync()
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(30): eager_step(di.clone())
sync()
eager_ms = (time.perf_counter()-t0)/30*1000

print(f"\n{'='*50}")
print(f"CUDA graph:      {cg_ms:.1f}ms/step")
print(f"Eager baseline:  {eager_ms:.1f}ms/step ({cg_ms/eager_ms:.2f}x)")
try:
    print(f"reduce-overhead: {comp_ms:.1f}ms/step ({cg_ms/comp_ms:.2f}x)")
except NameError:
    print("reduce-overhead: FAILED")
