"""Test torch.compile with eager backend (bypasses Inductor entirely)."""
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

# --- torch.compile with eager backend ---
print("\nTrying eager backend...", flush=True)
try:
    tm_eager = torch.compile(talker_model, dynamic=False, backend='eager')

    nh = getattr(tc, 'num_key_value_heads', tc.num_attention_heads)
    hd = getattr(tc, 'head_dim', tc.hidden_size // tc.num_attention_heads)
    mask_fn = create_causal_mask if talker_model.config.sliding_window is None else create_sliding_window_causal_mask

    cache_e = StaticCache(config=tc, max_cache_len=2048)
    dk = torch.zeros(1, nh, 1, hd, dtype=torch.bfloat16, device=device)
    for l in cache_e.layers:
        if not l.is_initialized: l.lazy_initialization(dk)
    for li in range(tc.num_hidden_layers):
        k,v = out.past_key_values[li]
        cache_e.update(k,v,li,{"cache_position": torch.arange(k.shape[2], device=device)})

    attn_e = mask_fn(config=tc, input_embeds=di, attention_mask=tam, cache_position=torch.tensor([53], device=device), past_key_values=cache_e)

    def eager_step(inp):
        cp = torch.tensor([53], device=device)
        pd = torch.full((3,1,1), 53.0, device=device)
        o = tm_eager(inputs_embeds=inp, attention_mask=attn_e, past_key_values=cache_e, cache_position=cp, position_ids=pd, use_cache=True)
        return o.last_hidden_state

    print("First run...", flush=True)
    with torch.inference_mode(): eager_step(di.clone())
    sync()
    print("Done!", flush=True)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(30): eager_step(di.clone())
    sync()
    eb_ms = (time.perf_counter()-t0)/30*1000
    print(f"eager backend: {eb_ms:.1f}ms/step ({cg_ms/eb_ms:.2f}x)", flush=True)

except Exception as e:
    print(f"ERROR eager backend: {e}", flush=True)
    traceback.print_exc()

# --- torch.compile with aot_eager backend ---
print("\nTrying aot_eager backend...", flush=True)
try:
    tm_aot = torch.compile(talker_model, dynamic=False, backend='aot_eager')

    cache_a = StaticCache(config=tc, max_cache_len=2048)
    for l in cache_a.layers:
        if not l.is_initialized: l.lazy_initialization(dk)
    for li in range(tc.num_hidden_layers):
        k,v = out.past_key_values[li]
        cache_a.update(k,v,li,{"cache_position": torch.arange(k.shape[2], device=device)})

    attn_a = mask_fn(config=tc, input_embeds=di, attention_mask=tam, cache_position=torch.tensor([54], device=device), past_key_values=cache_a)

    def aot_step(inp):
        cp = torch.tensor([54], device=device)
        pd = torch.full((3,1,1), 54.0, device=device)
        o = tm_aot(inputs_embeds=inp, attention_mask=attn_a, past_key_values=cache_a, cache_position=cp, position_ids=pd, use_cache=True)
        return o.last_hidden_state

    print("First run...", flush=True)
    with torch.inference_mode(): aot_step(di.clone())
    sync()
    print("AOT done!", flush=True)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(30): aot_step(di.clone())
    sync()
    aot_ms = (time.perf_counter()-t0)/30*1000
    print(f"aot_eager backend: {aot_ms:.1f}ms/step ({cg_ms/aot_ms:.2f}x)", flush=True)

except Exception as e:
    print(f"ERROR aot_eager: {e}", flush=True)
    traceback.print_exc()

print(f"\n{'='*50}")
print(f"CUDA graph (baseline):   {cg_ms:.1f}ms/step")
