"""Test torch.compile alternatives for sm_120."""
import torch, time, sys, os, traceback
sys.path.insert(0, r'G:\qwen-tts')

# Try forcing older arch before importing model
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'  # Ada Lovelace as fallback

from qwen_tts import Qwen3TTSModel
from fast_tts_v12 import TalkerGraph, _build_talker_inputs
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask

model_path = r'G:\Foundation\models\Qwen3-TTS'
device = 'cuda:0'

print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA arch: {torch.cuda.get_device_capability(0)}", flush=True)

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

# --- Eager baseline ---
nh = getattr(tc, 'num_key_value_heads', tc.num_attention_heads)
hd = getattr(tc, 'head_dim', tc.hidden_size // tc.num_attention_heads)
mask_fn = create_causal_mask if talker_model.config.sliding_window is None else create_sliding_window_causal_mask

cache_eager = StaticCache(config=tc, max_cache_len=2048)
dk = torch.zeros(1, nh, 1, hd, dtype=torch.bfloat16, device=device)
for l in cache_eager.layers:
    if not l.is_initialized: l.lazy_initialization(dk)
for li in range(tc.num_hidden_layers):
    k,v = out.past_key_values[li]
    cache_eager.update(k,v,li,{"cache_position": torch.arange(k.shape[2], device=device)})

attn_eager = mask_fn(config=tc, input_embeds=di, attention_mask=tam, cache_position=torch.tensor([53], device=device), past_key_values=cache_eager)

def eager_step(inp):
    cp = torch.tensor([53], device=device)
    pd = torch.full((3,1,1), 53.0, device=device)
    o = talker_model(inputs_embeds=inp, attention_mask=attn_eager, past_key_values=cache_eager, cache_position=cp, position_ids=pd, use_cache=True)
    return o.last_hidden_state

sync()
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(30): eager_step(di.clone())
sync()
eager_ms = (time.perf_counter()-t0)/30*1000
print(f"Eager baseline: {eager_ms:.1f}ms/step ({cg_ms/eager_ms:.2f}x faster with CUDA graph)", flush=True)

# --- Try torch.compile with suppress_errors (falls back to eager) ---
print("\nTrying torch.compile with suppress_errors...", flush=True)
torch._dynamo.config.suppress_errors = True
try:
    tm_compiled = torch.compile(talker_model, dynamic=False, mode='reduce-overhead')

    cache_c = StaticCache(config=tc, max_cache_len=2048)
    for l in cache_c.layers:
        if not l.is_initialized: l.lazy_initialization(dk)
    for li in range(tc.num_hidden_layers):
        k,v = out.past_key_values[li]
        cache_c.update(k,v,li,{"cache_position": torch.arange(k.shape[2], device=device)})

    attn_c = mask_fn(config=tc, input_embeds=di, attention_mask=tam, cache_position=torch.tensor([54], device=device), past_key_values=cache_c)

    def compiled_step(inp):
        cp = torch.tensor([54], device=device)
        pd = torch.full((3,1,1), 54.0, device=device)
        o = tm_compiled(inputs_embeds=inp, attention_mask=attn_c, past_key_values=cache_c, cache_position=cp, position_ids=pd, use_cache=True)
        return o.last_hidden_state

    print("First run...", flush=True)
    with torch.inference_mode(): compiled_step(di.clone())
    sync()
    print("Compiled!", flush=True)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(30): compiled_step(di.clone())
    sync()
    comp_ms = (time.perf_counter()-t0)/30*1000
    print(f"suppress_errors: {comp_ms:.1f}ms/step ({cg_ms/comp_ms:.2f}x)", flush=True)

except Exception as e:
    print(f"ERROR suppress_errors: {e}", flush=True)
    traceback.print_exc()

# --- Try torch.compile with eager backend (no Inductor at all) ---
print("\nTrying torch.compile with eager backend...", flush=True)
try:
    from torch._functorch import aot_autograd
    # Use AOTAutograd with eager as the fw_compiler and bw_compiler
    tm_eager_backend = torch.compile(
        talker_model, dynamic=False,
        backend='eager',  # Skip Inductor entirely
    )

    cache_e2 = StaticCache(config=tc, max_cache_len=2048)
    for l in cache_e2.layers:
        if not l.is_initialized: l.lazy_initialization(dk)
    for li in range(tc.num_hidden_layers):
        k,v = out.past_key_values[li]
        cache_e2.update(k,v,li,{"cache_position": torch.arange(k.shape[2], device=device)})

    attn_e2 = mask_fn(config=tc, input_embeds=di, attention_mask=tam, cache_position=torch.tensor([55], device=device), past_key_values=cache_e2)

    def eager_backend_step(inp):
        cp = torch.tensor([55], device=device)
        pd = torch.full((3,1,1), 55.0, device=device)
        o = tm_eager_backend(inputs_embeds=inp, attention_mask=attn_e2, past_key_values=cache_e2, cache_position=cp, position_ids=pd, use_cache=True)
        return o.last_hidden_state

    print("First run...", flush=True)
    with torch.inference_mode(): eager_backend_step(di.clone())
    sync()
    print("Done!", flush=True)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(30): eager_backend_step(di.clone())
    sync()
    eb_ms = (time.perf_counter()-t0)/30*1000
    print(f"eager backend: {eb_ms:.1f}ms/step ({cg_ms/eb_ms:.2f}x)", flush=True)

except Exception as e:
    print(f"ERROR eager backend: {e}", flush=True)
    traceback.print_exc()

# --- Try inductor with forced arch via env var (already set TORCH_CUDA_ARCH_LIST=8.9 above) ---
print("\nTrying torch.compile with TORCH_CUDA_ARCH_LIST=8.9...", flush=True)
try:
    tm_inductor = torch.compile(talker_model, dynamic=False)

    cache_i = StaticCache(config=tc, max_cache_len=2048)
    for l in cache_i.layers:
        if not l.is_initialized: l.lazy_initialization(dk)
    for li in range(tc.num_hidden_layers):
        k,v = out.past_key_values[li]
        cache_i.update(k,v,li,{"cache_position": torch.arange(k.shape[2], device=device)})

    attn_i = mask_fn(config=tc, input_embeds=di, attention_mask=tam, cache_position=torch.tensor([56], device=device), past_key_values=cache_i)

    def inductor_step(inp):
        cp = torch.tensor([56], device=device)
        pd = torch.full((3,1,1), 56.0, device=device)
        o = tm_inductor(inputs_embeds=inp, attention_mask=attn_i, past_key_values=cache_i, cache_position=cp, position_ids=pd, use_cache=True)
        return o.last_hidden_state

    print("First run...", flush=True)
    with torch.inference_mode(): inductor_step(di.clone())
    sync()
    print("Inductor compiled!", flush=True)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(30): inductor_step(di.clone())
    sync()
    ind_ms = (time.perf_counter()-t0)/30*1000
    print(f"inductor (arch 8.9): {ind_ms:.1f}ms/step ({cg_ms/ind_ms:.2f}x)", flush=True)

except Exception as e:
    print(f"ERROR inductor arch 8.9: {e}", flush=True)
    traceback.print_exc()

# Summary
print(f"\n{'='*50}")
print(f"CUDA graph (baseline):   {cg_ms:.1f}ms/step")
print(f"Eager baseline:          {eager_ms:.1f}ms/step ({cg_ms/eager_ms:.2f}x)")
