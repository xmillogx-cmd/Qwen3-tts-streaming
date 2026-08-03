"""Test torch.compile on talker single-step decode vs CUDA graph."""
import torch, time, sys
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
prefill_len = tg.prefill_kv(out.past_key_values)
tg.set_generation_state(tam, None)

di = torch.zeros(1, 1, tc.hidden_size, dtype=torch.bfloat16, device=device)
sync = torch.cuda.synchronize

# CUDA graph benchmark
for _ in range(5):
    tg.run(di, 50)
sync()
t0 = time.perf_counter()
for _ in range(30):
    tg.run(di, 50)
sync()
cg_ms = (time.perf_counter() - t0) / 30 * 1000
print(f"CUDA graph: {cg_ms:.1f}ms/step")

# --- torch.compile on talker model ---
print("\nCompiling talker model...", flush=True)
talker_compiled = torch.compile(talker_model, dynamic=False)

# Build fresh static cache for compiled version
cache = StaticCache(config=tc, max_cache_len=2048)
num_kv_heads = getattr(tc, 'num_key_value_heads', tc.num_attention_heads)
head_dim = getattr(tc, 'head_dim', tc.hidden_size // tc.num_attention_heads)
dummy_k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=torch.bfloat16, device=device)
for layer in cache.layers:
    if not layer.is_initialized:
        layer.lazy_initialization(dummy_k)

# Copy prefill KV
for li in range(tc.num_hidden_layers):
    k, v = out.past_key_values[li]
    cache_pos = torch.arange(k.shape[2], device=device)
    cache.update(k, v, li, {"cache_position": cache_pos})

# Build attention mask
mask_fn = create_causal_mask if talker_model.config.sliding_window is None else create_sliding_window_causal_mask
attn = mask_fn(config=tc, input_embeds=di, attention_mask=tam,
               cache_position=torch.tensor([52], device=device), past_key_values=cache)

rope_base = torch.zeros(1, 1, dtype=torch.float32, device=device)

@torch.compile(dynamic=False)
def compiled_step(input_embeds, position):
    cp = torch.tensor([position], device=device)
    delta = rope_base + cp.to(dtype=torch.float32)
    pos_ids = delta.unsqueeze(0).expand(3, -1, -1)
    out = talker_compiled(
        inputs_embeds=input_embeds, attention_mask=attn,
        past_key_values=cache, cache_position=cp,
        position_ids=pos_ids, use_cache=True,
    )
    return out.last_hidden_state

# Warmup (first run compiles)
print("First compiled run...", flush=True)
with torch.inference_mode():
    compiled_step(di.clone(), 52)
sync()
print("Compiled!", flush=True)

# Benchmark
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(30):
        compiled_step(di.clone(), 52)
sync()
comp_ms = (time.perf_counter() - t0) / 30 * 1000
print(f"torch.compile: {comp_ms:.1f}ms/step")

# Summary
print(f"\n{'='*40}")
print(f"CUDA graph:    {cg_ms:.1f}ms/step (baseline)")
print(f"torch.compile: {comp_ms:.1f}ms/step ({cg_ms/comp_ms:.2f}x speedup)" if comp_ms < cg_ms else f"torch.compile: {comp_ms:.1f}ms/step ({cg_ms/comp_ms:.2f}x slower)")
