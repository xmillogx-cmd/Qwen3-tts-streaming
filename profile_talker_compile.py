"""Test torch.compile on talker decode step vs CUDA graph."""
import torch, time, sys
sys.path.insert(0, r'G:\qwen-tts')

from qwen_tts import Qwen3TTSModel
from fast_tts_v12 import TalkerGraph, _build_talker_inputs
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask

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

# CUDA graph baseline
tg = TalkerGraph(talker.model, talker_config, device=device, max_seq_len=2048)
tg.capture(prefill_len=40, num_warmup=3)

tie, tam, tth, tts_pad = _build_talker_inputs(tts_model, text, language, speaker)

# Prefill to get hidden states for talker input
out = talker.forward(
    inputs_embeds=tie, attention_mask=tam, use_cache=True, output_hidden_states=True, return_dict=True,
    trailing_text_hidden=tth, tts_pad_embed=tts_pad,
    generation_step=None, past_hidden=None, past_key_values=None,
)

# Copy prefill KV into static cache
prefill_len = tg.prefill_kv(out.past_key_values)
tg.set_generation_state(tam, getattr(talker, "rope_deltas", None))

dummy_input = torch.zeros(1, 1, talker_config.hidden_size, dtype=torch.bfloat16, device=device)
sync = torch.cuda.synchronize

# CUDA graph baseline
with torch.inference_mode():
    for _ in range(5):
        tg.run(dummy_input, position=50)
sync()

t0 = time.perf_counter()
for _ in range(30):
    tg.run(dummy_input, position=50)
sync()
cuda_graph_ms = (time.perf_counter() - t0) / 30 * 1000
print(f"CUDA graph talker: {cuda_graph_ms:.1f}ms/step")

# --- Eager mode with torch.compile ---
print("\nBuilding compiled talker...", flush=True)

talker_model = talker.model
talker_model_compiled = torch.compile(talker_model, dynamic=False)

# Build static cache and attention mask
num_kv_heads = getattr(talker_config, 'num_key_value_heads', talker_config.num_attention_heads)
head_dim = getattr(talker_config, 'head_dim', talker_config.hidden_size // talker_config.num_attention_heads)

compiled_cache = StaticCache(config=talker_config, max_cache_len=2048)
dummy_k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=torch.bfloat16, device=device)
for layer in compiled_cache.layers:
    if not layer.is_initialized:
        layer.lazy_initialization(dummy_k)

# Copy prefill KV into compiled cache
for li in range(talker_config.num_hidden_layers):
    k, v = out.past_key_values[li]
    seq_len = k.shape[2]
    cache_pos = torch.arange(seq_len, device=device)
    compiled_cache.update(k, v, li, {"cache_position": cache_pos})

# Build attention mask for position 50
mask_fn = create_causal_mask if talker_model.config.sliding_window is None else create_sliding_window_causal_mask
dummy_1 = torch.zeros(1, 1, talker_config.hidden_size, dtype=torch.bfloat16, device=device)
pos_tensor = torch.tensor([50], device=device)

compiled_attn = mask_fn(config=talker_config, input_embeds=dummy_1, attention_mask=tam,
                        cache_position=pos_tensor, past_key_values=compiled_cache)

rope_deltas = getattr(talker, "rope_deltas", None)
if rope_deltas is None:
    rope_deltas = torch.zeros(1, 1, dtype=torch.float32, device=device)

def eager_talker_step(input_embeds, position):
    """Eager talker decode step with compiled model."""
    cache_position = torch.tensor([position], device=device)
    delta = rope_deltas + cache_position.to(dtype=torch.float32)
    position_ids = delta.unsqueeze(0).expand(3, -1, -1)

    out = talker_model_compiled(
        inputs_embeds=input_embeds, attention_mask=compiled_attn,
        past_key_values=compiled_cache, cache_position=cache_position,
        position_ids=position_ids, use_cache=True,
    )
    return out.last_hidden_state

# Warmup + compile
print("First run (compiling...)...", flush=True)
with torch.inference_mode():
    eager_talker_step(dummy_input.clone(), 50)
sync()
print("Compiled!", flush=True)

# Benchmark compiled
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(30):
        eager_talker_step(dummy_input.clone(), 50)
sync()
compiled_ms = (time.perf_counter() - t0) / 30 * 1000
print(f"Compiled talker: {compiled_ms:.1f}ms/step")

# Try with reduce_inductor_min_size for better fusion
print("\nTrying compile with max-autotune-no-cudagraphs...", flush=True)

talker_model_compiled2 = torch.compile(talker.model, mode='max-autotune-no-cudagraphs')

def eager_talker_step2(input_embeds, position):
    cache_position = torch.tensor([position], device=device)
    delta = rope_deltas + cache_position.to(dtype=torch.float32)
    position_ids = delta.unsqueeze(0).expand(3, -1, -1)

    out = talker_model_compiled2(
        inputs_embeds=input_embeds, attention_mask=compiled_attn,
        past_key_values=compiled_cache, cache_position=cache_position,
        position_ids=position_ids, use_cache=True,
    )
    return out.last_hidden_state

print("First run (compiling with max-autotune...)...", flush=True)
with torch.inference_mode():
    eager_talker_step2(dummy_input.clone(), 51)
sync()
print "Compiled with max-autotune!", flush=True)

t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(30):
        eager_talker_step2(dummy_input.clone(), 51)
sync()
compiled2_ms = (time.perf_counter() - t0) / 30 * 1000
print(f"Compiled max-autotune: {compiled2_ms:.1f}ms/step")

# Summary
print(f"\n{'='*50}")
print(f"CUDA graph (baseline):     {cuda_graph_ms:.1f}ms/step")
print(f"Compiled default:          {compiled_ms:.1f}ms/step ({cuda_graph_ms/compiled_ms:.2f}x)")
print(f"Compiled max-autotune:     {compiled2_ms:.1f}ms/step ({cuda_graph_ms/compiled2_ms:.2f}x)")
