"""Profile sample_logits in isolation."""
import torch, time, sys
sys.path.insert(0, r'G:\qwen-tts')
from fast_tts_v11 import sample_logits, apply_repetition_penalty

device = 'cuda:0'
vocab_size = 2151  # Qwen3-TTS vocab

# Pre-create suppress_mask
suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
suppress_start = max(0, vocab_size - 1024)
for i in range(suppress_start, vocab_size):
    if i != 2150:  # eos_id
        suppress_mask[i] = True

# Warmup
dummy_logits = torch.randn(vocab_size, device=device)
for _ in range(10):
    sample_logits(dummy_logits, temperature=0.9, top_k=50, top_p=1.0, do_sample=True, suppress_mask=suppress_mask)
torch.cuda.synchronize()

# Profile individual components
print("=== Component breakdown ===")

# 1. Just suppress + argmax (no sampling)
t0 = time.perf_counter()
for _ in range(100):
    out = sample_logits(dummy_logits.clone(), temperature=0.9, top_k=50, top_p=1.0, do_sample=False, suppress_mask=suppress_mask)
torch.cuda.synchronize()
print(f"argmax only: {(time.perf_counter()-t0)/100*1000:.2f}ms")

# 2. Sampling with top_k only (no top_p)
t0 = time.perf_counter()
for _ in range(100):
    out = sample_logits(dummy_logits.clone(), temperature=0.9, top_k=50, top_p=1.0, do_sample=True, suppress_mask=suppress_mask)
torch.cuda.synchronize()
print(f"top_k+top_p sampling: {(time.perf_counter()-t0)/100*1000:.2f}ms")

# 3. No suppress mask
t0 = time.perf_counter()
for _ in range(100):
    out = sample_logits(dummy_logits.clone(), temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
torch.cuda.synchronize()
print(f"no suppress mask: {(time.perf_counter()-t0)/100*1000:.2f}ms")

# 4. Just temperature + multinomial (no top_k/top_p)
def simple_sample(logits, temperature=0.9):
    import torch.nn.functional as F
    return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).squeeze(-1)

t0 = time.perf_counter()
for _ in range(100):
    out = simple_sample(dummy_logits.clone())
torch.cuda.synchronize()
print(f"simple sample (no top_k/p): {(time.perf_counter()-t0)/100*1000:.2f}ms")

# 5. Repetition penalty overhead
history = torch.randint(0, vocab_size, (20,), device=device)
logits_2d = dummy_logits.clone().unsqueeze(0)
t0 = time.perf_counter()
for _ in range(100):
    out = apply_repetition_penalty(logits_2d.clone(), history, 1.05)
torch.cuda.synchronize()
print(f"repetition penalty (20 tokens): {(time.perf_counter()-t0)/100*1000:.2f}ms")

# 6. torch.tensor creation overhead (what profile_decode does)
t0 = time.perf_counter()
src = torch.randint(0, vocab_size, (1,), device=device)
for _ in range(100):
    hist = torch.tensor([src[0].item()], device=device)  # CPU round-trip!
torch.cuda.synchronize()
print(f"torch.tensor from item(): {(time.perf_counter()-t0)/100*1000:.2f}ms")

# 7. Better: direct view
t0 = time.perf_counter()
for _ in range(100):
    hist = src.view(-1)  # no copy, just view
torch.cuda.synchronize()
print(f"view (no copy): {(time.perf_counter()-t0)/100*1000:.2f}ms")

# 8. Profile the full pipeline as in profile_decode
print("\n=== Full step (as in profile_decode) ===")
all_cb = torch.randint(0, vocab_size, (16,), device=device)
logits_2d = dummy_logits.clone().unsqueeze(0)
for step_idx in range(30):
    t0 = time.perf_counter()
    if step_idx > 0:
        history = torch.tensor([all_cb[0]], device=device)
        logits_2d = apply_repetition_penalty(logits_2d, history, 1.05)
    token = sample_logits(logits_2d.squeeze(0), temperature=0.9, top_k=50, top_p=1.0,
                          do_sample=True, suppress_mask=suppress_mask)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    if step_idx < 5 or step_idx % 5 == 0:
        print(f"Step {step_idx}: {ms:.1f}ms | token={token.item()}")
