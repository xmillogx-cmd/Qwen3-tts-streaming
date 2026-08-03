import torch, sys, time
sys.stdout.reconfigure(line_buffering=True)
from qwen_tts import Qwen3TTSModel
from faster_qwen3_tts import FasterQwen3TTS

model_path = r'G:\Foundation\models\Qwen3-TTS'

m1 = Qwen3TTSModel.from_pretrained(model_path, device_map='cuda:0', dtype=torch.bfloat16)
m1._init_cuda_graphs()
m1._warmup_cuda_graphs(50)

m2 = FasterQwen3TTS.from_pretrained(model_path, device='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')

text = 'Привет'
print('=== NATIVE ===', flush=True)
for i, (chunk1, sr1, t1) in enumerate(m1.generate_custom_voice_streaming(text=text, speaker='Sohee', language='Russian', chunk_size=8, max_new_tokens=20, backend='auto')):
    print(f'  chunk {i}: decode={t1["decode_ms"]:.0f}ms prefill={t1.get("prefill_ms",0):.0f}ms', flush=True)

print('=== FASTER ===', flush=True)
for i, (chunk2, sr2, t2) in enumerate(m2.generate_custom_voice_streaming(text=text, speaker='Sohee', language='Russian', chunk_size=8, max_new_tokens=20)):
    print(f'  chunk {i}: decode={t2["decode_ms"]:.0f}ms prefill={t2.get("prefill_ms",0):.0f}ms', flush=True)
