"""Isolated test of CUDA graph generation — no audio playback."""
import torch, time, sys
sys.path.insert(0, r'G:\qwen-tts')

from qwen_tts import Qwen3TTSModel
from fast_tts_v11 import (
    PredictorGraph, TalkerGraph, _build_talker_inputs, cuda_graph_generate_streaming,
)

model_path = r'G:\Foundation\models\Qwen3-TTS'
device = 'cuda:0'
text = "Привет мир."
language = 'Russian'
speaker = 'Sohee'

print(f"Loading model from {model_path}...", flush=True)
tts_model = Qwen3TTSModel.from_pretrained(
    model_path, device_map=device, dtype=torch.bfloat16,
)
inner = tts_model.model
talker = inner.talker
talker_config = inner.config.talker_config
predictor = talker.code_predictor
pred_config = predictor.config if hasattr(predictor, 'config') else predictor.model.config

print(f"Building PredictorGraph...", flush=True)
pg = PredictorGraph(
    predictor, pred_config, talker_config.hidden_size,
    device=device, dtype=torch.bfloat16,
)
pg.capture(num_warmup=3)
print("Predictor captured!", flush=True)

print(f"Building TalkerGraph...", flush=True)
tg = TalkerGraph(talker.model, talker_config, device=device, max_seq_len=2048)
tg.capture(prefill_len=40, num_warmup=3)
print("Talker captured!", flush=True)

# Build inputs
print(f"Building talker inputs...", flush=True)
tie, tam, tth, tts_pad = _build_talker_inputs(tts_model, text, language, speaker)
print(f"  tie: {tie.shape}, tam: {tam.shape}, tth: {tth.shape}, tts_pad: {tts_pad.shape}", flush=True)

# Run streaming generation
print(f"Starting cuda_graph_generate_streaming...", flush=True)
gen = cuda_graph_generate_streaming(
    talker, tie, tam, tth, tts_pad, talker_config,
    pg, tg,
    max_new_tokens=200, min_new_tokens=2,
    temperature=0.9, top_k=50, top_p=1.0,
    do_sample=True, repetition_penalty=1.05, chunk_size=8,
)

total_steps = 0
t0 = time.time()
try:
    for codec_chunk, timing in gen:
        print(f"  Chunk {timing['chunk_index']}: {codec_chunk.shape}, steps={timing['chunk_steps']}", flush=True)
        total_steps += timing['chunk_steps']
except Exception as e:
    import traceback
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()
finally:
    gen.close()

elapsed = time.time() - t0
print(f"\nDone! Total steps: {total_steps}, Time: {elapsed:.2f}s, Rate: {total_steps/elapsed:.1f} step/s", flush=True)
