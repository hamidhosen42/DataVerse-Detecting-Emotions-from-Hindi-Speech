"""Hindi ASR transcripts for every competition clip with a Whisper checkpoint (for a text-based emotion branch).

Usage: python -I transcribe.py <hf_model_id> <comp_paths.csv> <meta.csv> <out_csv> <hf_cache>
"""
import sys

import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

mid, lst, meta, out, cache = sys.argv[1:6]
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
proc = WhisperProcessor.from_pretrained(mid, cache_dir=cache)
model = WhisperForConditionalGeneration.from_pretrained(mid, cache_dir=cache).float().to(dev).eval()
if not hasattr(model.generation_config, "lang_to_id"):  # older fine-tunes ship an outdated generation config
    from transformers import GenerationConfig

    base = sys.argv[6] if len(sys.argv) > 6 else "openai/whisper-large-v2"
    model.generation_config = GenerationConfig.from_pretrained(base, cache_dir=cache)
    print("generation config taken from", base, flush=True)
paths = pd.read_csv(lst).path.tolist()
m = pd.read_csv(meta)
texts = []
with torch.no_grad():
    for i, p in enumerate(paths):
        y, sr = sf.read(p, dtype="float32")
        feats = proc(y, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
        ids = model.generate(feats, language="hi", task="transcribe", max_new_tokens=128)
        texts.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
        if i % 100 == 0:
            print(i, len(paths), texts[-1][:60], flush=True)
m["text"] = texts
m[["split", "Id", "emotion", "text"]].to_csv(out, index=False)
print("done", len(texts), flush=True)
