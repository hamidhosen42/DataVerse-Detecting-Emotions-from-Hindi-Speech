"""Extract frozen embeddings for all 1035 competition clips.

Usage: python -I embed.py <data_root> <meta.csv> <hf_model_id> <tag> <out_dir> [hf_cache]
tag 'sv'  -> WavLMForXVector speaker embedding (512-d)
other tag -> per-layer [mean, std] pooled hidden states, shape (N, L+1, 2D), float16
"""
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModel, WavLMForXVector

R, meta, model_id, tag, out = sys.argv[1:6]
cache = sys.argv[6] if len(sys.argv) > 6 else None
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
m = pd.read_csv(meta)
fe = AutoFeatureExtractor.from_pretrained(model_id, cache_dir=cache)
cls = WavLMForXVector if tag == "sv" else AutoModel
model = cls.from_pretrained(model_id, cache_dir=cache).to(dev).eval()
res = []
with torch.no_grad():
    for i, r in enumerate(m.itertuples()):
        y, sr = sf.read(f"{R}/{r.split}/audio_{r.Id:06d}.wav", dtype="float32")
        x = fe(y, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
        if tag == "sv":
            res.append(model(input_values=x).embeddings[0].float().cpu().numpy())
        else:
            h = torch.stack(model(input_values=x, output_hidden_states=True).hidden_states, 0)[:, 0]
            res.append(torch.cat([h.mean(1), h.std(1)], -1).float().cpu().numpy().astype(np.float16))
        if i % 250 == 0:
            print(tag, i, flush=True)
arr = np.stack(res)
np.save(f"{out}/{tag}.npy", arr)
print("done", tag, arr.shape, flush=True)
