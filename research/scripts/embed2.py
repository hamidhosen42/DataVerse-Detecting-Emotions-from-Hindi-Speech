"""Generalised frozen-embedding extractor (per-layer [mean, std] pooling, float16).

Handles: wav2vec2 / WavLM / HuBERT / XLS-R style models (input_values), w2v-BERT 2.0 (input_features),
and Whisper encoders (pools only the frames that cover real audio, never the 30 s zero padding).

Usage: python -I embed2.py <model_id> <tag> <out_dir> <cache_dir> <list_csv>
list_csv: a CSV with a 'path' column (competition clips or external clips, any sample rate).
Output: <out_dir>/<tag>.npy with shape (N, L+1, 2*D)
"""
import math
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2FeatureExtractor

model_id, tag, out, cache, lst = sys.argv[1:6]
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
paths = pd.read_csv(lst).path.tolist()
is_whisper = "whisper" in model_id.lower()
try:
    fe = AutoFeatureExtractor.from_pretrained(model_id, cache_dir=cache)
except Exception as e:  # pretrained-only wav2vec2 repos sometimes ship no preprocessor config
    print("feature extractor fallback:", type(e).__name__, flush=True)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
model = AutoModel.from_pretrained(model_id, cache_dir=cache)
if is_whisper:
    model = model.get_encoder()
model = model.float().to(dev).eval()  # some checkpoints (e.g. Whisper) load as fp16


def load(p):
    y, sr = sf.read(p, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != 16000:
        y = resample_poly(y, 16000, sr).astype(np.float32)
    return y


res = []
with torch.no_grad():
    for i, p in enumerate(paths):
        y = load(p)
        inputs = fe(y, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items() if k in ("input_values", "input_features", "attention_mask")}
        if is_whisper:
            inputs.pop("attention_mask", None)
            hs = model(**inputs, output_hidden_states=True).hidden_states
            n = max(1, min(hs[0].shape[1], math.ceil(len(y) / 16000 * 50)))  # 50 encoder frames per second
            h = torch.stack(hs, 0)[:, 0, :n]
        else:
            h = torch.stack(model(**inputs, output_hidden_states=True).hidden_states, 0)[:, 0]
        res.append(torch.cat([h.mean(1), h.std(1)], -1).float().cpu().numpy().astype(np.float16))
        if i % 250 == 0:
            print(tag, i, len(paths), flush=True)
arr = np.stack(res)
np.save(f"{out}/{tag}.npy", arr)
print("done", tag, arr.shape, flush=True)
