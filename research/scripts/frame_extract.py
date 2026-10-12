"""Frame-level Whisper encoder features (real-audio frames only) for attention-pooling heads.

Saves layers L1 and L2 concatenated per frame, downsampled 2x in time (mean of adjacent frames -> 25 Hz), float16:
  <out>/<tag>_frames.npy   (total_frames, 2*D)
  <out>/<tag>_offsets.npy  (N+1,) start offsets per clip
Usage: python -I frame_extract.py <hf_model_id> <tag> <out_dir> <hf_cache> <list_csv> [L1 L2]
"""
import math
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModel

mid, tag, out, cache, lst = sys.argv[1:6]
L1, L2 = (int(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (24, 32)
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
fe = AutoFeatureExtractor.from_pretrained(mid, cache_dir=cache)
enc = AutoModel.from_pretrained(mid, cache_dir=cache).get_encoder().float().to(dev).eval()
paths = pd.read_csv(lst).path.tolist()
chunks, offs = [], [0]
with torch.no_grad():
    for i, p in enumerate(paths):
        y, sr = sf.read(p, dtype="float32", always_2d=True)
        y = y.mean(1)
        x = fe(y, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
        hs = enc(input_features=x, output_hidden_states=True).hidden_states
        n = max(2, min(hs[0].shape[1], math.ceil(len(y) / 16000 * 50)))
        h = torch.cat([hs[L1][0, :n], hs[L2][0, :n]], -1)            # (n, 2D)
        n2 = n // 2 * 2
        h = h[:n2].reshape(n2 // 2, 2, -1).mean(1)                      # 25 Hz
        chunks.append(h.float().cpu().numpy().astype(np.float16))
        offs.append(offs[-1] + len(chunks[-1]))
        if i % 250 == 0:
            print(tag, i, len(paths), flush=True)
np.save(f"{out}/{tag}_frames.npy", np.concatenate(chunks))
np.save(f"{out}/{tag}_offsets.npy", np.array(offs, dtype=np.int64))
print("done", tag, offs[-1], "frames", flush=True)
