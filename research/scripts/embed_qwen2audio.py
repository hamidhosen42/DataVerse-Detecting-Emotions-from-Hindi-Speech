"""Frozen Qwen2-Audio audio-encoder features (Whisper-large-v3-initialised encoder further trained by Qwen on broad
audio understanding). Only shard 1 of Qwen/Qwen2-Audio-7B (which contains every audio_tower.* tensor) is needed.

Per-layer [mean, std] over real-audio frames (50 Hz before the encoder's final pooling), float16 -> <out>/<tag>.npy

Usage: python -I embed_qwen2audio.py <qwen_dir> <list_csv> <out_dir> <tag>
"""
import math
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from safetensors import safe_open
from transformers import Qwen2AudioConfig, WhisperFeatureExtractor
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder

qdir, lst, out, tag = sys.argv[1:5]
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
cfg = Qwen2AudioConfig.from_pretrained(qdir)
enc = Qwen2AudioEncoder(cfg.audio_config)
sd = {}
with safe_open(f"{qdir}/model-00001-of-00005.safetensors", "pt") as f:
    for k in f.keys():
        if k.startswith("audio_tower."):
            sd[k[len("audio_tower."):]] = f.get_tensor(k)
missing, unexpected = enc.load_state_dict(sd, strict=False)
print("loaded", len(sd), "tensors | missing", missing, "| unexpected", unexpected[:5], flush=True)
enc = enc.float().to(dev).eval()
fe = WhisperFeatureExtractor.from_pretrained(qdir)
res = []
with torch.no_grad():
    for i, p in enumerate(pd.read_csv(lst).path):
        y, sr = sf.read(p, dtype="float32", always_2d=True)
        y = y.mean(1)
        x = fe(y, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
        hs = enc(input_features=x, output_hidden_states=True).hidden_states
        n = max(2, min(hs[0].shape[1], math.ceil(len(y) / 16000 * 50)))
        h = torch.stack(hs[:-1], 0)[:, 0, :n]                  # embeddings + layers at 50 Hz
        last = hs[-1][0, : max(1, n // 2)]                      # final state is avg-pooled 2x (25 Hz) + layer-norm
        pooled = torch.cat([h.mean(1), h.std(1)], -1)
        pooled = torch.cat([pooled, torch.cat([last.mean(0), last.std(0)], -1)[None]], 0)
        res.append(pooled.float().cpu().numpy().astype(np.float16))
        if i % 250 == 0:
            print(tag, i, "| layers", len(hs), "frames", hs[0].shape[1], flush=True)
arr = np.stack(res)
np.save(f"{out}/{tag}.npy", arr)
print("done", tag, arr.shape, flush=True)
