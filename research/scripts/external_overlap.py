"""Overlap check between an external audio dataset and competition audio (before using it as training data).
Each external clip is resampled to 16 kHz mono and compared by max normalized cross-correlation with every
competition clip whose duration is within 0.15 s. Reports matches with NCC >= 0.9.

Usage: python -I external_overlap.py <ext_root> <name> <data_root> <meta.csv> <out_csv>
"""
import glob
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

ext, name, R, meta, out = sys.argv[1:6]
m = pd.read_csv(meta)
comp = []
for r in m.itertuples():
    y = sf.read(f"{R}/{r.split}/audio_{r.Id:06d}.wav", dtype="float32")[0]
    comp.append(((y - y.mean()) / (y.std() + 1e-9), r.split, r.Id, r.emotion))
lens = np.array([len(c[0]) for c in comp])
files = sorted(glob.glob(f"{ext}/**/*.wav", recursive=True))
hits, best = [], []
for k, f in enumerate(files):
    y, sr = sf.read(f, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != 16000:
        y = resample_poly(y, 16000, sr).astype(np.float32)
    y = (y - y.mean()) / (y.std() + 1e-9)
    cand = np.where(np.abs(lens - len(y)) < 0.15 * 16000)[0]
    s_best = 0.0
    for i in cand:
        c = comp[i][0]
        s = float(np.max(np.abs(fftconvolve(y, c[::-1], mode="full"))) / min(len(y), len(c)))
        s_best = max(s_best, s)
        if s >= 0.9:
            hits.append(dict(ext_file=f.split(ext)[-1], split=comp[i][1], Id=comp[i][2], comp_label=comp[i][3], ncc=round(s, 4)))
    best.append(s_best)
    if k % 500 == 0:
        print(name, k, len(files), flush=True)
pd.DataFrame(hits).to_csv(out, index=False)
print(f"{name}: {len(files)} files | matches >= 0.9: {len(hits)} | max NCC per file quantiles "
      f"{np.round(np.quantile(best, [0.5, 0.9, 0.99, 1.0]), 3)}")
