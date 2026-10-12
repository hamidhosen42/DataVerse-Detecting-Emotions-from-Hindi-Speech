"""Hidden-duplicate probe: nearest-neighbour structure in embedding space, excluding waveform twins.
If label agreement of the nearest non-twin neighbour jumps for very similar pairs, there are
transformed copies (speed/pitch-shifted, re-cut) that waveform correlation cannot see.

Usage: python -I embedding_neighbors.py <work_dir> <tag> <layer>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

W, tag, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = pd.read_csv(f"{W}/meta.csv")
m["dup_group"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
X = np.load(f"{W}/emb/{tag}.npy").astype(np.float32)[:, layer, : np.load(f"{W}/emb/{tag}.npy").shape[2] // 2]
X = normalize(X - X.mean(0))
S = X @ X.T
g = m.dup_group.values
S[g[:, None] == g[None, :]] = -2  # drop self + waveform twins
tr = np.where(m.split == "train")[0]
St = S[np.ix_(tr, tr)]
nn = St.argmax(1)
sim = St.max(1)
agree = m.emotion.values[tr][nn] == m.emotion.values[tr]
d = pd.DataFrame({"sim": sim, "agree": agree, "dur_a": m.dur.values[tr], "dur_b": m.dur.values[tr][nn]})
d["bucket"] = pd.qcut(d.sim, 10)
print(f"{tag} layer {layer}: train->train nearest NON-twin neighbour, label agreement by similarity decile")
print(d.groupby("bucket", observed=True).agg(n=("agree", "size"), agree=("agree", "mean"), dur_ratio=("dur_b", "median")).round(3).to_string())
print("top-1% similarity pairs agreement:", d[d.sim >= d.sim.quantile(0.99)].agree.mean().round(3))
te = np.where(m.split == "test")[0]
Ste = S[np.ix_(te, tr)]
print("test->train nearest non-twin similarity quantiles:", np.round(np.quantile(Ste.max(1), [0.1, 0.5, 0.9, 0.99]), 3),
      "| train->train:", np.round(np.quantile(sim, [0.1, 0.5, 0.9, 0.99]), 3))
