"""Head/feature variants for a frozen Whisper family, scored on Track B only (dup-group 5-fold x 3 seeds),
so the choice does not touch Track A (used for blend evaluation).

Usage: python -I whisper_head.py <research_dir> <tag>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R, tag = sys.argv[1], sys.argv[2]
W = f"{R}/work"
m = pd.read_csv(f"{W}/meta.csv")
m["dup_group"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
E = np.load(f"{W}/emb/{tag}.npy", mmap_mode="r")
tr = np.where(m.split == "train")[0]
y, g = m.emotion.values[tr], m.dup_group.values[tr]
L = E.shape[1] - 1
D = E.shape[2] // 2


def feat(spec):
    if spec == "last":
        return np.asarray(E[tr, L], np.float32)
    if spec == "last_mean_only":
        return np.asarray(E[tr, L, :D], np.float32)
    if spec.startswith("band"):
        lo = int(spec[4:])
        return np.asarray(E[tr, lo:L + 1], np.float32).mean(1)
    if spec.startswith("cat"):
        a, b = [int(v) for v in spec[3:].split("+")]
        return np.hstack([np.asarray(E[tr, a], np.float32), np.asarray(E[tr, b], np.float32)])
    raise ValueError(spec)


def trackb(X, C):
    s = []
    for seed in range(3):
        p = np.empty(len(y), dtype=object)
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            p[b] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(X[a], y[a]).predict(X[b])
        s.append(f1_score(y, p, average="macro"))
    return np.mean(s), np.std(s)


rows = []
for spec in ("last", "last_mean_only", f"band{L - 2}", f"band{L - 4}", f"cat{L - 8}+{L}"):
    X = feat(spec)
    for C in (0.01, 0.05, 0.2, 1.0):
        mu, sd = trackb(X, C)
        rows.append(dict(tag=tag, features=spec, C=C, trackB=round(mu, 4), sd=round(sd, 4)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{W}/whisper_head_{tag}.csv", index=False)
