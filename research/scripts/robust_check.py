"""Robustness of the frozen Whisper families and their equal blend across 6 different fold splits
(StratifiedKFold seeds 0-2 = random/LB-proxy; StratifiedGroupKFold by dup group seeds 0-2 = novel clips).

Usage: python -I robust_check.py <research_dir>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R = sys.argv[1]
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
m = pd.read_csv(f"{W}/meta.csv")
m["g"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
tr = np.where(m.split == "train")[0]
y, g = m.emotion.values[tr], m.g.values[tr]
fam = {t: np.asarray(np.load(f"{W}/emb/{t}.npy", mmap_mode="r")[tr], np.float32) for t in ("wlv2cat", "wlv3cat", "wvaanicat", "whhlv2cat")}
fam["wmed"] = np.asarray(np.load(f"{W}/emb/wmed.npy", mmap_mode="r")[tr, 22:25], np.float32).mean(1)
fam["hilarge"] = np.asarray(np.load(f"{W}/emb/hilarge.npy", mmap_mode="r")[tr, 16:21], np.float32).mean(1)
splits = [("random", s, StratifiedKFold(5, shuffle=True, random_state=s).split(tr, y)) for s in range(3)] + \
         [("dupgroup", s, StratifiedGroupKFold(5, shuffle=True, random_state=s).split(tr, y, g)) for s in range(3)]
rows = []
for scheme, seed, sp in splits:
    sp = list(sp)
    P = {}
    for t, X in fam.items():
        o = np.zeros((len(y), 4))
        for a, b in sp:
            o[b] = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced")).fit(X[a], y[a]).predict_proba(X[b])
        P[t] = o
    cands = dict(P)
    cands["B5 (5 Whisper)"] = np.mean([P[t] for t in ("wlv2cat", "wlv3cat", "wvaanicat", "whhlv2cat", "wmed")], 0)
    cands["B5 + hilarge"] = np.mean([P[t] for t in ("wlv2cat", "wlv3cat", "wvaanicat", "whhlv2cat", "wmed", "hilarge")], 0)
    for name, p in cands.items():
        rows.append(dict(scheme=scheme, seed=seed, model=name, f1=f1_score(y, CL[p.argmax(1)], average="macro")))
    print(scheme, seed, {k: round(r["f1"], 4) for k, r in zip(cands, rows[-len(cands):])}, flush=True)
d = pd.DataFrame(rows)
s = d.groupby(["model", "scheme"]).f1.agg(["mean", "std", "min"]).round(4).unstack()
print(s.to_string())
d.to_csv(f"{W}/robust_check.csv", index=False)
