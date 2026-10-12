"""Track B (novel recordings) for the frozen-only blend B5, from cached features (no training of fine-tuned members).

StratifiedGroupKFold(5) on near-duplicate groups, 3 seeds; per family the notebook's LR (StandardScaler + LogisticRegression
C=0.05 balanced); blend = equal mean of the 5 families' probabilities. The fine-tuned members of B8/B9 candidates cannot be
scored on these folds without re-training (5.5-7.5 h each on the local machine), so only frozen families and B5 are reported.

Usage: python -I b9_trackb_blend.py <repo_root>      -> prints and writes research/work/b9_trackb_blend.txt
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

P = sys.argv[1]
W = f"{P}/research/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
meta = pd.read_csv(f"{W}/meta.csv")
ids = {int(i) for v in json.load(open(f"{W}/canonical_folds.json")).values() for i in v}
TR = np.array(sorted(np.where((meta.split == "train") & meta.Id.isin(ids))[0]))
assert len(TR) == 824
y = pd.Categorical(meta.emotion.values[TR], categories=CL).codes
grp = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values[TR]
REC = {"wlv2": "cat", "wlv3": "cat", "wvaani": "cat", "whhlv2": "cat", "wmed": "band"}
X = {}
for t, r in REC.items():
    E = np.load(f"{W}/emb/{t}.npy", mmap_mode="r")
    X[t] = (np.hstack([np.asarray(E[TR, 24]), np.asarray(E[TR, 32])]) if r == "cat" else np.asarray(E[TR, 22:25]).astype(np.float32).mean(1)).astype(np.float32)   # float32 before the mean, as in the notebook
lines = []
res = {t: [] for t in REC} | {"B5 blend": []}
for seed in range(3):
    P_ = {t: np.zeros((len(TR), 4)) for t in REC}
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(TR, y, grp):
        assert not set(grp[a]) & set(grp[b])
        for t in REC:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000, class_weight="balanced")).fit(X[t][a], y[a])
            P_[t][b] = m.predict_proba(X[t][b])
    for t in REC:
        res[t].append(f1_score(y, P_[t].argmax(1), average="macro"))
    res["B5 blend"].append(f1_score(y, np.mean([P_[t] for t in REC], 0).argmax(1), average="macro"))
for k, v in res.items():
    s = f"{k:10s} Track B macro-F1 {np.mean(v):.4f} +- {np.std(v):.4f}  (seeds {[round(x, 4) for x in v]})"
    print(s)
    lines.append(s)
open(f"{W}/b9_trackb_blend.txt", "w").write("\n".join(lines) + "\n")
