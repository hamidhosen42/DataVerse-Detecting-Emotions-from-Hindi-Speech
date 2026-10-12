"""(1) Paired bootstrap: is blend(ens, frozen LR) really better than ens on OOF?  Resampling unit = dup group.
(2) Leaderboard noise: simulate public subsets of size N drawn like the hidden test (37% twin rows)
    and report SD of a single model's score and SD of the paired difference between two models.

Usage: python -I noise_and_significance.py <work_dir>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = sys.argv[1]
KD = f"{W}/kunal_train_output/dataverse_models"
CL = np.array(["angry", "happy", "neutral", "sad"])
m = pd.read_csv(f"{W}/meta.csv")
m["dup_group"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
key = m.set_index(["split", "Id"])
folds = pd.read_csv(f"{KD}/train_with_folds.csv")[["Id", "emotion", "fold"]]
P = {}
for name, sub in (("base+", "wavlm-base-plus_bcd60f08"), ("large", "wavlm-large_97ba7aec")):
    o = pd.read_csv(f"{KD}/{sub}/seed_42/oof_predictions.csv").set_index("Id").loc[folds.Id]
    P[name] = o[[f"prob_{c}" for c in CL]].values
ens = (P["base+"] + P["large"]) / 2
y = folds.emotion.values
fold = folds.fold.values
grp = key.loc[[("train", i) for i in folds.Id], "dup_group"].values
tw = np.array([np.isin(grp[i], grp[fold != fold[i]]) for i in range(len(y))])
X = np.load(f"{W}/emb/wbp.npy").astype(np.float32)[:, 7][[key.index.get_loc(("train", i)) for i in folds.Id]]
lr = np.zeros((len(y), 4))
for f in sorted(set(fold)):
    a, b = fold != f, fold == f
    lr[b] = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")).fit(X[a], y[a]).predict_proba(X[b])
pa, pb = CL[ens.argmax(1)], CL[(0.5 * ens + 0.5 * lr).argmax(1)]

rng = np.random.default_rng(0)
ug = np.unique(grp)
members = {g: np.where(grp == g)[0] for g in ug}
diffs = []
for _ in range(3000):
    idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
    diffs.append(f1_score(y[idx], pb[idx], average="macro") - f1_score(y[idx], pa[idx], average="macro"))
diffs = np.array(diffs)
print(f"(1) blend - ens OOF macroF1: point {f1_score(y, pb, average='macro') - f1_score(y, pa, average='macro'):+.4f} | "
      f"group-bootstrap mean {diffs.mean():+.4f}, 95% CI [{np.quantile(diffs, .025):+.4f}, {np.quantile(diffs, .975):+.4f}], P(>0)={np.mean(diffs > 0):.3f}")
print(f"    rows where the two disagree: {(pa != pb).sum()} | blend fixes {((pb == y) & (pa != y)).sum()}, breaks {((pa == y) & (pb != y)).sum()}")

print("\n(2) simulated leaderboard subsets (37% twin rows, like the hidden test):")
twin_idx, non_idx = np.where(tw)[0], np.where(~tw)[0]
for N in (52, 63, 84, 105, 125, 157):
    s_a, s_d = [], []
    for _ in range(3000):
        k = rng.binomial(N, 0.37)
        idx = np.concatenate([rng.choice(twin_idx, k), rng.choice(non_idx, N - k)])
        fa = f1_score(y[idx], pa[idx], average="macro")
        s_a.append(fa)
        s_d.append(f1_score(y[idx], pb[idx], average="macro") - fa)
    print(f"  N={N:3d}: single-model score SD={np.std(s_a):.4f} (mean {np.mean(s_a):.4f}) | paired difference SD={np.std(s_d):.4f}, "
          f"P(public shows blend worse or equal)={np.mean(np.array(s_d) <= 0):.2f}")
