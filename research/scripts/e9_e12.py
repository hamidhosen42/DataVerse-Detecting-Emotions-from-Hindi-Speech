"""E12 (external labelled Hindi data) and E9 (test pseudo-labels) for a frozen LR family, both measured
on Track A (public-pipeline folds) and Track B (dup-group folds) against the same family without them.

E12: training rows of each fold + external rows (vishlb, swastikatinkhede Hindi; no overlap verified) with
     sample weight w in {0.3, 1.0}.
E9 : inside each fold, the fold's own LR (trained on the fold's training rows only) labels NON-twin test rows;
     rows with max prob >= thr are added (weight 1) and the LR is refit; held-out rows are then scored.
     No information from held-out rows reaches the pseudo-labels (clean simulation).

Usage: python -I e9_e12.py <research_dir> <tag> <band_lo> <band_hi>
Needs work/emb/<tag>.npy and (for E12) work/emb/<tag>_ext.npy aligned with work/ext_paths.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

R, tag, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
m = pd.read_csv(f"{W}/meta.csv")
m["dup_group"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
X = np.load(f"{W}/emb/{tag}.npy").astype(np.float32)[:, lo:hi + 1].mean(1)
ext_ok = os.path.exists(f"{W}/emb/{tag}_ext.npy")
if ext_ok:
    Xe = np.load(f"{W}/emb/{tag}_ext.npy").astype(np.float32)[:, lo:hi + 1].mean(1)
    ye = pd.read_csv(f"{W}/ext_paths.csv").label.values
key = m.set_index(["split", "Id"])
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")[["Id", "emotion", "fold"]]
A_idx = np.array([key.index.get_loc(("train", i)) for i in folds.Id])
A_y, A_fold = folds.emotion.values, folds.fold.values
tr_all = np.where(m.split == "train")[0]
te_all = np.where(m.split == "test")[0]
twin_groups = set(m.dup_group.values[tr_all])
te_nontwin = np.array([i for i in te_all if m.dup_group.values[i] not in twin_groups])


def fit(Xt, yt, w=None):
    sc = StandardScaler().fit(Xt)
    clf = LogisticRegression(C=0.05, max_iter=4000, class_weight="balanced").fit(sc.transform(Xt), yt, sample_weight=w)
    return lambda Z: clf.predict_proba(sc.transform(Z))


def run(idx, yv, splits, mode, w=0.3, thr=0.9):
    pred = np.empty(len(idx), dtype=object)
    added = []
    for a, b in splits:
        Xt, yt, wt = X[idx[a]], yv[a], np.ones(len(a))
        if mode == "ext":
            Xt, yt, wt = np.vstack([Xt, Xe]), np.concatenate([yt, ye]), np.concatenate([wt, np.full(len(ye), w)])
        f = fit(Xt, yt, wt)
        if mode == "pl":
            p = f(X[te_nontwin])
            keep = p.max(1) >= thr
            added.append(int(keep.sum()))
            f = fit(np.vstack([Xt, X[te_nontwin][keep]]), np.concatenate([yt, CL[p[keep].argmax(1)]]))
        pred[b] = CL[f(X[idx[b]]).argmax(1)]
    return f1_score(yv, pred, average="macro"), added


trackA = list(zip(*[(np.where(A_fold != k)[0], np.where(A_fold == k)[0]) for k in sorted(set(A_fold))]))
trackA = list(zip(trackA[0], trackA[1]))
B_y, B_g = m.emotion.values[tr_all], m.dup_group.values[tr_all]
trackB = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X[tr_all], B_y, B_g))
rows = []
for name, mode, kw in [("base", "base", {}), ("E12 ext w=0.3", "ext", {"w": 0.3}), ("E12 ext w=1.0", "ext", {"w": 1.0}),
                       ("E9 PL thr=0.9", "pl", {"thr": 0.9}), ("E9 PL thr=0.8", "pl", {"thr": 0.8})]:
    if mode == "ext" and not ext_ok:
        continue
    fa, add = run(A_idx, A_y, trackA, mode, **kw)
    fb, _ = run(tr_all, B_y, trackB, mode, **kw)
    rows.append(dict(family=f"lr_{tag} L{lo}-{hi}", variant=name, trackA=round(fa, 4), trackB=round(fb, 4), pseudo_rows_per_fold=add or ""))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{W}/e9_e12_{tag}.csv", index=False)
