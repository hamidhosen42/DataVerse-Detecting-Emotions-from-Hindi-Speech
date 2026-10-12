"""Validation-scheme study with frozen SSL embeddings + logistic regression.

Scheme A  'random'  : StratifiedKFold(5), dup twins may straddle folds (mimics the organizers' random split).
Scheme B  'dupgroup': StratifiedGroupKFold(5) grouped by near-duplicate group (novel-clip generalization).
Each scheme is repeated over 3 seeds; OOF macro-F1 is reported as mean +- std over repeats, and for scheme A
also split into held-out rows that do / do not have a twin in the training folds.

Usage: python -I cv_frozen.py <work_dir> <tag> [layers e.g. "6,8,10" or "all"] [C]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W, tag = sys.argv[1], sys.argv[2]
layer_arg = sys.argv[3] if len(sys.argv) > 3 else "all"
C = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
m = pd.read_csv(f"{W}/meta.csv")
m["dup_group"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
X_all = np.load(f"{W}/emb/{tag}.npy").astype(np.float32)
tr = np.where(m.split == "train")[0]
y = m.loc[tr, "emotion"].values
groups = m.loc[tr, "dup_group"].values


def oof(X, splitter, seed):
    pred = np.empty(len(tr), dtype=object)
    has_twin = np.zeros(len(tr), bool)
    for a, b in splitter.split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(X[a], y[a])
        pred[b] = clf.predict(X[b])
        has_twin[b] = np.isin(groups[b], groups[a])
    return pred, has_twin


def evaluate(X, name):
    out = {}
    for scheme in ("random", "dupgroup"):
        f, f_tw, f_nt, tw_share = [], [], [], []
        for seed in range(3):
            sp = (StratifiedKFold(5, shuffle=True, random_state=seed) if scheme == "random"
                  else StratifiedGroupKFold(5, shuffle=True, random_state=seed))
            p, tw = oof(X, sp, seed)
            f.append(f1_score(y, p, average="macro"))
            if scheme == "random":
                f_tw.append(f1_score(y[tw], p[tw], average="macro"))
                f_nt.append(f1_score(y[~tw], p[~tw], average="macro"))
                tw_share.append(tw.mean())
        out[scheme] = f"{np.mean(f):.4f}+-{np.std(f):.4f}"
        if scheme == "random":
            out["random_twin"] = f"{np.mean(f_tw):.4f}"
            out["random_notwin"] = f"{np.mean(f_nt):.4f}"
            out["twin_share"] = f"{np.mean(tw_share):.3f}"
    print(name, out, flush=True)
    return out


if X_all.ndim == 2:
    evaluate(X_all[tr], f"{tag} (C={C})")
else:
    L = X_all.shape[1]
    layers = range(L) if layer_arg == "all" else [int(v) for v in layer_arg.split(",")]
    for l in layers:
        evaluate(X_all[tr, l], f"{tag} layer {l:02d} (C={C})")
