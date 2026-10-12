"""Train-time audio augmentation for a frozen Whisper LR family (features: layers 24+32 [mean,std]).
Augmented copies (speed 0.9x / 1.1x, +noise 20 dB) of a training clip are added ONLY when that clip is in the
fold's training part; validation always uses original clips. Track B (dup-group x3 seeds) + Track A.
If kept, writes OOF/test probabilities for blending (test predicted on original test clips).

Usage: python -I aug_eval.py <research_dir> <base_tag> <aug_tag>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R, base_tag, aug_tag = sys.argv[1:4]
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
m = pd.read_csv(f"{W}/meta.csv")
m["g"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
E = np.load(f"{W}/emb/{base_tag}.npy", mmap_mode="r")
X = np.hstack([np.asarray(E[:, 24]), np.asarray(E[:, 32])]).astype(np.float32)
A = np.load(f"{W}/emb/{aug_tag}.npy", mmap_mode="r")
XA = np.hstack([np.asarray(A[:, 24]), np.asarray(A[:, 32])]).astype(np.float32)
ap = pd.read_csv(f"{W}/aug_paths.csv")
lab = m.emotion.values


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))


def fit_with_aug(rows, kinds):
    sel = ap[ap.row.isin(rows) & ap.aug.isin(kinds)]
    Xt = np.vstack([X[rows], XA[sel.index.values]]) if len(sel) else X[rows]
    yt = np.concatenate([lab[rows], lab[sel.row.values]]) if len(sel) else lab[rows]
    return lr().fit(Xt, yt)


tr = np.where(m.split == "train")[0]
te = np.where(m.split == "test")[0]
yB, gB = lab[tr], m.g.values[tr]
key = m.set_index(["split", "Id"])
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")[["Id", "emotion", "fold"]]
A_idx = np.array([key.index.get_loc(("train", i)) for i in folds.Id])
yA, fA = folds.emotion.values, folds.fold.values
res = {}
for name, kinds in {"no aug": [], "+speed 0.9/1.1": ["sp09", "sp11"], "+noise 20dB": ["n20"], "+all 3": ["sp09", "sp11", "n20"]}.items():
    s = []
    for seed in range(3):
        p = np.empty(len(yB), dtype=object)
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, yB, gB):
            p[b] = fit_with_aug(tr[a], kinds).predict(X[tr[b]])
        s.append(f1_score(yB, p, average="macro"))
    o = np.zeros((len(yA), 4))
    for k in sorted(set(fA)):
        a, b = fA != k, fA == k
        o[b] = fit_with_aug(A_idx[a], kinds).predict_proba(X[A_idx[b]])
    res[name] = (o, fit_with_aug(A_idx, kinds).predict_proba(X[te]))
    print(f"{name:16s} Track B {np.mean(s):.4f} +- {np.std(s):.4f} | Track A {f1_score(yA, CL[o.argmax(1)], average='macro'):.4f}", flush=True)
np.savez(f"{W}/aug_probs_{base_tag}.npz", **{f"oof__{k}": v[0] for k, v in res.items()}, **{f"test__{k}": v[1] for k, v in res.items()})
