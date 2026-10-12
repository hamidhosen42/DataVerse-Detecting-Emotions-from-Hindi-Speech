"""Analyse the strongest public pipeline (kunalinfinite 01/02, WavLM Base+ & Large fine-tunes) with the
near-duplicate structure, and test two cheap upgrades on ITS OWN folds:
  (1) twin-label override on test rows whose waveform twin is in train,
  (2) blending with a frozen-feature logistic-regression model (WavLM Base+ layer 7).

Usage: python -I analyze_public_pipeline.py <work_dir>
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
CL = ["angry", "happy", "neutral", "sad"]
m = pd.read_csv(f"{W}/meta.csv")
m["dup_group"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
key = m.set_index(["split", "Id"])

folds = pd.read_csv(f"{KD}/train_with_folds.csv")[["Id", "emotion", "fold"]]
oof = {}
for name, sub in (("base+", "wavlm-base-plus_bcd60f08"), ("large", "wavlm-large_97ba7aec")):
    o = pd.read_csv(f"{KD}/{sub}/seed_42/oof_predictions.csv").set_index("Id").loc[folds.Id]
    oof[name] = o[[f"prob_{c}" for c in CL]].values
oof["ens"] = (oof["base+"] + oof["large"]) / 2
y = folds.emotion.values
grp = key.loc[[("train", i) for i in folds.Id], "dup_group"].values
fold = folds.fold.values
has_twin_in_train_folds = np.array([np.isin(grp[i], grp[fold != fold[i]]) for i in range(len(y))])
print(f"rows={len(y)} | held-out rows with a twin in training folds: {has_twin_in_train_folds.mean():.3f}")


def rep(name, P):
    pred = np.array(CL)[P.argmax(1)]
    tw = has_twin_in_train_folds
    return (f"{name:28s} OOF macroF1={f1_score(y, pred, average='macro'):.4f} | twin rows={f1_score(y[tw], pred[tw], average='macro'):.4f}"
            f" | non-twin rows={f1_score(y[~tw], pred[~tw], average='macro'):.4f}")


for k in ("base+", "large", "ens"):
    print(rep(k, oof[k]))

# frozen LR on the SAME folds (layer 7 mean+std)
X = np.load(f"{W}/emb/wbp.npy").astype(np.float32)[:, 7]
idx = np.array([key.index.get_loc(("train", i)) for i in folds.Id])
Xtr = X[idx]
lr = np.zeros((len(y), 4))
for f in sorted(set(fold)):
    a, b = fold != f, fold == f
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")).fit(Xtr[a], y[a])
    lr[b] = clf.predict_proba(Xtr[b])
print(rep("frozen LR wbp-L7", lr))
for w in (0.2, 0.33, 0.5):
    print(rep(f"ens*(1-{w}) + LR*{w}", (1 - w) * oof["ens"] + w * lr))
c = np.corrcoef(oof["ens"].argmax(1) == np.array([CL.index(v) for v in y]), lr.argmax(1) == np.array([CL.index(v) for v in y]))[0, 1]
print(f"correlation of correctness (ens vs LR): {c:.3f}")

# test-side: how many twin rows does the 0.87918 submission get 'wrong' w.r.t. its train twin's label?
sub = pd.read_csv(f"{W}/kunal_output/submission.csv")
te = m[m.split == "test"].merge(sub.rename(columns={"emotion": "pred"}), on="Id")
trn = m[m.split == "train"]
twin_lab = trn.groupby("dup_group").emotion.agg(lambda s: s.mode().iloc[0] if s.nunique() == 1 else None)
te["twin_label"] = te.dup_group.map(twin_lab)
t = te[te.twin_label.notna()]
print(f"\ntest rows with train twin (consistent label): {len(t)} | submission disagrees with twin label on {int((t.pred != t.twin_label).sum())}")
print(t[t.pred != t.twin_label][["Id", "pred", "twin_label"]].to_string(index=False))
te[["Id", "pred", "twin_label"]].to_csv(f"{W}/test_twin_labels.csv", index=False)
