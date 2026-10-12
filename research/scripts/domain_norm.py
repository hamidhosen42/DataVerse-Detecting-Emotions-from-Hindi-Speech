"""Domain-wise feature standardisation (PCM_16 vs PCM_32 recording batch) for frozen Whisper LR families.
Variants: 'global' (current), 'per-domain' (separate mean/std per batch, fit on the training part), 'global+flag'
(append a domain indicator), 'weight' (PCM_32 rows sample-weight 3). Reports Track B (3 seeds) and Track A,
overall and on PCM_32 / PCM_16 non-twin rows.
Usage: python -I domain_norm.py <research_dir> <tag> [tag...]
"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
R, TAGS = sys.argv[1], sys.argv[2:]
W = f"{R}/work"; CL = np.array(["angry", "happy", "neutral", "sad"])
m = pd.read_csv(f"{W}/meta.csv"); m["g"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
lab = pd.Categorical(m.emotion, categories=CL).codes; dom = (m.subtype == "PCM_32").values
key = m.set_index(["split", "Id"]); folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
A = key.index.get_indexer([("train", i) for i in folds.Id]); fA = folds.fold.values
tr = np.where(m.split == "train")[0]
gtr = m.g.values
def twin_mask(rows, splits):
    t = np.zeros(len(rows), bool)
    for a, b in splits: t[b] = np.isin(gtr[rows[b]], gtr[rows[a]])
    return t
def fit_pred(X, a_rows, b_rows, variant):
    Xa, Xb = X[a_rows].copy(), X[b_rows].copy()
    if variant == "per-domain":
        for dval in (False, True):
            ia, ib = dom[a_rows] == dval, dom[b_rows] == dval
            mu, sd = Xa[ia].mean(0), Xa[ia].std(0) + 1e-6
            Xa[ia] = (Xa[ia] - mu) / sd; Xb[ib] = (Xb[ib] - mu) / sd
    else:
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
        Xa, Xb = (Xa - mu) / sd, (Xb - mu) / sd
        if variant == "global+flag":
            Xa = np.hstack([Xa, 3 * dom[a_rows, None]]); Xb = np.hstack([Xb, 3 * dom[b_rows, None]])
    w = np.where(dom[a_rows], 3.0, 1.0) if variant == "weight" else None
    clf = LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced").fit(Xa, lab[a_rows], sample_weight=w)
    return clf.predict(Xb)
for tag in TAGS:
    E = np.load(f"{W}/emb/{tag}.npy").astype(np.float32)
    X = np.hstack([E[:, 24], E[:, 32]]) if E.ndim == 3 else E
    for variant in ("global", "per-domain", "global+flag", "weight"):
        sb = []
        for seed in range(3):
            sp = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, lab[tr], gtr[tr]))
            p = np.empty(len(tr), int)
            for a, b in sp: p[b] = fit_pred(X, tr[a], tr[b], variant)
            sb.append(f1_score(lab[tr], p, average="macro"))
        spA = [(np.where(fA != k)[0], np.where(fA == k)[0]) for k in range(5)]
        p = np.empty(len(A), int)
        for a, b in spA: p[b] = fit_pred(X, A[a], A[b], variant)
        nt = ~twin_mask(A, spA); d32 = dom[A]
        acc = lambda msk: (p[msk] == lab[A][msk]).mean()
        print(f"{tag:8s} {variant:12s} TrackB {np.mean(sb):.4f} | TrackA {f1_score(lab[A], p, average='macro'):.4f} | non-twin acc PCM_32 {acc(nt & d32):.3f} PCM_16 {acc(nt & ~d32):.3f}", flush=True)
