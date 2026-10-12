"""E7 (averaging rule + class-bias tuning) and E8 (stacking) on the B6 families, with NESTED evaluation:
for each outer fold k of the public-pipeline folds, anything tuned (bias vector, stacker) is fit on the other
4 folds' OOF predictions only, then applied to fold k. Compared with plain probability averaging.

Usage: python -I e7_e8.py <research_dir>
"""
import itertools
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

R = sys.argv[1]
d = np.load(f"{R}/work/family_probs.npz", allow_pickle=True)
CL = np.array(["angry", "happy", "neutral", "sad"])
B6 = ["ft_ens", "lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat", "lr_whhlv2cat", "lr_wmed"]
P = np.stack([d[f"oof__{f}"] for f in B6])  # (F, N, 4)
y, fold = d["y"], d["fold"]
eps = 1e-6
rules = {
    "prob mean (B6)": P.mean(0),
    "logit/log-prob mean": np.exp(np.log(P + eps).mean(0)),
    "rank mean": np.mean([np.argsort(np.argsort(p, axis=0), axis=0) for p in P], 0),
    "majority vote": np.stack([(P.argmax(2) == c).sum(0) + 1e-3 * P.mean(0)[:, c] for c in range(4)], 1),
}
for k, v in rules.items():
    print(f"{k:24s} OOF macro-F1 {f1_score(y, CL[v.argmax(1)], average='macro'):.4f}")

base = P.mean(0)
grid = np.linspace(-0.15, 0.15, 7)


def best_bias(Pp, yy):
    best, bv = -1, np.zeros(4)
    for b in itertools.product(grid, repeat=3):
        v = np.array([0.0, *b])
        s = f1_score(yy, CL[(np.log(Pp + eps) + v).argmax(1)], average="macro")
        if s > best + 1e-9:
            best, bv = s, v
    return bv


pred_bias = np.empty(len(y), dtype=object)
pred_stack = np.empty(len(y), dtype=object)
Z = np.log(P + eps).transpose(1, 0, 2).reshape(len(y), -1)  # stacker input: all families' log-probs
for k in sorted(set(fold)):
    a, b = fold != k, fold == k
    v = best_bias(base[a], y[a])
    pred_bias[b] = CL[(np.log(base[b] + eps) + v).argmax(1)]
    st = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(Z[a], y[a])
    pred_stack[b] = st.predict(Z[b])
    print(f"fold {k}: bias {np.round(v, 3)}", flush=True)
print(f"{'nested class-bias (E7)':24s} OOF macro-F1 {f1_score(y, pred_bias, average='macro'):.4f}")
print(f"{'nested LR stacker (E8)':24s} OOF macro-F1 {f1_score(y, pred_stack, average='macro'):.4f}")
print("note: stacker fit on OOF of the other folds, whose base models saw fold k in training -> mildly optimistic for E8")
