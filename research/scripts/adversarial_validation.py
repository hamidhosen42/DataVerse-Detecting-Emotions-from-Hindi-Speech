"""Adversarial validation (train vs test) + twin label-transfer accuracy + simple distribution tests.

Usage: python -I adversarial_validation.py <work_dir> [emb_tag layer]
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "wbp"
layer = int(sys.argv[3]) if len(sys.argv) > 3 else 6
m = pd.read_csv(f"{W}/meta.csv")
m["dup_group"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
is_test = (m.split == "test").astype(int).values

print("== KS tests train vs test")
for c in ("dur", "rms", "peak", "silence_frac"):
    a, b = m.loc[m.split == "train", c], m.loc[m.split == "test", c]
    print(f"{c:13s} KS={ks_2samp(a, b).statistic:.3f} p={ks_2samp(a, b).pvalue:.3f}")
print("PCM_32 share train/test:", round((m[m.split == "train"].subtype == "PCM_32").mean(), 3), round((m[m.split == "test"].subtype == "PCM_32").mean(), 3))


def psi(a, b, bins=10):
    q = np.quantile(a, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    pa = np.histogram(a, q)[0] / len(a) + 1e-6
    pb = np.histogram(b, q)[0] / len(b) + 1e-6
    return float(np.sum((pa - pb) * np.log(pa / pb)))


print("PSI dur:", round(psi(m.loc[m.split == "train", "dur"], m.loc[m.split == "test", "dur"]), 4))

X = np.load(f"{W}/emb/{tag}.npy").astype(np.float32)[:, layer]
X = np.hstack([X, m[["dur", "rms", "peak", "silence_frac"]].values, (m.subtype == "PCM_32").values[:, None]])
# group by dup group so a twin pair cannot let the classifier 'recognize' a file across folds
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)), X, is_test,
                      groups=m.dup_group.values, cv=cv, method="predict_proba")[:, 1]
print(f"\n== adversarial AUC ({tag} layer {layer} + clip stats, dup-grouped 5-fold): {roc_auc_score(is_test, p):.3f}")

tr = m[m.split == "train"]
sizes = tr.groupby("dup_group").size()
tw = tr[tr.dup_group.map(sizes) > 1]
correct = []
for _, r in tw.iterrows():
    others = tw[(tw.dup_group == r.dup_group) & (tw.Id != r.Id)].emotion
    correct.append(others.mode().iloc[0] == r.emotion)
print(f"\n== twin label transfer inside train: {np.sum(correct)}/{len(correct)} = {np.mean(correct):.4f} correct")
