"""Classifier variants on one frozen feature set, scored on Track B only (dup-group 5-fold x 3 seeds).

Usage: python -I classifier_variants.py <research_dir> <tag> [tag2 ...]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC

R = sys.argv[1]
W = f"{R}/work"
m = pd.read_csv(f"{W}/meta.csv")
m["g"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
tr = np.where(m.split == "train")[0]
y, g = m.emotion.values[tr], m.g.values[tr]
MODELS = {
    "LR C=0.05 (baseline)": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced")),
    "LDA shrinkage=auto": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    "Ridge alpha=1000": lambda: make_pipeline(StandardScaler(), RidgeClassifier(alpha=1000.0, class_weight="balanced")),
    "LinearSVC C=0.001": lambda: make_pipeline(StandardScaler(), LinearSVC(C=0.001, class_weight="balanced", max_iter=20000)),
    "SVC rbf C=10": lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced")),
    "kNN-15 cosine": lambda: make_pipeline(StandardScaler(), Normalizer(), KNeighborsClassifier(15, weights="distance")),
}
rows = []
for tag in sys.argv[2:]:
    X = np.asarray(np.load(f"{W}/emb/{tag}.npy", mmap_mode="r")[tr], np.float32)
    for name, mk in MODELS.items():
        s = []
        for seed in range(3):
            p = np.empty(len(y), dtype=object)
            for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
                p[b] = mk().fit(X[a], y[a]).predict(X[b])
            s.append(f1_score(y, p, average="macro"))
        rows.append(dict(tag=tag, model=name, trackB=round(np.mean(s), 4), sd=round(np.std(s), 4)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{W}/classifier_variants.csv", index=False)
