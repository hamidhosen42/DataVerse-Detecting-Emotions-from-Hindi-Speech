"""Pseudo-speaker analysis from WavLM-base-plus-sv x-vectors (no speaker IDs are provided).

Agglomerative clustering (cosine, average linkage) at several distance thresholds; for each, report
#clusters, train/test cluster sharing, and how predictive the cluster is of emotion (label purity).
Near-duplicate groups are merged first so twins never count as separate "speakers".

Usage: python -I speakers.py <work_dir>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

W = sys.argv[1]
m = pd.read_csv(f"{W}/meta.csv")
m["dup_group"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
X = normalize(np.load(f"{W}/emb/sv.npy"))
# nearest-neighbour cosine similarity profile (excluding self and twins)
S = X @ X.T
np.fill_diagonal(S, -1)
same_grp = m.dup_group.values[:, None] == m.dup_group.values[None, :]
S_nt = np.where(same_grp, -1, S)
print("NN cosine (non-twin) quantiles:", np.round(np.quantile(S_nt.max(1), [0.05, 0.25, 0.5, 0.75, 0.95]), 3))
for thr in (0.03, 0.05, 0.08, 0.12, 0.18):
    lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=thr).fit_predict(X)
    m["spk"] = lab
    sizes = m.spk.value_counts()
    tr, te = m[m.split == "train"], m[m.split == "test"]
    shared = te.spk.isin(set(tr.spk)).mean()
    big = sizes[sizes >= 5].index
    t = tr[tr.spk.isin(big)]
    purity = t.groupby("spk").emotion.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    w = t.groupby("spk").size()
    print(f"thr={thr}: clusters={len(sizes)} (>=5 rows: {len(big)}), largest={sizes.iloc[0]}, singletons={(sizes == 1).sum()}, "
          f"test rows in a cluster seen in train={shared:.2f}, weighted label purity of clusters>=5={np.average(purity, weights=w):.2f}")
    m[["split", "Id", "emotion", "spk"]].to_csv(f"{W}/pseudo_speakers_thr{thr}.csv", index=False)
