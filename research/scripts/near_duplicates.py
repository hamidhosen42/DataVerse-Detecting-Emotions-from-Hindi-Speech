"""All-pairs near-duplicate scan over competition audio (waveform normalized cross-correlation).

Pairs whose durations differ by < 0.25 s are compared; max |NCC| over all lags is the score.
Writes pairs with score > 0.5 and connected components at score >= 0.9 ("dup groups").

Usage: python -I near_duplicates.py <data_root> <meta.csv> <out_dir>
"""
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve

R, meta, OUT = sys.argv[1:4]
m = pd.read_csv(meta).reset_index(drop=True)
wav = []
for r in m.itertuples():
    y = sf.read(f"{R}/{r.split}/audio_{r.Id:06d}.wav", dtype="float32")[0]
    wav.append((y - y.mean()) / (y.std() + 1e-9))
n = len(m)
dur = m.dur.values
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        if abs(dur[i] - dur[j]) < 0.25:
            a, b = wav[i], wav[j]
            s = float(np.max(np.abs(fftconvolve(a, b[::-1], mode="full"))) / min(len(a), len(b)))
            if s > 0.5:
                pairs.append((i, j, s))
    if i % 200 == 0:
        print("row", i, "pairs so far", len(pairs), flush=True)
p = pd.DataFrame(pairs, columns=["i", "j", "ncc"])
for c in ("split", "Id", "emotion"):
    p[f"{c}_a"] = m.loc[p.i, c].values
    p[f"{c}_b"] = m.loc[p.j, c].values
p.sort_values("ncc", ascending=False).to_csv(f"{OUT}/near_dup_pairs.csv", index=False)

parent = list(range(n))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for r in p[p.ncc >= 0.9].itertuples():
    parent[find(r.i)] = find(r.j)
m["dup_group"] = [find(i) for i in range(n)]
m[["split", "Id", "emotion", "dup_group"]].to_csv(f"{OUT}/dup_groups.csv", index=False)

print("\nNCC histogram of compared pairs above 0.5:")
print(pd.cut(p.ncc, [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.01]).value_counts().sort_index().to_string())
hi = p[p.ncc >= 0.9]
print("\npairs >= 0.9 by split combo:", (hi.split_a + "-" + hi.split_b).value_counts().to_dict())
tt = hi[(hi.split_a == "train") & (hi.split_b == "train")]
print("train-train pairs >= 0.9:", len(tt), "| label disagreements:", int((tt.emotion_a != tt.emotion_b).sum()))
print(tt[tt.emotion_a != tt.emotion_b][["Id_a", "emotion_a", "Id_b", "emotion_b", "ncc"]].to_string(index=False))
sizes = m.groupby("dup_group").size()
multi = sizes[sizes > 1]
print("\ndup groups (>=2 members):", len(multi), "| rows in them:", int(multi.sum()))
g = m[m.dup_group.isin(multi.index)].groupby("dup_group")
test_with_train_twin = g.filter(lambda x: (x.split == "test").any() and (x.split == "train").any())
print("test rows with a train near-twin (ncc>=0.9):", int((test_with_train_twin.split == "test").sum()))
print("train-twin labels for those test rows:", test_with_train_twin[test_with_train_twin.split == "train"].emotion.value_counts().to_dict())
test_only = g.filter(lambda x: (x.split == "test").all())
print("test-only dup groups:", test_only.dup_group.nunique())
