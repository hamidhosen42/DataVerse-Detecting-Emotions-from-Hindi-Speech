"""Deep error analysis of B8 out-of-fold predictions (public-pipeline folds, 824 rows): where does it fail?

Segments: twin vs non-twin, duration, encoding batch (PCM_16/PCM_32), speaker novelty (max x-vector cosine to the
training part of the fold, twins excluded), transcript repetition (does the same normalised sentence occur in the
training part from a different recording, and does that label agree?), confidence calibration, class confusions.

Usage: python -I deep_error_analysis.py <research_dir>
"""
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

R = sys.argv[1]
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
d = np.load(f"{W}/family_probs.npz", allow_pickle=True)
z = np.load(f"{W}/lora_probs_wlv2.npz")
W5 = ["lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat", "lr_whhlv2cat", "lr_wmed"]
P = np.mean([z["oof"]] + [d[f"oof__{f}"] for f in W5], 0)
y, fold, tw = d["y"], d["fold"], d["tw"]
pred = CL[P.argmax(1)]
ok = pred == y
m = pd.read_csv(f"{W}/meta.csv")
m["g"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
idx = m.set_index(["split", "Id"]).index.get_indexer([("train", i) for i in folds.Id])
mm = m.iloc[idx].reset_index(drop=True)
t = pd.read_csv(f"{W}/transcripts_whhlv2.csv").fillna("")
norm = lambda s: re.sub(r"[^\w]", "", s.lower())
txt = t.text.map(norm).values[idx]
sv = normalize(np.load(f"{W}/emb/sv.npy")[idx])
grp = mm.g.values


def seg(name, mask):
    n = int(mask.sum())
    if n < 5:
        return
    f1 = f1_score(y[mask], pred[mask], average="macro") if len(set(y[mask])) > 1 else float("nan")
    print(f"  {name:38s} n={n:4d}  acc={ok[mask].mean():.3f}  macroF1={f1:.3f}  errors={int((~ok[mask]).sum())}")


print(f"B8 OOF macro-F1 {f1_score(y, pred, average='macro'):.4f} | errors {int((~ok).sum())}/{len(y)}")
print("\n[twin]"); seg("twin rows", tw); seg("non-twin rows", ~tw)
nt = ~tw
print("\n[duration, non-twin]")
for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 6), (6, 99)):
    seg(f"{lo}-{hi} s", nt & (mm.dur.values >= lo) & (mm.dur.values < hi))
print("\n[encoding batch, non-twin]")
for s in ("PCM_16", "PCM_32"):
    seg(s, nt & (mm.subtype.values == s))

# speaker novelty: nearest x-vector neighbour in the training part (twins excluded)
nov = np.zeros(len(y))
for k in range(5):
    a, b = np.where(fold != k)[0], np.where(fold == k)[0]
    S = sv[b] @ sv[a].T
    S[grp[b][:, None] == grp[a][None, :]] = -1
    nov[b] = S.max(1)
print("\n[speaker similarity to training part (x-vector cosine), non-twin]")
qs = np.quantile(nov[nt], [0, .25, .5, .75, 1])
for i in range(4):
    seg(f"q{i + 1}: {qs[i]:.3f}-{qs[i + 1]:.3f}", nt & (nov >= qs[i]) & (nov <= qs[i + 1] + 1e-9))

# transcript repetition in the training part (different recording)
rep_lab = np.empty(len(y), dtype=object)
for k in range(5):
    a, b = np.where(fold != k)[0], np.where(fold == k)[0]
    for i in b:
        same = a[(txt[a] == txt[i]) & (grp[a] != grp[i]) & (len(txt[i]) >= 6)]
        if len(same):
            rep_lab[i] = pd.Series(y[same]).mode().iloc[0]
has = np.array([v is not None for v in rep_lab])
print("\n[same sentence present in training part from another recording, non-twin]")
seg("sentence repeated", nt & has); seg("sentence new", nt & ~has)
if (nt & has).any():
    agree = np.array([rep_lab[i] == y[i] for i in np.where(nt & has)[0]])
    rule = np.array([rep_lab[i] == y[i] for i in np.where(nt & has)[0]])
    b8 = ok[nt & has]
    print(f"  text-match rule accuracy on those rows: {agree.mean():.3f} (n={agree.size}) vs B8 {b8.mean():.3f}; "
          f"rule fixes {int((rule & ~b8).sum())}, breaks {int((~rule & b8).sum())}")

print("\n[confidence calibration, all rows]")
conf = P.max(1)
for lo, hi in ((0, .5), (.5, .7), (.7, .9), (.9, 1.01)):
    seg(f"max prob {lo}-{hi}", (conf >= lo) & (conf < hi))
print("\n[confusions among errors, non-twin]")
e = nt & ~ok
print(pd.crosstab(pd.Series(y[e], name="true"), pd.Series(pred[e], name="pred")))
