"""How surprising are the top public scores, given our own B6 model?

Uses B6's out-of-fold predictions (public-pipeline folds) as a proxy for its per-clip correctness, and simulates
public subsets of size N drawn like the hidden test (37% rows with a train twin). Reports, for each N:
  - distribution of B6's simulated public score (mean, SD, 5/50/95 percentiles),
  - P(simulated score >= each leaderboard score), and where B6's actual 0.93060 falls,
  - the number of misclassified clips implied by each score (mean errors in simulated subsets with that score).

Usage: python -I top_score_reachability.py <research_dir>
"""
import sys

import numpy as np
from sklearn.metrics import f1_score

R = sys.argv[1]
d = np.load(f"{R}/work/family_probs.npz", allow_pickle=True)
CL = np.array(["angry", "happy", "neutral", "sad"])
B6 = ["ft_ens", "lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat", "lr_whhlv2cat", "lr_wmed"]
pred = CL[np.mean([d[f"oof__{f}"] for f in B6], 0).argmax(1)]
y, tw = d["y"], d["tw"]
twin_idx, non_idx = np.where(tw)[0], np.where(~tw)[0]
LB = {"top-3 (0.96544)": 0.96544, "rank 4-5 (0.95387)": 0.95387, "rank 7-9 (0.94230)": 0.94230, "B6 actual (0.93060)": 0.93060}
rng = np.random.default_rng(0)
for N in (52, 63, 84, 105):
    sc, err = [], []
    for _ in range(20000):
        k = rng.binomial(N, 0.373)
        idx = np.concatenate([rng.choice(twin_idx, k, replace=False), rng.choice(non_idx, N - k, replace=False)])
        sc.append(f1_score(y[idx], pred[idx], average="macro"))
        err.append(int((y[idx] != pred[idx]).sum()))
    sc, err = np.array(sc), np.array(err)
    q = np.percentile(sc, [5, 50, 95])
    print(f"N={N:3d}: B6 simulated public mean {sc.mean():.4f} SD {sc.std():.4f} | 5/50/95% {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f} | errors mean {err.mean():.1f}")
    for name, v in LB.items():
        near = np.abs(sc - v) < 0.004
        e = f"{err[near].mean():.1f}" if near.any() else "n/a"
        print(f"     P(B6 >= {name}) = {np.mean(sc >= v - 1e-6):.3f} | typical #errors at that score: {e}")
