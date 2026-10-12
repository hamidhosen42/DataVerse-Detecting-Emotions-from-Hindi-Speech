"""Does 'submit several near-equivalent candidates, keep the one with the best PUBLIC score' raise the expected
FINAL score (0.4*public + 0.6*private) here, or does it just overfit?

Simulation: draw a 209-clip 'test set' from the OOF rows (37.3% twin rows), split into public (N) / private (209-N),
score every candidate on both parts, pick the candidate with the best public score (ties -> B6), and compare its
final score with always submitting B6. Candidates use their OOF predictions on the public-pipeline folds.

Usage: python -I select_by_public_sim.py <research_dir>
"""
import sys

import numpy as np
from sklearn.metrics import f1_score

R = sys.argv[1]
d = np.load(f"{R}/work/family_probs.npz", allow_pickle=True)
CL = np.array(["angry", "happy", "neutral", "sad"])
y, tw = d["y"], d["tw"]
CANDS = {
    "B6": ["ft_ens", "lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat", "lr_whhlv2cat", "lr_wmed"],
    "B5": ["lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat", "lr_whhlv2cat", "lr_wmed"],
    "B3": ["lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat"],
    "B4": ["ft_ens", "lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat"],
    "B1": ["ft_ens", "lr_hilarge", "lr_w2vbert", "lr_whhlv2", "lr_whhlv2cat", "lr_wl", "lr_wlv2", "lr_wlv2cat", "lr_wlv3", "lr_wlv3cat",
           "lr_wmed", "lr_wser", "lr_wturbo", "lr_wvaani", "lr_wvaanicat", "lr_xlsr", "lr_xlsr1b"],
}
pred = {k: CL[np.mean([d[f"oof__{f}"] for f in v], 0).argmax(1)] for k, v in CANDS.items()}
for k in pred:
    print(f"{k}: OOF {f1_score(y, pred[k], average='macro'):.4f} | rows differing from B6: {(pred[k] != pred['B6']).sum()}")
twin_idx, non_idx = np.where(tw)[0], np.where(~tw)[0]
rng = np.random.default_rng(0)
for pool in (["B6", "B5", "B3", "B4"], ["B6", "B5", "B3", "B4", "B1"]):
    for N in (63, 84, 105):
        g_final, g_pub, g_priv = [], [], []
        for _ in range(4000):
            k = rng.binomial(209, 0.373)
            idx = np.concatenate([rng.choice(twin_idx, k, replace=False), rng.choice(non_idx, 209 - k, replace=False)])
            rng.shuffle(idx)
            pub, priv = idx[:N], idx[N:]
            sp = {c: f1_score(y[pub], pred[c][pub], average="macro") for c in pool}
            best = max(pool, key=lambda c: (sp[c], c == "B6"))
            spr = {c: f1_score(y[priv], pred[c][priv], average="macro") for c in (best, "B6")}
            g_pub.append(sp[best] - sp["B6"])
            g_priv.append(spr[best] - spr["B6"])
            g_final.append(0.4 * g_pub[-1] + 0.6 * g_priv[-1])
        print(f"pool {'+'.join(pool):18s} N={N:3d}: expected gain vs always-B6 -> public {np.mean(g_pub):+.4f}, private {np.mean(g_priv):+.4f}, "
              f"FINAL {np.mean(g_final):+.4f} (P(final gain<0)={np.mean(np.array(g_final) < 0):.2f})")
