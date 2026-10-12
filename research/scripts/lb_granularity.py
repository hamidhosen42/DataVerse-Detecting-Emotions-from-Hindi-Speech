"""Infer plausible public-LB subset sizes from the exact Macro-F1 values on the leaderboard.

For a candidate public size N and class counts near the train proportions (+-tol), enumerate all confusion
matrices with <= K errors and check which leaderboard scores are reproduced exactly (5-decimal rounding).
A size N is 'consistent' if every target score is reachable. Weak evidence only: many N can fit.

Usage: python -I lb_granularity.py
"""
import itertools

import numpy as np

targets = [0.96544, 0.95387, 0.94303, 0.94230]
K, tol = 4, 5
prior = np.array([0.248, 0.232, 0.263, 0.257])
cells = [(i, j) for i in range(4) for j in range(4) if i != j]
cfg = []
for k in range(1, K + 1):
    for comb in itertools.combinations_with_replacement(range(12), k):
        E = np.zeros((4, 4), int)
        for c in comb:
            E[cells[c]] += 1
        cfg.append(E)
E = np.stack(cfg)
fn, fp, nerr = E.sum(2), E.sum(1), E.sum((1, 2))
for N in [42, 52, 63, 73, 84, 94, 104, 105, 115, 125, 126, 146, 167]:
    base = np.round(prior * N).astype(int)
    rng = range(-tol, tol + 1)
    counts = []
    for da, db, dc in itertools.product(rng, rng, rng):
        n = base + np.array([da, db, dc, 0])
        n[3] = N - n[:3].sum()
        if (n > 0).all():
            counts.append(n)
    counts = np.array(counts)
    tp = counts[:, None, :] - fn[None]
    ok = (tp >= 0).all(2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mf = np.round((2 * tp / (2 * tp + fp[None] + fn[None])).mean(2), 5)
    res = {}
    for t in targets:
        hit = np.isclose(mf, t, atol=6e-6) & ok
        res[t] = int(nerr[np.where(hit)[1]].min()) if hit.any() else None
    flag = "CONSISTENT" if all(v is not None for v in res.values()) else ""
    print(f"N={N:3d} ({N / 209:.0%}) min errors per score: {res} {flag}", flush=True)
