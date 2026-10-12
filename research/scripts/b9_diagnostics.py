"""Extra B9 diagnostics, computed from cached OOF/test probabilities (no training):
  1. where the best candidate (B9d: lora30top replaces lora8) changes B8's OOF predictions (fixed / broken rows)
  2. diversity of the fine-tuned members (correctness correlation with the best frozen family and with each other)
  3. paired dup-group bootstrap restricted to NON-TWIN OOF rows (rows without a near-copy in their training folds)
  4. multiple-candidate optimism: White's Reality Check over the 7 alternatives of this round (B5, B9a-f) vs B8 --
     bootstrap gains are centred at their observed means (null: no candidate beats B8) and the max over candidates is
     compared with the observed best gain
  5. test-set agreement of B9d and B8 before the twin safety net
  6. member-level effects: 30-s vs 8-s LoRA, LP-FT vs its linear probe
Usage: python -I b9_diagnostics.py <research_dir>     -> prints and writes work/b9_diagnostics.txt
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

R = sys.argv[1]
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
d = np.load(f"{W}/family_probs.npz", allow_pickle=True)
y, fold, tw = d["y"], d["fold"], d["tw"].astype(bool)
oof = {k[5:]: d[k] for k in d.files if k.startswith("oof__")}
test = {k[6:]: d[k] for k in d.files if k.startswith("test__")}
c = np.load(f"{W}/b9_candidate_probs.npz")
for k in c.files:
    kind, name = k.split("__")
    (oof if kind == "oof" else test)[name] = c[k]
W5 = ["lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat", "lr_whhlv2cat", "lr_wmed"]
cands = {"B8": ["lora8"] + W5, "B5": W5, "B9a": ["lpft30"] + W5, "B9b": ["lora8", "lpft30"] + W5, "B9c": ["lpft30", "lp30"] + W5,
         "B9d": ["lora30top"] + W5, "B9e": ["lora8", "lora30top"] + W5, "B9f": ["lora30top", "lpft30"] + W5}
yi = pd.Categorical(y, categories=CL).codes
assert (yi >= 0).all() and len(yi) == 824
pred = {n: np.mean([oof[f] for f in fs], 0).argmax(1) for n, fs in cands.items()}
tpred = {n: np.mean([test[f] for f in fs], 0).argmax(1) for n, fs in cands.items()}


def mf1(t, p):   # fast macro-F1 (all 4 classes present in every resample used here)
    cm = np.bincount(4 * t + p, minlength=16).reshape(4, 4)
    tp = np.diag(cm)
    return float(np.mean(2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)))


assert abs(mf1(yi, pred["B8"]) - f1_score(yi, pred["B8"], average="macro")) < 1e-12
m = pd.read_csv(f"{W}/meta.csv")
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
A = m.set_index(["split", "Id"]).index.get_indexer([("train", i) for i in folds.Id])
grp = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values[A]
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def boot(mask, n=3000, seed=0):
    idx = np.where(mask)[0]
    ug = np.unique(grp[idx])
    mem = {g: idx[grp[idx] == g] for g in ug}
    rng = np.random.default_rng(seed)
    return [np.concatenate([mem[g] for g in rng.choice(ug, len(ug))]) for _ in range(n)]


# 1. B9d vs B8 on OOF rows
ch = pred["B9d"] != pred["B8"]
fixed, broken = (ch & (pred["B9d"] == yi)).sum(), (ch & (pred["B8"] == yi)).sum()
say(f"[1] OOF rows where B9d != B8: {ch.sum()} / 824 | B9d fixes {fixed}, breaks {broken}, both wrong {ch.sum() - fixed - broken} | "
    f"changed rows in twin / non-twin subsets: {(ch & tw).sum()} / {(ch & ~tw).sum()} | per fold {[int((ch & (fold == k)).sum()) for k in range(5)]}")
for n in ("B9e", "B9b", "B5", "B9a"):
    chn = pred[n] != pred["B8"]
    say(f"    {n}: changed {chn.sum()} rows, fixes {(chn & (pred[n] == yi)).sum()}, breaks {(chn & (pred['B8'] == yi)).sum()}")

# 2. diversity
ok = {f: (oof[f].argmax(1) == yi) for f in ("lora8", "lora30top", "lpft30", "lr_wlv2cat")}
cc = lambda a, b: np.corrcoef(ok[a], ok[b])[0, 1]
say(f"[2] correctness correlation: lora30top~lr_wlv2cat {cc('lora30top', 'lr_wlv2cat'):.3f} | lora8~lr_wlv2cat {cc('lora8', 'lr_wlv2cat'):.3f} | "
    f"lpft30~lr_wlv2cat {cc('lpft30', 'lr_wlv2cat'):.3f} | lora30top~lora8 {cc('lora30top', 'lora8'):.3f} | lora30top~lpft30 {cc('lora30top', 'lpft30'):.3f}")
for f in ("lora8", "lora30top", "lpft30"):
    say(f"    {f:10s} OOF acc {ok[f].mean():.4f} | rows it gets right that lr_wlv2cat gets wrong: {(ok[f] & ~ok['lr_wlv2cat']).sum()} | reverse: {(~ok[f] & ok['lr_wlv2cat']).sum()}")

# 3. non-twin-only paired bootstrap
bnt = boot(~tw)
say(f"[3] non-twin OOF rows: {(~tw).sum()} in {len(np.unique(grp[~tw]))} dup groups; B8 non-twin macro-F1 {mf1(yi[~tw], pred['B8'][~tw]):.4f}")
for n in ("B9d", "B9e", "B9b", "B9a", "B5"):
    dd = np.array([mf1(yi[i], pred[n][i]) - mf1(yi[i], pred["B8"][i]) for i in bnt])
    say(f"    {n} - B8 (non-twin): {dd.mean():+.4f} [{np.quantile(dd, .025):+.4f},{np.quantile(dd, .975):+.4f}] P(>0) {np.mean(dd > 0):.3f}")

# 4. White's Reality Check over this round's alternatives
ball = boot(np.ones(824, bool))
alts = ["B5", "B9a", "B9b", "B9c", "B9d", "B9e", "B9f"]
obs = np.array([mf1(yi, pred[n]) - mf1(yi, pred["B8"]) for n in alts])
G = np.array([[mf1(yi[i], pred[n][i]) - mf1(yi[i], pred["B8"][i]) for n in alts] for i in ball])
T = (G - G.mean(0)).max(1)                     # null-centred max over candidates
best = alts[int(obs.argmax())]
p_rc = float(np.mean(T >= obs.max()))
say(f"[4] White's Reality Check (3000 dup-group bootstrap draws, {len(alts)} alternatives vs B8): best = {best} observed gain {obs.max():+.4f}; "
    f"95th pct of null max {np.quantile(T, .95):+.4f}; p = {p_rc:.3f} -> {'significant' if p_rc < 0.05 else 'NOT significant'} at 0.05")
say("    (covers only this round's 7 alternatives; ~45 blends were tried across all rounds, so the true correction is stronger)")

# 5. test agreement
tt = tpred["B9d"] != tpred["B8"]
pb8 = np.mean([test[f] for f in cands["B8"]], 0)
srt = np.sort(pb8, 1)
say(f"[5] test argmax rows differing B9d vs B8 BEFORE the twin net: {tt.sum()} / 209 | mean |p_B9d - p_B8| {np.abs(np.mean([test[f] for f in cands['B9d']], 0) - pb8).mean():.4f} | "
    f"B8 test rows with top-2 margin < 0.10: {(srt[:, -1] - srt[:, -2] < 0.10).sum()}")
for n in alts:
    say(f"    {n}: test rows differing from B8 before twin net {(tpred[n] != tpred['B8']).sum()}")
# 6. member-level effects (single fine-tuned members, paired dup-group bootstrap)
for a, b, what in (("lora30top", "lora8", "context window 30 s vs 8 s (same LoRA recipe)"), ("lpft30", "lp30", "LP-FT vs its own linear probe (30 s)")):
    pa, pb = oof[a].argmax(1), oof[b].argmax(1)
    for nm, bb in (("all", ball), ("non-twin", bnt)):
        dd = np.array([mf1(yi[i], pa[i]) - mf1(yi[i], pb[i]) for i in bb])
        say(f"[6] {a} - {b} ({what}), {nm} rows: {dd.mean():+.4f} [{np.quantile(dd, .025):+.4f},{np.quantile(dd, .975):+.4f}] P(>0) {np.mean(dd > 0):.3f}")
open(f"{W}/b9_diagnostics.txt", "w").write("\n".join(lines) + "\n")
