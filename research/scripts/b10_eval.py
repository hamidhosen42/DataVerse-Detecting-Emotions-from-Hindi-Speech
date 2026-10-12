"""B10: does averaging two seeds of the native-30-s LoRA member (lora30top, seeds 0 and 1) give a reliable gain over B8?

Pre-declared before seed 1 finished (2026-10-12 04:35 UTC):
  primary   B10a = mean(lora30top s0, s1) replaces lora8 in B8      (+ 5 frozen families, equal-weight mean)
  secondary B10b = B8 + mean(lora30top s0, s1)
  keep rule: paired dup-group bootstrap gain vs B8 >= +0.005 and P(gain>0) >= 0.9, AND at least one test prediction changes
  multiplicity: White's Reality Check over all alternatives tried in B9 + B10 (B5, B9a-f, B10a, B10b)
Usage: python -I b10_eval.py <research_dir>  -> prints, writes work/b10_eval.txt, work/b10_eval.csv, submissions/sub_B10a.csv, sub_B10b.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

R = sys.argv[1]
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
d = np.load(f"{W}/family_probs.npz", allow_pickle=True)
y, fold, tw, test_ids = d["y"], d["fold"], d["tw"].astype(bool), d["test_ids"]
oof = {k[5:]: d[k] for k in d.files if k.startswith("oof__")}
test = {k[6:]: d[k] for k in d.files if k.startswith("test__")}
c = np.load(f"{W}/b9_candidate_probs.npz")
for k in c.files:
    kind, name = k.split("__")
    (oof if kind == "oof" else test)[name] = c[k]
z1 = np.load(f"{W}/loratop_probs_wlv2_s1.npz")
assert int(z1["folds_done"]) == 5, f"seed-1 run incomplete: {int(z1['folds_done'])}/5 folds"
oof["lora30top_s1"], test["lora30top_s1"] = z1["oof"], z1["test"]
oof["lora30top_avg"] = (oof["lora30top"] + oof["lora30top_s1"]) / 2
test["lora30top_avg"] = (test["lora30top"] + test["lora30top_s1"]) / 2
W5 = ["lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat", "lr_whhlv2cat", "lr_wmed"]
cands = {"B8": ["lora8"] + W5, "B5": W5, "B9a": ["lpft30"] + W5, "B9b": ["lora8", "lpft30"] + W5, "B9c": ["lpft30", "lp30"] + W5,
         "B9d": ["lora30top"] + W5, "B9e": ["lora8", "lora30top"] + W5, "B9f": ["lora30top", "lpft30"] + W5,
         "B10a": ["lora30top_avg"] + W5, "B10b": ["lora8", "lora30top_avg"] + W5}
yi = pd.Categorical(y, categories=CL).codes
pred = {n: np.mean([oof[f] for f in fs], 0).argmax(1) for n, fs in cands.items()}
tl = pd.read_csv(f"{W}/test_twin_labels.csv").set_index("Id").loc[test_ids, "twin_label"]
tpred = {}
for n, fs in cands.items():
    t = CL[np.mean([test[f] for f in fs], 0).argmax(1)]
    tpred[n] = np.where(tl.notna().values, tl.values, t)


def mf1(t, p):
    cm = np.bincount(4 * t + p, minlength=16).reshape(4, 4)
    return float(np.mean(2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)))


m = pd.read_csv(f"{W}/meta.csv")
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
A = m.set_index(["split", "Id"]).index.get_indexer([("train", i) for i in folds.Id])
grp = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values[A]


def boot(mask, n=3000, seed=0):
    idx = np.where(mask)[0]
    ug = np.unique(grp[idx])
    mem = {g: idx[grp[idx] == g] for g in ug}
    rng = np.random.default_rng(seed)
    return [np.concatenate([mem[g] for g in rng.choice(ug, len(ug))]) for _ in range(n)]


ball, bnt = boot(np.ones(len(yi), bool)), boot(~tw)
lines, rows = [], []


def say(s=""):
    print(s)
    lines.append(s)


say("== members (824 OOF rows)")
for f in ("lora8", "lora30top", "lora30top_s1", "lora30top_avg"):
    p = oof[f].argmax(1)
    say(f"{f:14s} OOF {mf1(yi, p):.4f} non-twin {mf1(yi[~tw], p[~tw]):.4f} folds {[round(mf1(yi[fold == k], p[fold == k]), 4) for k in range(5)]}")
for a_, b_ in (("lora30top_s1", "lora30top"), ("lora30top_avg", "lora30top"), ("lora30top_avg", "lora8")):
    pa, pb = oof[a_].argmax(1), oof[b_].argmax(1)
    dd = np.array([mf1(yi[i], pa[i]) - mf1(yi[i], pb[i]) for i in ball])
    say(f"  {a_} - {b_}: {dd.mean():+.4f} [{np.quantile(dd, .025):+.4f},{np.quantile(dd, .975):+.4f}] P(>0) {np.mean(dd > 0):.3f}")
say(f"  seed agreement (OOF argmax s0 vs s1): {(oof['lora30top'].argmax(1) == oof['lora30top_s1'].argmax(1)).mean():.3f}")
say("\n== candidates vs B8 (paired dup-group bootstrap, 3000 draws; test predictions include the twin safety net)")
for n in cands:
    p = pred[n]
    dd = np.array([mf1(yi[i], p[i]) - mf1(yi[i], pred["B8"][i]) for i in ball])
    dn = np.array([mf1(yi[i], p[i]) - mf1(yi[i], pred["B8"][i]) for i in bnt])
    diff = int((tpred[n] != tpred["B8"]).sum())
    keep = n.startswith("B10") and dd.mean() >= 0.005 and np.mean(dd > 0) >= 0.9 and diff > 0
    rows.append(dict(candidate=n, members="+".join(cands[n]), oof=round(mf1(yi, p), 4), non_twin=round(mf1(yi[~tw], p[~tw]), 4),
                     gain=round(dd.mean(), 4), ci95=f"[{np.quantile(dd, .025):+.4f},{np.quantile(dd, .975):+.4f}]", p_gt0=round(np.mean(dd > 0), 3),
                     gain_non_twin=round(dn.mean(), 4), p_non_twin=round(np.mean(dn > 0), 3), test_rows_diff_vs_B8=diff, keep_rule=keep))
    r = rows[-1]
    say(f"{n:5s} OOF {r['oof']:.4f} non-twin {r['non_twin']:.4f} | vs B8 {r['gain']:+.4f} {r['ci95']} P {r['p_gt0']:.3f} | non-twin {r['gain_non_twin']:+.4f} P {r['p_non_twin']:.3f} | test diff {diff} | keep={keep}")
alts = [n for n in cands if n != "B8"]
obs = np.array([mf1(yi, pred[n]) - mf1(yi, pred["B8"]) for n in alts])
G = np.array([[mf1(yi[i], pred[n][i]) - mf1(yi[i], pred["B8"][i]) for n in alts] for i in ball])
T = (G - G.mean(0)).max(1)
say(f"\nWhite's Reality Check over {len(alts)} alternatives (B9 + B10): best {alts[int(obs.argmax())]} {obs.max():+.4f}; null-max 95th pct {np.quantile(T, .95):+.4f}; p = {np.mean(T >= obs.max()):.3f}")
for n in ("B10a", "B10b"):
    say(f"\n{n} per class / confusion:")
    say(classification_report(yi, pred[n], target_names=CL, digits=4))
    say(str(pd.DataFrame(confusion_matrix(yi, pred[n]), index=CL, columns=CL)))
    ch = np.where(tpred[n] != tpred["B8"])[0]
    say(f"test rows changed vs B8: {[(int(test_ids[j]), tpred['B8'][j], tpred[n][j]) for j in ch]}")
    pd.DataFrame({"Id": test_ids, "emotion": tpred[n]}).to_csv(f"{R}/submissions/sub_{n}.csv", index=False)
pd.DataFrame(rows).to_csv(f"{W}/b10_eval.csv", index=False)
open(f"{W}/b10_eval.txt", "w").write("\n".join(lines) + "\n")
np.savez(f"{W}/b10_member_probs.npz", oof__lora30top_s1=oof["lora30top_s1"], test__lora30top_s1=test["lora30top_s1"],
         oof__lora30top_avg=oof["lora30top_avg"], test__lora30top_avg=test["lora30top_avg"])
