"""B9 final comparison: B8 vs candidates built from the native-30-s LP-FT run (and the frozen-only blend),
on the public-pipeline folds (824 rows) with a paired dup-group bootstrap, per-class F1, confusion matrices, fold variation,
non-twin scores; plus an optimism-aware decision (keep rule: mean gain >= +0.005, P(gain>0) >= 0.9; and a stricter
"many-candidates" rule P(gain>0) >= 0.99 for the ~45 blend candidates tried in this research).

Usage: python -I b9_eval.py <research_dir> [lpft_file=work/lpft_probs_wlv2_30s.npz]
Writes work/b9_eval.txt (stdout copy), work/b9_eval.csv, and research/submissions/sub_B9_<name>.csv for each candidate.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

R = sys.argv[1]
W = f"{R}/work"
LP = sys.argv[2] if len(sys.argv) > 2 else f"{W}/lpft_probs_wlv2_30s.npz"
CL = np.array(["angry", "happy", "neutral", "sad"])
d = np.load(f"{W}/family_probs.npz", allow_pickle=True)
y, fold, tw, test_ids = d["y"], d["fold"], d["tw"], d["test_ids"]
oof = {k[5:]: d[k] for k in d.files if k.startswith("oof__")}
test = {k[6:]: d[k] for k in d.files if k.startswith("test__")}
z = np.load(f"{W}/lora_probs_wlv2.npz")
oof["lora8"], test["lora8"] = z["oof"], z["test"]
z = np.load(LP)
nf = int(z["folds_done"])
assert nf == 5, f"LP-FT run incomplete: {nf}/5 folds"
oof["lp30"], test["lp30"] = z["oof_lp"], z["test_lp"]
oof["lpft30"], test["lpft30"] = z["oof_ft"], z["test_ft"]
tf = f"{W}/loratop_probs_wlv2.npz"
HAS_TOP = os.path.exists(tf) and int(np.load(tf)["folds_done"]) == 5
if HAS_TOP:
    zt = np.load(tf); oof["lora30top"], test["lora30top"] = zt["oof"], zt["test"]
W5 = ["lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat", "lr_whhlv2cat", "lr_wmed"]
cands = {
    "B8 (lora8 + 5 frozen)": ["lora8"] + W5,
    "B5 (5 frozen only)": W5,
    "B9a: lpft30 replaces lora8": ["lpft30"] + W5,
    "B9b: B8 + lpft30": ["lora8", "lpft30"] + W5,
    "B9c: lpft30 + lp30 + 5 frozen": ["lpft30", "lp30"] + W5,
}
if HAS_TOP:
    cands["B9d: lora30top replaces lora8"] = ["lora30top"] + W5
    cands["B9e: B8 + lora30top"] = ["lora8", "lora30top"] + W5
    cands["B9f: lora30top + lpft30 + 5 frozen"] = ["lora30top", "lpft30"] + W5
m = pd.read_csv(f"{W}/meta.csv")
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
A = m.set_index(["split", "Id"]).index.get_indexer([("train", i) for i in folds.Id])
grp = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values[A]
ug = np.unique(grp)
mem = {g: np.where(grp == g)[0] for g in ug}
rng = np.random.default_rng(0)
boots = [np.concatenate([mem[g] for g in rng.choice(ug, len(ug))]) for _ in range(3000)]
base = CL[np.mean([oof[f] for f in cands["B8 (lora8 + 5 frozen)"]], 0).argmax(1)]
bt = CL[np.mean([test[f] for f in cands["B8 (lora8 + 5 frozen)"]], 0).argmax(1)]
tl = pd.read_csv(f"{W}/test_twin_labels.csv").set_index("Id").loc[test_ids, "twin_label"]
lines, rows = [], []


def say(s=""):
    print(s)
    lines.append(s)


say("== single families (public-pipeline folds, 824 rows)")
say(f"{'family':12s} {'OOF':>7s} {'non-twin':>9s}  per-fold")
for f in ["lora8", "lp30", "lpft30"] + (["lora30top"] if HAS_TOP else []) + W5:
    p = CL[oof[f].argmax(1)]
    pf = [round(f1_score(y[fold == k], p[fold == k], average="macro"), 4) for k in range(5)]
    say(f"{f:12s} {f1_score(y, p, average='macro'):7.4f} {f1_score(y[~tw], p[~tw], average='macro'):9.4f}  {pf}")
c = lambda a, b: np.corrcoef(CL[oof[a].argmax(1)] == y, CL[oof[b].argmax(1)] == y)[0, 1]
say(f"correctness correlation: lpft30~lp30 {c('lpft30','lp30'):.3f} | lpft30~lora8 {c('lpft30','lora8'):.3f} | lpft30~lr_wlv2cat {c('lpft30','lr_wlv2cat'):.3f} | lora8~lr_wlv2cat {c('lora8','lr_wlv2cat'):.3f}")
say("\n== candidates vs B8 (paired dup-group bootstrap, 3000 draws)")
for n, fs in cands.items():
    P = np.mean([oof[f] for f in fs], 0)
    p = CL[P.argmax(1)]
    dd = np.array([f1_score(y[i], p[i], average="macro") - f1_score(y[i], base[i], average="macro") for i in boots])
    tp = CL[np.mean([test[f] for f in fs], 0).argmax(1)]
    tp = np.where(tl.notna().values, tl.values, tp)
    pf = [round(f1_score(y[fold == k], p[fold == k], average="macro"), 4) for k in range(5)]
    keep = n.startswith("B9") and dd.mean() >= 0.005 and np.mean(dd > 0) >= 0.9
    strict = keep and np.mean(dd > 0) >= 0.99
    rows.append(dict(candidate=n, oof=round(f1_score(y, p, average="macro"), 4), non_twin=round(f1_score(y[~tw], p[~tw], average="macro"), 4),
                     fold_min=min(pf), fold_max=max(pf), gain_vs_B8=round(dd.mean(), 4), ci95=f"[{np.quantile(dd, .025):+.4f},{np.quantile(dd, .975):+.4f}]",
                     p_gt0=round(np.mean(dd > 0), 3), test_rows_diff_vs_B8=int((tp != bt).sum()), keep_rule=keep, strict_rule=strict))
    say(f"{n:32s} OOF {rows[-1]['oof']:.4f} non-twin {rows[-1]['non_twin']:.4f} folds {pf} | vs B8 {dd.mean():+.4f} {rows[-1]['ci95']} P>0 {rows[-1]['p_gt0']:.3f} | test diff {rows[-1]['test_rows_diff_vs_B8']} | keep={keep} strict={strict}")
    pd.DataFrame({"Id": test_ids, "emotion": tp}).to_csv(f"{R}/submissions/sub_B9_{n.split(':')[0].split(' ')[0]}.csv", index=False)
    if n.startswith(("B8", "B9a", "B9b", "B9d")):
        say(classification_report(y, p, digits=4))
        say(str(pd.DataFrame(confusion_matrix(y, p, labels=CL), index=CL, columns=CL)))
pd.DataFrame(rows).to_csv(f"{W}/b9_eval.csv", index=False)
open(f"{W}/b9_eval.txt", "w").write("\n".join(lines) + "\n")
np.savez(f"{W}/b9_candidate_probs.npz", **{f"oof__{k}": v for k, v in oof.items() if k in ("lora8", "lp30", "lpft30", "lora30top")}, **{f"test__{k}": v for k, v in test.items() if k in ("lora8", "lp30", "lpft30", "lora30top")})
