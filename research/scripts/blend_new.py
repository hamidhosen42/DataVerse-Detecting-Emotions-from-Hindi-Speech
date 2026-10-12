"""Blend evaluation for the architecture round: B6 vs B6 + / with new-architecture families, on the public pipeline's
folds (Track A), dup-group paired bootstrap vs B6. Keep rule (pre-declared): mean gain >= +0.005 and P(gain>0) >= 0.9.

Sources: work/family_probs.npz (B6 families), work/heads_probs.npz (mlp_*, fuse_lr), work/attn_probs_*.npz (attn_*),
work/lora_probs_*.npz (lora_*; only if all 5 folds are done). Writes research/submissions/sub_<name>.csv for kept blends.

Usage: python -I blend_new.py <research_dir>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

R = sys.argv[1]
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
d = np.load(f"{W}/family_probs.npz", allow_pickle=True)
y, tw, test_ids = d["y"], d["tw"], d["test_ids"]
oof = {k[5:]: d[k] for k in d.files if k.startswith("oof__")}
test = {k[6:]: d[k] for k in d.files if k.startswith("test__")}
for f in [f"{W}/heads_probs.npz"] + glob.glob(f"{W}/attn_probs_*.npz"):
    if os.path.exists(f):
        z = np.load(f)
        oof.update({k[5:]: z[k] for k in z.files if k.startswith("oof__")})
        test.update({k[6:]: z[k] for k in z.files if k.startswith("test__")})
for f in glob.glob(f"{W}/lora_probs_*.npz"):
    z = np.load(f)
    if not np.isnan(z["oof"]).any():
        name = "lora_" + os.path.basename(f)[len("lora_probs_"):-4]
        oof[name], test[name] = z["oof"], z["test"]
print("families:", sorted(k for k in oof if not k.startswith("lr_") or k in ("lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat")))

B6 = ["ft_ens", "lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat", "lr_whhlv2cat", "lr_wmed"]
new = [k for k in oof if k.startswith(("mlp_", "attn_", "lora_", "fuse_"))]
cands = {"B6": B6}
for k in new:
    cands[f"B6+{k}"] = B6 + [k]
if all(f"mlp_{t}" in oof for t in ("wlv2", "wlv3", "wvaani")):
    cands["B6 with MLP instead of LR (v2,v3,vaani)"] = ["ft_ens", "mlp_wlv2", "mlp_wlv3", "mlp_wvaani", "lr_whhlv2cat", "lr_wmed"]
    cands["B6+all 3 MLP"] = B6 + ["mlp_wlv2", "mlp_wlv3", "mlp_wvaani"]
if all(f"attn_{t}" in oof for t in ("wlv2", "wlv3", "wvaani")):
    cands["B6+all 3 attn"] = B6 + ["attn_wlv2", "attn_wlv3", "attn_wvaani"]
    cands["B6+3 attn+3 MLP"] = B6 + ["attn_wlv2", "attn_wlv3", "attn_wvaani", "mlp_wlv2", "mlp_wlv3", "mlp_wvaani"]
if len(new) > 1:
    cands["B6+all new"] = B6 + new

folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
m = pd.read_csv(f"{W}/meta.csv")
m["g"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
grp = m.set_index(["split", "Id"]).loc[[("train", i) for i in folds.Id], "g"].values
ug = np.unique(grp)
mem = {g: np.where(grp == g)[0] for g in ug}
rng = np.random.default_rng(0)
boots = [np.concatenate([mem[g] for g in rng.choice(ug, len(ug))]) for _ in range(2000)]
base = CL[np.mean([oof[f] for f in B6], 0).argmax(1)]
base_test = CL[np.mean([test[f] for f in B6], 0).argmax(1)]
rows = []
for name, fs in cands.items():
    pred = CL[np.mean([oof[f] for f in fs], 0).argmax(1)]
    dd = np.array([f1_score(y[i], pred[i], average="macro") - f1_score(y[i], base[i], average="macro") for i in boots])
    tp = CL[np.mean([test[f] for f in fs], 0).argmax(1)]
    keep = name != "B6" and dd.mean() >= 0.005 and np.mean(dd > 0) >= 0.9
    rows.append(dict(blend=name, oof=round(f1_score(y, pred, average="macro"), 4), non_twin=round(f1_score(y[~tw], pred[~tw], average="macro"), 4),
                     gain=round(dd.mean(), 4), ci=f"[{np.quantile(dd, .025):+.4f},{np.quantile(dd, .975):+.4f}]", p_gt0=round(np.mean(dd > 0), 3),
                     test_rows_diff_vs_B6=int((tp != base_test).sum()), keep=keep))
    if keep:
        fn = "sub_" + name.replace("+", "_plus_").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "") + ".csv"
        pd.DataFrame({"Id": test_ids, "emotion": tp}).to_csv(f"{R}/submissions/{fn}", index=False)
r = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(r.to_string(index=False))
r.to_csv(f"{W}/blend_new_results.csv", index=False)
