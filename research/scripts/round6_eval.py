"""Round 6 (new architectures/backbones) evaluation vs B8, public-pipeline folds + dup-group bootstrap.
Keep rule (pre-declared): gain >= +0.005 and P(gain>0) >= 0.9.

New families (any that exist):
  lr_wlv1cat   : frozen Whisper-large (v1) layers 24+32 + LR
  lr_qwen2a*   : frozen Qwen2-Audio encoder (layers 24+32, or a Track-B-chosen band if given) + LR
  lpft_wlv2    : LP-FT Whisper-large-v2 (fine-tuned stage)
Candidates: B8 + each family; B8 with lpft replacing lora; B8 + all new.

Usage: python -I round6_eval.py <research_dir>
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R = sys.argv[1]
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
d = np.load(f"{W}/family_probs.npz", allow_pickle=True)
y, fold, tw, test_ids = d["y"], d["fold"], d["tw"], d["test_ids"]
oof = {k[5:]: d[k] for k in d.files if k.startswith("oof__")}
test = {k[6:]: d[k] for k in d.files if k.startswith("test__")}
z = np.load(f"{W}/lora_probs_wlv2.npz")
oof["lora_wlv2"], test["lora_wlv2"] = z["oof"], z["test"]
m = pd.read_csv(f"{W}/meta.csv")
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
A = m.set_index(["split", "Id"]).index.get_indexer([("train", i) for i in folds.Id])
TE = np.where(m.split == "test")[0]
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))
new = []
for tag in ("wlv1", "qwen2a"):
    f = f"{W}/emb/{tag}.npy"
    if os.path.exists(f):
        E = np.load(f).astype(np.float32)
        X = np.hstack([E[:, 24], E[:, 32]])
        o = np.zeros((len(A), 4))
        for k in range(5):
            a, b = fold != k, fold == k
            o[b] = mk().fit(X[A][a], y[a]).predict_proba(X[A][b])
        name = f"lr_{tag}cat"
        oof[name], test[name] = o, mk().fit(X[A], y).predict_proba(X[TE])
        new.append(name)
f = f"{W}/lpft_probs_wlv2.npz"
if os.path.exists(f):
    z = np.load(f)
    if int(z["folds_done"]) == 5:
        oof["lpft_wlv2"], test["lpft_wlv2"] = z["oof_ft"], z["test_ft"]
        oof["lp_wlv2"], test["lp_wlv2"] = z["oof_lp"], z["test_lp"]
        new.append("lpft_wlv2")
for n in new + (["lp_wlv2"] if "lp_wlv2" in oof else []):
    print(f"{n:14s} alone OOF {f1_score(y, CL[oof[n].argmax(1)], average='macro'):.4f} (non-twin {f1_score(y[~tw], CL[oof[n][~tw].argmax(1)], average='macro'):.4f})")
W5 = ["lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat", "lr_whhlv2cat", "lr_wmed"]
B8 = ["lora_wlv2"] + W5
cands = {"B8": B8}
for n in new:
    cands[f"B8+{n}"] = B8 + [n]
if "lpft_wlv2" in oof:
    cands["B8 with LP-FT instead of LoRA"] = ["lpft_wlv2"] + W5
if len(new) > 1:
    cands["B8+all new"] = B8 + new
mg = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values[A]
ug = np.unique(mg)
mem = {g: np.where(mg == g)[0] for g in ug}
rng = np.random.default_rng(0)
boots = [np.concatenate([mem[g] for g in rng.choice(ug, len(ug))]) for _ in range(2000)]
base = CL[np.mean([oof[f] for f in B8], 0).argmax(1)]
bt = CL[np.mean([test[f] for f in B8], 0).argmax(1)]
tl = pd.read_csv(f"{W}/test_twin_labels.csv").set_index("Id").loc[test_ids, "twin_label"]
os.makedirs(f"{R}/submissions", exist_ok=True)
for n, fs in cands.items():
    p = CL[np.mean([oof[f] for f in fs], 0).argmax(1)]
    dd = np.array([f1_score(y[i], p[i], average="macro") - f1_score(y[i], base[i], average="macro") for i in boots])
    tp = CL[np.mean([test[f] for f in fs], 0).argmax(1)]
    tp = np.where(tl.notna().values, tl.values, tp)
    keep = n != "B8" and dd.mean() >= 0.005 and np.mean(dd > 0) >= 0.9
    print(f"{n:32s} OOF {f1_score(y, p, average='macro'):.4f} non-twin {f1_score(y[~tw], p[~tw], average='macro'):.4f} | vs B8 {dd.mean():+.4f} "
          f"[{np.quantile(dd, .025):+.4f},{np.quantile(dd, .975):+.4f}] P>0 {np.mean(dd > 0):.3f} | test diff {int((tp != bt).sum())} | KEEP={keep}", flush=True)
    if n != "B8":
        fn = "sub_R6_" + n.replace("B8+", "B8_plus_").replace(" ", "_") + ".csv"
        pd.DataFrame({"Id": test_ids, "emotion": tp}).to_csv(f"{R}/submissions/{fn}", index=False)
np.savez(f"{W}/round6_probs.npz", **{f"oof__{k}": oof[k] for k in new}, **{f"test__{k}": test[k] for k in new})
