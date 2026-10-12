"""Round 5 evaluation vs B8 (pre-declared candidates, keep rule: gain >= +0.005 and P(gain>0) >= 0.9).
Candidates: B8 + lora_wvaani; B8 + lr_wlv2_v2 (4 kHz low-pass Whisper-v2 family); B8 + both.
Writes research/submissions/sub_<name>.csv for every candidate (kept or not) for inspection; prints test-row diffs.

Usage: python -I round5_eval.py <research_dir>
"""
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
for tag in ("wlv2", "wvaani"):
    z = np.load(f"{W}/lora_probs_{tag}.npz")
    assert not np.isnan(z["oof"]).any(), f"lora {tag} incomplete"
    oof[f"lora_{tag}"], test[f"lora_{tag}"] = z["oof"], z["test"]
m = pd.read_csv(f"{W}/meta.csv")
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
A = m.set_index(["split", "Id"]).index.get_indexer([("train", i) for i in folds.Id])
TE = np.where(m.split == "test")[0]
E = np.load(f"{W}/emb/wlv2_v2.npy").astype(np.float32)
X = np.hstack([E[:, 24], E[:, 32]])
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))
o = np.zeros((len(A), 4))
for k in range(5):
    a, b = fold != k, fold == k
    o[b] = mk().fit(X[A][a], y[a]).predict_proba(X[A][b])
oof["lr_wlv2_v2"], test["lr_wlv2_v2"] = o, mk().fit(X[A], y).predict_proba(X[TE])
print(f"lora_wvaani alone OOF {f1_score(y, CL[oof['lora_wvaani'].argmax(1)], average='macro'):.4f} | lora_wlv2 {f1_score(y, CL[oof['lora_wlv2'].argmax(1)], average='macro'):.4f}")
cc = lambda a, b: np.corrcoef(CL[oof[a].argmax(1)] == y, CL[oof[b].argmax(1)] == y)[0, 1]
print(f"correctness corr: lora_wvaani~lora_wlv2 {cc('lora_wvaani', 'lora_wlv2'):.3f} | lora_wvaani~lr_wvaanicat {cc('lora_wvaani', 'lr_wvaanicat'):.3f}")
W5 = ["lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat", "lr_whhlv2cat", "lr_wmed"]
cands = {"B8": ["lora_wlv2"] + W5, "B9a: B8+LoRA-vaani": ["lora_wlv2", "lora_wvaani"] + W5, "B9b: B8+lowpass": ["lora_wlv2", "lr_wlv2_v2"] + W5,
         "B9c: B8+both": ["lora_wlv2", "lora_wvaani", "lr_wlv2_v2"] + W5}
mg = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values[A]
ug = np.unique(mg)
mem = {g: np.where(mg == g)[0] for g in ug}
rng = np.random.default_rng(0)
boots = [np.concatenate([mem[g] for g in rng.choice(ug, len(ug))]) for _ in range(2000)]
base = CL[np.mean([oof[f] for f in cands["B8"]], 0).argmax(1)]
bt = CL[np.mean([test[f] for f in cands["B8"]], 0).argmax(1)]
tl = pd.read_csv(f"{W}/test_twin_labels.csv").set_index("Id").loc[test_ids, "twin_label"]
for n, fs in cands.items():
    p = CL[np.mean([oof[f] for f in fs], 0).argmax(1)]
    dd = np.array([f1_score(y[i], p[i], average="macro") - f1_score(y[i], base[i], average="macro") for i in boots])
    tp = CL[np.mean([test[f] for f in fs], 0).argmax(1)]
    tp = np.where(tl.notna().values, tl.values, tp)
    keep = n != "B8" and dd.mean() >= 0.005 and np.mean(dd > 0) >= 0.9
    print(f"{n:22s} OOF {f1_score(y, p, average='macro'):.4f} non-twin {f1_score(y[~tw], p[~tw], average='macro'):.4f} | vs B8 {dd.mean():+.4f} "
          f"[{np.quantile(dd, .025):+.4f},{np.quantile(dd, .975):+.4f}] P>0 {np.mean(dd > 0):.3f} | test rows diff vs B8 {int((tp != bt).sum())} | KEEP={keep}")
    if n != "B8":
        pd.DataFrame({"Id": test_ids, "emotion": tp}).to_csv(f"{R}/submissions/sub_{n.split(':')[0]}.csv", index=False)
np.savez(f"{W}/round5_probs.npz", oof_lowpass=oof["lr_wlv2_v2"], test_lowpass=test["lr_wlv2_v2"])
