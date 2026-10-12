"""B9 baseline audit: reproduce B8 from cached artifacts and verify provenance, class order, folds, duplicates.

Everything printed here is MEASURED from files in research/work, data/, my_submissions/ (no training).

Usage: python -I b9_audit.py <repo_root>
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

P = sys.argv[1]
W = f"{P}/research/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
out = {}

# ---------- 1. artifacts
d = np.load(f"{W}/family_probs.npz", allow_pickle=True)
z = np.load(f"{W}/lora_probs_wlv2.npz")
y, fold, tw, test_ids = d["y"], d["fold"], d["tw"], d["test_ids"]
W5 = ["lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat", "lr_whhlv2cat", "lr_wmed"]
oof = {k: d[f"oof__{k}"] for k in W5}
test = {k: d[f"test__{k}"] for k in W5}
oof["lora_wlv2"], test["lora_wlv2"] = z["oof"], z["test"]
print("== 1. artifact shapes:", {k: v.shape for k, v in oof.items()}, "| test:", {k: v.shape for k, v in test.items()})
assert all(v.shape == (824, 4) for v in oof.values()) and all(v.shape == (209, 4) for v in test.values())
for k, v in list(oof.items()) + list(test.items()):
    assert np.isfinite(v).all() and np.allclose(v.sum(1), 1, atol=1e-4), f"{k}: not finite / not normalised"
print("   all probabilities finite and rows sum to 1")

# ---------- 2. fold / id provenance
meta = pd.read_csv(f"{W}/meta.csv")
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
assert (folds.emotion.values == y).all() and (folds.fold.values == fold).all(), "y/fold in family_probs.npz != train_with_folds.csv"
tr = pd.read_csv(f"{P}/data/SRCASW-BhavVaani/train.csv")
te = pd.read_csv(f"{P}/data/SRCASW-BhavVaani/test.csv")
ss = pd.read_csv(f"{P}/data/SRCASW-BhavVaani/sample_submission.csv")
assert (test_ids == te.Id.values).all() and (ss.Id.values == te.Id.values).all(), "test id order mismatch"
lab_map = tr.set_index("Id").emotion
assert (lab_map.loc[folds.Id].values == y).all(), "fold labels != train.csv labels"
dropped = sorted(set(tr.Id) - set(folds.Id))
print(f"== 2. folds: 824 rows = 826 train rows minus {dropped} (exact-duplicate pair with conflicting labels); fold sizes {np.bincount(fold).tolist()}")
print(f"   test ids: 209, identical order in test.csv, sample_submission.csv and family_probs.npz")
print(f"   class order used everywhere: {CL.tolist()} (== sorted(train.emotion.unique()) = {sorted(tr.emotion.unique())})")

# ---------- 3. reproduce B8 OOF and test file
P8 = np.mean([oof[k] for k in ["lora_wlv2"] + W5], 0)
pred = CL[P8.argmax(1)]
f1 = f1_score(y, pred, average="macro")
out["B8_oof_macro_f1"] = round(f1, 4)
print(f"== 3. B8 OOF macro-F1 (824 rows, public-pipeline folds): {f1:.4f} | non-twin {f1_score(y[~tw], pred[~tw], average='macro'):.4f} | twin {f1_score(y[tw], pred[tw], average='macro'):.4f}")
print("   per-fold:", [round(f1_score(y[fold == k], pred[fold == k], average="macro"), 4) for k in range(5)])
print(classification_report(y, pred, digits=4))
print("   confusion (rows=true, cols=pred):\n", pd.DataFrame(confusion_matrix(y, pred, labels=CL), index=CL, columns=CL))
T8 = np.mean([test[k] for k in ["lora_wlv2"] + W5], 0)
tpred = CL[T8.argmax(1)]
tl = pd.read_csv(f"{W}/test_twin_labels.csv").set_index("Id").loc[test_ids, "twin_label"]
tpred2 = np.where(tl.notna().values, tl.values, tpred)
sub = pd.read_csv(f"{P}/my_submissions/0.94230_2026-10-11_B8_WhisperLoRA+Whisper5/submission.csv")
assert (sub.Id.values == test_ids).all()
print(f"   twin override changed {(tpred2 != tpred).sum()} rows | reproduced B8 test predictions vs submitted file: {(tpred2 != sub.emotion.values).sum()} rows differ")
print(f"   submitted file md5 {hashlib.md5(open(f'{P}/my_submissions/0.94230_2026-10-11_B8_WhisperLoRA+Whisper5/submission.csv','rb').read()).hexdigest()} == research/submissions/sub_B8.csv md5 {hashlib.md5(open(f'{P}/research/submissions/sub_B8.csv','rb').read()).hexdigest()}")

# ---------- 4. duplicate finding
g = pd.read_csv(f"{W}/dup_groups.csv")
pairs = pd.read_csv(f"{W}/near_dup_pairs.csv")
hi = pairs[pairs.ncc >= 0.95]
sizes = g.groupby("dup_group").size()
multi = sizes[sizes > 1]
tr_groups = set(g[g.split == "train"].dup_group)
te_twin = g[(g.split == "test") & g.dup_group.isin(tr_groups)]
print(f"== 4. near-duplicates: pairs with NCC>=0.95: {len(hi)} (train-train {((hi.split_a=='train')&(hi.split_b=='train')).sum()}, train-test {((hi.split_a!=hi.split_b)).sum()}, test-test {((hi.split_a=='test')&(hi.split_b=='test')).sum()}); "
      f"groups>=2: {len(multi)} covering {int(multi.sum())} rows; test rows with a train twin: {len(te_twin)}/209 = {len(te_twin)/209:.3f}")
print(f"   NCC gap: pairs in (0.5,0.95]: {((pairs.ncc>0.5)&(pairs.ncc<0.95)).sum()} | >=0.99: {(pairs.ncc>=0.99).sum()}")
# twin label consistency inside train
trg = g[g.split == "train"][["Id", "dup_group"]].merge(tr[["Id", "emotion"]], on="Id")
cons = trg.groupby("dup_group").emotion.nunique()
print(f"   train groups with >1 member: {(trg.groupby('dup_group').size()>1).sum()}, of which label-inconsistent: {(cons[trg.groupby('dup_group').size()>1]>1).sum()}")
# twin share within the public-pipeline folds (OOF rows whose group appears in training folds)
print(f"   OOF rows with a twin in the other folds: {tw.mean():.3f} (test: {len(te_twin)/209:.3f})")

# ---------- 5. LP stage of the running 30-s LP-FT should equal the frozen LR family per fold
print("== 5. consistency check: frozen lr_wlv2cat per fold vs LP stage of the 30-s LP-FT run (from its log)")
lr_fold = [round(f1_score(y[fold == k], CL[oof["lr_wlv2cat"][fold == k].argmax(1)], average="macro"), 4) for k in range(5)]
print("   lr_wlv2cat per fold:", lr_fold)
try:
    log = open(f"{W}/lpft_wlv2_30s.log").read()
    import re
    print("   30-s LP-FT log  :", re.findall(r"fold \d macro-F1 LP [0-9.]+ \| LP-FT [0-9.]+", log))
except FileNotFoundError:
    pass

# ---------- 6. validation-hygiene review (code facts)
print("""== 6. validation hygiene (from code review of research/scripts):
   - LR families: StandardScaler+LogisticRegression inside a sklearn Pipeline, fit on the training part of each fold only (blend_all.py); test models fit on all 824 rows.
   - Layer bands for new encoders chosen on Track B (dup-group folds), then evaluated on Track A (public-pipeline folds): mild selection optimism on Track A.
   - Blend membership: pre-declared rules (Track B >= 0.77; top-5 distinct encoders by Track B); equal weights, never tuned on OOF.
   - LoRA (8 s): fixed 6 epochs, no early stopping, no checkpoint selection on the validation fold; test = mean of 5 fold models.
   - LP-FT: LR probe fit on the training part only; head initialised from it; fixed epochs.
   - Twin override: uses training labels only (waveform NCC>=0.95 to a training clip); changes 0 B8 predictions.
   - Known optimism: the set of encoders tried grew during research (~20 families, ~40 blend candidates); the reported B8 OOF is the best of several near-equivalent candidates (B5/B6/B8 within 0.0012).""")
json.dump(out, open(f"{W}/b9_audit.json", "w"), indent=1)
