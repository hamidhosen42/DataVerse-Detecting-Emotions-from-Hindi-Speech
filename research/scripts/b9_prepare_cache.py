"""Prepare a B9-notebook cache directory from existing research artifacts, so the notebook can be executed in FULL mode
without re-training (cached reproduction). This verifies the notebook's data handling, fold mapping, frozen-family LR
fitting, ensembling, twin safety net and submission assertions against the known B8 / candidate outputs.

What is linked/converted:
  feat_<tag>.npy             <- research/work/emb/<tag>.npy           (frozen Whisper features, identical format/order)
  dup_groups.csv             <- research/work/dup_groups.csv          (near-duplicate groups, meta order)
  ft_<recipe>/fold<k>.npz    <- per-fold OOF rows of the chosen fine-tuned run, re-ordered to the notebook's row order;
                                test = that run's fold-mean test probabilities (fold-mean of identical arrays = same mean)

Usage: python -I b9_prepare_cache.py <repo_root> <cache_dir> <recipe> <probs_npz> <oof_key> <test_key>
  e.g. lora8  research/work/lora_probs_wlv2.npz        oof  test
       lpft30 research/work/lpft_probs_wlv2_30s.npz    oof_ft test_ft
"""
import os
import sys

import numpy as np
import pandas as pd

P, C, recipe, probs, ok, tk = sys.argv[1:7]
W = f"{P}/research/work"
os.makedirs(C, exist_ok=True)
for tag in ("wlv2", "wlv3", "wvaani", "whhlv2", "wmed"):
    dst = f"{C}/feat_{tag}.npy"
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(f"{W}/emb/{tag}.npy", dst)
pd.read_csv(f"{W}/dup_groups.csv")[["Id", "split", "dup_group"]].to_csv(f"{C}/dup_groups.csv", index=False)

meta = pd.concat([pd.read_csv(f"{P}/data/SRCASW-BhavVaani/train.csv").assign(split="train"),
                  pd.read_csv(f"{P}/data/SRCASW-BhavVaani/test.csv").assign(split="test")], ignore_index=True)
assert (meta.Id.values == pd.read_csv(f"{W}/meta.csv").Id.values).all(), "meta order differs from research meta.csv"
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
z = np.load(f"{P}/{probs}" if not probs.startswith("/") else probs)
oof, test = z[ok], z[tk]
assert oof.shape == (824, 4) and not np.isnan(oof).any() and test.shape == (209, 4)
id2row = {int(i): r for r, i in enumerate(meta.Id)}
TR = np.array(sorted(id2row[int(i)] for i in folds.Id))                     # notebook row order (sorted meta rows)
pos = {int(i): j for j, i in enumerate(folds.Id)}                           # research OOF row order (folds.Id)
fold_of = dict(zip(folds.Id.astype(int), folds.fold.astype(int)))
FOLD = np.array([fold_of[int(meta.Id[r])] for r in TR])
d = f"{C}/ft_{recipe}"
os.makedirs(d, exist_ok=True)
for k in range(5):
    rows = TR[FOLD == k]
    valid = np.stack([oof[pos[int(meta.Id[r])]] for r in rows])
    np.savez(f"{d}/fold{k}.npz", valid=valid, test=test, valid_rows=rows)
print(f"cache ready at {C}: 5 frozen feature links, dup_groups.csv, ft_{recipe}/fold0-4.npz from {probs}")
