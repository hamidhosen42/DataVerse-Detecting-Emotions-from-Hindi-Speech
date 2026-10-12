"""Evaluate every available model family and pre-declared blends on the public pipeline's folds (Track A),
with dup-group bootstrap vs the public fine-tuned ensemble, and write candidate submissions.

Families
  ft_ens            : public WavLM Base+ & Large fine-tunes (OOF + test probabilities from their outputs)
  ft_<name>         : any local fine-tune in research/local_ft/<name>_models (+ <name>_inference/test_probabilities.csv)
  lr_<tag>          : StandardScaler + LR(C=0.05, balanced) on the mean of a layer band of research/work/emb/<tag>.npy
Layer bands for new encoders are chosen on Track B (dup-group folds), NOT on Track A, to limit selection bias.

Pre-declared blends (equal weights)
  B0  ft_ens + lr_wbp + lr_wl + lr_xlsr                         (previous best, X10)
  B1  all FT families + every LR family whose Track-B macro-F1 >= QUAL (0.77)
  B2  B1 without FT families (frozen-only hedge)
Greedy forward selection is also reported, flagged OPTIMISTIC (selection on the evaluation folds).

Usage: python -I blend_all.py <research_dir> [QUAL]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R = sys.argv[1]
QUAL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.77
W = f"{R}/work"
KD = f"{W}/kunal_train_output/dataverse_models"
CL = np.array(["angry", "happy", "neutral", "sad"])
FIXED = {"wbp": range(6, 9), "wl": range(12, 19), "xlsr": range(10, 19),
         # bands below were selected on Track B in blend_all_final.txt (2026-10-10) and are now pinned for speed
         "hilarge": range(16, 21), "urdu": range(8, 13), "vakh": range(6, 11), "w2vbert": range(12, 17), "whhlv2": range(26, 31),
         "wlv2": range(30, 33), "wlv3": range(30, 33), "wmed": range(22, 25), "wser": range(24, 29), "wturbo": range(20, 25),
         "wvaani": range(30, 33), "xlsr1b": range(30, 35)}
m = pd.read_csv(f"{W}/meta.csv")
m["dup_group"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
key = m.set_index(["split", "Id"])
folds = pd.read_csv(f"{KD}/train_with_folds.csv")[["Id", "emotion", "fold"]]
y, fold = folds.emotion.values, folds.fold.values
tr_idx = np.array([key.index.get_loc(("train", i)) for i in folds.Id])
te = m[m.split == "test"]
te_idx = te.index.values
grp = m.dup_group.values[tr_idx]
tw = np.array([np.isin(grp[i], grp[fold != fold[i]]) for i in range(len(y))])
all_tr = np.where(m.split == "train")[0]
y_all, g_all = m.emotion.values[all_tr], m.dup_group.values[all_tr]


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000, class_weight="balanced"))


def trackb(X):
    """dup-group 5-fold OOF macro-F1 on all 826 train rows (seed 0)."""
    p = np.empty(len(y_all), dtype=object)
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y_all, g_all):
        p[b] = lr().fit(X[a], y_all[a]).predict(X[b])
    return f1_score(y_all, p, average="macro")


oof, test, info = {}, {}, {}
# ---- fine-tuned families
o = [pd.read_csv(f"{KD}/{s}/seed_42/oof_predictions.csv").set_index("Id").loc[folds.Id] for s in ("wavlm-base-plus_bcd60f08", "wavlm-large_97ba7aec")]
oof["ft_ens"] = np.mean([x[[f"prob_{c}" for c in CL]].values for x in o], 0)
kt = pd.read_csv(f"{W}/kunal_output/dataverse_inference/test_probabilities.csv").set_index("Id").loc[te.Id]
test["ft_ens"] = kt[[f"prob_{c}" for c in CL]].values
info["ft_ens"] = "public WavLM B+/L fine-tunes"
for d in sorted(glob.glob(f"{R}/local_ft/*_models")):
    name = os.path.basename(d).replace("_models", "")
    oo = glob.glob(f"{d}/*/seed_*/oof_predictions.csv")
    tp = f"{R}/local_ft/{name}_inference/test_probabilities.csv"
    if oo and os.path.exists(tp):
        x = pd.read_csv(oo[0]).set_index("Id").loc[folds.Id]
        oof[f"ft_{name}"] = x[[f"prob_{c}" for c in CL]].values
        test[f"ft_{name}"] = pd.read_csv(tp).set_index("Id").loc[te.Id][[f"prob_{c}" for c in CL]].values
        info[f"ft_{name}"] = f"local fine-tune {name}"

# ---- frozen LR families
for f in sorted(glob.glob(f"{W}/emb/*.npy")):
    tag = os.path.basename(f)[:-4]
    if tag == "sv" or "_ext" in tag:
        continue
    E = np.load(f, mmap_mode="r")
    if E.ndim == 3:
        if tag in FIXED:
            band = list(FIXED[tag])
        else:  # choose a 5-layer band centred on the best Track-B layer (even layers scanned)
            L = E.shape[1]
            scores = {l: trackb(np.asarray(E[all_tr, l], dtype=np.float32)) for l in range(2, L, 2)}
            best = max(scores, key=scores.get)
            band = list(range(max(0, best - 2), min(L, best + 3)))
            print(f"  {tag}: per-layer Track-B {dict((k, round(v, 3)) for k, v in scores.items())} -> band {band[0]}-{band[-1]}", flush=True)
        X = np.asarray(E[:, band], dtype=np.float32).mean(1)
        desc = f"layers {band[0]}-{band[-1]}"
    else:
        X = np.asarray(E, dtype=np.float32)
        desc = "utterance features"
    X = np.nan_to_num(X)
    tb = trackb(X[all_tr])
    oo = np.zeros((len(y), 4))
    for k in sorted(set(fold)):
        a, b = fold != k, fold == k
        oo[b] = lr().fit(X[tr_idx][a], y[a]).predict_proba(X[tr_idx][b])
    oof[f"lr_{tag}"] = oo
    test[f"lr_{tag}"] = lr().fit(X[tr_idx], y).predict_proba(X[te_idx])
    info[f"lr_{tag}"] = f"{desc}; TrackB={tb:.4f}"
    print(f"  lr_{tag}: {desc} TrackB={tb:.4f}", flush=True)


def tb_of(name):
    return float(info[name].split("TrackB=")[1]) if "TrackB=" in info[name] else 1.0


fams = list(oof)
ft = [f for f in fams if f.startswith("ft_")]
qual = [f for f in fams if f.startswith("lr_") and tb_of(f) >= QUAL]
WHISPER = {"wlv2", "wlv3", "wvaani", "whhlv2", "wmed", "wser", "wturbo"}


def encoder(f):  # several feature variants of one encoder count as one encoder
    return f.replace("lr_", "").replace("cat", "")


best_per_enc = {}
for f in sorted(qual, key=tb_of, reverse=True):
    best_per_enc.setdefault(encoder(f), f)
top3 = list(best_per_enc.values())[:3]
top5 = list(best_per_enc.values())[:5]
blends = {"B0 ft_ens+wbp+wl+xlsr": ["ft_ens", "lr_wbp", "lr_wl", "lr_xlsr"], f"B1 FT+LR(TrackB>={QUAL})": ft + qual, f"B2 LR(TrackB>={QUAL}) only": qual,
          "B3 top-3 LR by TrackB": top3, "B4 FT+top-3 LR": ft + top3, "B5 top-5 encoders": top5, "B6 FT+top-5 encoders": ft + top5,
          # B7 added after the 6-split robustness check (post-hoc): B6 + best non-Whisper encoder
          "B7 B6+best non-Whisper": ft + top5 + [max([f for f in qual if encoder(f) not in WHISPER], key=tb_of)]}
# greedy forward selection (optimistic)
pool, sel, best_score = set(fams), [], -1
while pool:
    cand = {f: f1_score(y, CL[np.mean([oof[s] for s in sel + [f]], 0).argmax(1)], average="macro") for f in pool}
    f, s = max(cand.items(), key=lambda kv: kv[1])
    if s <= best_score + 1e-9:
        break
    sel.append(f); pool.remove(f); best_score = s
blends["G greedy (OPTIMISTIC)"] = sel

rng = np.random.default_rng(0)
ug = np.unique(grp)
mem = {g: np.where(grp == g)[0] for g in ug}
boots = [np.concatenate([mem[g] for g in rng.choice(ug, len(ug))]) for _ in range(2000)]
base = CL[oof["ft_ens"].argmax(1)]
rows = []
for name, P in list(oof.items()) + [(b, np.mean([oof[f] for f in fs], 0)) for b, fs in blends.items() if fs]:
    pred = CL[P.argmax(1)]
    d = np.array([f1_score(y[i], pred[i], average="macro") - f1_score(y[i], base[i], average="macro") for i in boots])
    ft_ = f1_score(y[tw], pred[tw], average="macro"); fn_ = f1_score(y[~tw], pred[~tw], average="macro")
    rows.append(dict(model=name, members="+".join(blends.get(name, [name])), oof=round(f1_score(y, pred, average="macro"), 4), twin=round(ft_, 4),
                     non_twin=round(fn_, 4), est_test=round(0.37 * ft_ + 0.63 * fn_, 4), d_vs_ft=round(np.mean(d), 4),
                     ci95=f"[{np.quantile(d, .025):+.4f},{np.quantile(d, .975):+.4f}]", p_gt0=round(np.mean(d > 0), 3), info=info.get(name, "")))
r = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(r.drop(columns=["members"]).to_string(index=False))
r.to_csv(f"{W}/blend_all_results.csv", index=False)
print("\nblend members:", json.dumps({b: fs for b, fs in blends.items()}, indent=0))

tl = pd.read_csv(f"{W}/test_twin_labels.csv").set_index("Id").loc[te.Id, "twin_label"]
cur = pd.read_csv(f"{W}/kunal_output/submission.csv").set_index("Id").loc[te.Id, "emotion"].values
os.makedirs(f"{R}/submissions", exist_ok=True)
for b, fs in blends.items():
    if not fs or b.startswith("G "):
        continue
    pred = CL[np.mean([test[f] for f in fs], 0).argmax(1)]
    pred = np.where(tl.notna().values, tl.values, pred)
    fn = "sub_" + b.split()[0] + ".csv"
    pd.DataFrame({"Id": te.Id.values, "emotion": pred}).to_csv(f"{R}/submissions/{fn}", index=False)
    print(f"{fn}: {len(fs)} families | differs from current submission on {(pred != cur).sum()} rows | counts {pd.Series(pred).value_counts().to_dict()}")

# save per-family OOF (public-pipeline folds) and test probabilities for downstream experiments (E7/E8)
np.savez(f"{W}/family_probs.npz", **{f"oof__{k}": v for k, v in oof.items()}, **{f"test__{k}": v for k, v in test.items()},
         y=y, fold=fold, tw=tw, test_ids=te.Id.values)
print("saved family_probs.npz")
