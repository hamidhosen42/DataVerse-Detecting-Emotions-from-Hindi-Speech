"""Multi-encoder frozen-feature models + blending with the public fine-tuned ensemble, all on the public
pipeline's own 5 folds (train_with_folds.csv), then fit-on-all-train test predictions and candidate submissions.

Frozen model = StandardScaler + LogisticRegression(class_weight=balanced) on the mean of a fixed layer band
of per-layer [mean,std] pooled features. Bands were fixed from the per-layer study (cv_*_layers.txt):
  wbp 6-8, wl 12-18, xlsr 10-18.
Blends use EQUAL weights over model families (no weight tuning on OOF).
Significance: dup-group paired bootstrap vs the fine-tuned ensemble.

Usage: python -I blend_study.py <work_dir> <out_dir_for_submissions>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W, OUT = sys.argv[1], sys.argv[2]
KD = f"{W}/kunal_train_output/dataverse_models"
CL = np.array(["angry", "happy", "neutral", "sad"])
BANDS = {"wbp": range(6, 9), "wl": range(12, 19), "xlsr": range(10, 19)}
m = pd.read_csv(f"{W}/meta.csv")
m["dup_group"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
key = m.set_index(["split", "Id"])
folds = pd.read_csv(f"{KD}/train_with_folds.csv")[["Id", "emotion", "fold"]]
y, fold = folds.emotion.values, folds.fold.values
tr_idx = np.array([key.index.get_loc(("train", i)) for i in folds.Id])
te_rows = m[m.split == "test"]
te_idx = te_rows.index.values
grp = m.dup_group.values[tr_idx]
tw = np.array([np.isin(grp[i], grp[fold != fold[i]]) for i in range(len(y))])


def kunal(sub):
    o = pd.read_csv(f"{KD}/{sub}/seed_42/oof_predictions.csv").set_index("Id").loc[folds.Id]
    return o[[f"prob_{c}" for c in CL]].values


ens = (kunal("wavlm-base-plus_bcd60f08") + kunal("wavlm-large_97ba7aec")) / 2
kt = pd.read_csv(f"{W}/kunal_output/dataverse_inference/test_probabilities.csv").set_index("Id").loc[te_rows.Id]
ens_test = kt[[f"prob_{c}" for c in CL]].values

feats = {}
for tag, band in BANDS.items():
    E = np.load(f"{W}/emb/{tag}.npy").astype(np.float32)
    feats[tag] = E[:, list(band)].mean(1)
    del E


def lr_model(C=0.05):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))


# C sanity check on dup-group folds (twin-free, so C is not chosen for memorisation)
print("C check (dup-group 5-fold, OOF macro-F1):")
sgk = StratifiedGroupKFold(5, shuffle=True, random_state=0)
for tag in BANDS:
    X = feats[tag][tr_idx]
    res = []
    for C in (0.01, 0.05, 0.2):
        p = np.empty(len(y), dtype=object)
        for a, b in sgk.split(X, y, grp):
            p[b] = lr_model(C).fit(X[a], y[a]).predict(X[b])
        res.append(f"C={C}: {f1_score(y, p, average='macro'):.4f}")
    print(f"  {tag:5s}", " | ".join(res))

oof, test = {"ft_ens": ens}, {"ft_ens": ens_test}
for tag in BANDS:
    X = feats[tag]
    o = np.zeros((len(y), 4))
    for f in sorted(set(fold)):
        a, b = fold != f, fold == f
        o[b] = lr_model().fit(X[tr_idx][a], y[a]).predict_proba(X[tr_idx][b])
    oof[f"lr_{tag}"] = o
    test[f"lr_{tag}"] = lr_model().fit(X[tr_idx], y).predict_proba(X[te_idx])
oof["lr_all3"] = np.mean([oof[f"lr_{t}"] for t in BANDS], 0)
test["lr_all3"] = np.mean([test[f"lr_{t}"] for t in BANDS], 0)
for name, parts in {"ft+lr_wbp": ["ft_ens", "lr_wbp"], "ft+lr_xlsr": ["ft_ens", "lr_xlsr"],
                    "ft+lr_all3": ["ft_ens", "lr_all3"], "ft+wbp+wl+xlsr (4 equal)": ["ft_ens", "lr_wbp", "lr_wl", "lr_xlsr"]}.items():
    oof[name] = np.mean([oof[p] for p in parts], 0)
    test[name] = np.mean([test[p] for p in parts], 0)

rng = np.random.default_rng(0)
ug = np.unique(grp)
mem = {g: np.where(grp == g)[0] for g in ug}
boots = [np.concatenate([mem[g] for g in rng.choice(ug, len(ug))]) for _ in range(2000)]
base_pred = CL[ens.argmax(1)]
print("\nOn public-pipeline folds (824 rows; twin share of held-out rows %.3f):" % tw.mean())
rows = []
for name, P in oof.items():
    pred = CL[P.argmax(1)]
    f_all = f1_score(y, pred, average="macro")
    d = np.array([f1_score(y[i], pred[i], average="macro") - f1_score(y[i], base_pred[i], average="macro") for i in boots])
    est_lb = 0.37 * f1_score(y[tw], pred[tw], average="macro") + 0.63 * f1_score(y[~tw], pred[~tw], average="macro")
    rows.append(dict(model=name, oof=round(f_all, 4), twin=round(f1_score(y[tw], pred[tw], average="macro"), 4),
                     non_twin=round(f1_score(y[~tw], pred[~tw], average="macro"), 4), est_test=round(est_lb, 4),
                     delta_vs_ft=round(f_all - f1_score(y, base_pred, average="macro"), 4),
                     ci95=f"[{np.quantile(d, .025):+.4f},{np.quantile(d, .975):+.4f}]", p_gt0=round(np.mean(d > 0), 3)))
r = pd.DataFrame(rows)
print(r.to_string(index=False))
r.to_csv(f"{W}/blend_study_results.csv", index=False)

# candidate submissions (+ twin override safety net; it changes nothing if the model already agrees)
tl = pd.read_csv(f"{W}/test_twin_labels.csv").set_index("Id").loc[te_rows.Id, "twin_label"]
for name in ("ft+lr_xlsr", "ft+lr_all3", "ft+wbp+wl+xlsr (4 equal)", "lr_all3"):
    pred = CL[test[name].argmax(1)]
    over = np.where(tl.notna().values, tl.values, pred)
    fn = name.replace("+", "_").replace(" ", "").replace("(", "").replace(")", "")
    pd.DataFrame({"Id": te_rows.Id.values, "emotion": over}).to_csv(f"{OUT}/sub_{fn}.csv", index=False)
    cur = pd.read_csv(f"{W}/kunal_output/submission.csv").set_index("Id").loc[te_rows.Id, "emotion"].values
    print(f"{name:28s}: twin overrides changed {(over != pred).sum()} rows | differs from current 0.87918 submission on {(over != cur).sum()} rows | class counts {pd.Series(over).value_counts().to_dict()}")
np.savez(f"{W}/blend_probs.npz", **{k.replace(' ', '_'): v for k, v in test.items()})
