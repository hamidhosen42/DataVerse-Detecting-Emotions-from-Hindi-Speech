"""Local, real-data evaluation of the E11 package's experiments (DataVerse_Score_Improvement_Package.zip, third-party).

The package's notebook is not executed; its recipes are re-implemented here from its source (cells 8-14):
  features  legacy = layers 24+32 [mean,std] (medium: mean of 22-24); layer = 16+24+32 (medium: 20+24); late = 28+32 (medium: 24)
  models    lr_base LR(C=0.05, balanced) | lr_strong LR(C=0.3) | svm64 PCA64+RBF-SVC(C=3, probability) | knn PCA64+cosine 9-NN
  variants  E11 fast: wlv2 layer/late/lrC/svm/knn, wmed layer/svm; E11 full adds wlv3/wvaani/whhlv2 layer + svm
Features = research/work/emb/<tag>.npy (same recipe, bit-identical to fresh extraction - research/work/b9_frozen_parity.log).
Every scaler/PCA/model is fitted inside the training folds (canonical 5 folds, 824 rows).

Candidates vs B8 (= lora8 + 5 frozen LR families, equal mean; the submitted 0.94230 file):
  R  = B8 with the variant's legacy family replaced by the variant (equal weights)
  W  = E11's own rule: (1-w)*B8 + w*variant, w in {0.15, 0.30, 0.50} (weight picked on OOF -> optimistic)
Decision (same as B9/B10): gain >= +0.005 and P(gain>0) >= 0.9 (paired dup-group bootstrap, 2000 draws) and >= 1 test row
changes; White's Reality Check over ALL candidates evaluated here; Track B (dup-group-disjoint CV) for each variant vs legacy.
Usage: python -I e11_local_eval.py <research_dir>
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
R = sys.argv[1]
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
d = np.load(f"{W}/family_probs.npz", allow_pickle=True)
y, fold, tw, test_ids = d["y"], d["fold"], d["tw"].astype(bool), d["test_ids"]
yi = pd.Categorical(y, categories=CL).codes
oof = {k[5:]: d[k] for k in d.files if k.startswith("oof__")}
test = {k[6:]: d[k] for k in d.files if k.startswith("test__")}
c = np.load(f"{W}/b9_candidate_probs.npz")
oof["lora8"], test["lora8"] = c["oof__lora8"], c["test__lora8"]
meta = pd.read_csv(f"{W}/meta.csv")
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
A = meta.set_index(["split", "Id"]).index.get_indexer([("train", i) for i in folds.Id])          # research OOF row order
TE = meta.set_index(["split", "Id"]).index.get_indexer([("test", int(i)) for i in test_ids])
assert (fold == folds.fold.values).all() and (A >= 0).all() and (TE >= 0).all()
grp_all = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
grp = grp_all[A]
FAM = {"wlv2": "lr_wlv2cat", "wlv3": "lr_wlv3cat", "wvaani": "lr_wvaanicat", "whhlv2": "lr_whhlv2cat", "wmed": "lr_wmed"}
B8 = ["lora8"] + list(FAM.values())
tl = pd.read_csv(f"{W}/test_twin_labels.csv").set_index("Id").loc[test_ids, "twin_label"]


def feats(tag, recipe):
    E = np.load(f"{W}/emb/{tag}.npy", mmap_mode="r")
    g = lambda L: np.asarray(E[:, L]).astype(np.float32)
    if tag == "wmed":
        return {"legacy": lambda: np.asarray(E[:, 22:25]).astype(np.float32).mean(1), "layer": lambda: np.hstack([g(20), g(24)]), "late": lambda: g(24)}[recipe]()
    return {"legacy": lambda: np.hstack([g(24), g(32)]), "layer": lambda: np.hstack([g(16), g(24), g(32)]), "late": lambda: np.hstack([g(28), g(32)])}[recipe]()


def est(name):
    if name == "lr_base":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=4000))
    if name == "lr_strong":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=4000))
    if name == "svm64":
        return make_pipeline(StandardScaler(), PCA(n_components=64, svd_solver="randomized", random_state=42),
                             SVC(C=3.0, gamma="scale", probability=True, class_weight="balanced", random_state=42))
    if name == "knn":
        return make_pipeline(StandardScaler(), PCA(n_components=64, svd_solver="randomized", random_state=42), Normalizer(),
                             KNeighborsClassifier(n_neighbors=9, weights="distance", metric="cosine"))


def cv(X, method, f):
    o = np.zeros((len(A), 4))
    for k in range(5):
        a, b = f != k, f == k
        m = est(method).fit(X[A][a], yi[a])
        assert (m.classes_ == np.arange(4)).all()
        o[b] = m.predict_proba(X[A][b])
    return o, est(method).fit(X[A], yi).predict_proba(X[TE])


def mf1(t, p):
    cm = np.bincount(4 * t + p, minlength=16).reshape(4, 4)
    return float(np.mean(2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)))


ug = np.unique(grp)
mem = {g_: np.where(grp == g_)[0] for g_ in ug}
rng = np.random.default_rng(0)
BOOT = [np.concatenate([mem[g_] for g_ in rng.choice(ug, len(ug))]) for _ in range(2000)]
trackB = []
for s in range(3):
    f = np.zeros(len(A), int)
    for k, (_, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(A, yi, grp)):
        f[b] = k
    trackB.append(f)

# sanity: re-implemented legacy LR == the research family used in B8
X = feats("wlv2", "legacy")
o, t = cv(X, "lr_base", fold)
assert np.abs(o - oof["lr_wlv2cat"]).max() < 1e-6 and np.abs(t - test["lr_wlv2cat"]).max() < 1e-6, "legacy LR does not reproduce B8 family"
print("sanity: re-implemented legacy wlv2 LR reproduces B8's family exactly (OOF and test)", flush=True)

VARIANTS = [("wlv2", "layer", "lr_base"), ("wlv2", "late", "lr_base"), ("wlv2", "legacy", "lr_strong"), ("wlv2", "legacy", "svm64"),
            ("wlv2", "legacy", "knn"), ("wmed", "layer", "lr_base"), ("wmed", "legacy", "svm64")]
for t_ in ("wlv3", "wvaani", "whhlv2"):
    VARIANTS += [(t_, "layer", "lr_base"), (t_, "legacy", "svm64")]
p_b8 = np.mean([oof[f] for f in B8], 0)
t_b8 = np.mean([test[f] for f in B8], 0)
pred_b8 = p_b8.argmax(1)


def tlabels(P):
    return np.where(tl.notna().values, tl.values, CL[P.argmax(1)])


lab_b8 = tlabels(t_b8)
rows, cand_preds = [], {}
for tag, recipe, method in VARIANTS:
    t0 = time.time()
    X = feats(tag, recipe)
    o, t = cv(X, method, fold)
    tb = np.mean([mf1(yi, cv(X, method, f)[0].argmax(1)) for f in trackB])
    leg = FAM[tag]
    name = f"{tag}_{recipe}_{method}"
    base_tb = {"wlv2": 0.9000, "wlv3": 0.8948, "wvaani": 0.8917, "whhlv2": 0.8839, "wmed": 0.8478}[tag]
    cands = {f"R:{name}": (np.mean([o if f == leg else oof[f] for f in B8], 0), np.mean([t if f == leg else test[f] for f in B8], 0))}
    for w in (0.15, 0.30, 0.50):
        cands[f"W{w:.2f}:{name}"] = ((1 - w) * p_b8 + w * o, (1 - w) * t_b8 + w * t)
    for cn, (po, pt) in cands.items():
        p = po.argmax(1)
        dd = np.array([mf1(yi[i], p[i]) - mf1(yi[i], pred_b8[i]) for i in BOOT])
        diff = int((tlabels(pt) != lab_b8).sum())
        cand_preds[cn] = p
        rows.append(dict(candidate=cn, variant_oof=round(mf1(yi, o.argmax(1)), 4), legacy_oof=round(mf1(yi, oof[leg].argmax(1)), 4),
                         variant_trackB=round(tb, 4), legacy_trackB=base_tb, oof=round(mf1(yi, p), 4), non_twin=round(mf1(yi[~tw], p[~tw]), 4),
                         gain=round(dd.mean(), 4), ci95=f"[{np.quantile(dd, .025):+.4f},{np.quantile(dd, .975):+.4f}]", p_gt0=round(np.mean(dd > 0), 3),
                         test_rows_diff=diff, keep=bool(dd.mean() >= 0.005 and np.mean(dd > 0) >= 0.9 and diff > 0)))
    print(f"{name:26s} variant OOF {rows[-1]['variant_oof']:.4f} (legacy {rows[-1]['legacy_oof']:.4f}) | Track B {tb:.4f} (legacy {base_tb:.4f}) | {time.time() - t0:.0f}s", flush=True)
res = pd.DataFrame(rows)
names = list(cand_preds)
obs = np.array([mf1(yi, cand_preds[n]) - mf1(yi, pred_b8) for n in names])
G = np.array([[mf1(yi[i], cand_preds[n][i]) - mf1(yi[i], pred_b8[i]) for n in names] for i in BOOT[:1000]])
T = (G - G.mean(0)).max(1)
pd.set_option("display.width", 250)
print("\n== candidates vs B8 (sorted by OOF)")
print(res.sort_values("oof", ascending=False).to_string(index=False))
print(f"\nWhite's Reality Check over {len(names)} candidates: best {names[int(obs.argmax())]} {obs.max():+.4f}; null-max 95th pct {np.quantile(T, .95):+.4f}; p = {np.mean(T >= obs.max()):.3f}")
print("candidates passing the keep rule:", res[res.keep].candidate.tolist())
res.to_csv(f"{W}/e11_local_eval.csv", index=False)
