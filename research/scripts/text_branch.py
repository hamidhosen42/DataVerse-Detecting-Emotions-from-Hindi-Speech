"""Text branch: emotion from Hindi ASR transcripts (public Whisper ASR; no human labelling).

Families
  txt_tfidf : char_wb 2-5-gram TF-IDF + LR
  txt_e5    : intfloat/multilingual-e5-large mean-pooled embedding + LR
Scored on Track B (dup-group 5-fold x 3 seeds) and Track A (public-pipeline folds); then blended with the B6
families from family_probs.npz (equal weights) and compared with B6 by dup-group bootstrap.

Usage: python -I text_branch.py <research_dir> <transcripts.csv> <hf_cache>
"""
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

R, tpath, cache = sys.argv[1:4]
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
m = pd.read_csv(f"{W}/meta.csv")
m["g"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
t = pd.read_csv(tpath).fillna("")
assert (t.Id.values == m.Id.values).all() and (t.split.values == m.split.values).all()
text = t.text.values
key = m.set_index(["split", "Id"])
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")[["Id", "emotion", "fold"]]
A_idx = np.array([key.index.get_loc(("train", i)) for i in folds.Id])
yA, fA = folds.emotion.values, folds.fold.values
tr = np.where(m.split == "train")[0]
te = np.where(m.split == "test")[0]
yB, gB = m.emotion.values[tr], m.g.values[tr]

# ---- e5 embeddings
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large", cache_dir=cache)
enc = AutoModel.from_pretrained("intfloat/multilingual-e5-large", cache_dir=cache).float().to(dev).eval()
embs = []
with torch.no_grad():
    for i in range(0, len(text), 32):
        b = tok([f"query: {s}" for s in text[i:i + 32]], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
        h = enc(**b).last_hidden_state
        msk = b["attention_mask"].unsqueeze(-1).float()
        embs.append(((h * msk).sum(1) / msk.sum(1)).cpu().numpy())
E5 = np.vstack(embs).astype(np.float32)
np.save(f"{W}/emb/txt_e5.npy", E5)

MK = {
    "txt_tfidf": lambda: make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
                                       LogisticRegression(C=5.0, max_iter=5000, class_weight="balanced")),
    "txt_e5": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced")),
}
X = {"txt_tfidf": text, "txt_e5": E5}
oofA, testP = {}, {}
for name, mk in MK.items():
    Xn = X[name]
    s = []
    for seed in range(3):
        p = np.empty(len(yB), dtype=object)
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, yB, gB):
            p[b] = mk().fit(Xn[tr[a]], yB[a]).predict(Xn[tr[b]])
        s.append(f1_score(yB, p, average="macro"))
    o = np.zeros((len(yA), 4))
    for k in sorted(set(fA)):
        a, b = fA != k, fA == k
        o[b] = mk().fit(Xn[A_idx[a]], yA[a]).predict_proba(Xn[A_idx[b]])
    oofA[name] = o
    testP[name] = mk().fit(Xn[A_idx], yA).predict_proba(Xn[te])
    print(f"{name}: Track B {np.mean(s):.4f} +- {np.std(s):.4f} | Track A OOF {f1_score(yA, CL[o.argmax(1)], average='macro'):.4f}", flush=True)

d = np.load(f"{W}/family_probs.npz", allow_pickle=True)
assert (d["y"] == yA).all()
B6 = ["ft_ens", "lr_wlv2cat", "lr_wlv3cat", "lr_wvaanicat", "lr_whhlv2cat", "lr_wmed"]
P6 = np.mean([d[f"oof__{f}"] for f in B6], 0)
grp = m.g.values[A_idx]
ug = np.unique(grp)
mem = {g: np.where(grp == g)[0] for g in ug}
rng = np.random.default_rng(0)
boots = [np.concatenate([mem[g] for g in rng.choice(ug, len(ug))]) for _ in range(2000)]
base = CL[P6.argmax(1)]
tw = d["tw"]
for name, members in {"B6 + txt_tfidf": ["txt_tfidf"], "B6 + txt_e5": ["txt_e5"], "B6 + both text": ["txt_tfidf", "txt_e5"]}.items():
    P = np.mean([d[f"oof__{f}"] for f in B6] + [oofA[x] for x in members], 0)
    pred = CL[P.argmax(1)]
    dd = np.array([f1_score(yA[i], pred[i], average="macro") - f1_score(yA[i], base[i], average="macro") for i in boots])
    print(f"{name:16s} OOF {f1_score(yA, pred, average='macro'):.4f} (non-twin {f1_score(yA[~tw], pred[~tw], average='macro'):.4f}) | "
          f"delta vs B6 {dd.mean():+.4f} CI [{np.quantile(dd, .025):+.4f},{np.quantile(dd, .975):+.4f}] P>0 {np.mean(dd > 0):.3f}", flush=True)
np.savez(f"{W}/text_probs.npz", **{f"oof__{k}": v for k, v in oofA.items()}, **{f"test__{k}": v for k, v in testP.items()})
