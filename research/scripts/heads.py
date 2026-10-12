"""New heads on frozen pooled Whisper features (phase 1 of the architecture round).

Heads
  mlp      : standardized layers 24+32 [mean,std] -> MLP(256) with dropout, label smoothing; 3 inits averaged
  lw_lin   : SUPERB-style learned softmax weights over ALL layers' pooled [mean,std] -> linear classifier
  lw_mlp   : same learned layer weights -> MLP(256)
  fuse_lr  : early fusion: concat of several models' 24+32 features -> one LogisticRegression
Track B (dup-group 5-fold x 3 seeds) for selection; Track A (public-pipeline folds) OOF + test probabilities saved
to work/heads_probs.npz for blending.

Usage: python -I heads.py <research_dir> <tag> [tag ...]      (tags: wlv2 wlv3 wvaani whhlv2)
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R, TAGS = sys.argv[1], sys.argv[2:]
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
torch.set_num_threads(8)
m = pd.read_csv(f"{W}/meta.csv")
m["g"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
lab = pd.Categorical(m.emotion, categories=CL).codes
key = m.set_index(["split", "Id"])
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
A_idx = np.array([key.index.get_loc(("train", i)) for i in folds.Id])
fA = folds.fold.values
tr = np.where(m.split == "train")[0]
te = np.where(m.split == "test")[0]
B_splits = [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(tr, lab[tr], m.g.values[tr])) for s in range(3)]
A_splits = [(np.where(fA != k)[0], np.where(fA == k)[0]) for k in range(5)]


class LayerWeighted(nn.Module):
    def __init__(self, L, D, hidden):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(L))
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(D, 4) if hidden == 0 else nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Dropout(0.5), nn.Linear(hidden, 4))

    def forward(self, x):                      # x: (B, L, D)
        z = (x * self.w.softmax(0)[None, :, None]).sum(1)
        return self.head(self.drop(z))


def train_torch(Xtr, ytr, Xev_list, kind, seed, epochs=60):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-5
    f = lambda Z: torch.tensor((Z - mu) / sd, dtype=torch.float32)
    Xt, yt = f(Xtr), torch.tensor(ytr, dtype=torch.long)
    if kind == "mlp":
        net = nn.Sequential(nn.Dropout(0.2), nn.Linear(Xt.shape[1], 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 4))
    else:
        net = LayerWeighted(Xt.shape[1], Xt.shape[2], 0 if kind == "lw_lin" else 256)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(Xt), generator=g).split(64):
            opt.zero_grad()
            lossf(net(Xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return [net(f(Z)).softmax(-1).numpy() for Z in Xev_list]


def run_head(X, kind, rows, splits, test_rows=None):
    """returns OOF probs over `rows` (and test probs if test_rows given: model fit on all rows)."""
    o = np.zeros((len(rows), 4))
    for a, b in splits:
        if kind == "fuse_lr":
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced")).fit(X[rows[a]], lab[rows[a]])
            o[b] = clf.predict_proba(X[rows[b]])
        else:
            o[b] = np.mean([train_torch(X[rows[a]], lab[rows[a]], [X[rows[b]]], kind, s)[0] for s in range(3)], 0)
    t = None
    if test_rows is not None:
        if kind == "fuse_lr":
            t = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced")).fit(X[rows], lab[rows]).predict_proba(X[test_rows])
        else:
            t = np.mean([train_torch(X[rows], lab[rows], [X[test_rows]], kind, s)[0] for s in range(3)], 0)
    return o, t


out = {}
feats = {}
for tag in TAGS:
    E = np.load(f"{W}/emb/{tag}.npy").astype(np.float32)
    feats[tag] = E
    jobs = {f"mlp_{tag}": (np.hstack([E[:, 24], E[:, 32]]), "mlp"), f"lwlin_{tag}": (E, "lw_lin"), f"lwmlp_{tag}": (E, "lw_mlp")}
    for name, (X, kind) in jobs.items():
        sB = [f1_score(lab[tr], run_head(X, kind, tr, sp)[0].argmax(1), average="macro") for sp in B_splits]
        oA, tA = run_head(X, kind, A_idx, A_splits, te)
        out[f"oof__{name}"], out[f"test__{name}"] = oA, tA
        print(f"{name:16s} Track B {np.mean(sB):.4f} +- {np.std(sB):.4f} | Track A {f1_score(lab[A_idx], oA.argmax(1), average='macro'):.4f}", flush=True)
if len(TAGS) > 1:
    X = np.hstack([np.hstack([feats[t][:, 24], feats[t][:, 32]]) for t in TAGS])
    sB = [f1_score(lab[tr], run_head(X, "fuse_lr", tr, sp)[0].argmax(1), average="macro") for sp in B_splits]
    oA, tA = run_head(X, "fuse_lr", A_idx, A_splits, te)
    out["oof__fuse_lr"], out["test__fuse_lr"] = oA, tA
    print(f"{'fuse_lr(' + '+'.join(TAGS) + ')':16s} Track B {np.mean(sB):.4f} +- {np.std(sB):.4f} | Track A {f1_score(lab[A_idx], oA.argmax(1), average='macro'):.4f}", flush=True)
np.savez(f"{W}/heads_probs.npz", **out)
