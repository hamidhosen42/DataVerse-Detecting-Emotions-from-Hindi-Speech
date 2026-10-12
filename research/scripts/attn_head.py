"""Phase 2: attentive-statistics-pooling head on frozen frame-level Whisper features (layers 24+32, 25 Hz).

Model: per-frame standardisation (train-fold stats) -> Linear(2D,256)+GELU+Dropout -> attention scores
(Linear 256->128, tanh, ->1, masked softmax) -> weighted mean & std (512) -> Dropout -> Linear(512,4).
Training: AdamW lr 1e-3 wd 0.05, 30 epochs, batch 32, label smoothing 0.1, random 20% frame drop; 3 seeds averaged.
Track B (dup-group 5-fold x 3 seeds) and Track A (public-pipeline folds) + test probabilities -> work/attn_probs_<tag>.npz

Usage: python -I attn_head.py <research_dir> <tag>
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

R, tag = sys.argv[1], sys.argv[2]
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
torch.set_num_threads(8)
m = pd.read_csv(f"{W}/meta.csv")
m["g"] = pd.read_csv(f"{W}/dup_groups.csv").dup_group.values
lab = pd.Categorical(m.emotion, categories=CL).codes
F = np.load(f"{W}/frames/{tag}_frames.npy")
off = np.load(f"{W}/frames/{tag}_offsets.npy")
clips = [F[off[i]:off[i + 1]] for i in range(len(off) - 1)]
key = m.set_index(["split", "Id"])
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
A_idx = np.array([key.index.get_loc(("train", i)) for i in folds.Id])
fA = folds.fold.values
tr = np.where(m.split == "train")[0]
te = np.where(m.split == "test")[0]
B_splits = [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(tr, lab[tr], m.g.values[tr])) for s in range(3)]
A_splits = [(np.where(fA != k)[0], np.where(fA == k)[0]) for k in range(5)]


class AttnHead(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Dropout(0.3))
        self.att = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.out = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 4))

    def forward(self, x, mask):                 # x (B,T,D), mask (B,T) bool
        h = self.proj(x)
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1).unsqueeze(-1)
        mu = (a * h).sum(1)
        sd = ((a * (h - mu[:, None]) ** 2).sum(1) + 1e-5).sqrt()
        return self.out(torch.cat([mu, sd], -1))


def batchify(idx, mu, sd, drop=0.0, rng=None):
    xs = []
    for i in idx:
        c = clips[i]
        if drop and len(c) > 4:
            c = c[rng.random(len(c)) > drop]
        xs.append(c)
    T = max(len(c) for c in xs)
    X = np.zeros((len(xs), T, F.shape[1]), np.float32)
    M = np.zeros((len(xs), T), bool)
    for j, c in enumerate(xs):
        X[j, :len(c)] = (c.astype(np.float32) - mu) / sd
        M[j, :len(c)] = True
    return torch.tensor(X), torch.tensor(M)


def train_eval(train_rows, eval_sets, seed, epochs=30):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allf = np.concatenate([clips[i] for i in train_rows]).astype(np.float32)
    mu, sd = allf.mean(0), allf.std(0) + 1e-5
    net = AttnHead(F.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    y = torch.tensor(lab[train_rows], dtype=torch.long)
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(len(train_rows))
        for s in range(0, len(perm), 32):
            b = perm[s:s + 32]
            X, M = batchify(train_rows[b], mu, sd, 0.2, rng)
            opt.zero_grad()
            lossf(net(X, M), y[b]).backward()
            opt.step()
    net.eval()
    res = []
    with torch.no_grad():
        for rows in eval_sets:
            ps = []
            for s in range(0, len(rows), 64):
                X, M = batchify(rows[s:s + 64], mu, sd)
                ps.append(net(X, M).softmax(-1).numpy())
            res.append(np.vstack(ps))
    return res


def oof(rows, splits, test_rows=None):
    o = np.zeros((len(rows), 4))
    for a, b in splits:
        o[b] = np.mean([train_eval(rows[a], [rows[b]], s)[0] for s in range(3)], 0)
    t = np.mean([train_eval(rows, [test_rows], s)[0] for s in range(3)], 0) if test_rows is not None else None
    return o, t


sB = [f1_score(lab[tr], oof(tr, sp)[0].argmax(1), average="macro") for sp in B_splits]
oA, tA = oof(A_idx, A_splits, te)
print(f"attn_{tag}: Track B {np.mean(sB):.4f} +- {np.std(sB):.4f} ({', '.join(f'{v:.4f}' for v in sB)}) | Track A {f1_score(lab[A_idx], oA.argmax(1), average='macro'):.4f}", flush=True)
np.savez(f"{W}/attn_probs_{tag}.npz", **{f"oof__attn_{tag}": oA, f"test__attn_{tag}": tA})
