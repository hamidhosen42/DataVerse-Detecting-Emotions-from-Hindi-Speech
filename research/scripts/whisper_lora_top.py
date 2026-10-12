"""Native-30-s LoRA on the TOP layers of a Whisper encoder, trained from a cached fp32 copy of the frozen lower half.

Hypothesis (B9 queue #2): the B8 fine-tuned member (lora8: 8-s window, LoRA q/v r16, random head, lr 1e-4/1e-3,
6 epochs, OOF 0.9145) was held back by the 8-s window (frozen probe 0.908 at 8 s vs 0.936 at 30 s). The SAME recipe at
Whisper's native 30-s context should give a stronger, still-diverse member. Compute constraint: LoRA only in layers
K+1..32 (default K=16), so layers 1..K are frozen and deterministic -> cache their output once (fp32: fp16 caching
gave 2.2 % relative error at layer 32) and train only the top layers. Mathematically identical to full-model LoRA with
`layers_to_transform = K..31` (verified by --check against a full forward pass).

Pooling: masked mean+std over real-audio frames of hidden_states[L1] (output of layer L1) and hidden_states[32]
(= final layer_norm(output of layer 32)), exactly as in lora8/LP-FT. Head: Dropout(p) -> Linear(4D, 4), random init.
Training: AdamW(wd 0.01), OneCycle(max_lr=[lr_lora, lr_head], pct_start 0.1), label smoothing 0.1, effective batch 8,
fixed epochs, no early stopping, no checkpoint selection. Per-fold seed = 0 (as lora8). Gradient checkpointing per layer.
Output: work/loratop_probs_<tag>.npz with oof (824x4, public-pipeline fold order), test (mean of fold models),
folds_done; per-fold valid/test also saved so the run resumes.

Usage: python -I whisper_lora_top.py <research_dir> <hf_model> <tag> <hf_cache> <cache_dir>
         [--epochs 6] [--K 16] [--L1 24] [--lr_lora 1e-4] [--lr_head 1e-3] [--dropout 0.3] [--mb 4] [--folds 0,1,2,3,4]
         [--check] [--bench N] [--seed 0]
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from peft import LoraConfig, inject_adapter_in_model
from sklearn.metrics import f1_score
from transformers import AutoModel, WhisperFeatureExtractor

ap = argparse.ArgumentParser()
ap.add_argument("research_dir"); ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("hf_cache"); ap.add_argument("cache_dir")
ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--K", type=int, default=16); ap.add_argument("--L1", type=int, default=24)
ap.add_argument("--lr_lora", type=float, default=1e-4); ap.add_argument("--lr_head", type=float, default=1e-3); ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--mb", type=int, default=4); ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--check", action="store_true")
ap.add_argument("--bench", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)       # per-fold training seed; outputs of seed != 0 get a _s<seed> suffix
a = ap.parse_args()
W = f"{a.research_dir}/work"
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CL = np.array(["angry", "happy", "neutral", "sad"])
m = pd.read_csv(f"{W}/meta.csv")
lab = pd.Categorical(m.emotion, categories=CL).codes
paths = pd.read_csv(f"{W}/comp_paths.csv").path.values
N = len(m)
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
A_idx = m.set_index(["split", "Id"]).index.get_indexer([("train", i) for i in folds.Id])
fA = folds.fold.values
te = np.where(m.split == "test")[0]

enc = AutoModel.from_pretrained(a.model, cache_dir=a.hf_cache).get_encoder().float().eval()
fe = WhisperFeatureExtractor.from_pretrained(a.model, cache_dir=a.hf_cache)
T, D = enc.config.max_source_positions, enc.config.d_model
NF = np.array([max(1, min(T, math.ceil(len(sf.read(p, dtype="float32")[0]) / 16000 * 50))) for p in paths])

# ---------------------------------------------------------------- cache hidden_states[K] (fp32 memmap)
os.makedirs(a.cache_dir, exist_ok=True)
cfile, flag = f"{a.cache_dir}/h{a.K}_{a.tag}.f32", f"{a.cache_dir}/h{a.K}_{a.tag}.done"
if not os.path.exists(flag):
    t0 = time.time()
    H = np.lib.format.open_memmap(cfile, mode="w+", dtype=np.float32, shape=(N, T, D))
    enc_dev = enc.to(DEV)
    with torch.no_grad():
        for i, p in enumerate(paths):
            y = sf.read(p, dtype="float32")[0]
            hs = enc_dev(input_features=fe(y, sampling_rate=16000, return_tensors="pt").input_features.to(DEV), output_hidden_states=True).hidden_states
            H[i] = hs[a.K][0].float().cpu().numpy()
            if i % 200 == 0:
                print(f"cache {i}/{N} {time.time() - t0:.0f}s", flush=True)
    H.flush(); del H
    open(flag, "w").write("ok")
    enc = enc_dev.cpu()
    print(f"cached hidden_states[{a.K}] for {N} clips in {time.time() - t0:.0f}s", flush=True)
H = np.load(cfile, mmap_mode="r")


class Top(nn.Module):
    def __init__(self, enc, K):
        super().__init__()
        self.layers = nn.ModuleList([enc.layers[i] for i in range(K, len(enc.layers))])
        self.layer_norm = enc.layer_norm
        self.K = K


def build_top():
    base = AutoModel.from_pretrained(a.model, cache_dir=a.hf_cache).get_encoder().float()
    top = Top(base, a.K)
    del base
    inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"]), top)
    for n, p in top.named_parameters():
        p.requires_grad = "lora_" in n
    return top


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.top = build_top()
        self.drop = nn.Dropout(a.dropout)
        self.head = nn.Linear(4 * D, 4)

    def pooled(self, h, nfr, use_ckpt):
        mask = (torch.arange(h.shape[1], device=h.device)[None] < nfr[:, None]).float().unsqueeze(-1)
        keep = None
        for j, layer in enumerate(self.top.layers):
            h = ckpt.checkpoint(layer, h, None, use_reentrant=False) if use_ckpt else layer(h, None)
            if self.top.K + j + 1 == a.L1:
                keep = h
        h = self.top.layer_norm(h)
        out = []
        for x in (keep, h):
            mu = (x * mask).sum(1) / mask.sum(1)
            out += [mu, (((x - mu[:, None]) ** 2 * mask).sum(1) / mask.sum(1) + 1e-5).sqrt()]
        return torch.cat(out, -1)

    def forward(self, h, nfr, use_ckpt=False):
        return self.head(self.drop(self.pooled(h, nfr, use_ckpt)))


def batch(rows):
    return torch.from_numpy(np.stack([H[r] for r in rows])).to(DEV), torch.tensor(NF[rows], device=DEV)


if a.check:   # cached top path (LoRA at zero) == full forward pooled features
    model = Model().to(DEV).eval()
    full = enc.to(DEV).eval()
    rows = np.array([0, 5, 900])
    with torch.no_grad():
        p1 = model.pooled(*batch(rows), False)
        hs = [full(input_features=fe(sf.read(paths[r], dtype="float32")[0], sampling_rate=16000, return_tensors="pt").input_features.to(DEV), output_hidden_states=True).hidden_states for r in rows]
        p2 = []
        for r, h in zip(rows, hs):
            n = int(NF[r])
            for x in (h[a.L1][0, :n], h[-1][0, :n]):
                p2 += [x.mean(0), x.std(0, unbiased=False)]
        p2 = torch.stack([torch.cat(p2[4 * i:4 * i + 4]) for i in range(len(rows))])
    rel = float(((p1 - p2).abs().max() / p2.abs().max()).cpu())
    print(f"--check: max |cached-top pooled - full-forward pooled| relative = {rel:.2e}", flush=True)
    assert rel < 1e-3, "cached top path does not reproduce the full forward"
    enc = full.cpu()
    del model


def fit_predict(train_rows, eval_sets, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Model().to(DEV)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("top.")]
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr_lora}, {"params": model.head.parameters(), "lr": a.lr_head}], weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train_rows) / 8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_lora, a.lr_head], total_steps=steps, pct_start=0.1)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    y = torch.tensor(lab, dtype=torch.long, device=DEV)
    t0 = time.time()
    nstep = 0
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(train_rows)
        tot = 0.0
        for s in range(0, len(perm), 8):
            opt.zero_grad()
            for j in range(0, 8, a.mb):
                rows = perm[s + j:s + j + a.mb]
                if len(rows):
                    loss = lossf(model(*batch(rows), True), y[rows]) * len(rows) / len(perm[s:s + 8])
                    loss.backward()
                    tot += float(loss) * len(perm[s:s + 8])
            opt.step()
            sched.step()
            nstep += 1
            if a.bench and nstep >= a.bench:
                if DEV == "mps":
                    torch.mps.synchronize()
                print(f"bench: {a.bench} steps of 8 clips in {time.time() - t0:.1f}s -> {(time.time() - t0) / a.bench:.2f} s/step; "
                      f"est. epoch {math.ceil(len(train_rows) / 8) * (time.time() - t0) / a.bench / 60:.1f} min", flush=True)
                return None
        print(f"  epoch {ep + 1}/{a.epochs} train loss {tot / len(perm):.4f} | {time.time() - t0:.0f}s", flush=True)
    model.eval()
    res = []
    with torch.no_grad():
        for rows in eval_sets:
            res.append(np.vstack([model(*batch(rows[s:s + 8]), False).softmax(-1).cpu().numpy() for s in range(0, len(rows), 8)]))
    return res


folds_run = [int(v) for v in a.folds.split(",")]
if a.bench:
    fit_predict(A_idx[fA != 0], [A_idx[fA == 0][:8]])
    raise SystemExit
SUF = f"_s{a.seed}" if a.seed else ""
outf = f"{W}/loratop_probs_{a.tag}{SUF}.npz"
fold_dir = f"{W}/loratop_{a.tag}{SUF}_folds"
os.makedirs(fold_dir, exist_ok=True)
oof = np.full((len(A_idx), 4), np.nan)
tparts = []
for k in folds_run:
    b = np.where(fA == k)[0]
    ff = f"{fold_dir}/fold{k}.npz"
    if os.path.exists(ff):
        z = np.load(ff)
        pv, pt = z["valid"], z["test"]
        print(f"fold {k}: restored", flush=True)
    else:
        print(f"fold {k}: train {(fA != k).sum()} valid {len(b)}", flush=True)
        pv, pt = fit_predict(A_idx[fA != k], [A_idx[b], te], seed=a.seed)
        np.savez(ff, valid=pv, test=pt)
    oof[b] = pv
    tparts.append(pt)
    print(f"fold {k} macro-F1 {f1_score(lab[A_idx[b]], pv.argmax(1), average='macro'):.4f}", flush=True)
    np.savez(outf, oof=oof, test=np.mean(tparts, 0), folds_done=len(tparts))
done = ~np.isnan(oof[:, 0])
print(f"LORATOP {a.tag}{SUF}: OOF macro-F1 on {done.sum()} rows {f1_score(lab[A_idx][done], oof[done].argmax(1), average='macro'):.4f}", flush=True)
