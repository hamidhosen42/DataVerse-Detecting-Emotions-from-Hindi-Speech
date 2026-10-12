"""LP-FT (Kumar et al., ICLR 2022) for Whisper + LoRA SER, on the public pipeline's 5 folds (Track A).

Per fold:
  Stage LP : frozen encoder (LoRA disabled, 8-s window) -> masked mean+std of layers L1 and last -> StandardScaler+LR
             (C=0.05, balanced). The LR (with its scaler folded in) initialises the linear head exactly.
  Stage FT : LoRA (r=16, q/v) starts at zero, so training starts from the LP solution; EPOCHS epochs, LoRA lr 5e-5,
             head lr 1e-4, OneCycle (10% warm-up), label smoothing 0.1, batch 4 x accumulation 2, random crops.
Saves OOF/test for both stages: work/lpft_probs_<tag>.npz (keys oof_lp, test_lp, oof_ft, test_ft).

Usage: python -I whisper_lpft.py <research_dir> <hf_model_id> <tag> <hf_cache> [EPOCHS=4] [WIN=8] [L1=24]
"""
import math
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, WhisperFeatureExtractor

R, MID, TAG, CACHE = sys.argv[1:5]
EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
WIN = int(sys.argv[6]) if len(sys.argv) > 6 else 8
L1 = int(sys.argv[7]) if len(sys.argv) > 7 else 24
MB = int(sys.argv[8]) if len(sys.argv) > 8 else 4          # micro-batch; accumulation = 8 // MB
GC = len(sys.argv) > 9 and sys.argv[9] == "gc"          # gradient checkpointing
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
m = pd.read_csv(f"{W}/meta.csv")
lab = pd.Categorical(m.emotion, categories=CL).codes
WAV = [sf.read(p, dtype="float32")[0] for p in pd.read_csv(f"{W}/comp_paths.csv").path.values]
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
A_idx = m.set_index(["split", "Id"]).index.get_indexer([("train", i) for i in folds.Id])
fA = folds.fold.values
te = np.where(m.split == "test")[0]
NS, NPOS = WIN * 16000, WIN * 50
fe = WhisperFeatureExtractor.from_pretrained(MID, cache_dir=CACHE)


def mel(batch):
    return fe(batch, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=NS, truncation=True).input_features[:, :, :NPOS * 2]


def crop(y, train, rng):
    if len(y) <= NS:
        return y
    s = rng.integers(0, len(y) - NS + 1) if train else (len(y) - NS) // 2
    return y[s:s + NS]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        enc = AutoModel.from_pretrained(MID, cache_dir=CACHE).get_encoder().float()
        enc.embed_positions.weight.data = enc.embed_positions.weight.data[:NPOS].clone()
        enc.embed_positions.num_embeddings = NPOS
        enc.config.max_source_positions = NPOS
        if GC:
            enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            enc.enable_input_require_grads() if hasattr(enc, "enable_input_require_grads") else None
        self.enc = get_peft_model(enc, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"]))
        self.D = enc.config.d_model
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(4 * self.D, 4)

    def pooled(self, feats, nframes):
        hs = self.enc(input_features=feats, output_hidden_states=True).hidden_states
        mask = (torch.arange(hs[-1].shape[1], device=feats.device)[None] < nframes[:, None]).float().unsqueeze(-1)
        out = []
        for h in (hs[L1], hs[-1]):
            mu = (h * mask).sum(1) / mask.sum(1)
            out += [mu, (((h - mu[:, None]) ** 2 * mask).sum(1) / mask.sum(1) + 1e-5).sqrt()]
        return torch.cat(out, -1)

    def forward(self, feats, nframes):
        return self.head(self.drop(self.pooled(feats, nframes)))


def inputs(rows, train, rng):
    ys = [crop(WAV[i], train, rng) for i in rows]
    nfr = torch.tensor([min(NPOS, math.ceil(len(y) / 16000 * 50)) for y in ys], device=DEV)
    return mel(ys).to(DEV), nfr


def fit_predict(train_rows, eval_sets, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Model().to(DEV)
    # ---- stage LP: frozen features (adapters disabled), LR probe -> head init
    model.eval()
    feats = {}
    with torch.no_grad(), model.enc.disable_adapter():
        for name, rows in [("train", train_rows)] + [(f"e{i}", r) for i, r in enumerate(eval_sets)]:
            feats[name] = np.vstack([model.pooled(*inputs(rows[s:s + MB], False, rng)).cpu().numpy() for s in range(0, len(rows), MB)])
    sc = StandardScaler().fit(feats["train"])
    lr = LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced").fit(sc.transform(feats["train"]), lab[train_rows])
    lp_preds = [lr.predict_proba(sc.transform(feats[f"e{i}"])) for i in range(len(eval_sets))]
    Wt = lr.coef_ / sc.scale_[None, :]
    bt = lr.intercept_ - (lr.coef_ * sc.mean_[None, :] / sc.scale_[None, :]).sum(1)
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor(Wt, dtype=torch.float32))
        model.head.bias.copy_(torch.tensor(bt, dtype=torch.float32))
    # ---- stage FT: LoRA from zero (= LP solution at step 0)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": lora, "lr": 5e-5}, {"params": model.head.parameters(), "lr": 1e-4}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[5e-5, 1e-4], total_steps=EPOCHS * math.ceil(len(train_rows) / 8), pct_start=0.1)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    y = torch.tensor(lab, dtype=torch.long, device=DEV)
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(train_rows)
        tot = 0.0
        for s in range(0, len(perm), 8):
            opt.zero_grad()
            for half in [perm[s + j:s + j + MB] for j in range(0, 8, MB)]:
                if len(half):
                    loss = lossf(model(*inputs(half, True, rng)), y[half]) / (8 // MB)
                    loss.backward()
                    tot += float(loss) * (8 // MB) * len(half)
            opt.step()
            sched.step()
        print(f"  epoch {ep + 1}/{EPOCHS} train loss {tot / len(perm):.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        ft_preds = [np.vstack([model(*inputs(r[s:s + MB], False, rng)).softmax(-1).cpu().numpy() for s in range(0, len(r), MB)]) for r in eval_sets]
    return lp_preds, ft_preds


oof_lp, oof_ft = np.zeros((len(A_idx), 4)), np.zeros((len(A_idx), 4))
t_lp, t_ft = [], []
for k in range(5):
    a, b = np.where(fA != k)[0], np.where(fA == k)[0]
    print(f"fold {k}: train {len(a)} valid {len(b)}", flush=True)
    (lp_v, lp_t), (ft_v, ft_t) = fit_predict(A_idx[a], [A_idx[b], te])
    oof_lp[b], oof_ft[b] = lp_v, ft_v
    t_lp.append(lp_t)
    t_ft.append(ft_t)
    yb = lab[A_idx[b]]
    print(f"fold {k} macro-F1 LP {f1_score(yb, lp_v.argmax(1), average='macro'):.4f} | LP-FT {f1_score(yb, ft_v.argmax(1), average='macro'):.4f}", flush=True)
    np.savez(f"{W}/lpft_probs_{TAG}.npz", oof_lp=oof_lp, oof_ft=oof_ft, test_lp=np.mean(t_lp, 0), test_ft=np.mean(t_ft, 0), folds_done=k + 1)
yA = lab[A_idx]
print(f"LPFT {TAG}: OOF macro-F1 LP {f1_score(yA, oof_lp.argmax(1), average='macro'):.4f} | LP-FT {f1_score(yA, oof_ft.argmax(1), average='macro'):.4f}", flush=True)
