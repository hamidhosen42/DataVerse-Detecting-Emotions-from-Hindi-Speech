"""Phase 3: LoRA fine-tuning of a Whisper encoder for 4-class SER, on the public pipeline's 5 folds (Track A),
producing OOF + test probabilities for blending (work/lora_probs_<tag>.npz).

Design
- Input cropped/padded to WIN seconds (default 10 s): positional embeddings are sliced to WIN*50 positions, so the
  encoder runs ~3x faster than with 30 s padding. Clips longer than WIN: random crop in training, centre crop at eval.
- Features: masked mean+std pooling over real-audio frames of hidden layers L1 and last, concatenated -> Dropout ->
  Linear(4*D -> 4). Encoder frozen except LoRA (r=16, alpha=32, dropout 0.1) on q_proj and v_proj.
- Training: AdamW (LoRA lr 1e-4, head lr 1e-3), wd 0.01, EPOCHS epochs (no early stopping: no peeking at the fold),
  batch 4 x accumulation 2, label smoothing 0.1, fp32 on Apple MPS (or CUDA if present).

Usage: python -I whisper_lora.py <research_dir> <hf_model_id> <tag> <hf_cache> [EPOCHS=6] [WIN=10] [L1=24] [folds e.g. 0,1,2,3,4]
"""
import math
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from transformers import AutoModel, WhisperFeatureExtractor

R, MID, TAG, CACHE = sys.argv[1:5]
EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 6
WIN = int(sys.argv[6]) if len(sys.argv) > 6 else 10
L1 = int(sys.argv[7]) if len(sys.argv) > 7 else 24
FOLDS = [int(v) for v in sys.argv[8].split(",")] if len(sys.argv) > 8 else [0, 1, 2, 3, 4]
W = f"{R}/work"
CL = np.array(["angry", "happy", "neutral", "sad"])
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
m = pd.read_csv(f"{W}/meta.csv")
lab = pd.Categorical(m.emotion, categories=CL).codes
paths = pd.read_csv(f"{W}/comp_paths.csv").path.values
WAV = [sf.read(p, dtype="float32")[0] for p in paths]
key = m.set_index(["split", "Id"])
folds = pd.read_csv(f"{W}/kunal_train_output/dataverse_models/train_with_folds.csv")
A_idx = np.array([key.index.get_loc(("train", i)) for i in folds.Id])
fA = folds.fold.values
te = np.where(m.split == "test")[0]
NS = WIN * 16000
NPOS = WIN * 50
fe = WhisperFeatureExtractor.from_pretrained(MID, cache_dir=CACHE)


def mel(batch_wavs):
    x = fe(batch_wavs, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=NS, truncation=True).input_features
    return x[:, :, : NPOS * 2]


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
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
        self.enc = get_peft_model(enc, cfg)
        D = enc.config.d_model
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * D, 4))

    def forward(self, feats, nframes):
        hs = self.enc(input_features=feats, output_hidden_states=True).hidden_states
        T = hs[-1].shape[1]
        mask = (torch.arange(T, device=feats.device)[None, :] < nframes[:, None]).float().unsqueeze(-1)
        out = []
        for h in (hs[L1], hs[-1]):
            mu = (h * mask).sum(1) / mask.sum(1)
            sd = (((h - mu[:, None]) ** 2 * mask).sum(1) / mask.sum(1) + 1e-5).sqrt()
            out += [mu, sd]
        return self.head(torch.cat(out, -1))


def run_batch(model, rows, train, rng):
    ys = [crop(WAV[i], train, rng) for i in rows]
    nfr = torch.tensor([min(NPOS, math.ceil(len(y) / 16000 * 50)) for y in ys], device=DEV)
    return model(mel(ys).to(DEV), nfr)


def fit_predict(train_rows, eval_sets, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Model().to(DEV)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": lora, "lr": 1e-4}, {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    steps = EPOCHS * math.ceil(len(train_rows) / 8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=0.1)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    y = torch.tensor(lab, dtype=torch.long, device=DEV)
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(train_rows)
        tot = 0.0
        for s in range(0, len(perm), 8):
            opt.zero_grad()
            for half in (perm[s:s + 4], perm[s + 4:s + 8]):
                if len(half):
                    loss = lossf(run_batch(model, half, True, rng), y[half]) / 2
                    loss.backward()
                    tot += float(loss) * 2 * len(half)
            opt.step()
            sched.step()
        print(f"  epoch {ep + 1}/{EPOCHS} train loss {tot / len(perm):.4f}", flush=True)
    model.eval()
    res = []
    with torch.no_grad():
        for rows in eval_sets:
            res.append(np.vstack([run_batch(model, rows[s:s + 8], False, rng).softmax(-1).cpu().numpy() for s in range(0, len(rows), 8)]))
    return res


oof = np.full((len(A_idx), 4), np.nan)
test_parts = []
for k in FOLDS:
    a, b = np.where(fA != k)[0], np.where(fA == k)[0]
    print(f"fold {k}: train {len(a)} valid {len(b)}", flush=True)
    pv, pt = fit_predict(A_idx[a], [A_idx[b], te])
    oof[b] = pv
    test_parts.append(pt)
    print(f"fold {k} macro-F1 {f1_score(lab[A_idx[b]], pv.argmax(1), average='macro'):.4f}", flush=True)
    np.savez(f"{W}/lora_probs_{TAG}.npz", oof=oof, test=np.mean(test_parts, 0), folds_done=np.array(FOLDS[:len(test_parts)]))
done = ~np.isnan(oof[:, 0])
print(f"LoRA {TAG}: OOF macro-F1 on {done.sum()} rows {f1_score(lab[A_idx][done], oof[done].argmax(1), average='macro'):.4f}", flush=True)
