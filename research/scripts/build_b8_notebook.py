"""Write notebooks/B8_whisper_lora_blend.ipynb — the updated base notebook (B6 with the public WavLM fine-tune
replaced by our own LoRA-fine-tuned Whisper-large-v2). Self-contained: only competition data + public pretrained models.

Usage: python -I build_b8_notebook.py <out_ipynb>
"""
import json
import sys

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""
# DataVerse: Detecting Emotions from Hindi Speech — B8 (updated base notebook)

**What changed vs B6 (public LB 0.93060):** the fine-tuned component is now our own **LoRA-fine-tuned Whisper-large-v2
encoder** instead of the public WavLM Base+/Large fine-tunes from another participant's notebook. Everything is built
from the competition data and public pretrained models (Apache-2.0 / MIT), so the notebook is fully self-contained.

**Model = equal-weight probability average of 6 families**
1. **Whisper-large-v2 + LoRA** (r = 16 on q/v projections, 2.6 M trainable params), 8-s window (positional embeddings
   sliced), masked mean+std pooling of layers 24 and 32 → linear head; 5-fold models averaged for the test set.
2.–6. **Frozen Whisper encoders + logistic regression**: whisper-large-v2, whisper-large-v3,
   ARTPARK-IISc/whisper-large-v3-vaani-hindi, vasista22/whisper-hindi-large-v2 (layers 24+32 [mean,std], real-audio frames
   only) and whisper-medium (layers 22–24).
7. Twin override safety net (test clips that are waveform near-copies of consistently-labelled training clips).

**Validation (public pipeline's 5 folds, 824 rows):** LoRA alone OOF Macro-F1 **0.9145** (folds 0.908–0.920; the public
WavLM fine-tune ensemble: 0.862). Blend OOF **0.9433** vs B6 0.9421 (+0.0012, 95% CI [−0.007, +0.010] — statistically
the same). B8 differs from B6 on 3 of 209 test clips.

**Runtime:** Kaggle T4 ≈ 1 h for frozen features + ≈ 2–3 h for 5 LoRA folds. Features and LoRA predictions are cached
in `CACHE`, so re-runs are fast. Set `LORA_FOLDS`/`LORA_EPOCHS` smaller for a quick smoke test.
""")

code(r"""
import glob, hashlib, math, os
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn
from scipy.signal import fftconvolve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoFeatureExtractor, AutoModel, WhisperFeatureExtractor
from peft import LoraConfig, get_peft_model        # pip install peft (preinstalled on Kaggle images)

def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None

DATA = os.path.dirname(find("/kaggle/input/**/SRCASW-BhavVaani/train.csv") or find("../data/SRCASW-BhavVaani/train.csv")
                       or find("data/SRCASW-BhavVaani/train.csv"))
FOLDS_CSV = (find("/kaggle/input/**/dataverse_models/train_with_folds.csv")          # optional: identical folds to our CV
             or find("../research/work/kunal_train_output/dataverse_models/train_with_folds.csv"))
CACHE = "/kaggle/working/cache" if os.path.exists("/kaggle/working") else os.environ.get("B8_CACHE", "cache")
os.makedirs(CACHE, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CLASSES = np.array(["angry", "happy", "neutral", "sad"])
LORA_FOLDS = [int(v) for v in os.environ.get("LORA_FOLDS", "0,1,2,3,4").split(",")]
LORA_EPOCHS = int(os.environ.get("LORA_EPOCHS", "6"))
print("data:", DATA, "| folds:", FOLDS_CSV, "| cache:", CACHE, "| device:", DEV)
""")

code(r"""
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
meta = pd.concat([train.assign(split="train"), test.assign(split="test", emotion=np.nan)], ignore_index=True)
WAV = []
for s, f in zip(meta.split, meta.filename):
    y, sr = sf.read(f"{DATA}/{s}/{f}", dtype="float32", always_2d=True); assert sr == 16000
    WAV.append(y.mean(1))
LAB = pd.Categorical(meta.emotion, categories=CLASSES).codes

# training rows: drop exact-duplicate recordings whose copies carry conflicting labels; fold ids
meta["h"] = [hashlib.md5(w.tobytes()).hexdigest() for w in WAV]
if FOLDS_CSV:
    fdf = pd.read_csv(FOLDS_CSV).set_index("Id")
    TR = np.array([i for i in range(len(meta)) if meta.split[i] == "train" and meta.Id[i] in fdf.index])
    FOLD = fdf.loc[meta.Id.values[TR], "fold"].values
else:
    bad = meta[meta.split == "train"].groupby("h").emotion.nunique()
    TR = np.where((meta.split == "train") & ~meta.h.isin(bad[bad > 1].index))[0]
    FOLD = np.zeros(len(TR), int)
    for k, (_, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(TR, LAB[TR], meta.h.values[TR])):
        FOLD[b] = k
TE = np.where(meta.split == "test")[0]
print(len(TR), "training rows |", len(TE), "test rows | fold sizes", np.bincount(FOLD))
""")

code(r"""
# ---- frozen Whisper features: per-layer [mean, std] over real-audio frames (Whisper pads every clip to 30 s)
def whisper_features(model_id, tag):
    out = f"{CACHE}/{tag}.npy"
    if os.path.exists(out):
        return np.load(out)
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    enc = AutoModel.from_pretrained(model_id).get_encoder().float().to(DEV).eval()
    feats = []
    with torch.no_grad():
        for i, y in enumerate(WAV):
            x = fe(y, sampling_rate=16000, return_tensors="pt").input_features.to(DEV)
            hs = enc(input_features=x, output_hidden_states=True).hidden_states
            n = max(1, min(hs[0].shape[1], math.ceil(len(y) / 16000 * 50)))
            h = torch.stack(hs, 0)[:, 0, :n]
            feats.append(torch.cat([h.mean(1), h.std(1)], -1).float().cpu().numpy().astype(np.float16))
            if i % 250 == 0: print(tag, i, flush=True)
    arr = np.stack(feats); np.save(out, arr)
    del enc
    if DEV == "cuda": torch.cuda.empty_cache()
    return arr

MODELS = {"wlv2": ("openai/whisper-large-v2", "cat"), "wlv3": ("openai/whisper-large-v3", "cat"),
          "wvaani": ("ARTPARK-IISc/whisper-large-v3-vaani-hindi", "cat"), "whhlv2": ("vasista22/whisper-hindi-large-v2", "cat"),
          "wmed": ("openai/whisper-medium", "band")}
probs = {}
for tag, (mid, recipe) in MODELS.items():
    E = whisper_features(mid, tag).astype(np.float32)
    X = np.hstack([E[:, 24], E[:, 32]]) if recipe == "cat" else E[:, 22:25].mean(1)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000, class_weight="balanced")).fit(X[TR], LAB[TR])
    probs[tag] = clf.predict_proba(X[TE])
    print(tag, X.shape)
""")

code(r"""
# ---- Whisper-large-v2 + LoRA (8-s window), 5 fold models averaged on test
LORA_MID, WIN, L1 = "openai/whisper-large-v2", 8, 24
NS, NPOS = WIN * 16000, WIN * 50
wfe = WhisperFeatureExtractor.from_pretrained(LORA_MID)

def mel(batch):
    return wfe(batch, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=NS, truncation=True).input_features[:, :, :NPOS * 2]

def crop(y, train, rng):
    if len(y) <= NS: return y
    s = rng.integers(0, len(y) - NS + 1) if train else (len(y) - NS) // 2
    return y[s:s + NS]

class LoraSER(nn.Module):
    def __init__(self):
        super().__init__()
        enc = AutoModel.from_pretrained(LORA_MID).get_encoder().float()
        enc.embed_positions.weight.data = enc.embed_positions.weight.data[:NPOS].clone()
        enc.embed_positions.num_embeddings = NPOS
        enc.config.max_source_positions = NPOS
        self.enc = get_peft_model(enc, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"]))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * enc.config.d_model, 4))
    def forward(self, feats, nframes):
        hs = self.enc(input_features=feats, output_hidden_states=True).hidden_states
        mask = (torch.arange(hs[-1].shape[1], device=feats.device)[None] < nframes[:, None]).float().unsqueeze(-1)
        out = []
        for h in (hs[L1], hs[-1]):
            mu = (h * mask).sum(1) / mask.sum(1)
            out += [mu, (((h - mu[:, None]) ** 2 * mask).sum(1) / mask.sum(1) + 1e-5).sqrt()]
        return self.head(torch.cat(out, -1))

def run_batch(model, rows, train, rng):
    ys = [crop(WAV[i], train, rng) for i in rows]
    nfr = torch.tensor([min(NPOS, math.ceil(len(y) / 16000 * 50)) for y in ys], device=DEV)
    return model(mel(ys).to(DEV), nfr)

def fit_predict(train_rows, eval_rows, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = LoraSER().to(DEV)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": lora, "lr": 1e-4}, {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=LORA_EPOCHS * math.ceil(len(train_rows) / 8), pct_start=0.1)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1); y = torch.tensor(LAB, dtype=torch.long, device=DEV)
    for ep in range(LORA_EPOCHS):
        model.train(); perm = rng.permutation(train_rows); tot = 0.0
        for s in range(0, len(perm), 8):
            opt.zero_grad()
            for half in (perm[s:s + 4], perm[s + 4:s + 8]):            # batch 4 x accumulation 2
                if len(half):
                    loss = lossf(run_batch(model, half, True, rng), y[half]) / 2; loss.backward(); tot += float(loss) * 2 * len(half)
            opt.step(); sched.step()
        print(f"  epoch {ep + 1}/{LORA_EPOCHS} loss {tot / len(perm):.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        return np.vstack([run_batch(model, eval_rows[s:s + 8], False, rng).softmax(-1).cpu().numpy() for s in range(0, len(eval_rows), 8)])

cache_lora = f"{CACHE}/lora_wlv2_test_probs.npy"
if os.path.exists(cache_lora):
    probs["lora"] = np.load(cache_lora)
else:
    parts = []
    for k in LORA_FOLDS:
        print(f"LoRA fold {k}", flush=True)
        parts.append(fit_predict(TR[FOLD != k], TE))
    probs["lora"] = np.mean(parts, 0)
    if len(LORA_FOLDS) == 5: np.save(cache_lora, probs["lora"])
""")

code(r"""
blend = np.mean([probs[k] for k in ["lora", "wlv2", "wlv3", "wvaani", "whhlv2", "wmed"]], 0)
pred = CLASSES[blend.argmax(1)]

# twin override: test clips that are waveform near-copies (max NCC >= 0.95) of consistently-labelled training clips
def ncc(a, b):
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    return float(np.max(np.abs(fftconvolve(a, b[::-1], mode="full"))) / min(len(a), len(b)))
all_train = np.where(meta.split == "train")[0]; dur = np.array([len(w) / 16000 for w in WAV]); changed = 0
for j, t in enumerate(TE):
    labels = {meta.emotion[i] for i in all_train if abs(dur[i] - dur[t]) < 0.25 and ncc(WAV[t], WAV[i]) >= 0.95}
    if len(labels) == 1:
        lab = labels.pop(); changed += int(pred[j] != lab); pred[j] = lab
print("twin override changed", changed, "predictions")

sub = pd.DataFrame({"Id": meta.Id.values[TE], "emotion": pred}); sub.to_csv("submission.csv", index=False)
print(sub.emotion.value_counts().to_dict())
for ref in sorted(glob.glob("../my_submissions/*/submission.csv")) + sorted(glob.glob("../research/submissions/sub_B8.csv")):
    print(ref, "-> rows differing:", int((pd.read_csv(ref).emotion.values != sub.emotion.values).sum()))
sub.head()
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print("wrote", sys.argv[1], len(cells), "cells")
