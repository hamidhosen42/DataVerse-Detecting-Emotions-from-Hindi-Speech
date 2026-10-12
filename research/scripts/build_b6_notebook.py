"""Write notebooks/B6_whisper_blend.ipynb: a single, self-contained notebook that reproduces the B6 submission.

Usage: python -I build_b6_notebook.py <out_ipynb>
"""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})


md(r"""
# DataVerse: Detecting Emotions from Hindi Speech — B6 submission (public LB 0.93060)

**Method.** Equal-weight probability average of 6 model families:

1. Public fine-tuned WavLM Base+ & Large ensemble — test probabilities from the public notebook
   [`kunalinfinite/02-dataverse-inference`](https://www.kaggle.com/code/kunalinfinite/02-dataverse-inference)
   (attach it as an input; its training notebook is `kunalinfinite/01-dataverse-training`).
2. Five **frozen Whisper encoders** + logistic regression (no fine-tuning):
   `openai/whisper-large-v2`, `openai/whisper-large-v3`, `ARTPARK-IISc/whisper-large-v3-vaani-hindi`,
   `vasista22/whisper-hindi-large-v2`, `openai/whisper-medium`.
   Features = per-layer [mean, std] pooled **only over frames covering real audio** (not Whisper's 30 s padding);
   large models use layers 24 + 32 concatenated, Whisper-medium uses the mean of layers 22–24.
   Classifier: `StandardScaler` + `LogisticRegression(C=0.05, class_weight="balanced")`.
3. **Twin override (safety net):** a test clip whose waveform is a near-copy (max normalised cross-correlation ≥ 0.95)
   of training clips with one consistent label gets that training label. (For B6 this changed 0 predictions.)

Validation (see the research report): out-of-fold Macro-F1 0.942 on the public pipeline's 5 folds vs 0.862 for the
fine-tuned WavLM ensemble alone; dup-group CV (no near-duplicates across folds) 0.899 ± 0.008.
No external labelled data, no test labels, no hand labelling. All models are public (Apache-2.0 / MIT).

Runtime: ~1–1.5 h on a Kaggle T4 (Whisper feature extraction dominates); cached features are reused on re-runs.
""")

code(r"""
import glob, hashlib, math, os
import numpy as np, pandas as pd, soundfile as sf, torch
from scipy.signal import fftconvolve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoFeatureExtractor, AutoModel

def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None

# ---- paths: Kaggle inputs first, then a local project layout
DATA = os.path.dirname(find("/kaggle/input/**/SRCASW-BhavVaani/train.csv") or find("../data/SRCASW-BhavVaani/train.csv")
                       or find("data/SRCASW-BhavVaani/train.csv"))
FT_TEST_PROBS = (find("/kaggle/input/**/dataverse_inference/test_probabilities.csv")
                 or find("../research/work/kunal_output/dataverse_inference/test_probabilities.csv"))
FOLDS_CSV = (find("/kaggle/input/**/dataverse_models/train_with_folds.csv")
             or find("../research/work/kunal_train_output/dataverse_models/train_with_folds.csv"))
CACHE = "/kaggle/working/features" if os.path.exists("/kaggle/working") else os.environ.get("B6_CACHE", "features")
os.makedirs(CACHE, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CLASSES = np.array(["angry", "happy", "neutral", "sad"])
print("data:", DATA, "| FT probs:", FT_TEST_PROBS, "| folds:", FOLDS_CSV, "| device:", DEV)
""")

code(r"""
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
meta = pd.concat([train.assign(split="train"), test.assign(split="test", emotion=np.nan)], ignore_index=True)
paths = [f"{DATA}/{s}/{f}" for s, f in zip(meta.split, meta.filename)]

def load(p):
    y, sr = sf.read(p, dtype="float32", always_2d=True)
    assert sr == 16000
    return y.mean(1)

WAV = [load(p) for p in paths]

# training rows = those used by the public fine-tuned pipeline (exact duplicates with conflicting labels removed)
if FOLDS_CSV:
    keep_ids = set(pd.read_csv(FOLDS_CSV).Id)
else:
    h = [hashlib.md5(w.tobytes()).hexdigest() for w in WAV]
    meta["h"] = h
    bad = meta[meta.split == "train"].groupby("h").emotion.nunique()
    keep_ids = set(meta[(meta.split == "train") & ~meta.h.isin(bad[bad > 1].index)].Id)
TR = np.array([i for i in range(len(meta)) if meta.split[i] == "train" and meta.Id[i] in keep_ids])
TE = np.where(meta.split == "test")[0]
print(len(TR), "training rows |", len(TE), "test rows")
""")

code(r"""
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
            n = max(1, min(hs[0].shape[1], math.ceil(len(y) / 16000 * 50)))   # 50 encoder frames / s of real audio
            h = torch.stack(hs, 0)[:, 0, :n]
            feats.append(torch.cat([h.mean(1), h.std(1)], -1).float().cpu().numpy().astype(np.float16))
            if i % 250 == 0: print(tag, i, flush=True)
    arr = np.stack(feats); np.save(out, arr)
    del enc; torch.cuda.empty_cache() if DEV == "cuda" else None
    return arr

MODELS = {  # tag: (hub id, feature recipe)
    "wlv2": ("openai/whisper-large-v2", "cat24+32"),
    "wlv3": ("openai/whisper-large-v3", "cat24+32"),
    "wvaani": ("ARTPARK-IISc/whisper-large-v3-vaani-hindi", "cat24+32"),
    "whhlv2": ("vasista22/whisper-hindi-large-v2", "cat24+32"),
    "wmed": ("openai/whisper-medium", "band22-24"),
}
FEATS = {}
for tag, (mid, recipe) in MODELS.items():
    E = whisper_features(mid, tag).astype(np.float32)
    FEATS[tag] = np.hstack([E[:, 24], E[:, 32]]) if recipe == "cat24+32" else E[:, 22:25].mean(1)
    print(tag, FEATS[tag].shape)
""")

code(r"""
y_tr = meta.emotion.values[TR]
probs = {}
for tag, X in FEATS.items():
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000, class_weight="balanced"))
    clf.fit(X[TR], y_tr)
    assert list(clf.classes_) == list(CLASSES)
    probs[tag] = clf.predict_proba(X[TE])

ft = pd.read_csv(FT_TEST_PROBS).set_index("Id").loc[meta.Id.values[TE]]
probs["ft_ens"] = ft[[f"prob_{c}" for c in CLASSES]].values
blend = np.mean([probs[k] for k in ["ft_ens", "wlv2", "wlv3", "wvaani", "whhlv2", "wmed"]], 0)
pred = CLASSES[blend.argmax(1)]
""")

code(r"""
# twin override: test clips that are near-copies of training clips with a single consistent label
def ncc(a, b):
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    return float(np.max(np.abs(fftconvolve(a, b[::-1], mode="full"))) / min(len(a), len(b)))

all_train = np.where(meta.split == "train")[0]
dur = np.array([len(w) / 16000 for w in WAV])
changed = 0
for j, t in enumerate(TE):
    labels = {meta.emotion[i] for i in all_train if abs(dur[i] - dur[t]) < 0.25 and ncc(WAV[t], WAV[i]) >= 0.95}
    if len(labels) == 1:
        lab = labels.pop()
        changed += int(pred[j] != lab)
        pred[j] = lab
print("twin override changed", changed, "predictions")
""")

code(r"""
sub = pd.DataFrame({"Id": meta.Id.values[TE], "emotion": pred})
sub.to_csv("submission.csv", index=False)
print(sub.emotion.value_counts().to_dict())
ref = (glob.glob("../my_submissions/0.93060_*.csv") or [None])[0]
if ref:
    r = pd.read_csv(ref)
    print("rows differing from the submitted B6 file:", int((r.emotion.values != sub.emotion.values).sum()))
sub.head()
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print("wrote", sys.argv[1], len(cells), "cells")
