"""Build notebooks/B9_hindi_ser_final.ipynb — complete, self-contained train+inference pipeline.

Usage: python -I build_b9_notebook.py <out_ipynb> <canonical_folds.json> [selected.json]
selected.json (optional): {"ft_recipe": "lora8"|"lora30top"|"lpft30"|"none", "ensemble": [...family names...], "note": "..."}
Default selection = B8 composition (lora8 + 5 frozen Whisper LR families).
"""
import json
import sys

out, folds_json = sys.argv[1], sys.argv[2]
sel = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {"ft_recipe": "lora8", "ensemble": ["ft", "lr_wlv2", "lr_wlv3", "lr_wvaani", "lr_whhlv2", "lr_wmed"],
                                                               "note": "B8 composition (default)"}
FOLD_MAP = open(folds_json).read().strip()
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""
# DataVerse: Detecting Emotions from Hindi Speech — B9 final pipeline (train + inference)

Self-contained: needs only the competition data and five public pretrained Whisper checkpoints from the Hugging Face Hub
(all Apache-2.0, not gated; the exact commits are pinned in `REVISIONS`). No external labelled data, no other participant's
checkpoints, no hand labelling of test clips.

**Default = the B8 composition** (public LB 0.94230). No B9 candidate passed the pre-declared keep rule (table below), so the
default fine-tuned member stays the B8 recipe `lora8`. A fresh run re-trains every model; on another GPU and with this
notebook's per-fold seeds it can differ from the submitted B8 file on a few test clips (the submitted file is kept separately).

**Pipeline**
1. Data audit: exact-duplicate recordings with conflicting labels are dropped from training; near-duplicate recordings
   (waveform normalised cross-correlation ≥ 0.95) are grouped for *novel-recording* validation and for the twin safety net.
2. **Frozen Whisper encoder families** (`openai/whisper-large-v2`, `openai/whisper-large-v3`,
   `ARTPARK-IISc/whisper-large-v3-vaani-hindi`, `vasista22/whisper-hindi-large-v2`, `openai/whisper-medium`):
   per-layer [mean, std] pooled **only over frames that cover real audio** (Whisper pads every clip to 30 s), layers 24+32
   concatenated (medium: mean of layers 22–24) → `StandardScaler` + `LogisticRegression(C=0.05, class_weight="balanced")`.
3. **Fine-tuned family** (`FT_RECIPE`, default `lora8`): LoRA (r=16 on attention q/v) on the Whisper-large-v2 encoder, masked
   mean+std pooling of layers 24 and 32, linear head. `lora8` = the B8 recipe (8-s window); `lora30top` = the same recipe at the
   native 30-s context with LoRA in layers 17–32; `lpft30` = 30-s LP-FT (head initialised from the exact linear probe).
4. Equal-weight probability average of the selected families (`ENSEMBLE`), twin safety net, `submission.csv`.

**Validation** (all fitted inside training folds; no test labels anywhere): 5 canonical folds (embedded `FOLD_MAP`, the
public pipeline's folds, so numbers are comparable with the research report), OOF Macro-F1 overall / non-twin / per class,
dup-group-disjoint 5-fold for the frozen families, and a paired dup-group bootstrap of the ensemble vs. its components.

**Modes and options** (environment variables; set them in a cell *above* the configuration cell, e.g.
`import os; os.environ["B9_MODE"] = "smoke"`):
* `B9_MODE=smoke` — 96-clip subset, 2 frozen families, 1 fold × 1 epoch; checks that every stage runs and writes
  `submission_smoke.csv` (**never submit it**).
* `B9_MODE=full` (default) — everything; writes `submission.csv` and `b9_cache/b9_probs.npz` (OOF + test probabilities).
* `B9_FT_RECIPE` (`lora8` | `lora30top` | `lpft30` | `none`), `B9_ENSEMBLE` (JSON list), `B9_LORA_FOLDS` (e.g. `0,1` to spread the
  5 folds over several sessions), `B9_MICRO_BATCH`, `B9_GRAD_CKPT`, `B9_AMP` (CUDA fp16 autocast, default on; `0` = fp32).

Runtimes measured locally (Apple M4 Pro, MPS, fp32, research scripts with the same model): `lora30top` ≈ 66 min per fold with
its lower 16 layers pre-computed (this notebook recomputes them under `no_grad`, so it is slower), `lpft30` ≈ 90 min per fold.
Kaggle GPU runtimes have **not** been measured; if a session could exceed the time limit, train a subset of folds with
`B9_LORA_FOLDS` and resume. Every stage is cached under `CACHE`; to resume in a new Kaggle session, attach the previous
version's output and the cache is copied back automatically (`/kaggle/input/*/b9_cache`).
""")

md(r"""
### Selection evidence (measured locally, 5 canonical folds, 824 training rows; paired dup-group bootstrap vs B8)

| candidate (fine-tuned member + the 5 frozen families) | OOF Macro-F1 | non-twin | vs B8, 95 % CI | P(gain>0) | test rows ≠ B8 |
|---|---|---|---|---|---|
| **B8: `lora8`** (default of this notebook; public LB 0.94230) | **0.9433** | 0.9211 | — | — | 0 |
| B9d: `lora30top` (native 30 s, LoRA layers 17–32) | 0.9482 | 0.9279 | +0.0049 [+0.000, +0.012] | 0.95 | 0 |
| B9a: `lpft30` (native 30 s, LP-FT) | 0.9445 | 0.9230 | +0.0012 [−0.005, +0.008] | 0.60 | 1 |
| B5: no fine-tuned member | 0.9432 | 0.9214 | −0.0001 | 0.46 | 2 |

Single fine-tuned members: `lora8` 0.9145, `lpft30` 0.9302, `lora30top` 0.9242 (frozen Whisper-v2 probe: 0.9362).
Pre-declared keep rule: gain ≥ +0.005 **and** P ≥ 0.9, and P ≥ 0.99 to call it an upgrade after ~45 tried candidates.
No candidate passes, and the best ones produce **identical test predictions**, so the default stays the B8 composition.
Switch the fine-tuned member with `B9_FT_RECIPE=lora30top` (or `lpft30`, `none`).
""")

code(r"""
import os, sys, glob, math, json, time, random, shutil, hashlib, subprocess, inspect
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn
from scipy.signal import fftconvolve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoFeatureExtractor, AutoModel, WhisperFeatureExtractor
try:
    from peft import LoraConfig, inject_adapter_in_model, get_peft_model_state_dict
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=True)
    from peft import LoraConfig, inject_adapter_in_model, get_peft_model_state_dict
import torch.utils.checkpoint

# ----------------------------------------------------------------------------- configuration
MODE = os.environ.get("B9_MODE", "full").lower()                 # "smoke" | "full"
assert MODE in ("smoke", "full")
SEED = 42
FT_RECIPE = os.environ.get("B9_FT_RECIPE", "__FT_RECIPE__")      # "lora30top" | "lpft30" | "lora8" | "none"
ENSEMBLE = json.loads(os.environ.get("B9_ENSEMBLE", '__ENSEMBLE__'))   # "ft" = fine-tuned family, "lr_<tag>" = frozen family
SELECTION_NOTE = "__SELECTION_NOTE__"
TWIN_OVERRIDE = True            # copy the training label to a test clip that is a waveform near-copy (NCC>=0.95) of training clips
FROZEN = {                      # tag: (hub id, feature recipe)
    "wlv2":   ("openai/whisper-large-v2", "cat24+32"),
    "wlv3":   ("openai/whisper-large-v3", "cat24+32"),
    "wvaani": ("ARTPARK-IISc/whisper-large-v3-vaani-hindi", "cat24+32"),
    "whhlv2": ("vasista22/whisper-hindi-large-v2", "cat24+32"),
    "wmed":   ("openai/whisper-medium", "band22-24"),
}
FT_MODEL = "openai/whisper-large-v2"
REVISIONS = {   # Hugging Face commits behind every reported number (all Apache-2.0, not gated)
    "openai/whisper-large-v2": "ae4642769ce2ad8fc292556ccea8e901f1530655",
    "openai/whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1",
    "ARTPARK-IISc/whisper-large-v3-vaani-hindi": "63b5cc7c542373350b6be1d8f8823291dce54b26",
    "vasista22/whisper-hindi-large-v2": "f98540cee2b8cba5b3f7d1b123608dda0a5f5a95",
    "openai/whisper-medium": "abdf7c39ab9d0397620ccaea8974cc764cd0953e",
}
RECIPES = {   # fine-tuning recipes (identical model class; differ in context window, LoRA layers, head init and schedule)
    # lora_from=K: LoRA only in layers K+1..32; layers 1..K run frozen under no_grad (exactly equal to full-model LoRA restricted to those layers)
    "lora30top": dict(win=30, lora_from=16, head_init="random",       epochs=6, lr_lora=1e-4, lr_head=1e-3, dropout=0.3),
    "lpft30":    dict(win=30, lora_from=0,  head_init="linear_probe", epochs=3, lr_lora=5e-5, lr_head=1e-4, dropout=0.1),
    "lora8":     dict(win=8,  lora_from=0,  head_init="random",       epochs=6, lr_lora=1e-4, lr_head=1e-3, dropout=0.3),   # = B8 member
}
LORA = dict(r=16, alpha=32, dropout=0.1, targets=["q_proj", "v_proj"], L1=24, label_smoothing=0.1, eff_batch=8, warmup=0.1)
# ---- hardware / memory knobs
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MICRO_BATCH = int(os.environ.get("B9_MICRO_BATCH", "4" if DEV == "cuda" else "2"))   # per forward pass; grads accumulate to eff_batch
GRAD_CKPT = os.environ.get("B9_GRAD_CKPT", "1") == "1"                                 # gradient checkpointing (needed for 30 s on <=24 GB)
USE_AMP = DEV == "cuda" and os.environ.get("B9_AMP", "1") == "1"                       # fp16 autocast + GradScaler on CUDA only
LORA_FOLDS = [int(v) for v in os.environ.get("B9_LORA_FOLDS", "0,1,2,3,4").split(",")]
SAVE_ADAPTERS = os.environ.get("B9_SAVE_ADAPTERS", "1") == "1"
if MODE == "smoke":
    FROZEN = {k: FROZEN[k] for k in ("wlv2", "wmed")}
    LORA_FOLDS = [0]
    ENSEMBLE = [f for f in ENSEMBLE if f == "ft" or f[3:] in FROZEN]
    for r in RECIPES.values():
        r["epochs"] = 1

# ---- paths (Kaggle first, then a local repo layout)
def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None
DATA = os.path.dirname(find("/kaggle/input/**/SRCASW-BhavVaani/train.csv") or find("../data/SRCASW-BhavVaani/train.csv")
                       or find("data/SRCASW-BhavVaani/train.csv") or "")
assert DATA, "competition data not found (expected .../SRCASW-BhavVaani/train.csv)"
WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else os.environ.get("B9_WORK", os.getcwd())
CACHE = os.environ.get("B9_CACHE", f"{WORK}/b9_cache")
os.makedirs(CACHE, exist_ok=True)
for prev in glob.glob("/kaggle/input/*/b9_cache"):              # resume from an attached previous run
    shutil.copytree(prev, CACHE, dirs_exist_ok=True); print("restored cache from", prev)
SFX = "_smoke" if MODE == "smoke" else ""

def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
seed_all()
CLASSES = np.array(["angry", "happy", "neutral", "sad"])
print(f"MODE={MODE} | device={DEV} | AMP={USE_AMP} | grad-ckpt={GRAD_CKPT} | micro-batch={MICRO_BATCH} | FT_RECIPE={FT_RECIPE} | folds={LORA_FOLDS}")
print(f"ENSEMBLE={ENSEMBLE} ({SELECTION_NOTE})\ndata={DATA}\ncache={CACHE}")
print("versions:", {m.__name__: m.__version__ for m in (torch, np, pd)}, "transformers", __import__("transformers").__version__)
""")

md(r"""
## 1. Data, labels, canonical folds, near-duplicate groups
`FOLD_MAP` is the 5-fold assignment used throughout the research (824 training rows: the 826 train clips minus the exact-duplicate
pair Id 75 / Id 636 whose copies carry different labels). If an Id is missing from the map (e.g. a different data version),
a `StratifiedGroupKFold` on exact-hash groups is used instead.
""")

code(r"""
FOLD_MAP = json.loads('__FOLD_MAP__')      # {fold: [train Ids]}
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv"); sample = pd.read_csv(f"{DATA}/sample_submission.csv")
assert list(sample.columns) == ["Id", "emotion"] and (sample.Id.values == test.Id.values).all(), "sample_submission schema/order changed"
meta = pd.concat([train.assign(split="train"), test.assign(split="test", emotion=np.nan)], ignore_index=True)
WAV, DUR = [], []
for s, f in zip(meta.split, meta.filename):
    y, sr = sf.read(f"{DATA}/{s}/{f}", dtype="float32", always_2d=True); assert sr == 16000, f"{f}: {sr} Hz"
    y = y.mean(1); WAV.append(y); DUR.append(len(y) / sr)
DUR = np.array(DUR)
LAB = pd.Categorical(meta.emotion, categories=CLASSES).codes          # -1 for test
meta["h"] = [hashlib.md5(w.tobytes()).hexdigest() for w in WAV]
TE = np.where(meta.split == "test")[0]
id2row = {int(i): r for r, i in enumerate(meta.Id)}
fold_of = {int(i): int(k) for k, ids in FOLD_MAP.items() for i in ids}
if set(fold_of) <= set(meta.Id[meta.split == "train"]):
    TR = np.array(sorted(id2row[i] for i in fold_of))
    FOLD = np.array([fold_of[int(meta.Id[r])] for r in TR])
    print("canonical folds from FOLD_MAP")
else:
    bad = meta[meta.split == "train"].groupby("h").emotion.nunique()
    TR = np.where((meta.split == "train") & ~meta.h.isin(bad[bad > 1].index))[0]
    FOLD = np.zeros(len(TR), int)
    for k, (_, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(TR, LAB[TR], meta.h.values[TR])): FOLD[b] = k
    print("fallback folds: StratifiedGroupKFold on exact-hash groups")
if MODE == "smoke":    # stratified 96-clip subset keeps its canonical fold ids
    rng = np.random.default_rng(SEED); keep = np.concatenate([rng.choice(np.where(LAB[TR] == c)[0], 24, replace=False) for c in range(4)])
    TR, FOLD = TR[np.sort(keep)], FOLD[np.sort(keep)]
print(f"train rows {len(TR)} | test rows {len(TE)} | fold sizes {np.bincount(FOLD).tolist()} | durations {DUR.min():.2f}-{DUR.max():.2f} s")

# ---- near-duplicate groups (waveform NCC >= 0.95 between duration-matched clips); cached
def ncc(a, b):
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    return float(np.max(np.abs(fftconvolve(a, b[::-1], mode="full"))) / min(len(a), len(b)))
dup_file = f"{CACHE}/dup_groups.csv"
if MODE == "smoke":
    GROUP = pd.factorize(meta.h)[0]                                    # exact copies only (fast)
elif os.path.exists(dup_file):
    GROUP = pd.read_csv(dup_file).dup_group.values
else:
    t0 = time.time(); parent = list(range(len(meta)))
    def root(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    order = np.argsort(DUR)
    for ii in range(len(order)):
        i = order[ii]
        for jj in range(ii + 1, len(order)):
            j = order[jj]
            if DUR[j] - DUR[i] >= 0.25: break
            if ncc(WAV[i], WAV[j]) >= 0.95: parent[root(i)] = root(j)
    GROUP = np.array([root(i) for i in range(len(meta))]); GROUP = pd.factorize(GROUP)[0]
    pd.DataFrame({"Id": meta.Id, "split": meta.split, "dup_group": GROUP}).to_csv(dup_file, index=False)
    print(f"near-duplicate scan: {time.time() - t0:.0f} s")
sizes = pd.Series(GROUP).value_counts()
train_groups = set(GROUP[TR])
TEST_TWIN = np.array([GROUP[t] in train_groups for t in TE])
HAS_TWIN = np.array([np.isin(GROUP[TR[i]], GROUP[TR[FOLD != FOLD[i]]]) for i in range(len(TR))])   # OOF row has a twin in its training folds
print(f"dup groups with >=2 clips: {(sizes > 1).sum()} covering {int(sizes[sizes > 1].sum())} clips | test clips with a training twin: {TEST_TWIN.sum()}/{len(TE)} | OOF rows with twin: {HAS_TWIN.mean():.3f}")
""")

md(r"""
## 2. Frozen Whisper encoder families
Hidden states are pooled only over the first `ceil(duration·50)` encoder frames (the rest is padding). Features are cached
per model (`CACHE/feat_<tag>.npy`, float16, shape N×33×2560). OOF probabilities use the canonical folds; test probabilities
come from a model fitted on all training rows. `clf.classes_` is asserted to equal `CLASSES`.
""")

code(r"""
def whisper_features(model_id, tag, rows):
    out = f"{CACHE}/feat_{tag}{SFX}.npy"
    if os.path.exists(out):
        arr = np.load(out, mmap_mode="r")
        if arr.shape[0] == len(meta): return arr
    fe = AutoFeatureExtractor.from_pretrained(model_id, revision=REVISIONS[model_id])
    enc = AutoModel.from_pretrained(model_id, revision=REVISIONS[model_id]).get_encoder().float().to(DEV).eval()
    feats = np.zeros((len(meta), 33 if "medium" not in model_id else 25, 2 * enc.config.d_model), np.float16)
    done = np.zeros(len(meta), bool)
    with torch.no_grad():
        for n, i in enumerate(rows):
            y = WAV[i]
            x = fe(y, sampling_rate=16000, return_tensors="pt").input_features.to(DEV)
            with torch.autocast(DEV, dtype=torch.float16, enabled=USE_AMP):
                hs = enc(input_features=x, output_hidden_states=True).hidden_states
            nf = max(1, min(hs[0].shape[1], math.ceil(len(y) / 16000 * 50)))
            h = torch.stack(hs, 0)[:, 0, :nf].float()
            feats[i] = torch.cat([h.mean(1), h.std(1)], -1).cpu().numpy().astype(np.float16); done[i] = True
            if n % 250 == 0: print(f"  {tag}: {n}/{len(rows)}", flush=True)
    assert np.isfinite(feats[done].astype(np.float32)).all()
    if done.all(): np.save(out, feats)
    del enc; torch.cuda.empty_cache() if DEV == "cuda" else None
    return feats

def lr_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000, class_weight="balanced"))

ROWS = np.concatenate([TR, TE])
OOF, TESTP, FEATS = {}, {}, {}
y_tr = LAB[TR]
for tag, (mid, recipe) in FROZEN.items():
    t0 = time.time(); E = whisper_features(mid, tag, ROWS)
    X = np.hstack([np.asarray(E[:, 24]), np.asarray(E[:, 32])]).astype(np.float32) if recipe == "cat24+32" else np.asarray(E[:, 22:25]).astype(np.float32).mean(1)
    FEATS[tag] = X
    o = np.zeros((len(TR), 4))
    for k in sorted(set(FOLD)):
        a, b = FOLD != k, FOLD == k
        clf = lr_model().fit(X[TR][a], y_tr[a]); assert (clf.classes_ == np.arange(4)).all()
        o[b] = clf.predict_proba(X[TR][b])
    clf = lr_model().fit(X[TR], y_tr); assert (clf.classes_ == np.arange(4)).all()
    OOF[f"lr_{tag}"], TESTP[f"lr_{tag}"] = o, clf.predict_proba(X[TE])
    print(f"lr_{tag}: OOF macro-F1 {f1_score(y_tr, o.argmax(1), average='macro'):.4f} | {time.time() - t0:.0f} s")
""")

md(r"""
## 3. Fine-tuned family: Whisper-large-v2 encoder + LoRA (`FT_RECIPE`)
Masked mean+std pooling of hidden layer `L1`=24 and the final (layer-normed) output over real-audio frames → dropout → linear head.
Recipes (all: LoRA r=16, α=32 on attention q/v projections, AdamW, OneCycle with 10 % warm-up, label smoothing 0.1,
effective batch 8, fixed epochs — **no early stopping and no checkpoint selection on the validation fold**):

| recipe | context | LoRA layers | head init | epochs | lr (LoRA / head) |
|---|---|---|---|---|---|
| `lora30top` | native 30 s | 17–32 (layers 1–16 frozen, run under `no_grad`) | random | 6 | 1e-4 / 1e-3 |
| `lpft30` | native 30 s | 1–32 | exact linear-probe (LP-FT) | 3 | 5e-5 / 1e-4 |
| `lora8` (= B8 member) | 8-s window (positional embeddings sliced) | 1–32 | random | 6 | 1e-4 / 1e-3 |

The encoder forward is written out explicitly (convs → positions → layers → final layer norm) so frozen lower layers cost no
backward pass; in smoke mode it is asserted to match transformers' own forward. Each fold is cached
(`CACHE/ft_<recipe>/fold<k>.npz`) so an interrupted run resumes; LoRA + head weights are saved per fold. Test probabilities =
mean over the 5 fold models.
""")

code(r"""
if FT_RECIPE != "none":
    R = RECIPES[FT_RECIPE]; WIN = R["win"]; NS, NPOS = WIN * 16000, WIN * 50; L1 = LORA["L1"]
    wfe = WhisperFeatureExtractor.from_pretrained(FT_MODEL, revision=REVISIONS[FT_MODEL])
    FT_DIR = f"{CACHE}/ft_{FT_RECIPE}{SFX}"; os.makedirs(FT_DIR, exist_ok=True)

    def mel(batch):
        x = wfe(batch, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=NS, truncation=True).input_features
        return x[:, :, :NPOS * 2]

    def crop(y, train, rng):
        if len(y) <= NS: return y
        s = rng.integers(0, len(y) - NS + 1) if train else (len(y) - NS) // 2
        return y[s:s + NS]

    def run_layer(layer, h):
        # one WhisperEncoderLayer, independent of the transformers version: 4.x needs layer_head_mask and returns a tuple,
        # 5.x takes (hidden_states, attention_mask) and returns a tensor
        out = layer(h, None, None) if "layer_head_mask" in inspect.signature(layer.forward).parameters else layer(h, None)
        return out[0] if isinstance(out, (tuple, list)) else out

    class LoraSER(nn.Module):
        # Whisper encoder with LoRA in layers K+1..32 (K = recipe "lora_from"); manual forward that replicates
        # transformers' WhisperEncoder: hidden_states[L1] = output of layer L1, hidden_states[32] = final layer_norm(output of layer 32).
        # Embedding + layers 1..K have no trainable parameters -> run under no_grad (no stored activations, no backward through them).
        def __init__(self):
            super().__init__()
            enc = AutoModel.from_pretrained(FT_MODEL, revision=REVISIONS[FT_MODEL]).get_encoder().float()
            if NPOS < enc.config.max_source_positions:                      # shorter context window (lora8 recipe)
                enc.embed_positions.weight.data = enc.embed_positions.weight.data[:NPOS].clone()
                enc.embed_positions.num_embeddings = NPOS; enc.config.max_source_positions = NPOS
            self.K = R["lora_from"]; nL = len(enc.layers)
            # exact module names (not layers_pattern/layers_to_transform, whose matching differs across peft versions)
            targets = [f"layers.{i}.self_attn.{t}" for i in range(self.K, nL) for t in LORA["targets"]]
            inject_adapter_in_model(LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"], target_modules=targets), enc)
            for n, p in enc.named_parameters(): p.requires_grad = "lora_" in n
            n_lora = sum(p.numel() for p in enc.parameters() if p.requires_grad)
            assert n_lora == len(targets) * LORA["r"] * 2 * enc.config.d_model, f"unexpected LoRA parameter count {n_lora}"
            assert all(("lora_" in n) == p.requires_grad for n, p in enc.named_parameters())
            self.enc = enc; self.D = enc.config.d_model; self.n_lora = n_lora
            self.drop = nn.Dropout(R["dropout"]); self.head = nn.Linear(4 * self.D, 4)
        def hidden(self, feats):
            enc = self.enc
            with torch.no_grad():
                x = nn.functional.gelu(enc.conv1(feats)); x = nn.functional.gelu(enc.conv2(x)).permute(0, 2, 1)
                h = nn.functional.dropout(x + enc.embed_positions.weight[: x.shape[1]], p=enc.dropout, training=self.training)
                for layer in enc.layers[: self.K]: h = run_layer(layer, h)
            keep = None
            for i in range(self.K, len(enc.layers)):
                layer = enc.layers[i]
                if GRAD_CKPT and self.training and torch.is_grad_enabled():
                    h = torch.utils.checkpoint.checkpoint(run_layer, layer, h, use_reentrant=False)
                else:
                    h = run_layer(layer, h)
                if i + 1 == L1: keep = h
            return keep, enc.layer_norm(h)
        def pooled(self, feats, nframes):
            hL1, hlast = self.hidden(feats)
            mask = (torch.arange(hlast.shape[1], device=feats.device)[None] < nframes[:, None]).float().unsqueeze(-1)
            out = []
            for h in (hL1, hlast):
                h = h.float(); mu = (h * mask).sum(1) / mask.sum(1)
                out += [mu, (((h - mu[:, None]) ** 2 * mask).sum(1) / mask.sum(1) + 1e-5).sqrt()]
            return torch.cat(out, -1)
        def forward(self, feats, nframes):
            return self.head(self.drop(self.pooled(feats, nframes)))

    def inputs(rows, train, rng):
        ys = [crop(WAV[i], train, rng) for i in rows]
        nfr = torch.tensor([min(NPOS, math.ceil(len(y) / 16000 * 50)) for y in ys], device=DEV)
        return mel(ys).to(DEV), nfr

    def frozen_pooled(model, rows, rng):      # called right after construction: LoRA B matrices are zero -> exactly the frozen encoder
        model.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(rows), MICRO_BATCH):
                with torch.autocast(DEV, dtype=torch.float16, enabled=USE_AMP):
                    out.append(model.pooled(*inputs(rows[s:s + MICRO_BATCH], False, rng)).float().cpu().numpy())
        return np.vstack(out)

    def predict(model, rows, rng):
        model.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(rows), MICRO_BATCH):
                with torch.autocast(DEV, dtype=torch.float16, enabled=USE_AMP):
                    out.append(model(*inputs(rows[s:s + MICRO_BATCH], False, rng)).float().softmax(-1).cpu().numpy())
        return np.vstack(out)

    if MODE == "smoke":   # equivalence check: manual forward == transformers' WhisperEncoder forward (LoRA B = 0 at init)
        _m = LoraSER().to(DEV).eval(); _rows = TR[:2]
        with torch.no_grad():
            _f, _n = inputs(_rows, False, np.random.default_rng(0))
            _p1 = _m.pooled(_f, _n)
            _hs = _m.enc(input_features=_f, output_hidden_states=True).hidden_states
            _mask = (torch.arange(_hs[-1].shape[1], device=DEV)[None] < _n[:, None]).float().unsqueeze(-1); _p2 = []
            for _h in (_hs[L1], _hs[-1]):
                _mu = (_h * _mask).sum(1) / _mask.sum(1); _p2 += [_mu, (((_h - _mu[:, None]) ** 2 * _mask).sum(1) / _mask.sum(1) + 1e-5).sqrt()]
            _rel = float(((_p1 - torch.cat(_p2, -1)).abs().max() / torch.cat(_p2, -1).abs().max()).cpu())
        print(f"smoke check: manual forward vs transformers forward, max relative diff {_rel:.2e} | trainable LoRA params {_m.n_lora:,}")
        assert _rel < 1e-3, "manual encoder forward does not match transformers"
        del _m

    def train_fold(k, train_rows, valid_rows):
        seed_all(SEED + k); rng = np.random.default_rng(SEED + k)
        model = LoraSER().to(DEV); t0 = time.time()
        if R["head_init"] == "linear_probe":                                  # LP stage
            lp_file = f"{CACHE}/lp_feats_{FT_RECIPE}{SFX}.npy"
            if os.path.exists(lp_file) and np.load(lp_file, mmap_mode="r").shape[0] == len(meta):
                F = np.load(lp_file)
            else:
                F = np.zeros((len(meta), 4 * model.D), np.float32); F[ROWS] = frozen_pooled(model, ROWS, rng); np.save(lp_file, F)
            sc = StandardScaler().fit(F[train_rows]); lr = LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced").fit(sc.transform(F[train_rows]), LAB[train_rows])
            assert (lr.classes_ == np.arange(4)).all()
            Wt = lr.coef_ / sc.scale_[None]; bt = lr.intercept_ - (lr.coef_ * sc.mean_[None] / sc.scale_[None]).sum(1)
            with torch.no_grad():
                model.head.weight.copy_(torch.tensor(Wt, dtype=torch.float32)); model.head.bias.copy_(torch.tensor(bt, dtype=torch.float32))
            lp_valid = lr.predict_proba(sc.transform(F[valid_rows]))
            print(f"  fold {k}: LP (frozen probe) valid macro-F1 {f1_score(LAB[valid_rows], lp_valid.argmax(1), average='macro'):.4f} | {time.time() - t0:.0f} s", flush=True)
        lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("enc.")]
        opt = torch.optim.AdamW([{"params": lora, "lr": R["lr_lora"]}, {"params": model.head.parameters(), "lr": R["lr_head"]}], weight_decay=0.01)
        steps = R["epochs"] * math.ceil(len(train_rows) / LORA["eff_batch"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[R["lr_lora"], R["lr_head"]], total_steps=steps, pct_start=LORA["warmup"])
        scaler = (torch.amp.GradScaler("cuda", enabled=USE_AMP) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=USE_AMP)) if DEV == "cuda" else None
        lossf = nn.CrossEntropyLoss(label_smoothing=LORA["label_smoothing"]); yt = torch.tensor(LAB, dtype=torch.long, device=DEV)
        acc = LORA["eff_batch"] // MICRO_BATCH
        for ep in range(R["epochs"]):
            model.train(); perm = rng.permutation(train_rows); tot = 0.0
            for s in range(0, len(perm), LORA["eff_batch"]):
                opt.zero_grad()
                for j in range(0, LORA["eff_batch"], MICRO_BATCH):
                    half = perm[s + j:s + j + MICRO_BATCH]
                    if not len(half): continue
                    with torch.autocast(DEV, dtype=torch.float16, enabled=USE_AMP):
                        loss = lossf(model(*inputs(half, True, rng)).float(), yt[half]) / acc
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite training loss" + (" - rerun with B9_AMP=0 (fp32)" if USE_AMP else ""))
                    (scaler.scale(loss) if scaler else loss).backward(); tot += float(loss) * acc * len(half)
                if scaler: scaler.step(opt); scaler.update()
                else: opt.step()
                sched.step()
            print(f"  fold {k}: epoch {ep + 1}/{R['epochs']} train loss {tot / len(perm):.4f} | {time.time() - t0:.0f} s", flush=True)
        pv, pt = predict(model, valid_rows, rng), predict(model, TE, rng)
        if SAVE_ADAPTERS:
            torch.save({"lora": get_peft_model_state_dict(model.enc), "head": model.head.state_dict(), "recipe": R}, f"{FT_DIR}/fold{k}_adapter.pt")
        del model, opt; torch.cuda.empty_cache() if DEV == "cuda" else None
        return pv, pt

    ft_oof = np.full((len(TR), 4), np.nan); ft_test = []
    for k in LORA_FOLDS:
        a, b = np.where(FOLD != k)[0], np.where(FOLD == k)[0]
        f = f"{FT_DIR}/fold{k}.npz"
        if os.path.exists(f):
            z = np.load(f); pv, pt = z["valid"], z["test"]; print(f"fold {k}: restored from cache")
        else:
            print(f"fold {k}: train {len(a)} valid {len(b)}", flush=True)
            pv, pt = train_fold(k, TR[a], TR[b]); np.savez(f, valid=pv, test=pt, valid_rows=TR[b])
        ft_oof[b] = pv; ft_test.append(pt)
        print(f"fold {k}: FT valid macro-F1 {f1_score(LAB[TR[b]], pv.argmax(1), average='macro'):.4f}")
    OOF["ft"], TESTP["ft"] = ft_oof, np.mean(ft_test, 0)
    done = ~np.isnan(ft_oof[:, 0])
    print(f"ft ({FT_RECIPE}): OOF macro-F1 on {done.sum()} rows {f1_score(y_tr[done], ft_oof[done].argmax(1), average='macro'):.4f}")
""")

md(r"""
## 4. Validation report
OOF Macro-F1 per family and for the ensemble (overall, non-twin, per class, confusion, per fold); dup-group-disjoint 5-fold
for the frozen families (novel-recording generalisation); paired dup-group bootstrap of the ensemble against each component.
Only rows with predictions from every selected family are used (in `full` mode: all 824).
""")

code(r"""
fam = [f for f in ENSEMBLE if f in OOF]
assert fam, "no family available for the ensemble"
rows_ok = np.all([~np.isnan(OOF[f][:, 0]) for f in fam], axis=0)
yv, tv = y_tr[rows_ok], HAS_TWIN[rows_ok]
def mf1(p, m=None):
    m = np.ones(len(yv), bool) if m is None else m
    return f1_score(yv[m], p[m].argmax(1), average="macro")
P_ens = np.mean([OOF[f][rows_ok] for f in fam], 0)
print(f"rows evaluated: {rows_ok.sum()} | twin share {tv.mean():.3f}")
print(f"{'family':12s} {'OOF':>7s} {'non-twin':>9s} {'twin':>7s}  per-fold")
for f in fam + ["ENSEMBLE"]:
    p = P_ens if f == "ENSEMBLE" else OOF[f][rows_ok]
    pf = [round(f1_score(yv[FOLD[rows_ok] == k], p[FOLD[rows_ok] == k].argmax(1), average="macro"), 4) for k in sorted(set(FOLD[rows_ok]))]
    print(f"{f:12s} {mf1(p):7.4f} {mf1(p, ~tv):9.4f} {mf1(p, tv) if tv.any() else float('nan'):7.4f}  {pf}")
print("\nensemble per-class report:\n", classification_report(yv, P_ens.argmax(1), target_names=CLASSES, digits=4))
print("ensemble confusion (rows=true, cols=pred):\n", pd.DataFrame(confusion_matrix(yv, P_ens.argmax(1)), index=CLASSES, columns=CLASSES))

if MODE == "full":    # novel-recording CV for the frozen families (groups = near-duplicate groups)
    print("\ndup-group-disjoint 5-fold (3 seeds) macro-F1, frozen families:")
    for tag, X in FEATS.items():
        s = []
        for seed in range(3):
            p = np.zeros(len(TR), int)
            for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(TR, y_tr, GROUP[TR]):
                p[b] = lr_model().fit(X[TR][a], y_tr[a]).predict(X[TR][b])
            s.append(f1_score(y_tr, p, average="macro"))
        print(f"  lr_{tag}: {np.mean(s):.4f} +- {np.std(s):.4f}")

# paired dup-group bootstrap: ensemble minus each component / minus the frozen-only blend
g = GROUP[TR][rows_ok]; ug = np.unique(g); mem = {v: np.where(g == v)[0] for v in ug}
rng = np.random.default_rng(0); boots = [np.concatenate([mem[v] for v in rng.choice(ug, len(ug))]) for _ in range(1000)]
def paired(pa, pb):
    d = np.array([f1_score(yv[i], pa[i].argmax(1), average="macro") - f1_score(yv[i], pb[i].argmax(1), average="macro") for i in boots])
    return f"{d.mean():+.4f} [{np.quantile(d, .025):+.4f},{np.quantile(d, .975):+.4f}] P(>0)={np.mean(d > 0):.2f}"
print("\npaired dup-group bootstrap, ENSEMBLE minus:")
for f in fam: print(f"  {f:12s} {paired(P_ens, OOF[f][rows_ok])}")
frozen_only = [f for f in fam if f.startswith("lr_")]
if "ft" in fam and len(frozen_only) > 1: print(f"  frozen-only  {paired(P_ens, np.mean([OOF[f][rows_ok] for f in frozen_only], 0))}")
""")

md(r"""
## 5. Ensemble, twin safety net, submission
Equal-weight probability average of `ENSEMBLE` → argmax. Twin safety net: a test clip whose waveform is a near-copy of training
clips with one consistent label receives that label (uses training labels only — an automated 1-NN on the waveform, not hand
labelling; it changed 0 predictions for B8). Assertions check the schema, order, labels and probabilities before writing.
""")

code(r"""
P_test = np.mean([TESTP[f] for f in fam], 0)
assert P_test.shape == (len(TE), 4) and np.isfinite(P_test).all() and np.allclose(P_test.sum(1), 1, atol=1e-4)
pred = CLASSES[P_test.argmax(1)]
changed = 0
if TWIN_OVERRIDE:
    for j, t in enumerate(TE):
        if TEST_TWIN[j]:
            labels = {str(meta.emotion[i]) for i in TR if GROUP[i] == GROUP[t]}
            if len(labels) == 1:
                lab = labels.pop(); changed += int(pred[j] != lab); pred[j] = lab
print("twin safety net changed", changed, "predictions")
sub = pd.DataFrame({"Id": meta.Id.values[TE], "emotion": pred})
assert len(sub) == len(sample) == 209 and (sub.Id.values == sample.Id.values).all(), "row count / Id order"
assert sub.emotion.isin(CLASSES).all() and not sub.emotion.isna().any(), "invalid or missing label"
assert sub.Id.is_unique
OUT_CSV = "submission.csv" if MODE == "full" else "submission_smoke.csv"     # a smoke-mode file must never be submitted
sub.to_csv(OUT_CSV, index=False)
np.savez(f"{CACHE}/b9_probs{SFX}.npz", **{f"oof__{k}": v for k, v in OOF.items()}, **{f"test__{k}": v for k, v in TESTP.items()},
         y=y_tr, fold=FOLD, train_rows=TR, test_ids=meta.Id.values[TE], has_twin=HAS_TWIN, families=np.array(fam))
print(f"{OUT_CSV} written ({MODE} mode) | class counts {sub.emotion.value_counts().to_dict()}")
for ref in sorted(glob.glob("../my_submissions/*/submission.csv")):
    r = pd.read_csv(ref); print(f"  vs {ref.split('/')[-2]}: {(r.emotion.values != sub.emotion.values).sum()} rows differ")
sub.head()
""")

# Substitute placeholders in the Python cell sources BEFORE serialising the notebook (so JSON escaping is handled by json.dump).
assert "'" not in FOLD_MAP and "'" not in json.dumps(sel["ensemble"])
for c in cells:
    c["source"] = (c["source"].replace("__FOLD_MAP__", FOLD_MAP).replace("__FT_RECIPE__", sel["ft_recipe"])
                   .replace("__ENSEMBLE__", json.dumps(sel["ensemble"])).replace("__SELECTION_NOTE__", sel.get("note", "").replace('"', "'")))
    assert "__" + "FOLD_MAP__" not in c["source"] and "__" + "ENSEMBLE__" not in c["source"]
for i, c in enumerate(cells): c["id"] = f"b9-{i:02d}-{c['cell_type']}"          # nbformat 4.5 requires cell ids
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
print("wrote", out, len(cells), "cells | selection:", sel)
