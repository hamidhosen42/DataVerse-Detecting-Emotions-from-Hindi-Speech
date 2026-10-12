"""Render catalog tables (papers, repos, models, datasets, competitions) as markdown fragments for the report.

Usage: python -I build_report_tables.py <external_resources.csv> <out_dir>
"""
import re
import sys

import pandas as pd

a = pd.read_csv(sys.argv[1], dtype=str).fillna("").set_index("resource_id")
OUT = sys.argv[2]


def c(s, n=160):
    return str(s).replace("|", "/").replace("\n", " ").strip()[:n]


def grab(notes, key):
    m = re.search(key + r"[=:]\s*([^;]+)", notes)
    return m.group(1).strip() if m else "UNVERIFIED"


# ---- papers: (id, proposed experiment, difficulty, likely impact)
P = [("R111", "Frozen layer-wise probe across encoders (done: X03-X05)", "Low", "High"),
     ("R082", "Add Whisper-large-v3 encoder as frozen family (E5)", "Medium", "Medium-High"),
     ("R083", "emotion2vec+ frozen family (E5)", "Medium", "Medium"),
     ("R084", "Whisper vs SSL frozen comparison; logit adjustment for external data (E5/E12)", "Medium", "Medium"),
     ("R116", "Learned layer weights instead of fixed band for frozen probes", "Low", "Low-Medium"),
     ("R115", "Attentive stats pooling head on frozen frames", "Medium", "Low"),
     ("R189", "Interpret twin/speaker effects in CV (speaker-dependent vs independent gap)", "Low", "Medium (validation)"),
     ("R200", "LP-FT schedule for XLS-R fine-tune (E3)", "Medium", "Medium"),
     ("R117", "LoRA fine-tune of XLS-R / WavLM-large (E3 variant)", "Medium", "Medium"),
     ("R202", "Partial vs full WavLM fine-tuning", "Medium", "Low-Medium"),
     ("R100", "Use robust wav2vec2 SER features; valence/arousal as auxiliary signals", "Low", "Low"),
     ("R138", "Bootstrap CIs for every decision (done: X08-X10, X12)", "Low", "High (decision quality)"),
     ("R235", "Pooled OOF Macro-F1 (done)", "Low", "Medium"),
     ("R209", "Avoid tuning blend weights / thresholds on 824 rows", "Low", "High (private robustness)"),
     ("R234", "Confirm macro-F1 definition = mean of per-class F1", "Low", "Low"),
     ("R137", "Nested threshold tuning (E7)", "Low", "Low"),
     ("R184", "Post-hoc logit adjustment (E7)", "Low", "Low"),
     ("R136", "Temperature-scale FT and LR families before averaging (E7)", "Low", "Low"),
     ("R183", "Label-issue audit on OOF (E13)", "Low", "Low-Medium"),
     ("R211", "Fingerprint/correlation duplicate scan (done: X00)", "Low", "High (validation)"),
     ("R212", "Waveform-level near-dup detection beats hashing (done: X00)", "Low", "High (validation)"),
     ("R166", "Skip pitch-shift augmentation; low expected gain", "Low", "Low"),
     ("R121", "Speed perturbation if any augmentation is used in FT", "Low", "Low"),
     ("R126", "Mild SpecAugment in FT", "Low", "Low"),
     ("R127", "Mixup in FT runs", "Low", "Low"),
     ("R167", "CopyPaste augmentation in FT", "Medium", "Low"),
     ("R139", "Prior-weighted focal loss + model voting", "Medium", "Low-Medium"),
     ("R195", "Per-class threshold tuning (nested)", "Low", "Low"),
     ("R194", "Attention pooling over frozen features", "Medium", "Low-Medium"),
     ("R156", "Meta-classifier / model soup (only with nested CV)", "High", "Low"),
     ("R030", "Context: spontaneous Hindi SER is hard; acted portions dominate this dataset", "Low", "Background"),
     ("R032", "Prefer Indo-Aryan auxiliary corpora (E12)", "Medium", "Low-Medium"),
     ("R086", "Small fraction of target-language data matters most (E12)", "Medium", "Low"),
     ("R089", "Pseudo-labelling unlabeled target-language speech (E9)", "Medium", "Low-Medium"),
     ("R145", "Class-balanced top-k pseudo-label selection (E9)", "Low", "Low"),
     ("R128", "Speaker normalization (low priority: random split shares speakers)", "High", "Low"),
     ("R172", "Focal loss variant for FT", "Low", "Low"),
     ("R206", "SWA for FT checkpoints", "Low", "Low"),
     ("R205", "Label smoothing (already 0.02 in public pipeline)", "Low", "Low")]
L = ["| # | Paper | Authors | Year | Contribution | Why relevant here | Proposed experiment | Difficulty | Likely impact | Link |", "|---:|---|---|---|---|---|---|---|---|---|"]
for k, (i, e, d, imp) in enumerate([p for p in P if p[0] in a.index], 1):
    r = a.loc[i]
    L.append(f"| {k} | {c(r.title, 90)} | {c(r.authors, 50)} | {c(r.year, 10)} | {c(r.key_insight, 170)} | {c(r.method or r.task, 80)} | {e} | {d} | {imp} | {r.paper_url or r.url} |")
open(f"{OUT}/papers.md", "w").write("\n".join(L) + "\n")

# ---- repos
R = ["R386", "R377", "R378", "R380", "R390", "R351", "R354", "R350", "R355", "R346", "R374", "R301", "R315", "R345", "R405", "R411", "R428", "R423",
     "R424", "R416", "R353", "R435", "R438", "R441", "R356", "R357", "R370", "R368", "R220", "R215", "R447", "R446"]
L = ["| # | Repository | Stars (seen 2026-10-11) | Main method | Licence | Checkpoints | Direct usefulness | Experiment it enables | Link |", "|---:|---|---|---|---|---|---|---|---|"]
for k, i in enumerate([x for x in R if x in a.index], 1):
    r = a.loc[i]
    stars = grab(r.notes, "stars")
    ck = grab(r.notes, "checkpoints available")
    L.append(f"| {k} | {c(r.title, 70)} | {stars} | {c(r.method or r.category, 70)} | {c(r.license, 30)} | {c(ck, 25)} | {r.relevance} | {c(r.experiment_inspired or r.key_insight, 130)} | {r.github_url or r.url} |")
open(f"{OUT}/repos.md", "w").write("\n".join(L) + "\n")

# ---- models
M = ["R289", "R281", "R280", "R313", "R314", "R311", "R239", "R238", "R236", "R295", "R290", "R302", "R304", "R315", "R317", "R321", "R260", "R246",
     "R242", "R253", "R255", "R245", "R292", "R298", "R269", "R270", "R340", "R338"]
L = ["| # | Model | Architecture / size | Training data | Hindi in pretraining? | Licence | Compute (T4) | Rule risk | Recommended experiment | Link |", "|---:|---|---|---|---|---|---|---|---|---|"]
for k, i in enumerate([x for x in M if x in a.index], 1):
    r = a.loc[i]
    n = r.notes
    L.append(f"| {k} | {c(r.title, 60)} | {c(grab(n, 'params'), 45)} | {c(grab(n, 'pretraining data') if 'pretraining data' in n else grab(n, 'data'), 60)} | {c(grab(n, 'Hindi'), 30)} | "
             f"{c(r.license, 28)} | {c(grab(n, 'T4'), 50)} | {c(grab(n, 'rule risk'), 60)} | {c(r.experiment_inspired, 110)} | {r.model_url or r.url} |")
open(f"{OUT}/models.md", "w").write("\n".join(L) + "\n")

# ---- datasets
d = a[a.category.str.contains("dataset", case=False) & ~a.category.str.contains("repo", case=False)].copy()


def klass(n):
    for k in ("Competition-safe", "Competition-uncertain", "Research-only", "Not recommended"):
        if n.startswith(k) or k in n[:40]:
            return k
    return "Competition-uncertain"


d["cls"] = d.notes.map(klass)
order = {"Competition-safe": 0, "Competition-uncertain": 1, "Research-only": 2, "Not recommended": 3}
d = d.sort_values(["cls", "relevance"], key=lambda s: s.map(order) if s.name == "cls" else s.map({"High": 0, "Medium": 1, "Low": 2}))
L = ["| # | Dataset | Task / labels | Domain | Licence | Relevance | Classification | Notes (incl. potential use) | Link |", "|---:|---|---|---|---|---|---|---|---|"]
for k, (i, r) in enumerate(d.iterrows(), 1):
    L.append(f"| {k} | {c(r.title, 70)} | {c(r.task, 50)} | {c(r.domain, 40)} | {c(r.license, 35)} | {r.relevance} | **{r.cls}** | {c(r.notes, 170)} | {r.dataset_url or r.url} |")
open(f"{OUT}/datasets.md", "w").write("\n".join(L) + "\n")
print("dataset classes:", d.cls.value_counts().to_dict())

# ---- competitions / winning solutions
w = a[a.category.isin(["Previous competition", "Winning solution", "Winning solution code"])]
L = ["| # | Competition / solution | Year | Key idea (as stated by the source) | Transferable idea | Link |", "|---:|---|---|---|---|---|"]
for k, (i, r) in enumerate(w.iterrows(), 1):
    L.append(f"| {k} | {c(r.title, 90)} | {c(r.year, 10)} | {c(r.key_insight, 200)} | {c(r.experiment_inspired, 120)} | {r.url} |")
open(f"{OUT}/competitions.md", "w").write("\n".join(L) + "\n")
print("papers", len(P), "repos", len(R), "models", len(M), "datasets", len(d), "competitions", len(w))
