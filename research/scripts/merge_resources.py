"""Merge the four agent-built resource catalogs + competition-specific sources into external_resources.csv.

Dedup rule: identical normalized URL across different source files -> keep the first row (the
same page listed by two catalogs). Rows from the SAME catalog that share a URL are kept, because they
describe different resources hosted on one page (e.g. IITKGP Hindi vs Telugu corpora).

Usage: python -I merge_resources.py <resources_dir> <out_csv>
"""
import glob
import sys

import pandas as pd

d, out = sys.argv[1], sys.argv[2]
COLS = ["category", "title", "authors", "year", "resource_type", "venue", "url", "paper_url", "github_url",
        "dataset_url", "model_url", "task", "domain", "method", "relevance", "evidence_level", "reproducibility",
        "license", "competition_applicability", "key_insight", "experiment_inspired", "notes"]
K = "https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech"


def row(**kw):
    r = {c: "" for c in COLS}
    r.update(kw)
    return r


own = [
    row(category="Competition material", title="DataVerse: Detecting Emotions from Hindi Speech - Overview / Evaluation", authors="ENIAC, SRCASW (University of Delhi)", year="2026", resource_type="Competition page", venue="Kaggle Community", url=f"{K}/overview", task="4-class SER", domain="Hindi speech", relevance="High", evidence_level="Technical documentation", reproducibility="n/a", license="Kaggle competition rules", competition_applicability="Immediate experiment", key_insight="Metric is Macro-F1; final score = 0.40*public + 0.60*private; dataset of acted + spontaneous Hindi speech", experiment_inspired="Metric-aware decision rule; do not ignore public LB (40% weight)", notes="OFFICIAL FACT"),
    row(category="Competition material", title="DataVerse - Data description", authors="ENIAC, SRCASW", year="2026", resource_type="Competition page", venue="Kaggle Community", url=f"{K}/data", task="4-class SER", domain="Hindi speech", relevance="High", evidence_level="Technical documentation", reproducibility="n/a", license="Competition use + citation", competition_applicability="Immediate experiment", key_insight="826 train / 209 test wav files; submission columns Id,emotion with lowercase labels (sample_submission)", experiment_inspired="Submission format check", notes="OFFICIAL FACT. Evaluation page shows a different (AUDIO_ID / PREDICTED LABEL, capitalized) format; the sample_submission format is what was accepted for 46 teams"),
    row(category="Competition material", title="DataVerse - Rules", authors="ENIAC, SRCASW", year="2026", resource_type="Competition page", venue="Kaggle Community", url=f"{K}/rules", task="4-class SER", domain="Hindi speech", relevance="High", evidence_level="Technical documentation", reproducibility="n/a", license="", competition_applicability="Immediate experiment", key_insight="5 submissions/day, 2 final selections, teams <= 3, external data/models allowed if public & free; no hand-labeling of test; OSI open-source code requirement", experiment_inspired="Rule-compliance checklist", notes="OFFICIAL FACT"),
    row(category="Competition material", title="DataVerse - Public leaderboard", authors="Kaggle", year="2026", resource_type="Leaderboard", venue="Kaggle", url=f"{K}/leaderboard", task="4-class SER", domain="Hindi speech", relevance="High", evidence_level="Technical documentation", reproducibility="n/a", competition_applicability="Conceptual", key_insight="46 teams; top public 0.96544 (3-way tie); exact score values consistent with a public subset of ~52-105 clips", experiment_inspired="LB granularity analysis", notes="VERIFIED PUBLIC RESULT (snapshot 2026-10-10)"),
    row(category="Competition discussion", title="Registration (Unstop link) - organizer post", authors="Competition host", year="2026", resource_type="Discussion", venue="Kaggle forum", url=f"{K}/discussion/745561", relevance="Medium", evidence_level="Technical documentation", competition_applicability="Background", key_insight="Unstop registration is compulsory to claim prizes; register after submitting at least once", notes="ORGANIZER CLARIFICATION"),
    row(category="Competition discussion", title="Re: Registration - organizer reminder", authors="Competition host", year="2026", resource_type="Discussion", venue="Kaggle forum", url=f"{K}/discussion/747812", relevance="Medium", evidence_level="Technical documentation", competition_applicability="Background", key_insight="Only 8 teams had registered on Unstop as of 2026-10-09", notes="ORGANIZER CLARIFICATION"),
    row(category="Competition discussion", title="Will all participants receive a certificate?", authors="Participant + host (pgolephd)", year="2026", resource_type="Discussion", venue="Kaggle forum", url=f"{K}/discussion/746206", relevance="Medium", evidence_level="Technical documentation", competition_applicability="Background", key_insight="Certificates only for Unstop-registered; GitHub repo link required so hosts can verify predictions; best model can be updated until 14 Oct 12:00 AM", notes="ORGANIZER CLARIFICATION"),
    row(category="Public notebook", title="01_dataverse_training (WavLM Base+ / Large fine-tuning, 5-fold)", authors="Kunal Aarse", year="2026", resource_type="Kaggle notebook", venue="Kaggle", url="https://www.kaggle.com/code/kunalinfinite/01-dataverse-training", model_url="https://huggingface.co/microsoft/wavlm-large", task="4-class SER", domain="Hindi speech", method="WavLM fine-tune, learned layer weights, attentive statistics pooling, LLRD 0.95, 8 s crops, mild gain/noise aug, label smoothing 0.02, exact-dup-aware folds", relevance="High", evidence_level="Competition solution", reproducibility="Code public, no saved outputs; needs T4 + internet", license="Apache 2.0 (Kaggle default)", competition_applicability="Immediate experiment", key_insight="Strongest public end-to-end pipeline; self-reported OOF Macro-F1 0.789 for an earlier version (UNVERIFIED)", experiment_inspired="E0 baseline reproduction", notes="Paired inference notebook scores public 0.87918 (team Kunal Aarse; also the user's own submission)"),
    row(category="Public notebook", title="02_dataverse_inference (offline inference for 01)", authors="Kunal Aarse", year="2026", resource_type="Kaggle notebook", venue="Kaggle", url="https://www.kaggle.com/code/kunalinfinite/02-dataverse-inference", task="4-class SER", domain="Hindi speech", method="Equal-weight probability average over fold checkpoints, start/middle/end crops", relevance="High", evidence_level="Competition solution", reproducibility="Depends on 01 outputs", competition_applicability="Immediate experiment", key_insight="Public LB 0.87918 (VERIFIED: user's own submission of this notebook)", experiment_inspired="E0", notes=""),
    row(category="Public notebook", title="Acoustic Source-Filter Speech Affect Recognition", authors="Avik Das", year="2026", resource_type="Kaggle notebook", venue="Kaggle", url="https://www.kaggle.com/code/avikdas567/acoustic-source-filter-speech-affect-recognition", model_url="https://huggingface.co/superb/wav2vec2-base-superb-er", task="4-class SER", domain="Hindi speech", method="Hand-crafted prosodic/spectral features + frozen superb/wav2vec2-base-superb-er embeddings; MLP + LightGBM + XGBoost blend; Nelder-Mead class-threshold scaling on OOF", relevance="Medium", evidence_level="Competition solution", reproducibility="Code public, no saved outputs", competition_applicability="Potential experiment", key_insight="Notebook best public score 0.72790 (VERIFIED via Kaggle search API); thresholds tuned on the same OOF used to report the score (optimistic)", experiment_inspired="E7 class-bias tuning (nested)", notes="Most-voted notebook (6 votes)"),
    row(category="Public notebook", title="dataverse-dehs-baseline (MFCC/chroma/mel/contrast/ZCR + XGBoost)", authors="Shivam", year="2026", resource_type="Kaggle notebook", venue="Kaggle", url="https://www.kaggle.com/code/shivamgravity/dataverse-dehs-baseline", task="4-class SER", domain="Hindi speech", method="188 clip-level librosa features, XGBoost, 5-fold StratifiedKFold (accuracy)", relevance="Low", evidence_level="Competition solution", reproducibility="Code public, CPU", competition_applicability="Background", key_insight="Notebook best public score 0.74925 (VERIFIED via Kaggle search API)", experiment_inspired="Hand-crafted feature branch", notes=""),
    row(category="Participant repo", title="SRCASW-BhavVaani sample release (competition source dataset)", authors="Himanshi, Sonali Jha, Pragya Sethia, Swati Pal", year="2026", resource_type="GitHub repo", url="https://github.com/Himanshi7104/SRCASW-BhavVaani", dataset_url="https://github.com/Himanshi7104/SRCASW-BhavVaani", task="SER dataset", domain="Hindi speech", relevance="High", evidence_level="Official repository", reproducibility="20 wav files", license="research-only (README); no redistribution", competition_applicability="Background", key_insight="All 20 sample clips match competition audio (NCC>=0.99); 5 match TEST clips -> test-label leakage if used", experiment_inspired="Leakage audit only", notes="Not recommended: using its labels for test rows would violate the no-hand-labeling rule"),
]
dfs = []
for f in sorted(glob.glob(f"{d}/res_*.csv")):
    x = pd.read_csv(f, dtype=str).fillna("")
    x["_src"] = f.split("/")[-1]
    dfs.append(x)
a = pd.concat([pd.DataFrame(own).assign(_src="competition")] + dfs, ignore_index=True)
a["_u"] = a.url.str.strip().str.rstrip("/").str.lower().str.replace("http://", "https://").str.replace("www.", "")
first_src = a.groupby("_u")._src.transform("first")
a = a[a._src == first_src].copy()
a = a.drop_duplicates(subset=["_u", "title"])
a.insert(0, "resource_id", [f"R{i:03d}" for i in range(1, len(a) + 1)])
a[["resource_id"] + COLS].to_csv(out, index=False)
print("rows", len(a), "| by source", a._src.value_counts().to_dict())
print(a.category.value_counts().to_string())
