"""Build TOP_100_RESOURCES.md from external_resources.csv.

Ranks 1-N follow a hand-curated order (informed by the experiments in research/work); the rest are filled by
catalog score (relevance x2 + actionability), skipping near-duplicate titles. Custom 'how to use / impact /
difficulty' text for curated rows; derived text for the remainder.

Usage: python -I build_top100.py <external_resources.csv> <out_md>
"""
import re
import sys

import pandas as pd

a = pd.read_csv(sys.argv[1], dtype=str).fillna("").set_index("resource_id")
C = {  # id: (how to use, expected impact, difficulty)
    "R289": ("Frozen layer-band (10-18) probe already gives the best dup-group CV here (0.80); next: fine-tune with the public WavLM pipeline (--models facebook/wav2vec2-xls-r-300m)", "High (+0.01-0.03 non-twin F1)", "Medium"),
    "R008": ("E0 reference pipeline; verified OOF 0.862 / public 0.879; reuse its folds (train_with_folds.csv) as canonical", "High (baseline)", "Low (fork)"),
    "R003": ("Compliance: 5 subs/day, 2 finals, no hand-labelling of test, OSI-licensed code, public+free external data", "Critical (eligibility)", "Low"),
    "R001": ("Metric = Macro-F1; final = 0.4 public + 0.6 private -> public is not irrelevant but private dominates", "Critical", "Low"),
    "R281": ("Frozen band 12-18 gives 0.78 dup-group CV; already fine-tuned in the public pipeline - keep as a blend family", "Medium-High", "Low"),
    "R111": ("Template for the frozen probe used here (layer selection/weighting + light head)", "High", "Low"),
    "R082": ("Backbone shortlist: Whisper-large-v3 encoder ranked 1st on 23/32 corpora -> next frozen family to test (E5)", "Medium-High", "Medium"),
    "R313": ("Frozen encoder probe; trim the 30 s zero-padding frames before pooling (the public Whisper repo did not)", "Medium-High", "Medium"),
    "R239": ("Emotion-specialised embeddings via FunASR; label set covers the 4 classes -> frozen LR family + zero-shot sanity check", "Medium-High", "Medium (FunASR install; custom licence check)"),
    "R138": ("Read before trusting any CV/LB delta: with n~800 and LB subsets of ~50-100 clips, error bars are several points", "High (avoids bad decisions)", "Low"),
    "R235": ("Compute Macro-F1 on pooled OOF predictions (done in all scripts), not as a mean of fold F1s", "Medium", "Low"),
    "R209": ("Why HPO/threshold tuning on 824 rows overfits the selection criterion; justifies equal-weight blends", "High (private robustness)", "Low"),
    "R200": ("LP-FT: train the head on frozen features first, then unfreeze (reduces feature distortion on tiny data)", "Medium", "Medium"),
    "R117": ("LoRA on WavLM-large / XLS-R / Whisper encoder as a cheaper, less overfit-prone alternative to full FT", "Medium", "Medium"),
    "R280": ("Fast backbone for multi-seed runs; frozen L6-8 is a cheap diverse blend member (+0.018 blend gain verified)", "Medium", "Low"),
    "R295": ("Largest multilingual SSL with Hindi (MIT); frozen probe as an extra family", "Medium", "Medium"),
    "R301": ("Indic-only SSL; test as frozen family - Hindi in-domain pretraining may help non-twin clips", "Medium", "Medium (fairseq checkpoints)"),
    "R302": ("HF-ready Hindi wav2vec2 (gated, auto-approve) - simplest Indic frozen family", "Medium", "Low-Medium"),
    "R304": ("Hindi-only wav2vec2 (4,200 h); frozen probe", "Medium", "Medium"),
    "R315": ("Hindi Whisper encoder (includes spontaneous Gramvaani) as a frozen family", "Medium", "Medium"),
    "R211": ("Shows re-processed recordings are typical leakage; matches the 227 near-duplicate groups found here", "High (validation design)", "Low"),
    "R183": ("Run on blend OOF to list likely label errors (Id 75/636/878 conflict known); inspect, do not auto-relabel", "Low-Medium", "Low"),
    "R428": ("Implementation of confident learning for E13", "Low-Medium", "Low"),
    "R139": ("Odyssey 2024 winner: prior-weighted focal loss + model voting; macro-F1 metric like here", "Medium", "Medium"),
    "R195": ("Per-class thresholds gave ~+1% macro-F1 on a large imbalanced set; here classes are balanced -> expect less (E7, nested only)", "Low", "Low"),
    "R184": ("Post-hoc logit adjustment for balanced error - cheap E7 variant", "Low", "Low"),
    "R137": ("Theory for F1-optimal thresholds; use only inside nested CV", "Low", "Low"),
    "R234": ("Make sure the metric implementation is the average-of-per-class-F1 version (sklearn average='macro')", "Low (correctness)", "Low"),
    "R220": ("Cornell 1st place rose from 7th public to 1st private: robust CV + diverse voting beat LB chasing", "High (strategy)", "Low"),
    "R215": ("Freesound 1st: relabel suspicious clips, MLP stacker, geometric blending - stacking only if nested CV supports it", "Medium", "Medium"),
    "R194": ("Small non-English macro-F1 SER challenge won with attention pooling over frozen foundation features", "Medium", "Low"),
    "R115": ("Pooling head used by the public FT pipeline; candidate for a frozen-feature MLP head", "Low-Medium", "Low"),
    "R116": ("Learned layer-weighted wav2vec2 features for SER: alternative to fixed layer bands", "Low-Medium", "Low"),
    "R189": ("Speaker-dependent vs independent gap (~6.5 pts): here the split is random, so speaker overlap is expected in test", "Medium (validation)", "Low"),
    "R202": ("Which WavLM layers to fine-tune; partial FT as a regulariser", "Low-Medium", "Medium"),
    "R166": ("Evidence that pitch/speed augmentation gives only tenths of a point with strong SSL - do not prioritise", "Medium (saves time)", "Low"),
    "R121": ("Speed perturbation is the most robust augmentation if any is used", "Low", "Low"),
    "R126": ("SpecAugment-style masking (already mild in public pipeline)", "Low", "Low"),
    "R127": ("Mixup for FT runs; reduces memorisation of noisy labels", "Low", "Low"),
    "R167": ("CopyPaste augmentation idea (emotional + neutral concatenation)", "Low", "Medium"),
    "R411": ("Label-preserving waveform augmentation library", "Low", "Low"),
    "R246": ("4-class IEMOCAP model with the exact label set: zero-shot sanity check / extra frozen family", "Low-Medium", "Low"),
    "R242": ("Used by the Avik Das notebook; exact 4-class head (IEMOCAP English)", "Low", "Low"),
    "R260": ("Urdu 4-class model (same labels): closest-language zero-shot/initialisation", "Low-Medium", "Low"),
    "R253": ("Spontaneous-speech WavLM SER (MSP-Podcast); embeddings as a frozen family", "Low-Medium", "Low"),
    "R255": ("IS2025-challenge-derived WavLM-large SER; frozen family", "Low-Medium", "Low"),
    "R245": ("Arousal/valence/dominance outputs as 3 extra stacking features (CC BY-NC-SA: competition-uncertain under OSI rule)", "Low", "Low"),
    "R269": ("Fast emotion tags; zero-shot sanity check only (Chinese/English training)", "Low", "Low"),
    "R345": ("eGeMAPS 88 features as E2 hand-crafted family", "Low-Medium", "Low"),
    "R405": ("F0/jitter/shimmer/HNR for E2", "Low", "Low"),
    "R083": ("Paper behind emotion2vec; linear probe is the intended use", "Medium", "Low"),
    "R030": ("Hindi SER benchmark (BhavVani sitcom): spontaneous Hindi is hard (WF1 ~0.44) - this dataset is easier/acted-heavy", "Background", "Low"),
    "R032": ("Indo-Aryan cross-lingual transfer evidence -> prefer Urdu/Bangla over English as auxiliary data (E12)", "Low-Medium", "Low"),
    "R015": ("Acted Hindi, 4 classes, ODbL; checked for overlap (see report) -> E12 auxiliary data", "Low-Medium", "Low"),
    "R017": ("CC BY 4.0 Hindi/Marathi; E12 auxiliary data after overlap check", "Low", "Low"),
    "R051": ("UrduSER (CC BY 4.0) 4-class; E12", "Low-Medium", "Low"),
    "R045": ("SUBESCO (CC BY 4.0) Bangla 4-class; E12", "Low", "Low"),
    "R071": ("CREMA-D (ODbL) English 4-class; E12", "Low", "Low"),
    "R050": ("URDU spontaneous 4-class; README research-only -> competition-uncertain", "Low", "Low"),
    "R089": ("Semi-supervised cross-lingual SER with pseudo-labels -> E9 design", "Low-Medium", "Medium"),
    "R145": ("Class-balanced top-k pseudo-label selection -> E9", "Low", "Low"),
    "R136": ("Temperature scaling before averaging heterogeneous models (FT vs LR are calibrated differently)", "Low", "Low"),
    "R225": ("BirdCLEF 2024 1st-place pseudo-labelling/distillation; much larger unlabeled pool than here", "Low", "High"),
    "R012": ("Source dataset sample - audit only; using its labels for test rows breaks the no-hand-labelling rule", "Negative if misused", "-"),
    "R004": ("Score-granularity analysis: public subset ~52-105 clips; 1 error ~ 0.01-0.02", "High (strategy)", "Low"),
    "R009": ("Inference notebook whose test_probabilities.csv was reused for blending", "High", "Low"),
}
ORDER = ["R289", "R008", "R003", "R001", "R281", "R111", "R082", "R313", "R239", "R138", "R209", "R220", "R004", "R211", "R009", "R235",
         "R200", "R117", "R280", "R295", "R301", "R302", "R304", "R315", "R083", "R189", "R194", "R139", "R166", "R183", "R428", "R195", "R184",
         "R137", "R234", "R215", "R115", "R116", "R202", "R121", "R126", "R127", "R167", "R411", "R246", "R242", "R260", "R253", "R255", "R245",
         "R269", "R345", "R405", "R030", "R032", "R015", "R017", "R051", "R045", "R071", "R050", "R089", "R145", "R136", "R225", "R012"]
ORDER = [i for i in ORDER if i in a.index]
rs = {"High": 3, "Medium": 2, "Low": 1}
ap = {"Immediate experiment": 3, "Potential experiment": 2, "Conceptual": 1, "Background": 0}
a["score"] = a.relevance.map(rs).fillna(0) * 2 + a.competition_applicability.map(ap).fillna(0)


def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())[:40]


seen = {norm(a.loc[i, "title"]) for i in ORDER}
rest = []
for i in a.sort_values("score", ascending=False).index:
    if i in ORDER or norm(a.loc[i, "title"]) in seen:
        continue
    seen.add(norm(a.loc[i, "title"]))
    rest.append(i)
ids = (ORDER + rest)[:100]
imp = {"Immediate experiment": "Medium", "Potential experiment": "Low-Medium", "Conceptual": "Low", "Background": "Background"}
lines = ["# Top 100 resources - DataVerse: Detecting Emotions from Hindi Speech", "",
         "Ranked for usefulness to *this* competition (826 train / 209 test, Macro-F1, 37% of test rows have a train twin).",
         f"Ranks 1-{len(ORDER)} are hand-ordered using the experiments in `research/work`; ranks {len(ORDER) + 1}-100 are filled by catalog score "
         "(relevance x2 + actionability) with duplicate titles removed. Full metadata for every row: `research/external_resources.csv`.",
         "Expected impact refers to Macro-F1 on non-twin test clips unless stated; all impacts are estimates (YOUR HYPOTHESIS), not measured results, except where the text cites a measured number from this research.", "",
         "| Rank | ID | Resource | Type | Year | Key idea | Relevance | How to use it here | Expected impact | Difficulty | Link |",
         "|---:|---|---|---|---|---|---|---|---|---|---|"]
for k, i in enumerate(ids, 1):
    r = a.loc[i]
    how, impact, diff = C.get(i, (r.experiment_inspired or r.key_insight, imp.get(r.competition_applicability, "Low"),
                                  "Low" if r.category in ("Documentation", "Audio library", "Training framework") else "Medium"))
    cell = lambda s: str(s).replace("|", "/").replace("\n", " ")[:220]
    lines.append(f"| {k} | {i} | {cell(r.title)[:90]} | {cell(r.resource_type or r.category)[:30]} | {cell(r.year)[:12]} | {cell(r.key_insight)} | {r.relevance} | "
                 f"{cell(how)} | {cell(impact)} | {cell(diff)} | {r.url} |")
open(sys.argv[2], "w").write("\n".join(lines) + "\n")
print("top100 rows:", len(ids), "curated:", len(ORDER))
