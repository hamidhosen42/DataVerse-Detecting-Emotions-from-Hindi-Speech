# DataVerse: Detecting Emotions from Hindi Speech — Deep Competition Analysis

*Research snapshot: 2026-10-10 ~20:00 UTC (≈ 46.5 h before the 2026-10-12 18:30 UTC deadline). Prepared for Kaggle user `hosen42` (team "Md. Hamid Hosen", public rank 27 / 46, 0.87918, 1 submission used).*

**Evidence tags used throughout.** `OFFICIAL FACT` (competition pages/rules), `ORGANIZER CLARIFICATION` (host forum replies), `VERIFIED PUBLIC RESULT` (leaderboard, notebook output files, your own submission list), `PARTICIPANT CLAIM`, `EXTERNAL RESEARCH EVIDENCE` (papers/repos), `MEASURED HERE` (experiments run for this report; scripts + logs in `research/`), `YOUR HYPOTHESIS` (my inference), `YOUR RECOMMENDATION`, and `UNVERIFIED`.

## Table of contents

1. [Executive summary](#1-executive-summary) · [1b. Round-2 update](#1b-round-2-experiment-update-2026-10-10-2030-2330-utc--supersedes-the-candidate-choice-above)
2. [Competition breakdown & rules](#2-competition-breakdown--rules)
3. [Dataset analysis & schema](#3-dataset-analysis--schema)
4. [Target analysis](#4-target-analysis)
5. [Metric analysis](#5-metric-analysis)
6. [Leaderboard analysis](#6-leaderboard-analysis)
7. [Public notebooks, ranking & lineage](#7-public-notebooks-ranking--lineage)
8. [Discussions & organizer clarifications](#8-discussions--organizer-clarifications)
9. [External resources (453) & Top 100](#9-external-resources-453--top-100)
10. [Must-read papers](#10-must-read-papers)
11. [Top GitHub repositories](#11-top-github-repositories)
12. [Pretrained models](#12-pretrained-models)
13. [Related datasets & rule safety](#13-related-datasets--rule-safety)
14. [Related competitions & transferable ideas](#14-related-competitions--transferable-ideas)
15. [Validation strategy](#15-validation-strategy)
16. [Leakage analysis & automated tests](#16-leakage-analysis--automated-tests)
17. [Distribution shift](#17-distribution-shift)
18. [Duplicates](#18-duplicates)
19. [Seed / fold / leaderboard noise](#19-seed--fold--leaderboard-noise)
20. [Experiments run for this report](#20-experiments-run-for-this-report)
21. [Experiment roadmap E0–E13](#21-experiment-roadmap-e0e13)
22. [Top 10 experiments](#22-top-10-experiments)
23. [Final comparison table](#23-final-comparison-table)
24. [Recommended ensemble & final-submission selection](#24-recommended-ensemble--final-submission-selection)
25. [Approaches to avoid](#25-approaches-to-avoid)
26. [Private-leaderboard strategy](#26-private-leaderboard-strategy)
27. [Step-by-step execution plan (Stages 1–10)](#27-step-by-step-execution-plan-stages-110)
28. [Direct source links](#28-direct-source-links)
29. [Final conclusions](#29-final-conclusions)
30. [Reproducibility: files produced](#30-reproducibility-files-produced)

---

## 1. Executive summary

**What the competition is.** 4-class speech emotion recognition (angry / happy / neutral / sad) on short Hindi clips (826 train, 209 test, 16 kHz mono, median ≈ 2.9 s), scored by Macro-F1. The final ranking uses **0.40 × public + 0.60 × private** (`OFFICIAL FACT`). It is a small Kaggle Community competition (46 teams, $52 total prizes) run by ENIAC, SRCASW, University of Delhi; prize eligibility requires UG/PG student status, Unstop registration and a GitHub repo (`ORGANIZER CLARIFICATION`).

**The single most important finding (`MEASURED HERE`).** The dataset contains many re-processed copies of the same recording. A waveform cross-correlation scan found **227 near-duplicate groups covering 467 of 1,035 clips** (same length, same loudness, correlation 0.95–1.00; nothing between 0.5 and 0.95). **78 of the 209 test clips (37%) have a near-identical twin in train**, and train twins agree on the label 300/301 times. The test twin rate (0.373) matches a simulated random re-split (0.367 ± 0.031), so the organizers split randomly and public *and* private both contain ≈ 37% twins.

Consequences:
- Every reasonable model gets the twin rows right (the public fine-tuned WavLM ensemble: 100% on OOF twin rows; 0 of 78 test twin predictions disagree with the twin's train label). **The competition is decided on the ≈ 63% of clips without a twin.**
- Random K-fold CV is the faithful LB proxy (it reproduces the twin rate); a dup-group split measures novel-clip skill. Report both.
- A useful decomposition (`MEASURED HERE`): public score ≈ 0.37 × twin-F1 + 0.63 × non-twin-F1. For the public WavLM ensemble, 0.37 × 1.00 + 0.63 × 0.809 = 0.880 vs actual public 0.87918.

**Leaderboard reality (`VERIFIED PUBLIC RESULT` + `MEASURED HERE`).** Exact score values are consistent with a public subset of roughly 52–105 clips (25–50% of test; ≥ 60% is ruled out within 4 errors). On such a subset one model's public score has SD ≈ 0.03–0.05, and a real +0.018 improvement shows up as flat-or-worse 22–34% of the time. The top-10 public spread (0.9654 → 0.9423) is ≈ 2–4 misclassifications. How the 0.965 teams got there is `UNVERIFIED` (no public code); a lucky public draw is plausible.

**Best public, reproducible pipeline.** `kunalinfinite/01-dataverse-training` + `02-dataverse-inference` (WavLM Base+ & Large fine-tunes): verified OOF Macro-F1 **0.8620**, public **0.87918** — this is exactly your current submission.

**What beat it in controlled tests (`MEASURED HERE`, same folds, dup-group bootstrap).**

| Candidate | OOF Macro-F1 | Non-twin F1 | Δ vs public FT ensemble (95% CI) | Est. test score |
|---|---|---|---|---|
| Public FT ensemble (WavLM Base+ & Large) | 0.8620 | 0.8092 | — | 0.880 (public: 0.87918) |
| Frozen XLS-R (layers 10–18) + logistic regression | 0.8650 | 0.8143 | +0.003 [−0.021, +0.026] | 0.881 |
| 3 frozen encoders (WavLM Base+, WavLM Large, XLS-R) averaged | 0.8801 | 0.8368 | +0.018 [−0.001, +0.038] | 0.894 |
| **FT ensemble + 3 frozen-encoder LR, equal weights** | **0.8814** | **0.8351** | **+0.019 [+0.003, +0.037], P(>0) = 0.988** | **0.896** |

The equal-weight 4-family blend is ready as `research/submissions/sub_ft_wbp_wl_xlsr4equal.csv` (differs from your current submission on 12/209 rows). Nothing was submitted on your behalf.

**Biggest bottleneck.** Non-twin generalization on ~131 test clips, learned from only 672 distinct training recordings (826 rows after collapsing near-duplicates). **Most promising unexplored ideas:** fine-tuning XLS-R-300M (the best frozen encoder, Hindi in pretraining; +0.05 dup-group over WavLM Base+ when frozen), Whisper-large-v3 / emotion2vec+ frozen families, and keeping the ensemble equal-weighted.

**Biggest leakage risk.** The public GitHub repo `Himanshi7104/SRCASW-BhavVaani` (the dataset authors' 20-clip sample release) matches competition audio exactly; 5 of its clips are test clips with folder-name labels. Using them would violate the "no hand labeling / human prediction of test data" rule. **Do not use.**

**Top recommendation for the remaining ~46 h:** (1) submit the 4-family blend as a sanity check; (2) on a Kaggle T4, fork the public training notebook with `--models facebook/wav2vec2-xls-r-300m` (E3) and extract Whisper-large-v3 / emotion2vec+ frozen features (E5); (3) add only families that pass pre-declared OOF keep criteria, keep equal weights; (4) select finals by OOF evidence, not public deltas < 0.03; (5) register on Unstop and publish a reproducible GitHub repo by 14 Oct.

---

## 1b. Round-2 experiment update (2026-10-10, 20:30–23:30 UTC) — supersedes the candidate choice above

All numbers `MEASURED HERE` (logs: `research/work/cv_*_layers.txt`, `blend_all_final.txt`, `e9_e12_xlsr.csv`, `whisper_head_wlv3.csv`, `whisper_multilayer_wlv3.csv`; rows X15–X29 in `experiments.csv`). Same protocol: Track B (dup-group 5-fold × 3 seeds) for every choice of layers/features; Track A (public-pipeline folds) + dup-group bootstrap only for blend comparison.

**Frozen-encoder leaderboard (Track B = novel-clip Macro-F1, LR C = 0.05):**

| Encoder (Hugging Face id) | Hindi in training? | Best feature | Track B |
|---|---|---|---|
| openai/whisper-large-v2 | yes (multilingual ASR) | layers 24+32 concat | **0.896** |
| openai/whisper-large-v3 | yes | layers 24+32 concat | **0.894** (last layer alone 0.870) |
| ARTPARK-IISc/whisper-large-v3-vaani-hindi | Hindi fine-tune | layers 24+32 | **0.892** |
| vasista22/whisper-hindi-large-v2 | Hindi fine-tune | layers 24+32 | 0.876 |
| openai/whisper-medium | yes | layers 22–24 | 0.855 |
| Harveenchadha/hindi_large_wav2vec2 (Hindi ASR) | yes | layers 16–20 | 0.820 |
| facebook/w2v-bert-2.0 | yes | layers 12–16 | 0.801 |
| firdhokk/…whisper-large-v3 (English emotion fine-tune) | — | layers 24–28 | 0.803 |
| facebook/wav2vec2-xls-r-300m / -1b | yes | mid layers | 0.793 / 0.792 |
| openai/whisper-large-v3-turbo | yes | layers 20–24 | 0.786 |
| microsoft/wavlm-large | no | 12–18 | 0.780 |
| Harveenchadha/vakyansh_hindi_base_pretrained | Hindi only | 6–10 | 0.766 (L8 alone 0.791) |
| Talha/urdu-audio-emotions (Urdu 4-class SER) | Urdu | 8–12 | 0.753 |
| microsoft/wavlm-base-plus | no | 6–8 | 0.749 |
| emotion2vec+ large (FunASR) | ? | utterance embedding | 0.557 (zero-shot 0.485) |
| openSMILE eGeMAPSv02 | — | 88 functionals | 0.545 |

**Key findings.**
1. **Whisper encoders dominate** (EmoBox prediction confirmed): +0.07–0.10 Track B over the best SSL encoders. Pool only frames covering real audio (not the 30 s padding). Concatenating layer 24 and the last layer adds +0.02 over the last layer alone; 3–4 layers add nothing.
2. Hindi pretraining helps among wav2vec2-style models (Hindi-large 0.82, Vakyansh base L8 0.79 vs English WavLM-B+ 0.75), but emotion-fine-tuned checkpoints from other corpora (emotion2vec+, English-emotion Whisper, Urdu SER) transfer poorly.
3. **Rejected:** pseudo-labels (E9: +0.002 / +0.001, noise), external acted Hindi data (E12: −0.01 to −0.03, domain mismatch), XLS-R fine-tuning (E3: fold-0 0.806 vs WavLM 0.849; stopped), eGeMAPS, emotion2vec+, Whisper-turbo, XLS-R 1B as an upgrade.

**Final blends (equal weights, membership rules fixed before the run; Track A, 824 rows):**

| Blend | Members | OOF | Non-twin | Est. test | Δ vs current submission (95% CI) |
|---|---|---|---|---|---|
| current (public WavLM FT) | ft_ens | 0.862 | 0.809 | 0.880 | — |
| **B6 (primary)** | WavLM FT + Whisper-v2, Whisper-v3, Hindi-Whisper-v3, Hindi-Whisper-v2 (24+32 concat) + Whisper-medium | **0.942** | **0.920** | **0.949** | **+0.081 [+0.060, +0.104]** |
| B5 | B6 without the FT | 0.943 | 0.921 | 0.951 | +0.082 [+0.060, +0.105] |
| **B1 (hedge)** | WavLM FT + all 16 frozen families with Track B ≥ 0.77 | 0.927 | 0.898 | 0.936 | +0.065 [+0.044, +0.087] |

B5 vs B6 are indistinguishable; B6 is preferred because it keeps a fine-tuned, non-Whisper component (structural diversity). B1 is the hedge: much broader family mix, still +0.065. Caveat: the list of encoders tried grew during experimentation, so the absolute OOF of the best blend is mildly optimistic; the gap to the current submission (+0.08, CI lower bound +0.06) is far larger than that bias.

**Submitted:** B6 scored **0.93060 public** (2026-10-11 02:09 UTC; previous 0.87918), consistent with the OOF-based estimate given the public-subset noise (SD 0.03–0.05). A self-contained notebook reproducing it exactly (0/209 rows differ) is `notebooks/B6_whisper_blend.ipynb` (copy in `my_submissions/`).

**Round 3 (after the submission; rows X30–X36):** every further idea was inside the noise or worse — other classifiers on Whisper features (LR remains best), a text branch from Hindi ASR transcripts (alone 0.70–0.75; adds +0.001 to B6 — Whisper embeddings already carry the words), averaging rules / nested class-bias tuning / stacking (plain probability mean best; tuned biases flip sign across folds), train-time speed/noise augmentation (±0.003, tracks disagree), B7 (= B6 + Hindi-large; −0.005). A 6-split robustness check confirmed the blend (dup-group 0.899 ± 0.008 vs best single 0.894). Remaining errors concentrate on *neutral*; several confident errors look like label ambiguity.

**Top-team analysis (2026-10-11; full write-up `research/top_teams/TOP_TEAMS_ANALYSIS.md`).** No top team has published code, data, weights or a write-up (Kaggle profiles, all notebooks/forum threads, GitHub and the alternative rules-page slug were checked) — their methods are `UNVERIFIED`. Quantitatively, the top-3 (0.96544) are only ~2–3 public clips ahead of us; a B6-quality model reaches ≥ 0.96544 in 21–33% of simulated public draws and our 0.93060 sits in the lower fifth of B6's expected public range. Early leaders (SoloSid69, GanneKaJuice: top scores by 2026-09-25 with ≤ 10 submissions) fit "strong pretrained encoder found early" better than leaderboard probing. Because public counts 40% of the final score, picking the best-public among near-equivalent candidates is legitimate (simulated +0.0017 expected final), but our candidate files differ from B6 on only 0–3 test rows, so the realistic gain is ≤ 1 clip; the only substantive lever left is a new, stronger family (e.g. LoRA-fine-tuned Whisper).

**Round 4 — new architectures (rows X37–X40).** On frozen Whisper features: MLP heads (Track B up to 0.905), SUPERB-style learned layer weighting (0.83–0.85, clearly worse than the fixed layer-24+32 concatenation), early fusion (0.903) and attentive-statistics pooling over frame-level features (0.89–0.90) all match LR but add ≤ +0.0025 to B6 (P ≤ 0.82) — rejected. **LoRA fine-tuning of the Whisper-large-v2 encoder** (r = 16 on q/v, 8-s window, 2.6 M trainable parameters, ~1 h on Apple MPS) reaches OOF **0.9145** (folds 0.908–0.920), +0.05 over the public WavLM fine-tune; replacing the WavLM fine-tune in B6 with it (**B8**) gives OOF 0.9433 vs 0.9421 (+0.0012, CI [−0.007, +0.010]) — statistically tied, but self-contained (no other participant's checkpoints). `notebooks/B8_whisper_lora_blend.ipynb` is the updated base notebook; with cached features/LoRA predictions it reproduces `research/submissions/sub_B8.csv` exactly (3/209 rows differ from B6).

**B8 submitted:** public **0.94230** (2026-10-11 07:41 UTC; B6 0.93060) — on CV the two are tied, so the gain is mostly a favourable public draw on the 3 differing clips; selecting B8 is still right because public counts 40 %.

**Round 5 — deep error analysis → targeted experiments (rows X41–X47).** B8's 47 OOF errors are all non-twin; the **PCM_32 recording batch** (10 % of rows, band-limited audio with half the >4 kHz energy, louder, more leading silence) carries 30 % of them (accuracy 0.78 vs 0.94), at every clip duration. Neither domain-wise standardisation, silence trimming + loudness normalisation, 4 kHz low-pass equalisation, a nested PCM_32-specific bias, a repeated-sentence label rule, nor a second LoRA (Hindi Whisper-v3: OOF 0.894, unstable) improved B8 under the keep rule; the best (B8 + low-pass family, +0.0037, P 0.84) yields test predictions identical to B8. Remaining errors look like genuinely hard / ambiguous clips (mostly neutral confusions) rather than a fixable bias.

**Round 6 — more architectures/backbones (rows X48–X50).** Whisper-large (v1) frozen (dup-group 0.887, Track A 0.932) and the Qwen2-Audio audio encoder (loaded from shard 1 only; dup-group ≤ 0.80 — its audio-LLM training weakened the emotion signal relative to Whisper-v3) add +0.0024 and +0.0047 to B8 (P 0.76 / 0.84; ≤ 1 test row changed). LP-FT on Whisper-large-v2 (exact LR-probe head initialisation) reached 0.906 OOF, below plain LoRA (0.915). The decisive diagnostic: the frozen probe on the 8-s-window encoder scores 0.908 vs 0.936 with Whisper's native 30-s padded input, i.e. **Whisper representations depend on the 30-s context**; any future fine-tuning should keep 30-s inputs (≈3× slower). B8 remains the best submission; all candidates change 0–1 test predictions.

**Updated final-submission recommendation:** Final #1 = **B8** (`my_submissions/0.94230_2026-10-11_B8_WhisperLoRA+Whisper5/`, public 0.94230); Final #2 = **B6** (public 0.93060). Both statistically tied on CV. Both include the twin override (no-op here). Not submitted on the user's behalf.

---

## 2. Competition breakdown & rules

### 2.1 Core facts

| Item | Value | Evidence |
|---|---|---|
| Objective | Predict the emotion expressed in each unseen Hindi speech recording | OFFICIAL FACT (Overview/Description) |
| Real-world task | Speech emotion recognition (SER) for HCI, healthcare, education, call centres | OFFICIAL FACT |
| ML task type | Single-label 4-class audio classification | OFFICIAL FACT |
| Input | One mono WAV per sample (16 kHz; PCM_16 for 933 files, PCM_32 for 102) | MEASURED HERE |
| Target | `emotion` ∈ {angry, happy, neutral, sad} | OFFICIAL FACT |
| Output | One label per test Id | OFFICIAL FACT |
| Training data | 826 labelled clips (`train.csv`, `train/`) | OFFICIAL FACT + MEASURED |
| Test data | 209 unlabelled clips (`test.csv`, `test/`) | OFFICIAL FACT + MEASURED |
| Dataset | "SRCASW-BhavVaani", self-curated, acted + spontaneous Hindi speech; abstract says 1,036 samples (1,035 files are distributed) | OFFICIAL FACT; discrepancy MEASURED |
| Hidden test structure | Public/private split of the 209 test clips; sizes not disclosed ("see Data tab", which does not state it) | OFFICIAL FACT |
| Public LB size | Not stated; score granularity consistent with ~52–105 clips (25–50%) | YOUR HYPOTHESIS from MEASURED analysis (§6) |
| Metric | Macro-F1 over the 4 classes | OFFICIAL FACT |
| Final ranking | 0.40 × public + 0.60 × private | OFFICIAL FACT (Evaluation page) |
| Submission format | `sample_submission.csv` format: `Id,emotion` with lowercase labels, sorted as in test.csv. The Evaluation page describes a different format (`AUDIO_ID`, `PREDICTED LABEL`, capitalized, file `solutions.csv`), but the sample format is what 46 teams (including you) were scored with | OFFICIAL FACT + VERIFIED |
| Submissions | 5 per day; up to 2 final selections | OFFICIAL FACT (Rules §3) |
| Teams | Max 3 members; mergers allowed if combined submissions ≤ daily limit × days | OFFICIAL FACT |
| Timeline | Start 20-09-2026 (data files dated 2026-09-17), entry deadline 01-10-2026, final submission 12-10-2026 (Kaggle: 18:30 UTC) | OFFICIAL FACT |
| Code competition? | No: CSV submissions; your submission came from a notebook with internet enabled, which code competitions do not accept | VERIFIED (indirect) |
| Hardware / runtime | No runtime limit on inference (not a code competition); free Kaggle T4/P100 for training | YOUR HYPOTHESIS (no rule found) |
| Internet | Not restricted for training/inference | YOUR HYPOTHESIS (not a code competition) |
| External data | Allowed if publicly available, equally accessible, minimal cost ("Reasonableness Standard") | OFFICIAL FACT (Rules §2.5) |
| Pretrained models | Allowed ("use of external data and models is acceptable unless specifically prohibited") | OFFICIAL FACT |
| Open-source code | Must use OSI-approved licences that do not limit commercial use | OFFICIAL FACT (foundational rules §6c) |
| Hand labelling | Prohibited: "may not use … hand labeling or human prediction of the validation dataset or test data records" | OFFICIAL FACT (foundational rules) |
| Private sharing | Prohibited outside teams; public sharing must be on Kaggle | OFFICIAL FACT |
| Eligibility | UG/PG students only; professionals may play but cannot claim prizes | OFFICIAL FACT |
| Prizes | $26 / $16 / $10 + certificates (certificates only for Unstop-registered) | OFFICIAL FACT + ORGANIZER CLARIFICATION |
| Winner obligations | Top 3 notified by email, present at the college 15 Oct 10:00 IST (online possible), failure → disqualification | OFFICIAL FACT |
| Verification | Unstop form asks for a GitHub repo "so that we can verify predictions"; best model can be updated until 14 Oct 12:00 AM | ORGANIZER CLARIFICATION |
| Tie-break | Earlier-entered submission wins | OFFICIAL FACT (foundational rules) |
| Data use | Competition + educational use, with citation | OFFICIAL FACT |

### 2.2 Unusual scoring mechanics

1. **Public score counts for 40% of the final score.** Unlike standard Kaggle, the public LB is not just feedback. But the selection consequence is the same: choose the submissions with the best *expected* score on the whole test, and public-subset noise (SD 0.03–0.05) is larger than most real improvements. How the host combines the two final selections (best of two combined scores? per-submission?) is `UNVERIFIED`; assume each selected submission gets its own 0.4/0.6 score and the better counts.
2. **Leaderboard rank ≠ prize rank.** Only Unstop-registered students are eligible (only 8 teams had registered as of 2026-10-09, `ORGANIZER CLARIFICATION`).
3. **Format inconsistency** between the Evaluation page and `sample_submission.csv`. Keep the sample format that is already being scored.

### 2.3 Practical impact of each rule

| Rule | Practical impact | YOUR RECOMMENDATION |
|---|---|---|
| Macro-F1 | Every class counts equally; with near-balanced classes it behaves close to accuracy, but systematic under-prediction of one class is punished | Monitor per-class F1 and predicted class counts; no aggressive threshold tuning (§5) |
| 0.4 public + 0.6 private | Public is noisy *and* partly scored | Choose by OOF + twin decomposition; use public only for gross sanity (deltas > 0.03) |
| 5 subs/day, ~14 left | Enough for sanity checks, not for LB-driven tuning | Plan: ≤ 2 per day of genuinely different candidates |
| 2 final selections | One "best expected" + one structurally different hedge | §24 |
| External data/models allowed | All public HF encoders usable | Prefer Apache/MIT weights; treat CC-BY-NC weights (MMS, audEERING, mHuBERT) as competition-uncertain under the OSI rule |
| No hand labelling of test | Rules out copying test labels from the source repo (§16) and listening to test clips | Do not use `Himanshi7104/SRCASW-BhavVaani` labels |
| GitHub repo for verification | Hosts may re-run predictions | Keep every final submission reproducible from code + public weights |
| Team ≤ 3, merger limits | Merging late is possible only if combined submissions fit | Not relevant unless you merge |
| Tie → earlier submission | Many exact ties exist (§6) | Submit your final candidates early rather than at the deadline |

---

## 3. Dataset analysis & schema

### 3.1 Files and schema (`MEASURED HERE`, `research/work/audit_report.txt`)

```
SRCASW-BhavVaani/
├── train.csv              826 rows: Id (int), filename (str), emotion (str: angry|happy|neutral|sad)
├── test.csv               209 rows: Id, filename
├── sample_submission.csv  209 rows: Id, emotion (all "neutral")
├── train/audio_XXXXXX.wav 826 files
└── test/audio_XXXXXX.wav  209 files
```

| Field | Type | Notes |
|---|---|---|
| Id | int 1..1035 | train ∪ test = exactly 1..1035, disjoint; filename = `audio_{Id:06d}.wav` |
| filename | str | anonymized; no information beyond Id |
| emotion | category | train only |
| audio | WAV | 16 kHz, mono; PCM_16 (933) or PCM_32 (102) |

- **One sample = one utterance-level clip** (train and test alike); duration 1.0–11.2 s (train), 1.3–7.5 s (test); median 2.9 / 2.8 s.
- **Id order carries no label information**: adjacent-Id same-label rate 0.251 vs 0.250 ± 0.015 for shuffled labels; label mix is flat across Id quintiles.
- **No missing / corrupt files**; no clipping (clip fraction 0 for all class medians).
- **Encoding groups**: PCM_32 files are shorter (median 2.2 s vs 3.0 s) and less often duplicated (31% vs 47% in dup groups); spread evenly across classes (15–24 per class) and splits (9.7% train / 10.5% test). `YOUR HYPOTHESIS`: a second recording/export batch. Not label-informative.
- **Multiple rows per entity**: yes. 227 near-duplicate groups (217 pairs, 8 triples, 1 quadruple, 1 quintuple) cover 467 rows (§18). No speaker IDs are provided and x-vector clustering does not recover clean speakers (§3.3).
- **Train/test periods/sites/domains**: indistinguishable (adversarial AUC 0.454, §17).

### 3.2 Per-class acoustic summary (train medians, `MEASURED HERE`)

| Class | n | Duration (s) | RMS | Peak | Silence fraction | Share in dup groups |
|---|---|---|---|---|---|---|
| angry | 205 | 2.86 | 0.119 | 0.61 | 0.041 | 38% |
| happy | 192 | 2.73 | 0.101 | 0.67 | 0.061 | 36% |
| neutral | 217 | 2.58 | 0.074 | 0.46 | 0.054 | 49% |
| sad | 212 | 3.33 | 0.088 | 0.50 | 0.034 | 56% |

Energy (angry/happy louder) and duration (sad longer) carry signal, as expected from acoustic theory.

### 3.3 Speaker structure (`MEASURED HERE`, `speakers_report.txt`)

WavLM-base-plus-sv x-vectors: the median nearest non-twin neighbour has cosine 0.965, so speakers are not well separated. Agglomerative clustering gives no stable "speaker-sized" partition (697 clusters at distance 0.03 → 25 at 0.18); cluster label purity is low (0.36–0.58). Speaker embeddings alone predict emotion poorly (Macro-F1 0.55 random-CV / 0.47 dup-group). **Conclusion:** pseudo-speaker GroupKFold would be built on unreliable groups. Since the split was random, test speakers overlap train anyway (§15).

### 3.4 Relationships between files

- train.csv ↔ train/ and test.csv ↔ test/ via filename; sample_submission order = test.csv order.
- **Hidden relationship:** near-duplicate twins across and within splits (§18); a 20-clip public sample of the source dataset matches 19 train + 5 test clips (§16).

---

## 4. Target analysis

| Class | Train count | Share | Meaning (OFFICIAL) | Difficulty (MEASURED, public FT ensemble OOF F1) | Notes |
|---|---|---|---|---|---|
| angry | 205 | 24.8% | anger / aggressive state | 0.872 | high arousal; confused with happy |
| happy | 192 | 23.2% | joy / positive | 0.867 (FT) | high arousal; smallest class |
| neutral | 217 | 26.3% | no distinct emotion | 0.837 | hardest class; most confusions with happy/sad |
| sad | 212 | 25.7% | sorrow | 0.872 | lowest energy, longest clips; most duplicated |

- **Imbalance:** mild (23–26%); class weights and threshold tuning matter little.
- **Label noise:** one exact-duplicate group has conflicting labels (Id 75 = neutral, 636 = sad, 878 = neutral); everything else consistent across twins (300/301). Broader label noise is `UNVERIFIED`; spontaneous clips are inherently more ambiguous (`EXTERNAL RESEARCH EVIDENCE`: BhavVani sitcom Hindi has best WF1 ≈ 0.44).
- **Relationship between classes:** arousal axis (angry/happy vs neutral/sad) is easy; the valence split within high-arousal (angry vs happy) and neutral vs sad are where errors concentrate (FT ensemble confusion matrix in `kunal_train_output/.../cv_summary.json`).

---

## 5. Metric analysis

**Definition.** For class c: F1_c = 2·TP_c / (2·TP_c + FP_c + FN_c); Macro-F1 = ¼ Σ_c F1_c, computed on the whole public (or private) subset — sklearn `f1_score(average="macro")`.

**What it rewards / ignores.**
- Rewards balanced per-class precision *and* recall; ignores probabilities, confidence and calibration (only the argmax label is submitted).
- Ranking does not matter (no AUC); calibration matters only indirectly, when averaging models or tuning decision rules.
- Each error costs two F1 terms (an FN for the true class and an FP for the predicted class). With N clips and n ≈ N/4 per class, near-perfect F1 of each affected class drops by ≈ 1/(2n), so Macro-F1 drops by ≈ (1/(2n) + 1/(2n))/4 = 1/(4n) ≈ **1/N per error**: ≈ 0.019 at N = 52, 0.012 at N = 84, 0.010 at N = 105 (consistent with the observed top-LB steps of ≈ 0.0116, §6).

**Threshold / class-weight tuning.** With balanced classes and balanced predicted counts (FT ensemble predicts 54/46/53/56 on test), the expected gain from per-class bias tuning is small; Odyssey 2024 reported ≈ +1% from thresholds on a heavily imbalanced 8-class task (`EXTERNAL RESEARCH EVIDENCE`, R195). On 824 rows, tuned thresholds overfit the selection criterion (`EXTERNAL RESEARCH EVIDENCE`, Cawley & Talbot R209). The Avik Das notebook tunes divisors on the same OOF it reports — optimistic. `YOUR RECOMMENDATION`: only nested-CV bias tuning, adopt only if gain ≥ 0.005 with P > 0.9.

**Averaging choices.** Probability averaging of reasonably calibrated models is the default. Logit/geometric averaging matter only when one family is very over-confident (the fine-tuned models are more confident than the L2-regularised LR probes; `YOUR HYPOTHESIS`). Rank averaging is irrelevant (no ranking metric). Clipping irrelevant.

**Two models with the same accuracy, different Macro-F1 (illustration).** On 100 clips (25 per class), model A makes 10 errors spread evenly (accuracy 0.90, Macro-F1 ≈ 0.90). Model B makes 10 errors all by predicting "neutral" for true "happy" clips: happy recall 0.60 (F1 = 2·15/(30+10) = 0.75), neutral precision 25/35 (F1 = 50/60 = 0.833), angry & sad F1 = 1.0 → Macro-F1 = 0.896 at the same 0.90 accuracy. Macro-F1 penalises concentrated, one-directional confusions slightly more than spread errors; the effect is small at this class balance.

**Metric + twins.** Because ≈ 37% of test rows are twins that every decent model gets right, the metric's effective resolution is on ≈ 130 non-twin test clips (≈ 50 public, ≈ 80 private if public is 40%; `YOUR HYPOTHESIS`).

---

## 6. Leaderboard analysis

**Snapshot (2026-10-10 19:49 UTC, `VERIFIED PUBLIC RESULT`):** 46 teams. Top: 0.96544 (3-way tie: SoloSid69, thisray, Rin), 0.95387 (2 teams), 0.94303, 0.94230 (3 teams), 0.93124, 0.93120 … your team 0.87918 (rank 27, 1 submission, tied with Kunal Aarse — the author of the notebook you submitted).

**Public subset size (`MEASURED HERE`, `lb_granularity.txt`).** Enumerating confusion matrices with ≤ 4 errors and class counts near the train proportions, all four top scores (0.96544, 0.95387, 0.94303, 0.94230) are exactly reachable for public sizes N = 52, 63, 73, 84, 94, 105, but not for N ≥ 115 (most not even at N = 125–167). **Weak evidence that the public LB is 25–50% of the test set (~52–105 clips)**; the top scores correspond to 2–4 misclassifications.

**Noise (`MEASURED HERE`, `noise_and_significance.txt`).** Simulating public subsets with 37% twins from the verified OOF predictions:

| Public size N | SD of one model's score | SD of paired difference (two models, same subset) | P(public shows a real +0.018 gain as ≤ 0) |
|---|---|---|---|
| 52 | 0.047 | 0.032 | 0.34 |
| 63 | 0.042 | 0.028 | 0.30 |
| 84 | 0.036 | 0.025 | 0.25 |
| 105 | 0.032 | 0.022 | 0.22 |

Implications: rank volatility on the private LB will be high; public/private correlation is `UNVERIFIED` (no private scores exist yet). Ties are common, so the earlier-submission tie-break matters.

**Overfit risks ranked.** (1) choosing among near-identical submissions by public score (repeated-submission risk); (2) per-class threshold tuning on public feedback; (3) ensemble-weight tuning on public or on the 824-row OOF; (4) per-target tuning (n/a: single target).

**Decision framework (`YOUR RECOMMENDATION`).**

| OOF (both tracks) | Public | Interpretation | Action |
|---|---|---|---|
| ↑ significantly (bootstrap P > 0.9) | ↑ | strong candidate | keep |
| ↑ significantly | ↓ by < 0.03 | public noise | keep |
| ↑ significantly | ↓ by > 0.05 | check for a bug / format issue | investigate |
| ≈ (inside CI) | ↑ | likely public noise / overfit | do not select on this basis |
| ↓ | ↑ | public overfit | reject |
| only one fold/seed/track ↑ | any | insufficient evidence | needs more evidence |

---

## 7. Public notebooks, ranking & lineage

**Search coverage.** Kaggle Code tab via API (`kaggle kernels list --competition …`, all pages): 4 public notebooks. GitHub search: 1 participant repo (`kunalverma2512/DataVerse`) + the dataset authors' sample repo. No public datasets of model outputs, no forks. Notebook source and metadata: `research/notebooks/`; full table: `research/public_notebooks.csv`.

| Rank | Notebook | Author | Public LB | CV (verified?) | Model | Key technique | Overfit risk | Reproducible? | Link |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 01_dataverse_training + 02_dataverse_inference | Kunal Aarse | **0.87918** (VERIFIED: your submission 57050932 is this notebook's output) | **0.8620 OOF** (VERIFIED from its output files; optimistic: best epoch chosen on same fold) | WavLM Base+ & Large fine-tune | learned layer weights, attentive stats pooling, LLRD, exact-dup-aware folds, crops TTA | Low | Yes (T4) | [01](https://www.kaggle.com/code/kunalinfinite/01-dataverse-training) / [02](https://www.kaggle.com/code/kunalinfinite/02-dataverse-inference) |
| 2 | kunalverma2512/DataVerse (GitHub) | kunalverma2512 | README "76.68"; team "Kunal Verma" 0.76689 (PARTICIPANT CLAIM consistent with LB) | UNVERIFIED | Whisper-base frozen + LR/SVM/XGB | soft voting | Low | Medium | [repo](https://github.com/kunalverma2512/DataVerse) |
| 3 | dataverse-dehs-baseline | Shivam | 0.74925 (VERIFIED: notebook's own best public score) | UNVERIFIED (accuracy printed, outputs not saved) | XGBoost | 188 librosa clip stats | Low | Yes (CPU) | [nb](https://www.kaggle.com/code/shivamgravity/dataverse-dehs-baseline) |
| 4 | Acoustic Source-Filter Speech Affect Recognition | Avik Das | 0.72790 (VERIFIED: notebook's own best public score) | UNVERIFIED (outputs not saved) | MLP + LGBM + XGB on hand-crafted + frozen wav2vec2-SUPERB-ER | Nelder-Mead class divisors on same OOF | Medium | Partly | [nb](https://www.kaggle.com/code/avikdas567/acoustic-source-filter-speech-affect-recognition) |

**A. Verified high-scoring notebooks:** only #1. **B. Claimed but unverified:** #2 (claim matches LB), the training notebook's docstring claims "OOF 0.789 / public 0.83266" for an *earlier* version (UNVERIFIED). **C. Strong CV/training notebooks:** #1. **D. EDA/preprocessing:** #4 (extensive acoustic EDA, F0/formant features). **E. Forks/reused stacks:** none found; your submission reuses Family A outputs.

**Lineage / solution families (`MEASURED HERE` by reading code and metadata).**
- **Family A** – `kunalinfinite/01` (trains WavLM Base+/Large, saves checkpoints + OOF) → `kunalinfinite/02` (pure inference over A's checkpoints; declared as `kernel_sources`) → your submission (A's output file). 01's docstring references an even earlier private/legacy notebook (`kunalinfinite/dataverse-eda-and-training`, not public) whose checkpoints 02 can also load.
- **Family B** – Avik Das: independent (hand-crafted source-filter features + SUPERB-ER embeddings).
- **Family C** – Shivam: independent classical baseline.
- **Family D** – kunalverma2512: independent (Whisper-base embeddings).
- No shared checkpoints, prediction files, or ensemble weights across families. Only Family A is a genuinely strong, reproducible pipeline; B–D are weaker and mutually independent.

**Why Family A works / what is redundant (`YOUR HYPOTHESIS` + `MEASURED HERE`).** Strong components: SSL backbone fine-tuning with layer weighting (frozen mid-layers already reach 0.75–0.78 dup-group), memorisation of twins (twin F1 1.000). Weak or redundant components: WavLM Large adds little over Base+ (0.8499 vs 0.8555 OOF; ensemble 0.8620); 8-second crops are irrelevant for 3-second clips; single seed. Missing diversity: no multilingual/Hindi-pretrained backbone. Remaining headroom: non-twin F1 0.809 → 0.84+ already reached by adding frozen families (X10); XLS-R fine-tuning is the most likely next step.

---

## 8. Discussions & organizer clarifications

All 3 forum topics and every message were read via the Kaggle API (`research/official/discussions.json`). There are no technical discussions (no validation, leakage, or model threads). Full table: `research/important_discussions.csv`.

| Title | Author | Date | Type | Key claim | Practical implication |
|---|---|---|---|---|---|
| [Registration](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/discussion/745561) | host | 2026-10-03 (+ reply 10-05) | ORGANIZER CLARIFICATION | Unstop registration is compulsory to claim prizes; register if you submitted at least once | Register on Unstop |
| [Re: Registration](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/discussion/747812) | host | 2026-10-09 | ORGANIZER CLARIFICATION | Only 8 teams had registered | Many LB teams are not prize-eligible |
| [Will all participants receive a certificate?](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/discussion/746206) | participant + host (@pgolephd) | 2026-10-06/07 | ORGANIZER CLARIFICATION | Certificates only for registered participants; Unstop form needs a GitHub repo so hosts can verify predictions; best model can be updated until 14 Oct 12:00 AM | Keep a reproducible repo; deadline 14 Oct |

---

## 9. External resources (453) & Top 100

**Count.** **453 unique, relevant resources** were catalogued (`research/external_resources.csv`, 23 columns as requested). The 500 target was not reached; per your instruction, the list was not padded. Sources: 4 parallel research passes (Hindi/Indic & benchmark datasets + papers; pretrained models; methods + previous competitions; repos/libraries/docs) plus competition-specific sources. Each URL was seen in a search result or fetched page; most model entries were confirmed via the Hugging Face API, repos via the GitHub API (stars as of 2026-10-11), arXiv IDs via export.arxiv.org. Fields that could not be confirmed say `UNVERIFIED`; a few author/venue fields marked "from memory; verify" in `notes` should be double-checked before citing.

| Breakdown | Count |
|---|---|
| Relevance High / Medium / Low | 124 / 201 / 137 (before dedup; see CSV) |
| Evidence: Peer-reviewed / Official repository / Preprint / Community / Competition solution / Technical documentation / Unverified | 164 / 120 / 63 / 58 / 32 / 22 / 3 (before dedup) |
| Largest categories | SER benchmark datasets (28), winning solutions (27+6 code), SER fine-tuned models (27), SSL encoders (25), ASR encoders (23), Indic SER datasets (21), SER repos (21), previous competitions (20), Hindi SER datasets (16) |

**Top 100** (ranked, with how-to-use, expected impact, difficulty): see **`research/TOP_100_RESOURCES.md`**. Ranks 1–66 are hand-ordered using the experiments in this report; 67–100 by catalog score. The ten most useful:

1. XLS-R 300M (best frozen encoder here; next to fine-tune) 2. Public training notebook (verified baseline) 3. Competition rules 4. Overview/evaluation 5. WavLM Large 6. SUPERB (frozen layer-probe protocol) 7. EmoBox (Whisper-large-v3 ranked 1st on 23/32 corpora) 8. Whisper-large-v3 encoder 9. emotion2vec+ large 10. Varoquaux 2018 (CV error bars at small n).

---

## 10. Must-read papers

39 papers selected from the catalog for direct relevance. "Proposed experiment" maps to the roadmap IDs in §21.

| # | Paper | Authors | Year | Contribution | Why relevant here | Proposed experiment | Difficulty | Likely impact | Link |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | SUPERB: Speech processing Universal PERformance Benchmark | Shu-wen Yang, Hung-yi Lee, et al. (21 authors) | 2021 | Frozen SSL encoders with a learnable softmax-weighted sum of all layers and a light head is a strong, cheap protocol for small labeled sets. | Frozen SSL upstream + learnable weighted sum of hidden layers + lightweight task | Frozen layer-wise probe across encoders (done: X03-X05) | Low | High | https://arxiv.org/pdf/2105.01051 |
| 2 | EmoBox: Multilingual Multi-corpus Speech Emotion Recognition Toolkit and Benchmark | Ziyang Ma, Mingjie Chen, Hezhao Zhang, Zhisheng Zh | 2024 | Whisper large-v3 encoder ranked 1st on 23 of 32 datasets (per search summary) - strongest general frozen backbone for SER across languages | Standardized splits; frozen-encoder + linear probe for 10 SSL/ASR models; cross- | Add Whisper-large-v3 encoder as frozen family (E5) | Medium | Medium-High | https://arxiv.org/abs/2406.07162 |
| 3 | emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation | Ziyang Ma, Zhisheng Zheng, Jiaxin Ye, Jinchao Li,  | 2023 | Emotion-specific foundation model; linear probe beats generic SSL on IEMOCAP and generalizes across languages; outputs 9 classes incl. angry/happy/neutral/sad | Self-supervised online distillation on emotion data; emotion2vec+ seed/base/larg | emotion2vec+ frozen family (E5) | Medium | Medium | https://arxiv.org/abs/2312.15185 |
| 4 | SER Evals: In-domain and Out-of-domain Benchmarking for Speech Emotion Recognition | Mohamed Osman, Daniel Z. Kaplan, Tamer Nadeem | 2024 | Whisper (ASR model) outperforms dedicated SSL models on cross-lingual SER; logit adjustment handles label-prior shift | Large in/out-of-domain benchmark; logit adjustment for class-distribution shift | Whisper vs SSL frozen comparison; logit adjustment for external data (E5/E12) | Medium | Medium | https://arxiv.org/abs/2408.07851 |
| 5 | Emotion Recognition from Speech Using Wav2vec 2.0 Embeddings | Leonardo Pepino, Pablo Riera, Luciana Ferrer | 2021 | Combining all wav2vec2 layers via trainable weights with a shallow head: ~67.2% avg recall IEMOCAP, ~84.3% RAVDESS; pre-trained (non-ASR-finetuned) model better. | Frozen wav2vec2 features, trainable layer weights, shallow downstream nets | Learned layer weights instead of fixed band for frozen probes | Low | Low-Medium | https://arxiv.org/pdf/2104.03502 |
| 6 | Attentive Statistics Pooling for Deep Speaker Embedding | Koji Okabe, Takafumi Koshinaka, Koichi Shinoda | 2018 | Attention-weighted mean and std pooling reduced EER 7.5% (NIST SRE12) and 8.1% (VoxCeleb) vs plain pooling. | Attention-weighted mean + std pooling of frame features | Attentive stats pooling head on frozen frames | Medium | Low | https://arxiv.org/abs/1803.10963v1 |
| 7 | A Fine-tuned Wav2vec 2.0/HuBERT Benchmark For Speech Emotion Recognition, Speaker Verifica | Yingzhi Wang, Abdelmoumene Boumadane, Abdelwahab H | 2021 | Partial/full fine-tuning with simple heads: IEMOCAP 79.58% WA speaker-dependent vs 73.01% speaker-independent (shows ~6.5 pt speaker leakage inflation). | Partial vs full fine-tuning of wav2vec2/HuBERT with simple heads | Interpret twin/speaker effects in CV (speaker-dependent vs independent gap) | Low | Medium (validation) | https://arxiv.org/pdf/2111.02735 |
| 8 | Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (LP-FT) | Ananya Kumar, Aditi Raghunathan, Robbie Jones, Ten | 2022 | Full FT with random head distorts features: ~2% better ID but ~7% worse OOD than linear probing; LP-FT gives ~1% better ID and ~10% better OOD than FT. | Linear probe first, then full fine-tune (LP-FT) | LP-FT schedule for XLS-R fine-tune (E3) | Medium | Medium | https://arxiv.org/pdf/2202.10054 |
| 9 | PEFT-SER: On the Use of Parameter Efficient Transfer Learning Approaches For Speech Emotio | Tiantian Feng, Shrikanth Narayanan | 2023 | LoRA generally best among PEFT methods (released checkpoints use rank 16); WavLM Base+ best on IEMOCAP, Whisper Small best on other datasets. | Adapters, embedding prompt tuning, LoRA on wav2vec2/WavLM/Whisper | LoRA fine-tune of XLS-R / WavLM-large (E3 variant) | Medium | Medium | https://arxiv.org/pdf/2306.05350 |
| 10 | A Comprehensive Exploration of Fine-Tuning WavLM for Enhancing Speech Emotion Recognition | Fadel Ali, Aniati Murni Arymurthy, Radityo Eko Pra | 2023 | WavLM beats wav2vec2/HuBERT; tuning only top layers gives significant gain, full-layer tuning best. | Fine-tuning different WavLM layer groups vs wav2vec2/HuBERT | Partial vs full WavLM fine-tuning | Medium | Low-Medium | https://scholar.ui.ac.id/en/publications/a-comprehensive-exploration-of-fine-tuning-wavlm-for-enhancing-sp/ |
| 11 | Dawn of the transformer era in speech emotion recognition: closing the valence gap | Johannes Wagner, Andreas Triantafyllopoulos, Hagen | 2023 | Valence CCC .638 on MSP-Podcast without text; transformers implicitly learn linguistic info; robust to perturbations | Fine-tuned wav2vec2/HuBERT on arousal/dominance/valence | Use robust wav2vec2 SER features; valence/arousal as auxiliary signals | Low | Low | https://arxiv.org/abs/2203.07378v4 |
| 12 | Cross-validation failure: small sample sizes lead to large error bars | Gael Varoquaux | 2018 | With ~100 samples CV accuracy error bars are about +/-10 points; fold standard error underestimates true uncertainty. | Analysis of CV error bars | Bootstrap CIs for every decision (done: X08-X10, X12) | Low | High (decision quality) | https://arxiv.org/pdf/1706.07581 |
| 13 | Apples-to-Apples in Cross-Validation Studies: Pitfalls in Classifier Performance Measureme | George Forman, Martin Scholz | 2010 | Averaging F1 over folds is biased (esp. with imbalance); pool TP/FP/FN (i.e., pooled OOF predictions) before computing F1. | Compare ways of computing F-measure/AUC under CV | Pooled OOF Macro-F1 (done) | Low | Medium | https://api.crossref.org/works/10.1145%2F1882471.1882479 |
| 14 | On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation | Gavin C. Cawley, Nicola L. C. Talbot | 2010 | Overfitting the model-selection criterion can be as large as differences between algorithms; selection must be redone inside each resample (nested CV). | Nested CV / treating selection as part of fitting | Avoid tuning blend weights / thresholds on 824 rows | Low | High (private robustness) | https://jmlr2020.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf |
| 15 | Macro F1 and Macro F1 | Juri Opitz, Sebastian Burst | 2019 | Two different formulas are both called macro-F1 and can differ by up to 0.5; under optimal thresholding rare labels get pushed to be predicted frequently. | Compare averaged-F1 vs F1-of-averages definitions | Confirm macro-F1 definition = mean of per-class F1 | Low | Low | https://arxiv.org/pdf/1911.03347 |
| 16 | Thresholding Classifiers to Maximize F1 Score | Zachary C. Lipton, Charles Elkan, Balakrishnan Nar | 2014 | For calibrated probabilities the F1-optimal threshold equals half the optimal F1; macro vs micro averaging lead to different optimal thresholds. | Optimal decision threshold for F1 | Nested threshold tuning (E7) | Low | Low | https://arxiv.org/pdf/1402.1892 |
| 17 | Long-tail learning via logit adjustment | Aditya Krishna Menon, Sadeep Jayasumana, Ankit Sin | 2021 | Subtracting tau*log(prior) from logits post-hoc (or in the loss) targets balanced error, a close proxy for macro-F1. | Post-hoc logit adjustment by log class priors, or logit-adjusted loss | Post-hoc logit adjustment (E7) | Low | Low | https://arxiv.org/abs/2007.07314v2 |
| 18 | On Calibration of Modern Neural Networks | Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinber | 2017 | Modern nets are miscalibrated; single-parameter temperature scaling is simple and often most effective; does not change argmax. | Temperature scaling | Temperature-scale FT and LR families before averaging (E7) | Low | Low | https://arxiv.org/abs/1706.04599v1 |
| 19 | Confident Learning: Estimating Uncertainty in Dataset Labels (cleanlab) | Curtis Northcutt, Lu Jiang, Isaac Chuang | 2021 | Out-of-fold predicted probabilities can identify likely mislabeled examples under class-conditional noise; outperforms other noisy-label methods on CIFAR. | Estimate joint noisy/true label distribution from OOF probabilities; prune/rank  | Label-issue audit on OOF (E13) | Low | Low-Medium | https://arxiv.org/abs/1911.00068v4 |
| 20 | Data leakage in cross-modal retrieval training: A case study | Benno Weck, Xavier Serra | 2023 | Fingerprinting found 3,601 train/test matching pairs incl. exact duplicates, reprocessed recordings and partial overlaps -> overly optimistic benchmarks. | Panako audio fingerprinting to find train/test duplicates; regroup splits by rec | Fingerprint/correlation duplicate scan (done: X00) | Low | High (validation) | https://arxiv.org/pdf/2302.12258 |
| 21 | How Far Do Foundation Models Transfer to Infant Signals? A Cross-Dataset Transfer Audit wi | UNVERIFIED | 2026 | Exact SHA-1 hashing missed waveform-identical re-exports; log-mel simhash found 8-42 near-duplicate pairs per corpus; cross-correlation caught cases hashing missed. | SHA-1 exact hashing, 64-bit log-mel simhash, waveform cross-correlation | Waveform-level near-dup detection beats hashing (done: X00) | Low | High (validation) | https://arxiv.org/pdf/2608.08989 |
| 22 | Improving Speech Emotion Recognition with Unsupervised Speaking Style Transfer (EmoAug) | Leyuan Qu, Wei Wang, Cornelius Weber, Pengcheng Yu | 2024 | Direct pitch-shift evidence (Table 1, HuBERT-Large, IEMOCAP WA/UA): baseline 70.24/71.13; +pitch shift (+-2 semitones) 70.47/71.24; +speed 0.9-1.1 70.35/71.19; +CopyPaste | Style-transfer augmentation vs speed, pitch shift, CopyPaste, VTLP, emotional vo | Skip pitch-shift augmentation; low expected gain | Low | Low | https://arxiv.org/pdf/2211.08843 |
| 23 | A Comparative Study of Data Augmentation Techniques for Deep Learning Based Emotion Recogn | Ravi Shankar, Abdouh Harouna Kenfack, Arjun Somaya | 2022 | Speed/rate augmentation gives the most robust gain across models; long-range dependencies matter for emotion; Copy-Paste (same-label concatenation) exploits segmental emo | Speed/rate, Copy-Paste and other augmentations; repeated 5-fold CV | Speed perturbation if any augmentation is used in FT | Low | Low | https://arxiv.org/pdf/2211.05047 |
| 24 | SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition | Daniel S. Park, William Chan, et al. | 2019 | Masking time/frequency bands is cheap, needs no extra data, and fights overfitting with small data (6.8% WER LibriSpeech test-other w/o LM). | Time warp, frequency masking, time masking on filterbanks | Mild SpecAugment in FT | Low | Low | https://arxiv.org/pdf/1904.08779 |
| 25 | mixup: Beyond Empirical Risk Minimization | Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, Da | 2018 | Mixup improves generalization, reduces memorization of corrupt labels; tested on speech commands too. | Convex combinations of example pairs and labels | Mixup in FT runs | Low | Low | https://arxiv.org/pdf/1710.09412 |
| 26 | CopyPaste: An Augmentation Method for Speech Emotion Recognition | Raghavendra Pappagari, Jesus Villalba, Piotr Zelas | 2021 | Emotional+neutral concatenation can keep the emotional label (non-neutral emotion dominates perception); all 3 CopyPaste schemes improved SER on 3 datasets, better than n | Concatenate emotional utterance with neutral (or same-emotion) utterance, keep e | CopyPaste augmentation in FT | Medium | Low | https://arxiv.org/pdf/2010.14602 |
| 27 | 1st Place Solution to Odyssey Emotion Recognition Challenge Task1: Tackling Class Imbalanc | Mingjie Chen, Hezhao Zhang, Yuanchao Li, Jiachen L | 2024 | Ranked 1st of 68 submissions: Macro-F1 35.69%, acc 37.32%; focal loss weighted by class priors trades major-class performance for macro-F1; majority vote of 7 diverse mod | Focal loss + prior-based class weights; 7-model ensemble with majority voting | Prior-weighted focal loss + model voting | Medium | Low-Medium | https://arxiv.org/abs/2405.20064v1 |
| 28 | Double Multi-Head Attention Multimodal System for Odyssey 2024 Speech Emotion Recognition  | Federico Costa, Miquel India, Javier Hernando | 2024 | Per-class thresholds tuned to maximize macro-F1 (fallback to next class) gave ~1% absolute validation macro-F1; 3-model hard-voting ensemble 33.80% val (baseline 30.70%); | Frozen wav2vec2/XLS-R/HuBERT/WavLM; WCE vs focal; per-class one-vs-rest threshol | Per-class threshold tuning (nested) | Low | Low | https://arxiv.org/pdf/2406.10598 |
| 29 | BSC-UPC at EmoSPeech-IberLEF2024: Attention Pooling for Emotion Recognition | Marc Casals-Salvador, Federico Costa, Miquel India | 2024 | 1st place in EmoSPeech 2024 with 86.69% macro-F1 using attention pooling on pre-trained speech+text features (small non-English dataset). | Attention pooling over frozen pre-trained speech and text model features | Attention pooling over frozen features | Medium | Low-Medium | https://ceur-ws.org/Vol-3756/EmoSPeech2024_paper1.pdf |
| 30 | MEDUSA: A Multimodal Deep Fusion Multi-Stage Training Framework for Speech Emotion Recogni | Georgios Chatzichristodoulou, Despoina Kosmopoulou | 2025 | Ranked 1st in IS2025 Task 1: ensemble + meta-classifier, Manifold MixUp regularization, soft targets, balanced sampling, weight-averaged model soup. | Cross-modal transformer fusion of SSL acoustic+text; Manifold MixUp; soft target | Meta-classifier / model soup (only with nested CV) | High | Low | https://arxiv.org/pdf/2506.09556 |
| 31 | Exploring Multilingual Unseen Speaker Emotion Recognition: Leveraging Co-Attention Cues in | Arnav Goel, Medha Hira, Anubha Gupta | 2024 | Whisper encoder is the strongest frozen backbone across languages; on Hindi BhavVani Whisper-Medium WF1 0.439 vs wav2vec2 0.317, HuBERT 0.332, WavLM 0.305 | Co-attention fusion + multitask learning on top of Whisper/HuBERT/wav2vec2/WavLM | Context: spontaneous Hindi SER is hard; acted portions dominate this dataset | Low | Background | https://arxiv.org/abs/2406.08931 |
| 32 | Investigating the Impact of Cross-lingual Acoustic-Phonetic Similarities on Multilingual S | Syeda Tamanna Alam Monisha, Sadia Sultana | 2024 | Models trained on same-family (Indo-Aryan: Bangla/Hindi/Odia) languages transfer better to each other than to English/German; on IITKGP-SEHSC (4 emotions: anger, happines | Handcrafted acoustic features; SVM/RF/MLP/XGBoost/CNN; monolingual, cross-lingua | Prefer Indo-Aryan auxiliary corpora (E12) | Medium | Low-Medium | https://ajse.aiub.edu/index.php/ajse/article/download/1422/244/5592 |
| 33 | Cross Lingual Speech Emotion Recognition: Urdu vs. Western Languages | Siddique Latif, Adnan Qayyum, Muhammad Usman, Juna | 2018 | Multilingual training is comparable to in-language baseline, and adding even a fraction of target-language data helps | Cross-lingual and multilingual training scenarios | Small fraction of target-language data matters most (E12) | Medium | Low | https://arxiv.org/abs/1812.10411 |
| 34 | Semi-supervised cross-lingual speech emotion recognition | Mirko Agarla, Simone Bianco, Luigi Celona, Paolo N | 2022 | Pseudo-labelling unlabeled target-language speech gives large gains (avg +40% UA vs prior SOTA) in speaker-independent setup | Transformer with hard/soft pseudo-labelling of unlabeled target-language utteran | Pseudo-labelling unlabeled target-language speech (E9) | Medium | Low-Medium | https://arxiv.org/abs/2207.06767 |
| 35 | Semi-supervised multimodal emotion recognition with class-balanced pseudo-labeling (MER202 | Chen, Guo, Li, Zhang, Jiang (UNVERIFIED first name | 2023 | Selecting top-k confident pseudo-labels per class (class-balanced) avoids majority-class drift; weighted F1 88.53% on MER2023-SEMI test. | Class-balanced top-k high-confidence pseudo-label selection per class | Class-balanced top-k pseudo-label selection (E9) | Low | Low | https://pure.nwpu.edu.cn/en/publications/semi-supervised-multimodal-emotion-recognition-with-class-balance/ |
| 36 | Speaker Normalization for Self-supervised Speech Emotion Recognition | Itai Gat, Hagai Aronowitz, Weizhong Zhu, Edmilson  | 2022 | SER classifiers exploit speaker shortcuts; adversarial speaker normalization on SSL features gave new SOTA on IEMOCAP (speaker-independent and dependent). | Gradient-reversal adversary removing speaker info from SSL representations | Speaker normalization (low priority: random split shares speakers) | High | Low | https://arxiv.org/pdf/2202.01252 |
| 37 | Focal Loss for Dense Object Detection | Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming  | 2017 | Dynamically scaled CE focusing training on hard examples; used by the Odyssey 2024 SER winner with prior-based class weights. | Focal loss down-weighting easy examples | Focal loss variant for FT | Low | Low | https://arxiv.org/pdf/1708.02002 |
| 38 | Averaging Weights Leads to Wider Optima and Better Generalization (SWA) | Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov | 2018 | Averaging weights over the training trajectory finds wider optima and better generalization at almost no extra cost. | Stochastic weight averaging along SGD trajectory | SWA for FT checkpoints | Low | Low | https://arxiv.org/abs/1803.05407v3 |
| 39 | When Does Label Smoothing Help? | Rafael Muller, Simon Kornblith, Geoffrey Hinton | 2019 | Label smoothing improves generalization and calibration but hurts knowledge distillation from the smoothed teacher. | Label smoothing | Label smoothing (already 0.02 in public pipeline) | Low | Low | https://arxiv.org/pdf/1906.02629 |

---

## 11. Top GitHub repositories

Stars were read from the GitHub API on 2026-10-11 (via `notes`). "Direct usefulness" = catalog relevance.

| # | Repository | Stars (seen 2026-10-11) | Main method | Licence | Checkpoints | Direct usefulness | Experiment it enables | Link |
|---:|---|---|---|---|---|---|---|---|
| 1 | Hugging Face Transformers | 167158 (seen 2026-10-11 via GitHub API) | AutoModelForAudioClassification, Wav2Vec2/WavLM/HuBERT/Whisper ForSequ | Apache-2.0 | yes | High | Core fine-tuning code for all SSL experiments | https://github.com/huggingface/transformers |
| 2 | WavLM (microsoft/unilm) | 22229 (seen 2026-10-11 via GitHub API) | Masked speech denoising + prediction SSL | MIT | yes | High | WavLM-large/base-plus weighted-layer probe and partial fine-tune (freeze CNN feature encoder, LLRD) | https://github.com/microsoft/unilm |
| 3 | XLS-R / wav2vec 2.0 (fairseq) | 32206 (seen 2026-10-11 via GitHub API) | Cross-lingual wav2vec2 pretraining | MIT | yes (HF apache-2.0) | High | XLS-R-300m fine-tune with weighted layers; compare to indicwav2vec-hindi | https://github.com/facebookresearch/fairseq |
| 4 | OpenAI Whisper | 110285 (seen 2026-10-11 via GitHub API) | Weakly-supervised encoder-decoder | MIT | yes | High | Whisper-small/medium encoder mean+attentive pooling probe; LoRA on encoder; transcript-based features | https://github.com/openai/whisper |
| 5 | PEFT (LoRA etc.) | 21779 (seen 2026-10-11 via GitHub API) | LoRA/IA3/adapters on attention/FFN projections | Apache-2.0 | no | High | LoRA r=8-16 on q/v (and FFN) of WavLM-large / XLS-R / Whisper encoder vs full fine-tune vs frozen probe | https://github.com/huggingface/peft |
| 6 | SpeechBrain IEMOCAP emotion-recognition recipe | 11863 (seen 2026-10-11 via GitHub API) | ECAPA-TDNN on fbanks and wav2vec2/HuBERT fine-tuning with pooling head | Apache-2.0 | yes (HF speechbrain/emoti | High | Port train_with_wav2vec2.yaml to the 826 Hindi clips (swap encoder for indicwav2vec-hindi / XLS-R / WavLM), 5-fold group CV; also  | https://github.com/speechbrain/speechbrain |
| 7 | s3prl (SUPERB toolkit) - emotion downstream | 2568 (seen 2026-10-11 via GitHub API) | Frozen SSL upstream, learnable weighted-sum of layers, projector + mea | Apache-2.0 | yes (upstreams via hub | High | Implement learnable softmax-weighted layer pooling over WavLM/XLS-R/indicwav2vec hidden states + attentive pooling head; compare w | https://github.com/s3prl/s3prl |
| 8 | FunASR toolkit (emotion2vec / SenseVoice inference) | UNVERIFIED | AutoModel loader for emotion2vec(+)/SenseVoice | MIT code; model weights under  | UNVERIFIED | Medium | Batch-extract emotion2vec+ embeddings/logits offline and cache as Kaggle dataset. | https://github.com/modelscope/FunASR |
| 9 | audEERING w2v2-how-to (dimensional emotion model) | 563 (seen 2026-10-11 via GitHub API) | wav2vec2-large-robust pruned to 12 layers, fine-tuned on MSP-Podcast v | Repo MIT; model weights CC BY- | yes (HF + Zenodo) | High | Add A/V/D scores + embeddings as features in a GBDT/LR stack; check the NC licence against competition rules | https://github.com/audeering/w2v2-how-to |
| 10 | EmoBox: multilingual multi-corpus SER toolkit and benchmark | UNVERIFIED | Standardised splits + SSL feature benchmarking | UNVERIFIED (repo) | UNVERIFIED | High | Pretrain/auxiliary-train on URDU + SUBESCO + CREMA-D mapped to 4 classes, then fine-tune on Hindi. | https://github.com/emo-box/EmoBox |
| 11 | Vakyansh Indic wav2vec2 models | 329 (seen 2026-10-11 via GitHub API) | wav2vec2 pretrained/fine-tuned on Indic data | MIT | yes | Medium | Second Hindi encoder for ensemble | https://github.com/Open-Speech-EkStep/vakyansh-models |
| 12 | IndicWav2Vec (Base/Large pretrained, 40 Indian languages) | UNVERIFIED | wav2vec2 pretraining on Indic speech (fairseq checkpoints) | mit | UNVERIFIED | High | Convert IndicWav2Vec Large to HF; fine-tune for 4-class SER; compare vs XLS-R-300M. | https://github.com/AI4Bharat/IndicWav2Vec |
| 13 | IndicWhisper / Vistaar (12 Indian-language Whisper fine-tunes) | UNVERIFIED | Whisper (medium per Hindi example path) fine-tuned on 10.7k h Indic la | mit | UNVERIFIED | High | Use Hindi IndicWhisper encoder as SER backbone vs vanilla whisper-medium. | https://github.com/AI4Bharat/vistaar |
| 14 | openSMILE (opensmile-python): eGeMAPSv02 / ComParE_2016 | UNVERIFIED | eGeMAPSv02 88 functionals (25 LLDs); ComParE_2016 6373 functionals | Dual: free for research/educat | UNVERIFIED | High | eGeMAPS functionals + LightGBM/SVM baseline; concat with SSL pooled embeddings; per-speaker z-normalisation. | https://github.com/audeering/opensmile-python |
| 15 | Parselmouth (Praat in Python) | 1293 (seen 2026-10-11 via GitHub API) | Praat pitch, formants, jitter, shimmer, HNR, intensity | GPL-3.0 | no | High | Voice-quality feature block for GBDT branch | https://github.com/YannickJadoul/Parselmouth |
| 16 | audiomentations | 2323 (seen 2026-10-11 via GitHub API) | Gain, noise (SNR), pitch shift, time stretch, RIR, band-pass, room sim | MIT | no | High | Ablate aug policies (gain+noise+RIR vs +pitch/tempo) on group CV macro-F1 | https://github.com/iver56/audiomentations |
| 17 | cleanlab | 11694 (seen 2026-10-11 via GitHub API) | Confident learning on out-of-fold probabilities; Datalab (label, outli | Apache-2.0 | no | High | OOF probs from SSL probe -> Datalab find_issues -> inspect/drop/relabel/down-weight; rerun CV | https://github.com/cleanlab/cleanlab |
| 18 | FAISS | 41124 (seen 2026-10-11 via GitHub API) | Exact/ANN kNN | MIT | no | Medium | Cosine kNN on SSL embeddings to flag duplicates / leakage; kNN-label features | https://github.com/facebookresearch/faiss |
| 19 | Chromaprint | 1394 (seen 2026-10-11 via GitHub API) | Chroma-based fingerprints (fpcalc) | Other/custom (GitHub: NOASSERT | no | Medium | fpcalc fingerprints to detect duplicated clips | https://github.com/acoustid/chromaprint |
| 20 | pyannote.audio | 10628 (seen 2026-10-11 via GitHub API) | Segmentation + WeSpeaker ResNet34 embeddings | MIT | yes | High | Second speaker-embedding view to confirm pseudo-speaker clusters before GroupKFold | https://github.com/pyannote/pyannote-audio |
| 21 | SpeechBrain ECAPA-TDNN speaker recognition (VoxCeleb recipe + spkrec-e | 11863 (seen 2026-10-11 via GitHub API) | ECAPA-TDNN 192-d embeddings, cosine scoring | Apache-2.0 | yes | High | Extract ECAPA embeddings -> agglomerative/HDBSCAN clustering -> pseudo-speaker IDs -> StratifiedGroupKFold; also test per-speaker  | https://github.com/speechbrain/speechbrain |
| 22 | scikit-learn | 67522 (seen 2026-10-11 via GitHub API) | LogisticRegression/SVC/Ridge on embeddings, pipelines, CV | BSD-3-Clause | no | High | LR/SVM grid over C with StandardScaler on each embedding bank | https://github.com/scikit-learn/scikit-learn |
| 23 | LightGBM | 18854 (seen 2026-10-11 via GitHub API) | GBDT (multiclass, class weights) | MIT | no | High | LightGBM on eGeMAPS+ComParE+Parselmouth features and as level-2 stacker | https://github.com/lightgbm-org/LightGBM |
| 24 | Optuna | 14904 (seen 2026-10-11 via GitHub API) | TPE sampler, pruning | MIT | no | Medium | Tune head LR/dropout/LoRA rank and GBDT params on repeated CV | https://github.com/optuna/optuna |
| 25 | ser-with-w2v2 (Pepino et al., wav2vec 2.0 embeddings for SER) | 140 (seen 2026-10-11 via GitHub API) | Frozen wav2vec2 layer outputs combined with trainable weights + downst | None detected | no | Medium | Reproduce layer-weighted frozen-feature baseline on Hindi clips as a quick, low-variance model | https://github.com/habla-liaa/ser-with-w2v2 |
| 26 | soxan - wav2vec2 for speech classification | 277 (seen 2026-10-11 via GitHub API) | Wav2Vec2 + custom classification head with HF Trainer (notebooks) | Apache-2.0 | yes (author's HF models) | Medium | Template for quick XLS-R/indicwav2vec fine-tune with HF Trainer | https://github.com/m3hrdadfi/soxan |
| 27 | MSP-Podcast Challenge (Odyssey 2024) baselines | 31 (seen 2026-10-11 via GitHub API) | WavLM-based categorical and attribute baselines | None detected | yes | Medium | Use WavLM categorical baseline as a feature extractor / warm start | https://github.com/msplabresearch/MSP-Podcast_Challenge |
| 28 | MERTools (MER challenge toolkits) | 336 (seen 2026-10-11 via GitHub API) | Feature extraction for many acoustic/visual/lexical encoders + fusion  | None detected | no | Medium | Reuse feature-extraction scripts for multi-encoder embedding bank | https://github.com/zeroQiaoba/MERTools |
| 29 | kaggle-birdsong-recognition: 1st place Cornell Birdcall Identification | UNVERIFIED | PANNs-style SED with DenseNet121 backbone, tanh attention, smaller Att | UNVERIFIED | UNVERIFIED | High | Small head on pretrained backbone; tune decision thresholds for F1; trust CV over public LB (209-clip test). | https://github.com/ryanwongsa/kaggle-birdsong-recognition |
| 30 | argus-freesound: 1st place solution, Freesound Audio Tagging 2019 (Kag | UNVERIFIED | 2D CNN with attention/skip/aux classifiers on log-mel + deltas; SpecAu | UNVERIFIED | UNVERIFIED | High | SpecAugment + mixup in fine-tuning; inspect OOF low-confidence clips for relabel/drop; geometric mean of fold/model probabilities. | https://github.com/lRomul/argus-freesound |
| 31 | BirdCLEF 2024 - 3rd place (Theo Viel part) | 10 (seen 2026-10-11 via GitHub API) | Pseudo-labelling unlabeled soundscapes + distillation; 224x224 log-mel | MIT | no | Medium | Pseudo-label the test set with ensemble, retrain (self-training) if allowed | https://github.com/TheoViel/kaggle_birdclef2024 |
| 32 | BirdCLEF 2024 - 1st place | 2 (seen 2026-10-11 via GitHub API) | See Kaggle writeup linked in README; released checkpoints on Kaggle da | Apache-2.0 | yes (Kaggle dataset) | Medium | Template for Kaggle-only training | https://github.com/arpoyda/BirdCLEF_2024 |

---

## 12. Pretrained models

Fields from Hugging Face model cards / API; "UNVERIFIED" where the card does not state it. **Transferability measured here** for three of them (frozen probe, dup-group CV Macro-F1): XLS-R 300M 0.80 > WavLM Large 0.78 > WavLM Base+ 0.75 (`MEASURED HERE`). Domain mismatch: WavLM/HuBERT are English-only; XLS-R, MMS, w2v-BERT 2.0, Whisper, IndicWav2Vec, Vakyansh include Hindi.

| # | Model | Architecture / size | Training data | Hindi in pretraining? | Licence | Compute (T4) | Rule risk | Recommended experiment | Link |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | XLS-R 300M | 300M (card) | 436k h, 128 languages (card) | yes | apache-2.0 | OK fp16 | none. | Full fine-tune with weighted layer pooling; compare vs WavLM-large to measure value of Hindi pretraining. | https://huggingface.co/facebook/wav2vec2-xls-r-300m |
| 2 | WavLM Large | UNVERIFIED (~316M) | 94k h English (card) | no | UNVERIFIED on HF card (micro | fine-tune OK with batch 8-16 of 3 s | none. | Full fine-tune (fp16, grad-ckpt, LLRD) and frozen weighted-layer probe; compare with XLS-R (Hindi-aware). | https://huggingface.co/microsoft/wavlm-large |
| 3 | WavLM Base+ | UNVERIFIED (~95M) | 60k h Libri-Light + 10k h GigaSpeech + 24k h VoxPopuli (card | no | UNVERIFIED on HF card (micro | easy | none. | Fine-tune with weighted layer sum + attentive pooling; 5-fold CV; use as fast workhorse for augmentation ablat | https://huggingface.co/microsoft/wavlm-base-plus |
| 4 | Whisper large-v3 | 1550M total (card table | large-v3 card: 1M h weak + 4M h pseudo-labeled | yes (hi in card) | apache-2.0 | encoder FT with fp16/LoRA | none. Tip: Whisper pads to 30 s - truncate encoder input/pos | Frozen encoder layer-wise features -> probe; LoRA fine-tune of encoder (~635M) on T4. | https://huggingface.co/openai/whisper-large-v3 |
| 5 | Whisper large-v3-turbo | ~809M (HF API) | UNVERIFIED | yes | mit | encoder FT with LoRA/fp16 | none. | Equivalent to large-v3 encoder features; pick one of the two for compute. | https://huggingface.co/openai/whisper-large-v3-turbo |
| 6 | Whisper medium | 769M total (card table | large-v3 card: 1M h weak + 4M h pseudo-labeled | yes (hi in card) | apache-2.0 | encoder FT with fp16/LoRA | none. Tip: Whisper pads to 30 s - truncate encoder input/pos | Fine-tune encoder with fp16 + grad-ckpt or LoRA; layer-weighted pooling. | https://huggingface.co/openai/whisper-medium |
| 7 | emotion2vec+ large (emotion2vec_plus_large) | ~300M (card) | 42,526 h (card) | unknown | other (FunASR model-license) | inference/feature extraction easy, fine-tune OK wi | low (public), license 'other'. | Top priority: zero-shot 4-class mapping baseline, then frozen-feature head and partial fine-tune; use as ensem | https://huggingface.co/emotion2vec/emotion2vec_plus_large |
| 8 | emotion2vec+ base (emotion2vec_plus_base) | ~90M (card) | 4788 h (card) | unknown | other (FunASR model-license) | easy (fine-tune fits) | low, license 'other'. | Use as frozen embedding + logits extractor; stack its 9-class posteriors with a WavLM/XLS-R fine-tune. | https://huggingface.co/emotion2vec/emotion2vec_plus_base |
| 9 | emotion2vec_base (self-supervised emotion representation) | UNVERIFIED (base-size, 768-d) | open-source unlabeled emotion data (hours not on card) | unknown | other (FunASR model-license) | inference trivial, full fine-tune feasible | license 'other' (FunASR MODEL_LICENSE) - publicly available, | Extract frozen utterance+frame features (granularity=frame) for all 1035 clips; train linear/attentive-pooling | https://huggingface.co/emotion2vec/emotion2vec_base |
| 10 | w2v-BERT 2.0 | ~580M (HF API) | 4.5M h, >143 languages (card) | yes | mit | FT feasible small batch | none. | Fine-tune with audio-classification head (fp16, grad-ckpt) or frozen weighted-layer probe. | https://huggingface.co/facebook/w2v-bert-2.0 |
| 11 | XLS-R 1B | 1B (card) | 436k h (card) | yes | apache-2.0 | frozen extraction fp16 fine, FT needs LoRA + grad- | none. | Frozen per-layer mean-pooled features -> layer selection -> MLP/SVM. | https://huggingface.co/facebook/wav2vec2-xls-r-1b |
| 12 | ai4bharat/indicwav2vec-hindi | UNVERIFIED (~317M large) | UNVERIFIED | yes | apache-2.0 | OK | GATED (auto-approve, contact info) - still publicly availabl | Use encoder (drop CTC head) as SER backbone; frozen layer-wise probe then fine-tune. | https://huggingface.co/ai4bharat/indicwav2vec-hindi |
| 13 | Vakyansh Hindi base pretrained (4200 h) | UNVERIFIED (base ~95M) | 4200 h Hindi (card) | yes | apache-2.0 | easy | none. | Fine-tune for 4-class SER; compare monolingual Hindi vs multilingual XLS-R. | https://huggingface.co/Harveenchadha/vakyansh_hindi_base_pretrained |
| 14 | IndicWhisper / Vistaar (12 Indian-language Whisper fine-tune | UNVERIFIED (~769M if medium) | >10,700 h labelled across 12 languages (page) | yes | mit | encoder FT w/ fp16 | none (MIT). | Use Hindi IndicWhisper encoder as SER backbone vs vanilla whisper-medium. | https://indicwhisper.objectstore.e2enetworks.net/hindi_models.zip |
| 15 | whisper-hindi-medium (SPRING Lab IIT Madras) | UNVERIFIED (whisper-medium class) | UNVERIFIED | yes | apache-2.0 | encoder FT w/ fp16/LoRA | none. | Encoder as SER backbone; compare to vanilla Whisper same size. | https://huggingface.co/vasista22/whisper-hindi-medium |
| 16 | ARTPARK-IISc/whisper-large-v3-vaani-hindi | ~1.54B (HF API) | UNVERIFIED | yes | apache-2.0 | frozen/LoRA | none. | Frozen encoder features or LoRA. | https://huggingface.co/ARTPARK-IISc/whisper-large-v3-vaani-hindi |
| 17 | Talha/urdu-audio-emotions | ~316M (XLSR-53 large) | UNVERIFIED | no (Urdu yes) | apache-2.0 | OK fp16 | none. | Zero-shot predict on Hindi train set (no label mapping needed); then use as initialisation for Hindi fine-tune | https://huggingface.co/Talha/urdu-audio-emotions |
| 18 | speechbrain/emotion-recognition-wav2vec2-IEMOCAP | UNVERIFIED (wav2vec2-base ~95M) | IEMOCAP | no | apache-2.0 | easy | none. | Zero-shot 4-class baseline; use SpeechBrain recipe as fine-tuning template on Hindi data. | https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP |
| 19 | SUPERB ER head: wav2vec2-base-superb-er | UNVERIFIED (backbone wav2vec2-base) | IEMOCAP 4-class (session1 acc 0.6343 on card) | no | apache-2.0 | easy | none (IEMOCAP-trained weights are public). | Zero-shot predict on train to measure cross-lingual transfer; use as init for full fine-tuning (head already 4 | https://huggingface.co/superb/wav2vec2-base-superb-er |
| 20 | 3loi/SER-Odyssey-Baseline-WavLM-Categorical | ~319M (HF API) | UNVERIFIED | no | mit | OK fp16 | none (MIT). | Zero-shot mapping Angry/Happy/Neutral/Sad; fine-tune as WavLM-large init with class-balanced loss. | https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Categorical |
| 21 | tiantiaf/wavlm-large-categorical-emotion (Vox-Profile) | ~316M (HF API) | UNVERIFIED | no | openrail | OK | OpenRAIL use restrictions only. | Zero-shot mapping; embeddings for probe; fine-tune with same recipe on Hindi. | https://huggingface.co/tiantiaf/wavlm-large-categorical-emotion |
| 22 | audeering wav2vec2-large-robust-12-ft-emotion-msp-dim | ~165M (safetensors count via HF API) | MSP-Podcast | no | cc-by-nc-sa-4.0 (commercial  | easy | NON-COMMERCIAL (CC BY-NC-SA) - fine for Kaggle research but  | Extract 3-d AVD + 1024-d pooled embedding; add as stacking features; also fine-tune with a 4-class head. | https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim |
| 23 | MMS-300m (pretrained) | ~300M | ~500k h, 1,400+ languages (card) | yes | cc-by-nc-4.0 | OK | NON-COMMERCIAL license (fine for competition, note). | Fine-tune alongside XLS-R-300M; ensemble. | https://huggingface.co/facebook/mms-300m |
| 24 | mHuBERT-147 | 95M (card) | 90k h, 147 languages (card) | yes (hi in card) | cc-by-nc-sa-4.0 | easy | non-commercial share-alike. | Fine-tune as lightweight Hindi-aware backbone; ensemble member. | https://huggingface.co/utter-project/mHuBERT-147 |
| 25 | SenseVoiceSmall (FunAudioLLM) | UNVERIFIED for Small | UNVERIFIED | no (Small supports zh, yue, en | other (FunASR MODEL_LICENSE; | trivial | license 'other' - public. | Zero-shot emotion-tag predictions as features; extract encoder embeddings and train a 4-class head. | https://huggingface.co/FunAudioLLM/SenseVoiceSmall |
| 26 | Qwen2-Audio-7B-Instruct | ~8.4B (HF API) | UNVERIFIED | unknown (card lists en) | apache-2.0 | needs 4-bit quantisation for 16GB, slow but 1035 c | low (Apache-2.0). | Prompt 'Classify the speaker's emotion: angry/happy/neutral/sad' and use probs as features; or extract audio-e | https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct |
| 27 | microsoft/wavlm-base-plus-sv | UNVERIFIED (~95M) | UNVERIFIED | unknown | UNVERIFIED (UniSpeech LICENS | trivial | none. | Pseudo-speaker clustering; speaker-adversarial training ablation. | https://huggingface.co/microsoft/wavlm-base-plus-sv |
| 28 | SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb) | UNVERIFIED (~20M) | VoxCeleb | unknown (VoxCeleb contains som | apache-2.0 | trivial | none. | Cluster ECAPA embeddings -> GroupKFold; per-speaker feature standardisation; concat embedding to SER head. | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb |

**Competition-rule risk summary (`YOUR RECOMMENDATION`).** Safe: Apache-2.0 / MIT weights (XLS-R, Whisper, w2v-BERT 2.0, HuBERT, SUPERB heads, SpeechBrain, Vakyansh). Uncertain: non-commercial weights (MMS CC-BY-NC, audEERING CC-BY-NC-SA, mHuBERT-147), FunASR custom licence (emotion2vec, SenseVoice), gated-but-free (ai4bharat IndicWav2Vec/IndicConformer), Llama-3 licence (Shuka). The OSI clause in the rules is written for code; weights licences are a grey zone — prefer the safe set for final submissions.

---

## 13. Related datasets & rule safety

**Rule test (`OFFICIAL FACT`):** external data must be publicly available, equally accessible and minimal cost. Datasets behind signed declarations/agreements (IITKGP-SEHSC, IEMOCAP, MSP-Podcast, ESD) fail "equally accessible"; paid ones fail cost. No public Hindi emotion corpus with these exact four classes *and* spontaneous speech is freely available (`EXTERNAL RESEARCH EVIDENCE`).

**Overlap checks before using any external audio (`MEASURED HERE`):**
- vishlb/speech-emotion-recognition-hindi (ODbL): `vishlb: 3200 files | matches >= 0.9: 0 | max NCC per file quantiles [0.154 0.218 0.278 0.339]` → **no overlap** (true twins score ≥ 0.95).
- swastikatinkhede/multilingual-speech-dataset, Hindi subset (CC BY 4.0): `swast_hindi: 250 files | matches >= 0.9: 0 | max NCC per file quantiles [0.144 0.189 0.245 0.255]` → **no overlap** (true twins score ≥ 0.95).
- github.com/Himanshi7104/SRCASW-BhavVaani (sample of the competition's source data): 20/20 clips match competition audio (NCC ≥ 0.99), 5 of them test rows → **Not recommended / rule violation** (§16).
- Earlier hash-based check (exact PCM MD5 + coarse quantized fingerprint) on the same two Kaggle datasets also found 0 matches.

| # | Dataset | Task / labels | Domain | Licence | Relevance | Classification | Notes (incl. potential use) | Link |
|---:|---|---|---|---|---|---|---|---|
| 1 | speech emotion recognition Hindi (Kaggle; 8 actors, 8 emotions) | Speech emotion recognition | Hindi, acted read speech | ODbL (Kaggle: 'Database: Open Datab | High | **Competition-safe** | Competition-safe (public Kaggle, ODbL) but verify: Kaggle description is empty, so content details come from a Plaksha student deck; check overlap with competition audio  | https://www.kaggle.com/datasets/vishlb/speech-emotion-recognition-hindi |
| 2 | Indian Speech Emotion Recognition Dataset (Marathi/English/Hindi) | Speech emotion recognition | Marathi (500), English (250), Hindi (250 | CC BY 4.0 | High | **Competition-safe** | Competition-safe (CC BY 4.0, public). Provenance/annotation quality unverified; dataset updated June 2026 (v8); check overlap with competition audio. | https://www.kaggle.com/datasets/swastikatinkhede/multilingual-speech-dataset |
| 3 | Rasa: expressive multilingual Indian TTS dataset (incl. Hindi, 6 Ekman | Expressive TTS (usable for SER) | 22 Indian languages incl. Hindi, studio  | CC BY 4.0 (gated: accept conditions | High | **Competition-safe** | Competition-safe (CC BY 4.0, auto-gated). Caveat: few speakers per language, studio TTS style -> domain gap; 388 GB total so download Hindi subset only. | https://huggingface.co/datasets/ai4bharat/Rasa |
| 4 | SUBESCO: SUST Bangla Emotional Speech Corpus | Speech emotion recognition | Bangla, acted, 20 professional actors (1 | CC BY 4.0 | High | **Competition-safe** | Competition-safe (CC BY 4.0, direct Zenodo/HF download). Kaggle mirror sushmit0109 mislabels license as CC BY-NC-SA. | https://huggingface.co/datasets/sustcsenlp/bn_emotion_speech_corpus |
| 5 | BanglaSER: A Bangla speech emotion recognition dataset | Speech emotion recognition | Bangla, non-professional speakers | CC BY 4.0 | High | **Competition-safe** | Competition-safe (CC BY 4.0). Earlier Mendeley versions (B-SER v3/v4) had 4 classes / 1,224 files. | https://data.mendeley.com/datasets/t9h6p943xy/5 |
| 6 | UrduSER: A Dataset for Urdu Speech Emotion Recognition | Speech emotion recognition | Urdu, drama-serial style, 10 professiona | CC BY 4.0 | High | **Competition-safe** | Competition-safe (CC BY 4.0). | https://data.mendeley.com/datasets/jcpfjnk5c2/3 |
| 7 | CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset | Speech emotion recognition | English, 91 actors (48M/43F), 12 sentenc | ODbL 1.0 (+DbCL 1.0); Kaggle copy l | High | **Competition-safe** | Competition-safe (ODbL, direct download). Language mismatch (English). | https://www.kaggle.com/datasets/ejlok1/cremad |
| 8 | hindi-emotion-voice-references (Rasa Hindi/Urdu + RAVDESS clips) | Voice-cloning references (usable for SER) | Hindi (from Rasa) + English (RAVDESS) | Hindi: CC BY 4.0 (Rasa); English: C | Medium | **Competition-safe** | Competition-safe (inherits Rasa CC BY 4.0) but no Hindi neutral class. | https://huggingface.co/datasets/sumedhu/hindi-emotion-voice-references |
| 9 | Indian English Speech Emotion Dataset | Speech emotion recognition | Indian-accented English, multiple region | CC BY 4.0 | Medium | **Competition-safe** | Competition-safe (CC BY 4.0) - language mismatch (English). | https://data.mendeley.com/datasets/mtk28hgc6x |
| 10 | MaSERC: Dual-Scenario Marathi Emotional Speech Corpus | Speech emotion recognition | Marathi, scripted (carrier sentence + mo | CC BY 4.0 | Medium | **Competition-safe** | Competition-safe (CC BY 4.0). | https://data.mendeley.com/datasets/b9jrrpdxgb |
| 11 | ShEMO: Sharif Emotional Speech Database (Persian) | Speech emotion recognition | Persian, semi-natural radio plays | Apache-2.0 (HF card); paper: free f | Medium | **Competition-safe** | Competition-safe (HF Apache-2.0); note modified version github.com/aliyzd95/modified_shemo fixes label inconsistencies. | https://huggingface.co/datasets/pariajm/sharif_emotional_speech_dataset |
| 12 | nEMO: Dataset of Emotional Speech in Polish | Speech emotion recognition | Polish, 9 actors | CC BY-NC-SA 4.0 | Medium | **Competition-safe** | Competition-safe (public HF, NC-SA). | https://huggingface.co/datasets/amu-cai/nEMO |
| 13 | Emozionalmente: Italian crowdsourced emotional speech corpus | Speech emotion recognition | Italian, 431 amateur actors, 16 kHz, avg | CC BY 4.0 (OpenSLR SLR161) | Medium | **Competition-safe** | Competition-safe (CC BY 4.0). | https://openslr.org/161/ |
| 14 | ASVP-ESD: Audio, Speech and Vision Processing Lab Emotional Sound data | Speech emotion recognition (speech + non-speech) | Mixed languages; movies, TV, YouTube; >1 | CC BY 4.0 | Medium | **Competition-safe** | Competition-safe (CC BY 4.0); labels noisy, includes non-speech vocalizations - filter to speech. | https://zenodo.org/records/7132783 |
| 15 | RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song | Speech emotion recognition | North American English, 24 actors, 2 sen | CC BY-NC-SA 4.0 | Medium | **Competition-safe** | Competition-safe (public Zenodo/Kaggle; NC-SA). Language mismatch. | https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio |
| 16 | Berlin EmoDB (Berlin Database of Emotional Speech) via audb | Speech emotion recognition | German, 10 actors, 10 sentences | CC0-1.0 (audeering audb package) | Medium | **Competition-safe** | Competition-safe (CC0 via audb). | https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb |
| 17 | MELD: Multimodal EmotionLines Dataset (Friends TV) | Emotion recognition in conversation | English TV sitcom dialogue, avg 3.6 s | GPL-3.0 (repository) | Medium | **Competition-safe** | Competition-safe (public, GPL-3.0); English, noisy labels. | https://huggingface.co/datasets/declare-lab/MELD |
| 18 | M2H2: Multimodal Multiparty Hindi Dataset for Humor Recognition | Humor recognition | Hindi TV series, 13 episodes, ~4.46 h | MIT (repository) | Low | **Competition-safe** | Competition-safe as unlabeled audio (MIT repo), but no emotion labels. | https://github.com/declare-lab/M2H2-dataset |
| 19 | C-BESD: Children Bilingual (Telugu-English) Emotion Speech Dataset | Speech emotion recognition | Telugu + English, children 6-12 | CC BY-SA 4.0 | Low | **Competition-safe** | Competition-safe (CC BY-SA) but child voices; low expected benefit. | https://www.kaggle.com/datasets/pranuthi19/bilingual-emotion-speech-datasetbesd |
| 20 | BANSpEmo: A Bangla Emotional Speech Recognition Dataset | Speech emotion recognition | Bangla, read sentences | CC BY 4.0 | Low | **Competition-safe** | Competition-safe (CC BY 4.0) but only 3 of 4 classes. | https://data.mendeley.com/datasets/rdwn4bs5ky |
| 21 | BanglaMOOD: 8-emotion Bangla speech dataset | Speech emotion recognition | Bangla | CC BY-SA 4.0 (Kaggle label) | Low | **Competition-safe** | Competition-safe per Kaggle label; contents not verified. | https://www.kaggle.com/datasets/mostakimhossain52/banglamood-8-emotion-bangla-speech-dataset |
| 22 | MESD: Mexican Emotional Speech Database | Speech emotion recognition | Mexican Spanish, adult + child voices | CC BY 4.0 | Low | **Competition-safe** | Competition-safe (CC BY 4.0). | https://data.mendeley.com/datasets/cy34mh68j9/5 |
| 23 | CaFE: Canadian French Emotional speech dataset | Speech emotion recognition | Canadian French, 12 actors, 2 intensitie | CC BY-NC-SA 4.0 | Low | **Competition-safe** | Competition-safe (public, NC-SA). | https://zenodo.org/record/1478765 |
| 24 | JL-Corpus: Open Source Emotional Speech Corpus for HRI (New Zealand En | Speech emotion recognition | NZ English, acted | CC0 (Kaggle label); GitHub states n | Low | **Competition-safe** | Competition-safe per Kaggle CC0 label. | https://www.kaggle.com/datasets/tli725/jl-corpus |
| 25 | TESS: Toronto Emotional Speech Set | Speech emotion recognition | Canadian English, 2 female actresses, si | CC BY-NC 4.0 (Borealis); Kaggle cop | Low | **Competition-safe** | Competition-safe (public) but low value; NC/ND licensing. | https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess |
| 26 | voiceagent-ser-open (EmoDB + SUBESCO + nEMO bundle) | Speech emotion recognition | German, Bangla, Polish | Per-corpus upstream licenses (CC0 / | Low | **Competition-safe** | Competition-safe per component licenses. | https://huggingface.co/datasets/Pomoika24/voiceagent-ser-open |
| 27 | BhavVani (Hindi SER dataset from sitcom 'Sarabhai vs Sarabhai') | Speech emotion recognition | Hindi, TV sitcom (semi-natural, multi-sp | UNVERIFIED (no license stated; acce | High | **Competition-uncertain** | Competition-uncertain: gated by request form with no stated license; check that it counts as 'publicly and freely available' before use. Repo says contents moved to githu | https://forms.gle/9AqxS2oY4XVSeH1UA |
| 28 | SHEMuD master dataset (Hindi-English multimodal; 7 emotions) - Kaggle  | Speech emotion recognition / multimodal ERC | Hindi TV comedy 'Shrimaan Shrimati Phir  | CC BY-NC-SA 4.0 (Kaggle label); sou | High | **Competition-uncertain** | Competition-uncertain: third-party re-upload of TV audio; emotion labels annotated in a humor-recognition study (possibly text-driven); NC license. | https://www.kaggle.com/datasets/hbistechie/shemud-master-dataset |
| 29 | URDU dataset (Cross Lingual Speech Emotion Recognition: Urdu vs. Weste | Speech emotion recognition | Urdu, spontaneous talk-show speech from  | Research purposes only (README); no | High | **Competition-uncertain** | Competition-uncertain: README restricts to research use; widely mirrored (Kaggle kingabzpro/urdu-emotion-dataset). Paper: adding a fraction of target-language data to mul | https://github.com/siddiquelatif/URDU-Dataset |
| 30 | IEMOCAP: Interactive Emotional Dyadic Motion Capture database | Speech emotion recognition | English dyadic, scripted + improvised | License PDF via electronic release  | High | **Competition-uncertain** | Competition-uncertain: requires release-form approval, so may not be 'publicly and freely available'. Kaggle copy (columbine/iemocap) likely violates terms - do not rely  | https://sail.usc.edu/iemocap/iemocap_release.htm |
| 31 | Indian Emotional Speech Corpora (IESC) | Speech emotion recognition | North-Indian speakers (language not stat | Unknown (Kaggle license field 'Unkn | Medium | **Competition-uncertain** | Competition-uncertain: license 'Unknown'. Associated paper: 'A lightweight 2D CNN based approach for speaker-independent emotion recognition from speech...' (Multimedia T | https://www.kaggle.com/datasets/ybsingh/indian-emotional-speech-corpora-iesc |
| 32 | MNITJ-SEHSD: A Hindi Emotional Speech Database | Speech emotion recognition | Hindi, acted (neutral sentences) | UNVERIFIED | Medium | **Competition-uncertain** | Competition-uncertain: no public download location found; details only from a secondary course deck. | https://ai3011.plaksha.edu.in/Spring 2026/PDFs/Frostyllusion.pdf |
| 33 | EmoTa: A Tamil Emotional Speech Dataset | Speech emotion recognition | Sri Lankan Tamil, acted, 48 kHz | EmoTa Academic-Commercial License v | Medium | **Competition-uncertain** | Competition-uncertain: academic license with redistribution ban; Dravidian (not Indo-Aryan). | https://huggingface.co/datasets/aaivu-labs/EmoTa |
| 34 | KBES: KUET Bangla Emotional Speech dataset with intensity levels | Speech emotion recognition | Bangla TV/drama dialogue (semi-natural), | UNVERIFIED (Kaggle mirror says Apac | Medium | **Competition-uncertain** | Competition-uncertain: original license not verified; several Kaggle re-uploads with conflicting licenses. | https://www.kaggle.com/datasets/rimondevnath/kuet-bangla-emotional-speech-kbes |
| 35 | ESD: Emotional Speech Dataset (English + Mandarin, parallel) | Speech emotion recognition / emotional voice conve | English + Mandarin, 20 speakers, studio | Repo MIT, but data 'research purpos | Medium | **Competition-uncertain** | Competition-uncertain: requires emailing a signed license; research-only clause. | https://github.com/HLTSingapore/Emotional-Speech-Data |
| 36 | Speech Emotion Recognition (en) - combined CREMA-D/RAVDESS/SAVEE/TESS  | Speech emotion recognition | English acted | Data files (c) Original Authors (in | Medium | **Competition-uncertain** | Competition-uncertain: SAVEE portion redistributed against its license; prefer official sources. | https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en |
| 37 | CAMEO: Collection of Multilingual Emotional Speech Corpora | SER / multilingual benchmark | Multilingual | Per-corpus UNVERIFIED | Medium | **Competition-uncertain** | Competition-uncertain: per-corpus licenses vary; HF URL not shown on abstract page. | https://arxiv.org/abs/2505.11051 |
| 38 | Audio Emotion Detection Dataset (English+Hindi YouTube clips) | Speech emotion recognition | English+Hindi YouTube speech, ~60 s clip | CC BY 4.0 (card); YouTube source ri | Low | **Competition-uncertain** | Competition-uncertain: YouTube-derived content, unclear labeling procedure. | https://huggingface.co/datasets/RapidOrc121/audio-emotion-detection-dataset |
| 39 | indian-tts-emotion-60min (Hindi + Indian English YouTube clips with em | TTS/ASR with emotion tags | Hindi + Indian English, real YouTube spe | CC BY 4.0 (card); per-clip sources  | Low | **Competition-uncertain** | Competition-uncertain: tiny, weakly labeled. | https://huggingface.co/datasets/sarthwa8/indian-tts-emotion-60min |
| 40 | EmoInHindi: Multi-label Emotion and Intensity Dataset in Hindi Dialogu | Text emotion recognition in dialogue | Hindi text (no audio) | UNVERIFIED | Low | **Competition-uncertain** | Competition-uncertain / text-only, not audio. | https://aclanthology.org/2022.lrec-1.627 |
| 41 | IIIT-H TEMD: Semi-Natural Telugu Emotional Speech Database from Profes | Speech emotion recognition | Telugu, semi-natural drama scenarios | UNVERIFIED (availability not stated | Low | **Competition-uncertain** | Competition-uncertain: no public download found. | https://aclanthology.org/2020.lrec-1.192 |
| 42 | TaMaR-EmoDB: Multi-Lingual Speech Corpora for Emotion Analysis (Tamil, | Speech emotion recognition | Malayalam, Tamil, Ravula; acted | UNVERIFIED (availability not stated | Low | **Competition-uncertain** | Competition-uncertain: no download found. | https://www.isca-archive.org/interspeech_2019/rajan19_interspeech.html |
| 43 | Grusha Kannada Emotional TTS dataset | Emotional TTS / SER | Kannada, single female speaker, 16 kHz c | CC BY-NC 4.0 | Low | **Competition-uncertain** | Competition-uncertain: public but NC license and gated by contact sharing; single speaker limits value. | https://huggingface.co/datasets/grushaaaaa/kannada-emotional-tts |
| 44 | NITK-KLESC: Kannada Language Emotional Speech Corpus for Speaker Recog | Speaker recognition under emotion | Kannada, acted | UNVERIFIED | Low | **Competition-uncertain** | Competition-uncertain: no download terms found. | https://idr.nitk.ac.in/handle/123456789/29265 |
| 45 | Telugu Emotion Speech (Kaggle) | Speech emotion recognition | Telugu | Apache 2.0 (Kaggle label) | Low | **Competition-uncertain** | Competition-uncertain: no description of provenance/labels. | https://www.kaggle.com/datasets/jettysowmith/telugu-emotion-speech |
| 46 | Urdu-Sindhi Speech Emotion Corpus | Speech emotion recognition | Urdu and Sindhi | UNVERIFIED | Low | **Competition-uncertain** | Competition-uncertain: availability not verified. | https://ftp.saiconference.com/Downloads/Volume11No4/Paper_104-Introducing_the_Urdu_Sindhi_Speech_Emotion_Corpus.pdf |
| 47 | Rasa Urdu Emotions Dataset (Kaggle repack of AI4Bharat Rasa Urdu) | Expressive TTS (usable for SER) | Urdu | CC0 (Kaggle label; upstream Rasa is | Low | **Competition-uncertain** | Competition-uncertain: third-party repack; prefer the official Rasa HF source for license clarity. | https://www.kaggle.com/datasets/anvayjainmm24b023/rasa-urdu-final-v1 |
| 48 | ASED: Amharic Speech Emotion Dataset | Speech emotion recognition | Amharic, 4 dialects | UNVERIFIED ('freely available for r | Low | **Competition-uncertain** | Competition-uncertain: license unverified; language far from Hindi. | https://arxiv.org/abs/2201.02710 |
| 49 | EMOVO: Italian emotional speech corpus | Speech emotion recognition | Italian, 6 actors, 14 sentences | UNVERIFIED | Low | **Competition-uncertain** | Competition-uncertain: official license/download not verified (DagsHub mirror exists). | https://aclanthology.org/L14-1478/ |
| 50 | AESDD: Acted Emotional Speech Dynamic Database (Greek) | Speech emotion recognition | Greek, 5 actors | UNVERIFIED | Low | **Competition-uncertain** | Competition-uncertain: license not verified. | https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/ |
| 51 | RESD: Russian Emotional Speech Dialogs | Speech emotion recognition | Russian acted dialogues, 44.1 kHz | No license in metadata | Low | **Competition-uncertain** | Competition-uncertain: no license stated. | https://huggingface.co/datasets/Aniemore/resd |
| 52 | EmoV-DB: The Emotional Voices Database | Emotional TTS | English, 4 speakers, CMU Arctic text | Custom non-commercial (per secondar | Low | **Competition-uncertain** | Competition-uncertain: non-commercial custom license; poor class match. | https://www.openslr.org/115/ |
| 53 | SAVEE: Surrey Audio-Visual Expressed Emotion database | Speech emotion recognition | British English, 4 male speakers | Custom license: use within licensee | Low | **Competition-uncertain** | Competition-uncertain: registration + no-redistribution license; Kaggle mirrors are unofficial. | https://www.kaggle.com/datasets/barelydedicated/savee-database |
| 54 | EmoDB 2.0 (extension with ambiguous samples, glottograms, naturalness  | Speech emotion recognition | German acted | UNVERIFIED | Low | **Competition-uncertain** | Competition-uncertain: license of 2.0 release not verified. | https://www.isca-archive.org/interspeech_2025/burkhardt25_interspeech.html |
| 55 | ssi-speech-emotion-recognition (CREMA-D/TESS/RAVDESS/SAVEE bundle) | Speech emotion recognition | English (tagged en+hi, Hindi portion not | Tagged MIT (conflicts with upstream | Low | **Competition-uncertain** | Competition-uncertain: license relabelling of upstream data. | https://huggingface.co/datasets/stapesai/ssi-speech-emotion-recognition |
| 56 | EmoNet-Voice: synthetic multilingual fine-grained speech emotion bench | SER (fine-grained, synthetic) | Synthetic voices, 4 languages (en, de, e | UNVERIFIED | Low | **Competition-uncertain** | Competition-uncertain: release status/license unverified; no Hindi; synthetic. | https://arxiv.org/abs/2506.09827v3 |
| 57 | IIIT-H Telugu emotion database (Interspeech 2013, memory-recall elicit | Speech emotion recognition | Telugu, 7 students (2F/5M), induced via  | UNVERIFIED | Low | **Competition-uncertain** | Competition-uncertain: availability not stated. | https://www.isca-archive.org/interspeech_2013/gangamohan13_interspeech.pdf |
| 58 | IITKGP-SEHSC: Hindi Speech Corpus for Emotion Analysis | Speech emotion recognition | Hindi, acted/simulated, studio 16 kHz | Academic-only declaration: academic | High | **Research-only** | Research-only: access requires a signed declaration forwarded by head of an academic institution and forbids redistribution, so it is not 'publicly and freely available'; | https://cse.iitkgp.ac.in/~ksrao/pdf/declaration.pdf |
| 59 | MSP-Podcast corpus | Speech emotion recognition | English natural podcast speech | Free academic license signed by ins | Medium | **Research-only** | Research-only: institution must sign license; not freely available to all participants. | https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html |
| 60 | Indian EmoSpeech Command Dataset (keywords incl. Hindi 'bachao') | Emotion-aware keyword spotting | Indian crowd-sourced, real-world noise,  | Non-commercial; original via emo-sp | Low | **Research-only** | Research-only (original access requires educational email request); Kaggle mirror by an author is CC BY-NC-SA. | https://www.kaggle.com/datasets/ujjwal9/emospeech |
| 61 | IITKGP-SESC: Telugu simulated emotion speech corpus | Speech emotion recognition | Telugu, acted | UNVERIFIED (same IITKGP declaration | Low | **Research-only** | Research-only: IIT Kharagpur academic access process. | https://cse.iitkgp.ac.in/~ksrao/res.html |
| 62 | MSP-IMPROV corpus | Speech emotion recognition | English improvised dyadic | Academic license form signed by aut | Low | **Research-only** | Research-only: signed academic agreement required. | https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html |
| 63 | IITG Politician Speech Corpus (Analysis of Emotional Content in Indian | Speech emotion recognition | Hindi, spontaneous public speeches from  | UNVERIFIED | High | **Not recommended** | Not recommended as data (no public release found). Useful evidence that 4-class spontaneous Hindi SER is feasible ~79% UAR. | https://arxiv.org/abs/2007.13325 |
| 64 | OdiaESC-MCS: Odia / Odia-English code-switched emotional speech corpus | Speech emotion recognition | Odia + Odia-English code-switched, acted | Research/educational use; IEEE Data | Low | **Not recommended** | Not recommended: requires paid IEEE DataPort subscription (not freely available). | https://ieee-dataport.org/documents/odiaesc-mcs-emotional-speech-corpus-multilingual-code-switching-low-resource-indigenous |
| 65 | Indian TTS dataset (Hindi + Indian English YouTube, rule-based emotion | TTS (weak emotion tags) | Hindi + Indian English YouTube | UNVERIFIED | Low | **Not recommended** | Not recommended for supervised SER (labels not human-annotated). | https://huggingface.co/datasets/Itsharshi/indian-tts-dataset |

---

## 14. Related competitions & transferable ideas

**Most similar challenges (as their sources state; `EXTERNAL RESEARCH EVIDENCE`).**

| Challenge | Similarity | Winner / top approach | Transferable idea |
|---|---|---|---|
| Odyssey 2024 SER Challenge, Task 1 (MSP-Podcast, 8 classes, Macro-F1) | same metric, natural speech | 1st (Chen et al., arXiv 2405.20064): prior-weighted focal loss + majority vote over 7 models with different features/losses (Macro-F1 35.69%); 2nd TalTech: w2v-BERT audio + LLaMA-2 text; 3rd UPC: frozen XLS-R / wav2vec2 / WavLM with attention pooling, weighted CE, per-class thresholds, 3-model vote | multi-backbone frozen/fine-tuned voting; thresholds only nested |
| Interspeech 2025 SER in Naturalistic Conditions | Macro-F1, SSL backbones | MEDUSA (arXiv 2506.09556): cross-modal fusion, Manifold MixUp, soft targets, balanced sampling, meta-classifier, model soups | soft labels, balanced sampling (low priority here) |
| EmoSPeech 2024 (IberLEF; 3,000 train / 750 test, Spanish, Macro-F1) | small, non-English | 1st BSC-UPC (arXiv 2407.12467): attention pooling over frozen pre-trained speech + text features, 86.69% Macro-F1 | frozen foundation features + light head is enough at this scale |
| ComParE 2023 Emotion Share | SSL layer choice | Aalto: choose the right wav2vec2 model and its best single layer | per-layer probe (done: X03–X05) |
| Freesound Audio Tagging 2019 (Kaggle) | small, noisy labels | 1st argus-freesound: log-mel CNNs, SpecAugment + MixUp, noise-robust loss, relabelling, MLP stackers, geometric blend | relabel only conflicting duplicates; stacking only with nested CV |
| Cornell Birdcall 2020 (Kaggle) | ≤ 100 samples/class, LB shake-up | 1st (ryanwongsa): reduced attention head, noise augmentation, F1-tuned thresholds, 13-model k-of-n voting; **7th public → 1st private** | trust CV and diversity over the public LB |
| BirdCLEF 2024 (Kaggle) | pseudo-labelling | 1st-team parts: ensemble pseudo-labels on unlabeled soundscapes + distillation; BirdCLEF 2023 2nd place: pseudo-labelling "doesn't improve the score" | pseudo-labelling helps with large unlabeled pools; here only 131 non-twin test clips → low priority |
| PolEval 2025 SER (cross-lingual, Macro-F1) | cross-lingual transfer | winners not found; Qwen2-Audio baseline Macro-F1 ≈ 0.14–0.18 | audio-LLM zero-shot is weak; do not spend time there |

No previous Kaggle competition on Hindi or Indic speech emotion was found (closest: Bengali.AI Speech, which is ASR).

**All catalogued previous-competition / winning-solution entries:**

| # | Competition / solution | Year | Key idea (as stated by the source) | Transferable idea | Link |
|---:|---|---|---|---|---|
| 1 | 1st Place Solution to Odyssey Emotion Recognition Challenge Task1: Tackling Class Imbalanc | 2024 | Ranked 1st of 68 submissions: Macro-F1 35.69%, acc 37.32%; focal loss weighted by class priors trades major-class performance for macro-F1; majority vote of 7 diverse models beat all single models. | Class-weighted focal loss; ensemble of models differing in SSL backbone and loss; majority vote vs prob averaging. | https://arxiv.org/abs/2405.20064v1 |
| 2 | TalTech Odyssey 2024 SER challenge system (Wav2Vec2-BERT + LLaMA2 text) | 2024 | Ranked 2nd: macro-F1 0.354 on evaluation data, combining multilingual Wav2Vec2-BERT audio and LLM text model. | Wav2Vec2-BERT (w2v-BERT 2.0, multilingual incl. Hindi) as a backbone; optionally Hindi ASR transcripts + text model. | https://www.isca-archive.org/odyssey_2024/harm24_odyssey.html |
| 3 | The Interspeech 2025 Challenge on Speech Emotion Recognition in Naturalistic Conditions | 2025 | Combining audio and text foundation models was effective; record participation (search summary: 93 teams). | Consider a text branch from Hindi ASR (e.g., Whisper/IndicWhisper) fused with audio. | https://www.isca-archive.org/interspeech_2025/naini25_interspeech.html |
| 4 | Developing a High-performance Framework for Speech Emotion Recognition in Naturalistic Con | 2025 | Top in Task 2: text embeddings + gender auxiliary task + using ambiguous samples; best result from a simple two-system ensemble. | Auxiliary gender-prediction head (pseudo-labeled by a gender model) as multitask regularizer. | https://www.isca-archive.org/interspeech_2025/lertpetchpun25_interspeech.html |
| 5 | MER 2023: Multi-label Learning, Modality Robustness, and Semi-Supervised Learning | 2023 | MER-SEMI track explicitly tests semi-supervised learning with abundant unlabeled data; MER-NOISE tests robustness to noise. | Use unlabeled audio (test set + external Hindi speech) for pseudo-labeling/continued pretraining. | https://arxiv.org/abs/2304.08981v2 |
| 6 | The ACII 2022 Affective Vocal Bursts (A-VB) Workshop & Competition | 2022 | Type task (8-class) baseline UAR 0.4172; A-VB High winner test CCC 0.7295 (Anonymous winners per results table). |  | https://www.competitions.hume.ai/avb2022 |
| 7 | Predicting Affective Vocal Bursts with Finetuned wav2vec 2.0 | 2022 | Full fine-tuning of wav2vec2 effective on small paralinguistic datasets. |  | https://arxiv.org/pdf/2209.13146 |
| 8 | Winning the ACII Affective Vocal Bursts (A-VB) Challenge (Type task) - University of Augsb | 2022 | 1st place in A-VB Type task (8-class). |  | https://intranet.uni-augsburg.de/en/fakultaet/fai/informatik/prof/hcm/news/winning-acii-affective-vocal-bursts-vb-challenge |
| 9 | The INTERSPEECH 2009 Emotion Challenge | 2009 | First SER challenge; enforced speaker independence; unweighted average recall as metric for imbalanced classes. Participant 2-stage fusion got 41.59% (5-class) / 67.90% (2-class) recall. | Unweighted metrics require class-balanced training; openSMILE functionals as cheap extra branch. | https://www.isca-archive.org/interspeech_2009/schuller09_interspeech.html |
| 10 | The INTERSPEECH 2018 Computational Paralinguistics Challenge: Atypical & Self-Assessed Aff | 2018 | ComParE series: small, speaker-independent paralinguistic tasks scored with UAR; fusion of diverse systems typical. |  | https://www.isca-archive.org/interspeech_2018/schuller18_interspeech.html |
| 11 | Cogito entry to ComParE 2018 Self-Assessed Affect (LSTM with attention) | 2018 | Attention over frame-level features for utterance-level affect. |  | https://www.isca-archive.org/interspeech_2018/gorrostieta18_interspeech.html |
| 12 | Overview of EmoSPeech at IberLEF 2024: Multimodal Speech-text Emotion Recognition in Spani | 2024 | Small dataset (3,000 train / 750 test), non-English, macro-F1 metric -> structurally very similar; text-task top macro-F1 0.6719 (ITST). | Look at multimodal (speech+text) top systems for small non-English data. | https://portalcientifico.uned.es/documentos/670967f4002d1b09ea02261c |
| 13 | PolEval 2025 Speech Emotion Recognition task (Polish, cross-lingual) | 2025 | Cross-lingual SER with macro-F1: train 29,714 clips from 12 CAMEO corpora (7 languages), evaluate on unseen Polish (nEMO); no target-language data allowed. Very weak LLM baseline (F1-macro ~0.14-0.18) | Measure zero-shot transfer from multilingual public SER corpora to Hindi before fine-tuning on 826 clips. | https://github.com/poleval/2025-speech-emotion |
| 14 | MSP-Podcast SER Challenge 2024: L'antenne du Ventoux Multimodal Self-Supervised Learning f | 2024 | Ensemble of independently fine-tuned SSL models fused by an SVM at score level; dev macro-F1 ~0.35. | Stacking: logistic regression/SVM on OOF probabilities of several backbones. | https://arxiv.org/abs/2407.05746 |
| 15 | MEDUSA: A Multimodal Deep Fusion Multi-Stage Training Framework for Speech Emotion Recogni | 2025 | Ranked 1st in IS2025 Task 1: ensemble + meta-classifier, Manifold MixUp regularization, soft targets, balanced sampling, weight-averaged model soup. | Manifold mixup in head; class-balanced sampler; average weights of fold models (soup) or average predictions; stack with | https://arxiv.org/abs/2506.09556 |
| 16 | The ACM Multimedia 2023 Computational Paralinguistics Challenge: Emotion Share & Requests | 2023 | Multilingual real-world emotion task; wav2vec2 among baselines; participant reports HuBERT-Large + light self-attention improved baseline by 4.6%. |  | https://arxiv.org/abs/2304.14882v2 |
| 17 | Effect of Attention and Self-Supervised Speech Embeddings on Non-Semantic Speech Tasks | 2023 | Frozen HuBERT-Large + lightweight self-attention head improved over the challenge baseline by 4.6%. | Small transformer/attention head on frozen frame-level SSL features. | https://ar5iv.labs.arxiv.org/html/2308.14359 |
| 18 | The INTERSPEECH 2020 Computational Paralinguistics Challenge: Elderly Emotion, Breathing & | 2020 | Small-data emotion sub-challenge with speaker-independent split, UAR metric. |  | https://www.isca-archive.org/interspeech_2020/schuller20_interspeech.html |
| 19 | Is Everything Fine, Grandma? Acoustic and Linguistic Modeling for Robust Elderly Speech Em | 2020 | Winning small-data emotion system combined acoustic and linguistic (transcript) models. | Add a text branch from Hindi ASR transcripts. | https://www.uu.nl/en/news/utrecht-researchers-win-two-out-of-three-grand-challenges-in-computational-paralinguistics |
| 20 | The ICML 2022 Expressive Vocalizations Workshop and Competition (ExVo) | 2022 | Few-shot emotion track with speaker personalization; multi-task baseline S_MTL 0.335; few-shot mean CCC 0.444. |  | https://www.competitions.hume.ai/exvo2022 |
| 21 | MER 2024: Semi-Supervised Learning, Noise Robustness, and Open-Vocabulary Multimodal Emoti | 2024 | Semi-supervised track with large unlabeled set; noise track uses MUSAN speech as additive noise. | MUSAN noise augmentation; semi-supervised on unlabeled Hindi speech. | https://arxiv.org/pdf/2404.17113 |
| 22 | BSC-UPC at EmoSPeech-IberLEF2024: Attention Pooling for Emotion Recognition | 2024 | 1st place in EmoSPeech 2024 with 86.69% macro-F1 using attention pooling on pre-trained speech+text features (small non-English dataset). | Attention pooling head over frozen multilingual SSL features (+ optional Hindi ASR text embeddings). | https://arxiv.org/abs/2407.12467 |
| 23 | Double Multi-Head Attention Multimodal System for Odyssey 2024 Speech Emotion Recognition  | 2024 | Per-class thresholds tuned to maximize macro-F1 (fallback to next class) gave ~1% absolute validation macro-F1; 3-model hard-voting ensemble 33.80% val (baseline 30.70%); 3rd place, 34.41% test macro- | Per-class threshold/bias tuning on OOF probs; ensemble of frozen XLS-R, wav2vec2, WavLM heads with weighted CE. | https://arxiv.org/html/2406.10598v1 |
| 24 | Speech Emotion Recognition Using Fine-Tuned DWFormer: A Study on Track 1 of the IERPChalle | 2025 | 1st place in IERP 2024 Track 1; search snippet: Chinese-wav2vec2-Large features beat Chinese-HuBERT-Large and WavLM-Large -> language-matched SSL backbone can win. | Test Hindi/Indic-pretrained wav2vec2 (e.g., IndicWav2Vec, MMS) vs English WavLM. | https://arxiv.org/abs/2508.11371 |
| 25 | Aalto team wins ComParE 2023 Emotion Share sub-challenge (wav2vec2 + best layer + Bayesian | 2023 | Winning factors: selecting the right wav2vec2 model and its best-performing transformer layer, plus a Bayesian linear layer. | Layer sweep: train a linear/LR probe per layer on frozen features; pick best layer(s) per backbone. | https://teflon.aalto.fi/?p=458 |
| 26 | Enhancing Speech Emotion Recognition with Graph-Based Multimodal Fusion and Prosodic Featu | 2025 | Macro-F1 39.79% test / 42.20% val; explicit prosodic features (F0) added to SSL features. | Concatenate pooled SSL embedding with eGeMAPS/F0 statistics. | https://arxiv.org/abs/2506.02088 |
| 27 | Advancing Emotion Recognition via Ensemble Learning: Integrating Speech, Context, and Text | 2025 | 6th place in IS2025 SER challenge; +7.4% macro-F1 over official baseline. |  | https://www.isca-archive.org/interspeech_2025/shi25e_interspeech.pdf |
| 28 | Audio tagging with noisy labels and minimal supervision (Freesound Audio Tagging 2019 / DC | 2019 | Kaggle audio competition with a small curated set, label noise and acoustic mismatch between noisy train and test. |  | https://arxiv.org/pdf/1906.02975 |
| 29 | argus-freesound: 1st place solution, Freesound Audio Tagging 2019 (Kaggle) | 2019 | 1st place: heavy augmentation (SpecAugment+MixUp), noise-robust loss for noisy labels, relabeling suspicious curated clips, 5-fold CV, stacking + geometric-mean blending. | SpecAugment + mixup in fine-tuning; inspect OOF low-confidence clips for relabel/drop; geometric mean of fold/model prob | https://github.com/lRomul/argus-freesound |
| 30 | 2nd place Freesound Audio Tagging 2019 | 2019 | Silence trimming + random crops; single model CV 0.89763, ensemble CV 0.9108 (lwlrap). | Trim silence before SSL feature extraction; random-crop TTA. | https://github.com/qrfaction/2nd-Freesound-Audio-Tagging-2019 |
| 31 | Freesound Audio Tagging 2019 4th place solution (OsciiArt) | 2019 | Search snippet of team paper: semi-supervised relabeling of noisy data via model predictions, multitask learning, ensemble over clip lengths; 3rd public (lwlrap 0.750) / 4th private. | Ensemble over different crop lengths (e.g., 2 s, 3 s, full). | https://github.com/OsciiArt/Freesound-Audio-Tagging-2019 |
| 32 | General-purpose Tagging of Freesound Audio with AudioSet Labels: Task Description, Dataset | 2018 | Kaggle audio classification with only ~3,700 of 9,500 train clips manually verified. |  | https://arxiv.org/abs/1807.09902v3 |
| 33 | Training general-purpose audio tagging networks with noisy labels and iterative self-verif | 2018 | Iterative self-verification of noisy labels; ensemble MAP@3 0.951, 2nd of 558 submissions. | Iteratively drop/relabel train clips whose OOF prediction confidently disagrees with label. | https://research.jku.at/en/publications/training-general-purpose-audio-tagging-networks-with-noisy-labels/ |
| 34 | kaggle-birdsong-recognition: 1st place Cornell Birdcall Identification | 2020 | 1st private (from 7th public -> shake-up): reduce head capacity for <=100 samples/class, noise augmentation, F1-tuned thresholds, k-of-n voting ensemble. Private 0.681 / public 0.616. | Small head on pretrained backbone; tune decision thresholds for F1; trust CV over public LB (209-clip test). | https://github.com/ryanwongsa/kaggle-birdsong-recognition |
| 35 | Cornell Birdcall Identification (Kaggle 2020) - competition description | 2020 | F1-based audio competition where thresholding decided ranks. |  | https://wildlabs.net/en/funding-opportunity/competition-cornell-birdcall-identification |
| 36 | Recognizing bird species in diverse soundscapes under weak supervision (BirdCLEF 2021 2nd  | 2021 | Full use of pretrained CNNs + augmentation + ensembling under weak labels. |  | https://arxiv.org/abs/2107.07728v1 |
| 37 | BirdCLEF 2021 1st place summary (Japanese slides) | 2021 | Mixup of short crops used to counter weak clip-level labels. |  | https://speakerdeck.com/startjapan/birdclef2021matome |
| 38 | BirdCLEF 2024 1st Solution explained (Zenn article) | 2024 | Winner corrected primary labels using model predictions + secondary labels and added soft pseudo-labels from an external pretrained classifier at small weight (0.05). | Use an external pretrained SER model's soft predictions (e.g., emotion2vec) as low-weight auxiliary targets (knowledge d | https://zenn.dev/yuto_mo/articles/ad43c630729073 |
| 39 | birdclef-2024 (CPMP part of 1st-place team solution) | 2024 | Pseudo-labeling unlabeled target-domain audio with a large ensemble and distilling into small models was key; balance pseudo-labeled and labeled samples; CPMP part alone private 0.6896 (would place 3r | Ensemble teacher -> pseudo-label test (and any unlabeled Hindi audio) -> retrain students with similar count of pseudo v | https://github.com/jfpuget/birdclef-2024 |
| 40 | Overview of BirdCLEF 2024: Acoustic Identification of Under-studied Bird Species in the We | 2024 | Top private ROC-AUC 0.690 (public 0.738); top-10 within 1.5% -> public/private shake-up typical with small/shifted test. |  | https://ceur-ws.org/Vol-3740/paper-184.pdf |
| 41 | Tackling domain shift in bird audio (BirdCLEF+ 2025 2nd place) | 2025 | 2nd private (ROC-AUC 0.928): domain-shift handled by in-domain transfer + distillation-based semi-supervised learning on unlabeled target audio. | Self-distillation on unlabeled test audio. | https://ceur-ws.org/Vol-4038/paper_256.pdf |
| 42 | BirdCLEF 2025 top-3 solution review (slides) | 2025 | Reviewer's synthesis: self-training on unlabeled target-domain audio recurs across top BirdCLEF solutions. |  | https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/Birdclef_2025_top3_solution_review.pdf |
| 43 | birdclef2023-2nd-place-solution | 2023 | README notes pseudo-labeling is optional and 'doesn't improve the score' -> pseudo-labeling is not universally helpful; external data + background noise aug were used. | Validate pseudo-labeling gains in CV before adopting. | https://github.com/LIHANG-HONG/birdclef2023-2nd-place-solution |
| 44 | OOD-Speech: A Large Bengali Speech Recognition Dataset for Out-of-Distribution Benchmarkin | 2023 | Largest Indic-speech Kaggle competition: OOD test domains (TV drama, talk shows, sermons) -> domain shift between train and test is the core problem; 1st place (team Chimege) released Whisper-medium-b | Expect channel/domain mismatch between train and test Hindi clips; use adversarial validation; consider Indic-pretrained | https://arxiv.org/abs/2305.09688v1 |
| 45 | Bengali.AI Speech Recognition (Kaggle) discussion thread | 2023 | Indic speech Kaggle competition; 1st place Whisper-medium fine-tune mirrored on HF (Apache-2.0). | Whisper encoder fine-tuned on Indic speech may carry Hindi-relevant acoustics. | https://kaggle.com/c/bengaliai-speech/discussion/447961 |
| 46 | Deep Learning Approaches for Understanding Simple Speech Commands (TensorFlow Speech Recog | 2018 | Many architectures performed similarly; data preparation drove further improvements. | Focus on data cleaning/dedup/augmentation over architecture search. | https://arxiv.org/pdf/1810.02364 |
| 47 | Dacon Monthly Speech Emotion Recognition AI Challenge | UNVERIFIED | Closest found platform-hosted SER classification competition (anonymized TRAIN_/TEST_ wav files, csv submission); acted corpus with many repeated sentences/speakers -> speaker/sentence leakage risks s | Check whether test speakers/sentences overlap train via speaker-embedding clustering. | https://dacon.io/en/competitions/official/236105/data |
| 48 | Freesound Audio Tagging 2019 - 3rd team (ebouteillon) | 2019 (repo | Well-documented small-data audio pipeline | Training-schedule ideas | https://github.com/ebouteillon/freesound-audio-tagging-2019 |
| 49 | Freesound General-Purpose Audio Tagging 2018 - 8th place | 2018 (repo | Trimming silence + combining CNN with summary-statistic features | Trim + hybrid CNN/feature model | https://github.com/sainathadapa/kaggle-freesound-audio-tagging |
| 50 | BirdCLEF 2021 - 2nd place | 2021 (repo | Clean config-driven Kaggle training template | Template for experiment management | https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place |
| 51 | BirdCLEF 2023 - 1st place | 2023 (repo | Pretraining on related external data before target fine-tune | Pre-fine-tune on external SER corpora (Urdu/CREMA-D) then Hindi | https://github.com/VSydorskyy/BirdCLEF_2023_1st_place |
| 52 | BirdCLEF 2024 - 1st place | 2024 (repo | Kaggle-kernel-reproducible pipeline | Template for Kaggle-only training | https://github.com/arpoyda/BirdCLEF_2024 |
| 53 | BirdCLEF 2024 - 3rd place (Theo Viel part) | 2024 (repo | README: key ingredient was pseudo-labelling unlabeled data and distillation into small models | Pseudo-label the test set with ensemble, retrain (self-training) if allowed | https://github.com/TheoViel/kaggle_birdclef2024 |

---

## 15. Validation strategy

**How the hidden test was generated (`MEASURED HERE`).** Random split of 1,035 files, no grouping: test twin rate 0.373 vs simulated random re-split 0.367 ± 0.031; adversarial AUC 0.454; flat Id/label structure. Speakers and recordings are therefore shared between train and test exactly as in a random split.

**Chosen scheme — two tracks on frozen fold assignments (`YOUR RECOMMENDATION`):**

| Track | Split | Why | What it estimates |
|---|---|---|---|
| A. LB proxy | Public pipeline's `train_with_folds.csv` (5-fold, exact-duplicates grouped, near-twins free) — canonical, never changed | Same generative process as the hidden test (random; twin share 0.27–0.31 vs 0.37 in test) | Expected public/private score; use the twin decomposition 0.37·F1_twin + 0.63·F1_non-twin to correct for the twin-share difference |
| B. Novel clips | `StratifiedGroupKFold(5, shuffle, seed 0)` grouped by near-duplicate group (`dup_groups.csv`) | Removes memorisation; measures what decides the ranking | Non-twin skill; used for model/feature selection |

Always report: pooled OOF Macro-F1 (not mean of folds), per-class F1, twin vs non-twin F1, repeated-seed mean ± SD (3 repeats for cheap models), and a dup-group paired bootstrap against the incumbent.

**Schemes rejected and why.**
- Plain StratifiedKFold *as the only metric*: inflates by ≈ 0.09 over dup-group (e.g. WavLM Base+ L7: 0.841 vs 0.753) and rewards memorisation.
- GroupKFold by pseudo-speaker: x-vector clusters are unstable (§3.3) and the real test is not speaker-disjoint — it would optimise the wrong target.
- TimeSeriesSplit / purged / embargo: no temporal structure.
- Leave-one-domain-out (PCM_16 vs PCM_32): encodings are evenly mixed across classes/splits and not shifted; could be an extra robustness check only.
- Nested CV: required only for any tuned post-processing (thresholds, stacker weights).
- Repeated CV: yes for cheap probes (3 seeds); for fine-tuning, use ≥ 2 seeds before claiming gains < 0.01.

---

## 16. Leakage analysis & automated tests

| # | Route | Status | Evidence | Verdict |
|---|---|---|---|---|
| 1 | **Source-dataset sample repo** `github.com/Himanshi7104/SRCASW-BhavVaani` (20 wav, folder = label) | **Present** | 20/20 clips match competition audio with NCC ≥ 0.99; 19 train matches all label-consistent; **5 test matches (Ids withheld)** | Using these labels = human/external labelling of test records → violates the no-hand-labelling rule and the repo's research-only licence. **Do not use.** Expected gain would be tiny anyway (FT models likely already get several right). |
| 2 | Train↔test near-duplicate twins | Present (78 test rows) | §18 | Not leakage in the rule sense — these are training labels — and every model already memorises them. Twin override = harmless safety net (0 changes for FT submissions). |
| 3 | Duplicate leakage inside CV | Present | twin rows straddle random folds | Handled by Track B validation |
| 4 | Conflicting duplicate labels | Present (Id 75/636/878) | exact PCM duplicates, labels neutral/sad/neutral | drop or down-weight in training; cannot be "correct" both ways |
| 5 | Filename/Id leakage | Absent | Id order random; filename = Id | — |
| 6 | Metadata leakage (bit depth, duration, loudness) | Weak | PCM_32 balanced across classes; duration/energy are genuine acoustic cues | — |
| 7 | External-dataset overlap (vishlb, swastikatinkhede) | See §13 | NCC scan | Use only if no overlap |
| 8 | Speaker leakage | Present by design (random split) | x-vector analysis | Not exploitable beyond what models learn; do not "fix" it |
| 9 | Pseudo-label leakage | Risk if test pseudo-labels are used inside CV folds incorrectly | — | Simulate PL inside CV (E9) |
| 10 | Validation leakage from checkpoint selection | Present in public FT pipeline (best epoch chosen on the fold being scored) | code | OOF 0.862 is optimistic; compare with fixed-epoch runs when possible |
| 11 | Public-LB feedback leakage | Risk | ~14 submissions left, subset 52–105 clips | Follow the §6 decision table |
| 12 | Target-encoding / train-test concatenation | n/a | no tabular features | — |

**Automated tests (scripts in `research/scripts/`, all run for this report):**
- `audit_data.py` — exact PCM/file MD5 duplicates, Id-order label autocorrelation, format/duration audit.
- `near_duplicates.py` — all-pairs waveform max-NCC for duration-matched clips → `near_dup_pairs.csv`, `dup_groups.csv`.
- `embedding_neighbors.py` — hidden transformed duplicates in WavLM space (nearest non-twin neighbour label agreement by similarity decile: flat 0.46–0.63 → none found).
- `source_repo_overlap.py`, `external_overlap.py` — external audio vs competition audio.
- `adversarial_validation.py` — train-vs-test classifier, KS, PSI.

---

## 17. Distribution shift

`MEASURED HERE` (`adversarial_report.txt`): KS train vs test — duration 0.062 (p = 0.52), RMS 0.056 (p = 0.65), peak 0.072 (p = 0.34), silence fraction 0.095 (p = 0.09); PSI(duration) = 0.039; PCM_32 share 9.7% vs 10.5%; adversarial AUC (WavLM Base+ L6 + clip stats, dup-grouped 5-fold) **0.454**; nearest non-twin embedding-similarity quantiles test→train [0.43, 0.58, 0.75, 0.88] vs train→train [0.43, 0.58, 0.73, 0.87]. **No shift.** Implication: no domain adaptation needed; model selection can rely on CV; private-vs-public differences will be sampling noise, not shift.

---

## 18. Duplicates

`MEASURED HERE` (`near_dup_report.txt`, `dup_nature_report.txt`):

- Exact (identical decoded PCM): 21 groups, 43 rows (12 train–train pairs, 1 train triple, 8 train–test pairs — all 8 train twins labelled sad).
- Near-duplicates (max NCC ≥ 0.95 at some lag; same sample count in 100% of pairs; RMS ratio ≤ 1.04): 257 pairs → 227 groups, 467 rows. NCC histogram: 0 pairs in (0.5, 0.95], 10 in (0.95, 0.99], 65 in (0.99, 0.999], 182 > 0.999 → cleanly bimodal. `YOUR HYPOTHESIS`: augmented or re-exported copies (e.g. light noise / bit-depth change) of the same recording entered the pool before the split.
- Pairs by split: train–train 162, train–test 86, test–test 9; 8 test-only groups.
- Share of rows in groups by class (train): sad 56%, neutral 49%, angry 38%, happy 36%.
- Test rows with a train twin: **78** (twin labels: sad 29, angry 22, neutral 19, happy 15).
- Twin label consistency inside train: 300/301 (the only miss is the 75/636/878 conflict).

**How to use this legitimately (`YOUR RECOMMENDATION`).** (1) Group by `dup_group` for Track-B CV. (2) Apply the twin override as a safety net (it uses train labels only). (3) De-duplicate inside each training fold for fine-tuning (the public pipeline drops exact duplicates; near-twins effectively double-weight some recordings — harmless, but adds nothing).

---

## 19. Seed / fold / leaderboard noise

- Fold-assignment noise (3 repeated 5-fold splits, frozen LR probes): SD 0.002–0.012 (random) and 0.003–0.017 (dup-group) — `cv_*_layers.txt`.
- Seed noise of fine-tuning: `UNVERIFIED` (public pipeline ran one seed). Literature and experience suggest ±0.01 on OOF at this size (`YOUR HYPOTHESIS`); run ≥ 2 seeds before trusting < 0.01 gains.
- Bootstrap CI of OOF differences (dup-group resampling, 2,000–3,000 draws): ±0.017–0.02 half-width for blends vs the FT ensemble.
- Public-subset noise: §6 (SD 0.03–0.05).
- Rule: **do not accept improvements smaller than ≈ 0.005 OOF without P(>0) > 0.9 on the paired bootstrap, and never on public deltas < 0.03.**

---

## 20. Experiments run for this report

All on CPU/Apple-MPS (no GPU fine-tuning was run here). Logs in `research/work/`; rows X00–X14 in `research/experiments.csv`.

| ID | Experiment | Result | Decision |
|---|---|---|---|
| X00 | Near-duplicate scan | 227 groups / 467 rows; 78 test rows with train twin; twin label agreement 300/301 | KEEP (defines `dup_group`) |
| X01 | Adversarial validation | AUC 0.454; KS all p > 0.08 | KEEP (no shift) |
| X02 | Speaker x-vector probe / clustering | F1 0.55 / 0.47; no stable speaker clusters | REJECT speaker-grouped CV |
| X03 | WavLM Base+ per-layer frozen probe | best L7: 0.841 random / 0.753 dup-group | KEEP band 6–8 |
| X04 | WavLM Large per-layer | L16 0.856 / 0.781; L12 dup-group 0.784 | KEEP band 12–18 |
| X05 | XLS-R 300M per-layer | L16 0.866 / **0.802** | KEEP band 10–18 (best) |
| X06 | Verify public FT pipeline | OOF 0.8620; twin 1.000 / non-twin 0.809; public 0.87918 ≈ decomposition 0.880 | KEEP as baseline |
| X07 | Twin override on current submission | 0/78 changes | KEEP as safety net only |
| X08 | FT ens + WavLM-B+ LR (50/50) | 0.8803; Δ +0.0185 [+0.0018, +0.0363]; fixes 31 / breaks 16 rows; correctness corr 0.51 | KEEP |
| X09 | 3 frozen LR families averaged | 0.8801; non-twin 0.837 | KEEP |
| X10 | **FT ens + 3 frozen LR, equal weights** | **0.8814; Δ +0.0194 [+0.0027, +0.0372], P = 0.988; est. test 0.896** | **KEEP (final candidate #1)** |
| X11 | LR C ∈ {0.01, 0.05, 0.2} | differences ≤ 0.007 | REJECT tuning |
| X12 | Public-LB noise simulation | SD 0.032–0.047 | KEEP as decision rule |
| X13 | External Hindi dataset overlap (vishlb 3,200 + swastikatinkhede Hindi 250) | 0 matches; max NCC 0.34 / 0.26 | KEEP (both usable for E12) |
| X14 | Source sample repo overlap | 20/20 matched, 5 test | REJECT (leakage) |

---

## 21. Experiment roadmap E0–E13

Every experiment uses the frozen Track-A folds (`train_with_folds.csv`) and Track-B folds (dup-group SGKF seed 0), reports both, and is compared with a dup-group paired bootstrap against its parent. Full fields (hypothesis, change, keep/reject criteria) in `research/experiments.csv`.

| ID | Parent | Hypothesis | Exact change | Model / features | Seeds | Exp. CV impact | Exp. public | Exp. private | Overfit risk | Leakage risk | Compute | Keep if | Reject if |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | — | Public FT pipeline is the reliable baseline | none (verified from outputs) | WavLM B+/L FT | 42 | 0.862 OOF | 0.879 | ≈ 0.88 | Low | Low | done | OOF within 0.01 | — |
| E1 | E0 | Two-track validation measures the hidden test | freeze folds; twin split; bootstrap | any | — | — | — | — | — | — | done | — | — |
| E2 | E1 | Prosody adds signal | eGeMAPS (88) + Parselmouth → LR/LGBM family | openSMILE | 0–2 | +0–0.01 blend | noise | +0–0.01 | Low | Low | CPU min | blend non-twin +≥0.005, P > 0.8, both tracks | otherwise |
| E3 | E1 | Fine-tuned XLS-R > fine-tuned WavLM | public 01 notebook with `--models facebook/wav2vec2-xls-r-300m`, backbone LR ≈ 3e-5 | XLS-R 300M FT | 42 (+2026) | +0.01–0.03 non-twin | noise | +0.01–0.02 | Medium (1 seed) | Low | 2–3 h T4 | OOF non-twin ≥ 0.81 and blend +≥0.005 | below either |
| E4 | E3 | Small HPO helps | layer band ± 2, C, FT LR × {0.5, 2} | — | 0 | ≤ 0.007 | noise | ≈ 0 | **High** | Low | min–h | gain > 2 × CV SD | default |
| E5 | E1 | Whisper / emotion2vec add diversity | frozen probes: Whisper-large-v3 (trim padding), emotion2vec+ large, w2v-BERT 2.0 | LR on band means | 0–2 | +0.005–0.02 blend | noise | +0.005–0.015 | Low | Low | 30–90 min GPU each | dup-group ≥ 0.78 and blend +≥0.005 | otherwise |
| E6 | X10 | Equal-weight family blend | mean of probabilities | — | — | +0.019 (done) | noise | +0.01–0.02 | Low | Low | min | done | — |
| E7 | E6 | Logit/geometric mean or class bias helps | nested 5×5 | — | — | < 0.005 | noise | ≈ 0 | Medium | Low | min | nested +≥0.005, P > 0.9 | otherwise |
| E8 | E6 | Stacker beats equal weights | multinomial LR (C ≤ 0.1) on OOF probs, nested | — | — | ≤ 0.01 | noise | ≈ 0 | **High** | Medium | min | nested +≥0.01 | otherwise |
| E9 | E6 | Conservative test pseudo-labels help LR families | add non-twin test rows with blend p ≥ 0.9 and all families agree; retrain LR only; simulate inside CV first | — | — | +0–0.01 | noise | +0–0.01 | Medium | Medium | min | in-CV simulation +≥0.005 non-twin | otherwise |
| E10 | E6 | Final robust ensemble | equal blend of families passing criteria + twin override | — | — | best OOF | — | best expected | Low | Low | min | — | — |
| E11 | X07 | Twin override safety net | apply to every submission | rule | — | 0 for FT | 0 | 0 to + | None | Low | s | always | — |
| E12 | E1 | Public 4-class corpora help LR families | add UrduSER / SUBESCO / CREMA-D / vishlb (if no overlap) to LR training, weight 0.3 | LR families | 0–2 | +0–0.01 | noise | +0–0.01 | Low | Medium (licence/overlap) | 1–2 h | dup-group +≥0.01 | otherwise |
| E13 | E1 | A few labels are wrong | cleanlab on blend OOF; inspect top 15; drop only conflicting duplicates | — | — | ≈ 0 | — | ≈ 0 | Medium | Low | min | clear errors found | none found |

---

## 22. Top 10 experiments

| Priority | Experiment | Why | Expected upside (non-twin → overall) | Private robustness | Compute | Risk | Dependencies |
|---|---|---|---|---|---|---|---|
| 1 | **Submit X10 (4-family equal blend)** | significant OOF gain, no tuning | +0.019 OOF → est. +0.016 test | High | none | Low | none (file ready) |
| 2 | **E3 XLS-R 300M fine-tune** (fork public 01 notebook) | best frozen encoder (+0.05 dup-group vs WavLM B+ frozen), Hindi in pretraining | +0.01–0.03 non-twin | Medium–High | 2–3 h T4 | Medium (1 seed) | Kaggle GPU |
| 3 | **E5 Whisper-large-v3 encoder frozen family** | EmoBox top encoder; Hindi in pretraining | +0.005–0.02 blend | High | ~45 min T4 | Low | none |
| 4 | **E5 emotion2vec+ large frozen family** | emotion-specialised, 4 classes covered | +0.005–0.02 blend | High | ~30 min | Low (licence check) | FunASR install |
| 5 | E10 re-blend with equal weights + twin override | combine whatever passes | cumulative | High | min | Low | 2–4 |
| 6 | E2 eGeMAPS/prosody family | cheap, different inductive bias | +0–0.01 | High | CPU | Low | none |
| 7 | E3b second seed of the best fine-tune | halves seed noise | +0.003–0.01 | High | 2–3 h | Low | 2 |
| 8 | E12 Indo-Aryan/English 4-class auxiliary data for LR families | more distinct recordings | +0–0.01 | Medium | 1–2 h | Medium | overlap check |
| 9 | E9 conservative pseudo-labels (simulated in CV first) | uses 131 non-twin test clips | +0–0.01 | Medium | min | Medium | 5 |
| 10 | E7 nested class-bias / logit averaging check | confirm nothing is left on the table | < 0.005 | — | min | Low | 5 |

---

## 23. Final comparison table

Only verified or measured scores; everything else `UNVERIFIED`.

| Rank | Approach / notebook | Public LB | CV (pooled OOF Macro-F1) | Model | Key technique | Overfit risk | Reproducible? |
|---|---|---|---|---|---|---|---|
| — | Top public teams (SoloSid69, thisray, Rin) | 0.96544 (VERIFIED) | UNVERIFIED | UNVERIFIED | UNVERIFIED (no public code) | UNVERIFIED | No |
| 1 | **X10: FT ensemble + 3 frozen LR, equal weights (this report)** | not submitted | **0.8814** (MEASURED, public-pipeline folds) | WavLM B+/L FT + LR on WavLM B+ / WavLM L / XLS-R bands | family diversity, no tuned weights | Low | Yes |
| 2 | X09: 3 frozen LR families (this report) | not submitted | 0.8801 (MEASURED) | LR on frozen SSL bands | layer-band probes | Low | Yes (CPU/MPS ≈ 1 h) |
| 3 | X08: FT ensemble + WavLM B+ LR | not submitted | 0.8803 (MEASURED) | — | 2-family blend | Low–Medium | Yes |
| 4 | X05: XLS-R L10–18 LR alone | not submitted | 0.8650 (MEASURED; dup-group 0.80) | frozen XLS-R | — | Low | Yes |
| 5 | Family A: kunalinfinite 01/02 (your current submission) | **0.87918** (VERIFIED) | **0.8620** (VERIFIED, optimistic) | WavLM B+ & L FT | layer weights + ASP + LLRD | Low | Yes |
| 6 | Family D: kunalverma2512 (GitHub) | 0.76689 team (PARTICIPANT CLAIM) | UNVERIFIED | Whisper-base frozen + LR/SVM/XGB | soft vote | Low | Medium |
| 7 | Family C: Shivam baseline | 0.74925 (VERIFIED) | UNVERIFIED | XGBoost on librosa stats | — | Low | Yes |
| 8 | Family B: Avik Das | 0.72790 (VERIFIED) | UNVERIFIED | MLP/LGBM/XGB + SUPERB-ER | OOF-tuned divisors | Medium | Partly |

**Answers to the requested summary questions.**
- **Best current public approach:** Family A (WavLM Base+ & Large fine-tunes), public 0.87918.
- **Best reproducible baseline:** same (verified OOF 0.862).
- **Best validation strategy:** two-track (random folds = LB proxy with twin decomposition; dup-group SGKF = novel-clip selection), pooled OOF, dup-group bootstrap (§15).
- **Biggest bottleneck:** non-twin generalization from 672 distinct training recordings (826 rows); public LB too small to guide choices.
- **Most promising unexplored ideas:** XLS-R fine-tune; Whisper-large-v3 and emotion2vec+ frozen families; equal-weight multi-family blending.
- **Best single-model strategy:** XLS-R 300M — frozen band probe now (0.865 OOF), fine-tune next with the public pipeline (E3), LP-FT or LoRA as alternatives.
- **Best ensemble strategy:** equal-weight probability mean over *families* (fine-tuned SSL, frozen-probe per encoder), twin override, no tuned weights.
- **Best pseudo-label strategy:** only if the in-CV simulation shows gain: class-balanced, high-confidence (≥ 0.9, all families agree), non-twin test rows, LR families only (E9).
- **Biggest leakage risks:** source sample repo (5 test labels), twins inside CV, checkpoint selection on validation folds.
- **Biggest private-LB risks:** picking finals by public deltas; single-seed fine-tunes; tuned thresholds/weights; licence-uncertain weights if the host verifies.

---

## 24. Recommended ensemble & final-submission selection

**Recommended ensemble now (`MEASURED HERE`, file ready):** mean of 4 probability sets — (1) public FT ensemble (WavLM Base+ & Large, 10 fold models), (2) LR on WavLM Base+ layers 6–8, (3) LR on WavLM Large layers 12–18, (4) LR on XLS-R 300M layers 10–18 (LR: StandardScaler + `LogisticRegression(C=0.05, class_weight="balanced")`, fit on all 826 rows for test) → argmax → twin override. `research/submissions/sub_ft_wbp_wl_xlsr4equal.csv`.

**Target ensemble by the deadline (`YOUR RECOMMENDATION`):** the same recipe plus any family that passes the E3/E5 keep criteria (fine-tuned XLS-R; Whisper-large-v3 frozen; emotion2vec+ frozen), still equal weights over families.

**Final two selections:**
1. **Best expected score:** the largest equal-weight blend whose every family passed its keep criterion (OOF Track A and Track B both improved, bootstrap P > 0.9).
2. **Hedge:** a structurally different but strong submission. If (1) contains new fine-tunes, choose the X10 blend; if (1) is X10, choose the 3-frozen-family LR average (X09; no fine-tuned component). Do not use the current FT-only submission as a final unless nothing else validates — X10 dominates it on OOF.

Submit both early (tie-break = earlier submission).

---

## 25. Approaches to avoid

- **Weak public tricks:** choosing between near-identical submissions by public score (SD 0.03–0.05); resubmitting with small threshold tweaks.
- **Likely leakage / rule violations:** labels from `github.com/Himanshi7104/SRCASW-BhavVaani`; listening to or hand-labelling test clips; any external corpus that overlaps competition audio.
- **Unstable models:** single-seed fine-tunes judged on < 0.01 gains; very large encoders (XLS-R 1B/2B, Whisper-large FT) fully fine-tuned on 826 clips without LoRA/LP-FT.
- **Expensive low-ROI approaches:** audio-LLMs (Qwen2-Audio, Shuka) zero-shot (PolEval baselines ≈ 0.14–0.18 Macro-F1); training CNNs from scratch on spectrograms; heavy augmentation search (EmoAug: pitch/speed gains ≤ 0.2 points with strong SSL).
- **Leaderboard hacks:** probing the public subset with crafted submissions.
- **Redundant ensembles:** more folds/seeds of the same WavLM Base+ fine-tune instead of new families; WavLM Large FT adds little over Base+.
- **Invalid validation:** plain random-KFold accuracy as the selection metric (inflated by twins; accuracy ≠ Macro-F1); thresholds tuned on the OOF used for reporting (Avik Das notebook); speaker-grouped folds on unreliable clusters.
- **Questionable external data:** IEMOCAP/MSP-Podcast/IITKGP (gated); unofficial Kaggle mirrors with relabelled licences; non-commercial weights for final submissions.
- **Noise-amplifying pseudo-labels:** low-confidence or single-model pseudo-labels; pseudo-labelling twin rows (already correct, adds nothing but bias toward sad).
- **Padding mistakes:** Whisper features pooled over 30 s of zero padding (Family D).

---

## 26. Private-leaderboard strategy

1. **Expected-score thinking.** Final = 0.4·public + 0.6·private; both subsets are random draws with ≈ 37% twins. The submission with the best OOF non-twin F1 maximises *expected* final score; the public number is one noisy draw of the same quantity.
2. **Conservative selection.** Equal weights; no tuned thresholds; fixed folds; only families with bootstrap P > 0.9 on both tracks.
3. **CV confidence intervals.** Report every candidate as mean ± bootstrap CI; prefer the candidate whose CI lower bound is highest when means are close.
4. **Seed robustness.** ≥ 2 seeds for any fine-tune that enters the final blend; frozen LR families are deterministic.
5. **Diversity.** Families with different pretraining (English WavLM vs multilingual XLS-R vs Whisper ASR vs emotion2vec) — measured correctness correlation FT vs frozen LR 0.51.
6. **Limited submission feedback.** ≈ 14 submissions left; use ≤ 2/day; each new submission must be a genuinely different candidate.
7. **Holdout discipline.** Never tune on the public LB; never re-split folds.
8. **Domain robustness.** No shift detected; robustness here means robustness to sampling noise, not to domain.

---

## 27. Step-by-step execution plan (Stages 1–10)

Time budget assumes now ≈ 2026-10-10 20:00 UTC, deadline 2026-10-12 18:30 UTC; daily submission counter resets 00:00 UTC.

| Stage | When | Deliverables | Status |
|---|---|---|---|
| 1 Research | done | this report; `external_resources.csv` (453); `TOP_100_RESOURCES.md`; notebooks + discussions read | ✅ |
| 2 Data audit | done | `meta.csv`, `exact_duplicates.csv`, `near_dup_pairs.csv`, `dup_groups.csv`, `audit_report.txt`, overlap checks | ✅ |
| 3 Validation | done | frozen folds (Track A = `train_with_folds.csv`, Track B = dup-group SGKF seed 0), twin decomposition, bootstrap script | ✅ |
| 4 Baseline | done | E0 verified (OOF 0.862 / public 0.879) | ✅ |
| 5 Controlled experiments | done (frozen) | X03–X11 | ✅ |
| 6 Strong single models | Oct 11, 00:00–06:00 UTC | **E3**: fork `kunalinfinite/01-dataverse-training`, set `NOTEBOOK_ARGS = ["--models", "facebook/wav2vec2-xls-r-300m", "--backbone-lr", "3e-5", "--folds-csv", <public train_with_folds.csv>]`, run on T4; save OOF + test probabilities | to do |
| 7 Diversity | Oct 11, in parallel on a second notebook | **E5**: run `research/scripts/embed.py` on Kaggle GPU for `openai/whisper-large-v3` (encoder; trim padded frames) and emotion2vec+ large (FunASR); per-layer probe with `cv_frozen.py`; **E2** eGeMAPS on CPU | to do |
| 8 Ensembling | Oct 11 evening | rerun `blend_study.py` with new families (equal weights), bootstrap vs X10; write new submission files | to do |
| 9 Robustness tests | Oct 12 morning | 2nd seed of the best fine-tune; nested E7 check; class-count sanity; re-check twin override; verify file format (`Id,emotion`, lowercase, 209 rows, test order) | to do |
| 10 Final submissions | Oct 12 before ~16:00 UTC | submit the two finals early; select them in "My Submissions"; register on Unstop; push GitHub repo (code + model links + README) by 14 Oct | to do |

**Submission schedule (`YOUR RECOMMENDATION`):** Oct 10 UTC: X10 (sanity). Oct 11: best blend including E3/E5 results; optionally X09 for diversity information. Oct 12: final best blend; keep 2–3 spare submissions for format fixes.

---

## 28. Direct source links

- Competition: [overview](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/overview) · [data](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/data) · [rules](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/rules) · [leaderboard](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/leaderboard) · [discussion](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/discussion)
- Discussions: [745561](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/discussion/745561) · [747812](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/discussion/747812) · [746206](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/discussion/746206)
- Notebooks: [kunalinfinite/01-dataverse-training](https://www.kaggle.com/code/kunalinfinite/01-dataverse-training) · [kunalinfinite/02-dataverse-inference](https://www.kaggle.com/code/kunalinfinite/02-dataverse-inference) · [avikdas567/acoustic-source-filter-speech-affect-recognition](https://www.kaggle.com/code/avikdas567/acoustic-source-filter-speech-affect-recognition) · [shivamgravity/dataverse-dehs-baseline](https://www.kaggle.com/code/shivamgravity/dataverse-dehs-baseline)
- Participant / source repos: [kunalverma2512/DataVerse](https://github.com/kunalverma2512/DataVerse) · [Himanshi7104/SRCASW-BhavVaani](https://github.com/Himanshi7104/SRCASW-BhavVaani) (audit only)
- Models used here: [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus) · [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large) · [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) · [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv)
- External datasets checked: [vishlb/speech-emotion-recognition-hindi](https://www.kaggle.com/datasets/vishlb/speech-emotion-recognition-hindi) · [swastikatinkhede/multilingual-speech-dataset](https://www.kaggle.com/datasets/swastikatinkhede/multilingual-speech-dataset)
- All other sources: `research/external_resources.csv` (url, paper_url, github_url, dataset_url, model_url columns).

---

## 29. Final conclusions

1. This competition is **two problems in one**: 37% of test clips are re-processed copies of training clips (trivially solved by any model), and 63% are genuinely new recordings where models differ. Optimise and validate on the second; report both.
2. The public leaderboard is **too small to rank strong submissions** (SD 0.03–0.05; top-10 spread ≈ 2–4 errors). With 0.6 weight on private, the right objective is OOF non-twin Macro-F1 under fixed folds with bootstrap CIs.
3. **Diversity beats a single stronger model here.** Frozen multilingual SSL probes (XLS-R best) are as good as the public fine-tuned WavLM ensemble and are only half-correlated with it; an untuned equal blend gives a statistically supported +0.019 OOF (est. test ≈ 0.896 vs 0.880).
4. **Highest-value next steps:** XLS-R fine-tune, Whisper-large-v3 and emotion2vec+ frozen families, equal-weight re-blend, two early final selections, Unstop registration and a reproducible GitHub repo.
5. **Do not touch** the source sample repo's labels or any test-labelling shortcut; do not tune on public.

---

## 30. Reproducibility: files produced

| Path | Content |
|---|---|
| `DataVerse_Hindi_SER_Deep_Analysis.md` | this report |
| `research/external_resources.csv` | 453 resources, 23 columns |
| `research/TOP_100_RESOURCES.md` | ranked Top 100 |
| `research/public_notebooks.csv` | notebook catalogue |
| `research/important_discussions.csv` | all forum threads |
| `research/experiments.csv` | X00–X14 (run) + E0–E13 (planned) |
| `research/submissions/*.csv` | candidate submissions (not submitted) |
| `research/scripts/*.py` | all analysis code (run with `python -I script.py …`) |
| `research/work/` | logs, OOF-derived tables, dup groups, embeddings (`emb/*.npy`), downloaded public-notebook outputs |
| `research/official/` | competition pages, discussions JSON, leaderboard snapshot, kernel list |
| `research/notebooks/` | public notebook sources |
| `data/` | competition data |

Environment: Python 3.12, torch 2.x (Apple MPS), transformers ≥ 4.45, scikit-learn, Kaggle CLI 2.2.4; embeddings: `embed.py <data_root> research/work/meta.csv <hf_model> <tag> research/work/emb`.
